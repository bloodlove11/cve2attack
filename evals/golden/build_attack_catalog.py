"""Build the compact ATT&CK catalog + revoked-by map from official STIX.

Pin a version (default 19.2) so regen is reproducible. Vendors compact JSON
under ``evals/golden/raw/``, never the 50MB+ STIX bundles.

Examples::

    python -m evals.golden.build_attack_catalog --from-dir /tmp/attack-stix
    python -m evals.golden.build_attack_catalog --download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from evals.golden.attack_catalog import (
    REVISION_PATH,
    REVOKED_BY_PATH,
    TECHNIQUES_PATH,
)

ATTACK_VERSION = "19.2"
STIX_BASE = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master"
DOMAINS: tuple[tuple[str, str], ...] = (
    ("enterprise-attack", "enterprise"),
    ("mobile-attack", "mobile"),
    ("ics-attack", "ics"),
)
DESC_MAX = 360

_CITATION_RE = re.compile(r"\(Citation:[^)]*\)", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS_RE = re.compile(r"\s+")
_TECH_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)


def _tech_id_from_object(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references") or []:
        if not isinstance(ref, dict):
            continue
        eid = str(ref.get("external_id") or "").strip().upper()
        if _TECH_ID_RE.fullmatch(eid):
            return eid
    return None


def clean_description(text: str, *, max_len: int = DESC_MAX) -> str:
    """Strip citations / markdown links and truncate for lexical RAG."""
    s = _CITATION_RE.sub(" ", text or "")
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def parse_stix_bundle(
    bundle: dict[str, Any] | list[Any],
    *,
    domain: str,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Extract current techniques + revoked-by edges from one STIX bundle."""
    objects = bundle.get("objects") if isinstance(bundle, dict) else bundle
    if not isinstance(objects, list):
        raise ValueError("STIX bundle missing objects list")

    by_stix: dict[str, dict[str, Any]] = {}
    patterns: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"domain": domain, "attack_version": "", "n_objects": len(objects)}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        oid = obj.get("id")
        if isinstance(oid, str):
            by_stix[oid] = obj
        if obj.get("type") == "x-mitre-collection":
            meta["attack_version"] = str(obj.get("x_mitre_version") or "")
            meta["collection"] = str(obj.get("name") or "")

    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "attack-pattern":
            continue
        tid = _tech_id_from_object(obj)
        if not tid:
            continue
        patterns.append(
            {
                "id": tid,
                "stix_id": obj.get("id"),
                "name": str(obj.get("name") or "").strip(),
                "description": clean_description(str(obj.get("description") or "")),
                "domain": domain,
                "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique") or ("." in tid)),
                "parent_id": tid.split(".", 1)[0] if "." in tid else None,
                "deprecated": bool(obj.get("x_mitre_deprecated")),
                "revoked": bool(obj.get("revoked")),
            }
        )

    revoked_edges: dict[str, str] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "revoked-by":
            continue
        src = by_stix.get(str(obj.get("source_ref") or ""))
        tgt = by_stix.get(str(obj.get("target_ref") or ""))
        if not src or not tgt:
            continue
        if src.get("type") != "attack-pattern" or tgt.get("type") != "attack-pattern":
            continue
        sid = _tech_id_from_object(src)
        did = _tech_id_from_object(tgt)
        if sid and did and sid != did:
            revoked_edges[sid] = did

    current = [p for p in patterns if not p["revoked"]]
    meta["n_attack_patterns"] = len(patterns)
    meta["n_current"] = len(current)
    meta["n_revoked"] = sum(1 for p in patterns if p["revoked"])
    meta["n_revoked_by"] = len(revoked_edges)
    return current, revoked_edges, meta


def _chain_revoked(edges: dict[str, str]) -> dict[str, str]:
    """Follow revoked-by until a terminal (non-revoked or missing) ID."""
    out: dict[str, str] = {}
    for src in edges:
        seen: set[str] = {src}
        cur = src
        while cur in edges and edges[cur] not in seen:
            cur = edges[cur]
            seen.add(cur)
        if cur != src:
            out[src] = cur
    return dict(sorted(out.items()))


def merge_domains(
    parsed: list[tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Union current techniques (first-seen ID wins) and chain revoked-by."""
    techs: list[dict[str, Any]] = []
    seen: set[str] = set()
    edges: dict[str, str] = {}
    metas: list[dict[str, Any]] = []
    for current, revoked_edges, meta in parsed:
        metas.append(meta)
        edges.update(revoked_edges)
        for row in current:
            tid = row["id"]
            if tid in seen:
                continue
            seen.add(tid)
            techs.append(
                {
                    "id": tid,
                    "name": row["name"],
                    "description": row["description"],
                    "domain": row["domain"],
                    "is_subtechnique": row["is_subtechnique"],
                    "parent_id": row["parent_id"],
                    "deprecated": row["deprecated"],
                }
            )
    techs.sort(key=lambda r: r["id"])
    return techs, _chain_revoked(edges), metas


def load_bundle(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def download_bundle(domain_folder: str, version: str, dest: Path) -> Path:
    url = f"{STIX_BASE}/{domain_folder}/{domain_folder}-{version}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned MITRE URL)
        dest.write_bytes(resp.read())
    return dest


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_catalog(
    techniques: list[dict[str, Any]],
    revoked_by: dict[str, str],
    metas: list[dict[str, Any]],
    *,
    version: str = ATTACK_VERSION,
) -> dict[str, Any]:
    """Write vendored catalog files and return the revision record."""
    TECHNIQUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tech_text = json.dumps(techniques, indent=2, ensure_ascii=False) + "\n"
    rev_text = json.dumps(revoked_by, indent=2, ensure_ascii=False) + "\n"
    TECHNIQUES_PATH.write_text(tech_text, encoding="utf-8")
    REVOKED_BY_PATH.write_text(rev_text, encoding="utf-8")
    parents = sum(1 for t in techniques if not t.get("is_subtechnique"))
    revision = {
        "source": "https://github.com/mitre-attack/attack-stix-data",
        "attack_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "domains": list(metas),
        "n_current": len(techniques),
        "n_parents": parents,
        "n_subtechniques": len(techniques) - parents,
        "n_revoked_mapped": len(revoked_by),
        "files": {
            "techniques": TECHNIQUES_PATH.name,
            "revoked_by": REVOKED_BY_PATH.name,
            "techniques_sha256": _sha256_text(tech_text),
            "revoked_by_sha256": _sha256_text(rev_text),
        },
        "urls": {
            folder: f"{STIX_BASE}/{folder}/{folder}-{version}.json"
            for folder, _short in DOMAINS
        },
    }
    REVISION_PATH.write_text(
        json.dumps(revision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return revision


def build_from_dir(stix_dir: Path, *, version: str = ATTACK_VERSION) -> dict[str, Any]:
    parsed: list[tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]] = []
    for folder, short in DOMAINS:
        path = stix_dir / f"{folder}-{version}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing STIX bundle: {path}")
        current, edges, meta = parse_stix_bundle(load_bundle(path), domain=short)
        parsed.append((current, edges, meta))
    techs, revoked, metas = merge_domains(parsed)
    return write_catalog(techs, revoked, metas, version=version)


def build_from_download(
    dest_dir: Path,
    *,
    version: str = ATTACK_VERSION,
) -> dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for folder, _short in DOMAINS:
        download_bundle(folder, version, dest_dir / f"{folder}-{version}.json")
    return build_from_dir(dest_dir, version=version)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build compact ATT&CK catalog from STIX")
    p.add_argument("--version", default=ATTACK_VERSION, help="ATT&CK version (default 19.2)")
    p.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help="Directory with {enterprise,mobile,ics}-attack-{version}.json",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download pinned STIX bundles from attack-stix-data (network)",
    )
    p.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/tmp/attack-stix"),
        help="Where to write downloaded STIX (not vendored)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.from_dir:
        rev = build_from_dir(args.from_dir, version=args.version)
    elif args.download:
        rev = build_from_download(args.download_dir, version=args.version)
    else:
        print("pass --from-dir PATH or --download", file=sys.stderr)
        return 2
    print(
        f"wrote {rev['n_current']} techniques "
        f"({rev['n_parents']} parents, {rev['n_subtechniques']} subs), "
        f"{rev['n_revoked_mapped']} revoked-by maps → {TECHNIQUES_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
