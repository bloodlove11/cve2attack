"""Eval-time retrieval helpers for Live agent golden evals.

The Live-agent ATT&CK path prefers ATT&CK technique-documentation retrieval plus
NVD/Zenodo CVE enrichment over the target CVE's CTID gold mapping. Similarity-based
leave-one-out neighbors (NVD description + CWEs query) supply two-head few-shot ICL
and a soft ``prefer_ids`` bias, never target gold. Under ``protocol=circl_test``
the neighbor pool is CIRCL train only.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from evals.golden.attack_docs import (
    format_technique_context,
    retrieve_techniques,
)
from evals.golden.ctid_index import get_index, retrieve, technique_prior
from evals.golden.nvd_enrich import get_cve_enrichment

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_BM25_K1 = 1.5
_BM25_B = 0.75


def build_attack_context(
    cve_id: str,
    *,
    k: int = 5,
    exclude_cve: bool = True,
) -> str:
    """Build compact CTID-neighbor ATT&CK context (leave-one-out by default).

    Kept for Hybrid LLM fallback / optional secondary neighbor blocks.
    Live agent prefers :func:`build_attack_llm_context`.
    """
    target = (cve_id or "").strip().upper()
    if not target:
        return ""

    excluded = {target} if exclude_cve else set()
    neighbors = retrieve(
        cve_id=target,
        query_text=f"{target} ATT&CK techniques exploitation",
        k=k,
        exclude_cve_ids=excluded,
        allow_exact=not exclude_cve,
    )

    lines: list[str] = [
        "### Retrieved CTID neighbor mappings (leave-one-out; not the target CVE)",
    ]
    if neighbors:
        for hit in neighbors:
            techs = hit.get("attack_techniques") or []
            tech_s = ", ".join(techs) if techs else "(none)"
            phase = hit.get("phase") or ""
            score = hit.get("score")
            extra = []
            if phase:
                extra.append(str(phase))
            if score is not None and hit.get("match") != "exact":
                extra.append(f"score={score}")
            suffix = f" ({'; '.join(extra)})" if extra else ""
            lines.append(f"- {hit.get('cve_id')}: [{tech_s}]{suffix}")
    else:
        lines.append("- (no neighbor hits)")

    priors = technique_prior(top_n=8)
    if priors:
        lines.append("")
        lines.append(
            "### Global CTID technique frequency hint (corpus prior, not a label)"
        )
        lines.append("- Common techniques: " + ", ".join(priors))

    lines.append("")
    lines.append(
        "Prefer technique IDs that appear among retrieved CTID neighbors when "
        "they are relevant to the target CVE; still output only valid T-IDs."
    )
    return "\n".join(lines).strip()


def build_attack_llm_context(
    cve_id: str,
    *,
    enrichment: dict[str, Any] | None = None,
    tech_k: int = 12,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Primary Live-agent ATT&CK context: NVD/Zenodo enrichment + ATT&CK docs.

    Does not inject the target CVE's own CTID techniques.
    Returns ``(context_str, technique_hits, enrichment_dict)``.
    """
    target = (cve_id or "").strip().upper()
    enrich = enrichment if enrichment is not None else get_cve_enrichment(target)
    description = (enrich.get("description") or "").strip()
    cwes = enrich.get("cwes") or []
    cvss = enrich.get("cvss")

    query_parts = [description, target]
    if cwes:
        query_parts.append(" ".join(str(c) for c in cwes))
    query = " ".join(p for p in query_parts if p).strip() or f"{target} exploitation"
    hits = retrieve_techniques(query, k=tech_k)

    lines: list[str] = ["### CVE enrichment (not CTID gold labels)"]
    lines.append(f"- cve_id: {target}")
    if description:
        desc = description if len(description) <= 900 else description[:899] + "…"
        lines.append(f"- description: {desc}")
    else:
        lines.append("- description: (unavailable offline)")
    if cwes:
        lines.append(f"- cwes: {', '.join(str(c) for c in cwes)}")
    if cvss is not None:
        lines.append(f"- cvss: {cvss}")
    src = enrich.get("source")
    if src:
        lines.append(f"- enrichment_source: {src}")

    tech_block = format_technique_context(hits)
    if tech_block:
        lines.append("")
        lines.append(tech_block)

    return "\n".join(lines).strip(), hits, enrich


def build_triage_context(case_input: dict) -> str:
    """Restate triage signals + short advisory policy hint."""
    inp = case_input or {}
    kev = inp.get("in_kev", inp.get("kev", False))
    cvss = inp.get("cvss_score", inp.get("cvss"))
    epss = inp.get("epss")
    critical = inp.get("critical_asset", inp.get("critical", False))
    asset = inp.get("asset_context", inp.get("asset"))
    description = inp.get("description")

    lines: list[str] = [
        "### Structured triage signals",
        f"- in_kev / KEV: {kev}",
        f"- cvss_score: {cvss}",
        f"- epss: {epss}",
        f"- critical_asset: {critical}",
    ]
    if asset is not None:
        lines.append(f"- asset_context: {asset}")
    if description:
        desc = str(description).strip()
        if len(desc) > 400:
            desc = desc[:399] + "…"
        lines.append(f"- description: {desc}")

    lines.extend(
        [
            "",
            "### Policy hint (advisory guidance — not a forced label)",
            "- KEV / known exploited → lean ACT (remediate now).",
            "- High EPSS (≥ ~0.3) with high CVSS or critical asset → ACT or ATTEND.",
            "- Critical asset + CVSS ≥ ~7 (even with low EPSS) → lean ATTEND.",
            "- Otherwise → TRACK (monitor).",
            "Use the numeric signals above; this hint is advisory only.",
        ]
    )

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Similarity-based labeled neighbor ICL (Live-safe; never target CTID gold)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2]


def _make_neighbor_doc(rec: dict[str, Any], enrich: dict[str, Any]) -> dict[str, Any]:
    text = _neighbor_doc_text(rec, enrich)
    tokens = _tokenize(text)
    return {
        "cve_id": rec["cve_id"],
        "text": text,
        "tokens": tokens,
        "tf": Counter(tokens),
        "attack_techniques": list(rec["attack_techniques"]),
        "techniques": {k: list(v) for k, v in rec["techniques"].items()},
        "phase": rec.get("phase"),
        "has_enrichment": bool((enrich.get("description") or "").strip()),
        "source": rec.get("source") or enrich.get("source"),
    }


def _neighbor_doc_text(rec: dict[str, Any], enrich: dict[str, Any]) -> str:
    """Searchable text for neighbor retrieval: description + CWEs only.

    Intentionally omits gold technique IDs and ATT&CK lexicon expansion so
    BM25/embeddings match on vulnerability text rather than label vocabulary.
    Technique IDs remain available on the returned neighbor records for
    few-shot formatting after retrieval.
    """
    parts: list[str] = []
    desc = (enrich.get("description") or "").strip()
    if desc:
        parts.append(desc)
    cwes = enrich.get("cwes") or []
    if cwes:
        parts.append(" ".join(str(c) for c in cwes))
    # Phase is metadata (not a technique id); keep for light topical signal.
    phase = (rec.get("phase") or "").strip()
    if phase:
        parts.append(phase)
    return " ".join(p for p in parts if p).strip()


@lru_cache(maxsize=1)
def _neighbor_corpus() -> tuple[dict[str, Any], ...]:
    """CTID labeled records with searchable text (offline enrichment only)."""
    _, records = get_index()
    return tuple(
        _make_neighbor_doc(rec, get_cve_enrichment(rec["cve_id"], fetch_nvd=False))
        for rec in records
    )


def clear_neighbor_corpus_cache() -> None:
    """Drop cached neighbor corpora (tests)."""
    _neighbor_corpus.cache_clear()
    _cached_circl_train_neighbor_corpus.cache_clear()


def _circl_train_neighbor_corpus(
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Build BM25-ready neighbor docs from CIRCL train records (descriptions in-row)."""
    out: list[dict[str, Any]] = []
    for rec in records:
        enrich = {
            "cve_id": rec["cve_id"],
            "description": rec.get("description") or "",
            "cwes": [],
            "cvss": None,
            "source": "circl_train",
        }
        # Prefer offline NVD/Zenodo when available for better lexical match
        offline = get_cve_enrichment(rec["cve_id"], fetch_nvd=False)
        if (offline.get("description") or "").strip():
            enrich = offline
        elif (rec.get("description") or "").strip():
            enrich = {
                "cve_id": rec["cve_id"],
                "description": rec.get("description") or "",
                "cwes": list(offline.get("cwes") or []),
                "cvss": offline.get("cvss"),
                "source": "circl_train",
            }
        out.append(_make_neighbor_doc(rec, enrich))
    return tuple(out)


@lru_cache(maxsize=1)
def _cached_circl_train_neighbor_corpus() -> tuple[dict[str, Any], ...]:
    """Memoized CIRCL train corpus.

    The builder takes unhashable dict records, so it cannot be cached directly.
    Rebuilding it per case dominated neighbor retrieval (~0.33s of ~0.52s) and
    the inputs are pinned to an HF revision, so once per process is enough.
    """
    from evals.golden.circl_import import circl_train_neighbor_records

    return _circl_train_neighbor_corpus(circl_train_neighbor_records())




def _bm25_scores(
    query_tokens: list[str],
    corpus: tuple[dict[str, Any], ...],
) -> list[float]:
    """Okapi BM25 over pre-tokenized neighbor docs."""
    n = len(corpus)
    if n == 0 or not query_tokens:
        return [0.0] * n
    avgdl = sum(len(d["tokens"]) for d in corpus) / max(1, n)
    df: Counter[str] = Counter()
    for d in corpus:
        df.update(set(d["tokens"]))
    scores = [0.0] * n
    q_tf = Counter(query_tokens)
    for term, qf in q_tf.items():
        n_qi = df.get(term, 0)
        if n_qi == 0:
            continue
        idf = math.log(1.0 + (n - n_qi + 0.5) / (n_qi + 0.5))
        for i, d in enumerate(corpus):
            f = d["tf"].get(term, 0)
            if f <= 0:
                continue
            dl = len(d["tokens"]) or 1
            denom = f + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / max(avgdl, 1e-9))
            scores[i] += idf * (f * (_BM25_K1 + 1.0) / denom) * qf
    return scores


@lru_cache(maxsize=2)
def _cached_sentence_transformer(model_name: str) -> Any:
    """Optional dense encoder; ImportError if sentence-transformers is absent."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_retrieve(
    query: str,
    corpus: tuple[dict[str, Any], ...],
    *,
    k: int,
    excluded: set[str],
) -> list[dict[str, Any]] | None:
    """Dense retrieval when sentence-transformers is installed; else None."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not query.strip() or not corpus:
        return None
    try:
        model = _cached_sentence_transformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception:  # noqa: BLE001 (optional heavy deps)
        return None
    doc_texts = [d["text"] or d["cve_id"] for d in corpus]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = doc_emb @ q_emb
    order = np.argsort(-scores)
    hits: list[dict[str, Any]] = []
    for idx in order:
        i = int(idx)
        doc = corpus[i]
        cid = doc["cve_id"]
        if cid in excluded:
            continue
        hits.append(_neighbor_hit(doc, float(scores[i]), match="embedding"))
        if len(hits) >= k:
            break
    return hits


def _neighbor_hit(doc: dict[str, Any], score: float, *, match: str) -> dict[str, Any]:
    return {
        "cve_id": doc["cve_id"],
        "attack_techniques": list(doc["attack_techniques"]),
        "techniques": {k: list(v) for k, v in doc["techniques"].items()},
        "phase": doc.get("phase"),
        "source": doc.get("source"),
        "match": match,
        "score": round(float(score), 4),
        "has_enrichment": bool(doc.get("has_enrichment")),
    }


def _lexical_neighbor_row(r: dict[str, Any]) -> dict[str, Any]:
    """Shape a ``ctid_index.retrieve`` row like a ``_neighbor_hit`` record."""
    return {
        "cve_id": r["cve_id"],
        "attack_techniques": list(r.get("attack_techniques") or []),
        "techniques": {
            kk: list(vv) for kk, vv in (r.get("techniques") or {}).items()
        },
        "phase": r.get("phase"),
        "source": r.get("source"),
        "match": r.get("match") or "lexical",
        "score": r.get("score"),
    }


def query_text_from_enrichment(
    cve_id: str,
    enrichment: dict[str, Any] | None = None,
) -> str:
    """Build neighbor-retrieval query from NVD description + CWEs (+ CVE id)."""
    target = (cve_id or "").strip().upper()
    enrich = enrichment if enrichment is not None else get_cve_enrichment(target)
    parts: list[str] = []
    desc = (enrich.get("description") or "").strip()
    if desc:
        parts.append(desc)
    cwes = enrich.get("cwes") or []
    if cwes:
        parts.append(" ".join(str(c) for c in cwes))
    if target:
        parts.append(target)
    return " ".join(p for p in parts if p).strip()


def retrieve_labeled_neighbors(
    cve_id: str,
    *,
    enrichment: dict[str, Any] | None = None,
    k: int = 3,
    use_embeddings: bool | None = None,
    exclude_cve_ids: set[str] | frozenset[str] | None = None,
    protocol: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve ``k`` labeled neighbors using NVD description + CWEs.

    The target CVE is never returned (no gold leak). Prefers dense embeddings
    when available; falls back to BM25. The result is a soft bias only: callers
    must not copy neighbor labels as the answer.

    When ``protocol`` is ``circl_test`` (or ``circl`` / ``circl_train`` pool),
    neighbors are drawn from the CIRCL train split only and every CIRCL test
    CVE id is hard-excluded (in addition to the target).
    """
    target = (cve_id or "").strip().upper()
    if k <= 0:
        return []
    enrich = enrichment if enrichment is not None else (
        get_cve_enrichment(target) if target else {}
    )
    query = query_text_from_enrichment(target, enrich)
    excluded: set[str] = {target} if target else set()
    if exclude_cve_ids:
        excluded |= {str(x).strip().upper() for x in exclude_cve_ids if x}

    proto = (protocol or "").strip().lower()
    use_circl_train = proto in {
        "circl_test",
        "circl",
        "circl_train",
        "circl_gold",
    }
    if use_circl_train:
        from evals.golden.circl_import import circl_test_cve_ids

        excluded |= set(circl_test_cve_ids())
        corpus = _cached_circl_train_neighbor_corpus()
    else:
        corpus = _neighbor_corpus()
    if not corpus:
        return []

    want_embed = bool(use_embeddings)
    if want_embed and query:
        try:
            emb_hits = _embed_retrieve(query, corpus, k=k, excluded=excluded)
            if emb_hits:
                return emb_hits
        except Exception:  # noqa: BLE001 (optional heavy deps / runtime)
            pass

    # BM25 fallback (default offline path)
    q_tokens = _tokenize(query) if query else []
    if not q_tokens:
        if use_circl_train:
            return []
        # Degenerate: year/id Jaccard via existing retrieve helper
        rows = retrieve(
            cve_id=target or None,
            query_text=query or target,
            k=k,
            exclude_cve_ids=excluded,
            allow_exact=False,
        )
        return [
            _lexical_neighbor_row(r) for r in rows if r.get("cve_id") != target
        ][:k]

    scores = _bm25_scores(q_tokens, corpus)
    def _rank_key(i: int) -> tuple:
        doc = corpus[i]
        techs = doc.get("techniques") or {}
        has_heads = 1 if (techs.get("exploitation") or techs.get("primary")) else 0
        has_enrich = 1 if doc.get("has_enrichment") else 0
        # Prefer higher BM25, then two-head labels, then enriched docs
        return (-scores[i], -has_heads, -has_enrich, doc["cve_id"])

    ranked = sorted(range(len(corpus)), key=_rank_key)
    hits: list[dict[str, Any]] = []
    for i in ranked:
        doc = corpus[i]
        if doc["cve_id"] in excluded:
            continue
        if scores[i] <= 0.0:
            continue
        hits.append(_neighbor_hit(doc, scores[i], match="bm25"))
        if len(hits) >= k:
            break
    # If BM25 found nothing useful, fall back to leave-one-out lexical retrieve
    # (CTID corpus). Never do this under CIRCL protocol: train pool only.
    if not hits:
        if use_circl_train:
            return []
        rows = retrieve(
            cve_id=target or None,
            query_text=query or target,
            k=k,
            exclude_cve_ids=excluded,
            allow_exact=False,
        )
        for r in rows:
            if r.get("cve_id") == target:
                continue
            hits.append(_lexical_neighbor_row(r))
            if len(hits) >= k:
                break
    # Final hard filter (target + protocol excludes)
    return [
        h
        for h in hits
        if str(h.get("cve_id") or "").upper() not in excluded
    ][:k]


def neighbor_technique_ids(neighbors: list[dict[str, Any]]) -> list[str]:
    """Collect unique technique IDs from neighbors for soft ``prefer_ids`` bias."""
    out: list[str] = []
    seen: set[str] = set()
    for n in neighbors or []:
        techs = n.get("techniques") or {}
        ordered: list[str] = []
        for key in ("exploitation", "primary", "secondary", "uncategorized", "all"):
            ordered.extend(techs.get(key) or [])
        ordered.extend(n.get("attack_techniques") or [])
        for tid in ordered:
            s = str(tid).strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def neighbors_as_few_shot(neighbors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape neighbor hits for :func:`evals.golden.live_attack.format_few_shot`."""
    out: list[dict[str, Any]] = []
    for n in neighbors or []:
        techs = n.get("techniques") or {}
        out.append(
            {
                "cve_id": n.get("cve_id"),
                "attack_techniques": list(n.get("attack_techniques") or []),
                "techniques": {
                    "primary": list(techs.get("primary") or []),
                    "secondary": list(techs.get("secondary") or []),
                    "exploitation": list(techs.get("exploitation") or []),
                    "uncategorized": list(techs.get("uncategorized") or []),
                    "all": list(
                        techs.get("all") or n.get("attack_techniques") or []
                    ),
                },
            }
        )
    return out
