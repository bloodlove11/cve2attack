"""Shared Live ATT&CK inference pipeline (Chat + golden ``mode=agent``).

CVE→ATT&CK path used by golden eval ``mode=agent`` and the Chat product surface,
so demos and eval numbers come from the same code:

1. Enrich CVE (Zenodo / NVD); never CTID gold for the target CVE
2. ATT&CK technique-doc RAG (lexical / candidate T-IDs)
3. Similarity neighbor ICL (k=3 labeled neighbors, target excluded)
4. Two-head LLM predict (exploitation + primary_impact)
5. Exploitation-refine LLM call (second LLM pass on the weak head)
6. Candidate-constrain + grounded evidence-gate + T-ID postprocess

The pipeline does not look up or inject CTID gold for the target CVE, and the
final technique heads always come from LLM output (filtered, never replaced by
a classifier or KB lookup).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from evals.golden.ctid_index import build_few_shot_exemplars
from evals.golden.eval_rag import (
    build_attack_llm_context,
    neighbor_technique_ids,
    neighbors_as_few_shot,
    retrieve_labeled_neighbors,
)
from evals.golden.nvd_enrich import get_cve_enrichment

_TECH_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# High-prior false positives: require grounded evidence quote or drop in postprocess
HIGH_PRIOR_FP_IDS = frozenset({"T1190", "T1203", "T1059", "T1055", "T1068"})
# Minimum length for an evidence quote to count as grounded (blocks "." / "remote")
MIN_GROUNDED_QUOTE_LEN = 8
# Exploitation refine may keep at most this many IDs (precision > coverage)
MAX_EXPLOIT_REFINE = 2
# Bump when gates / ICL / candidate filters change. Of-record CIRCL 0.504
# artifacts predate this version (no enterprise filter / CVE-only grounding).
LIVE_PIPELINE_VERSION = "3"
_EVIDENCE_GATE_SYSTEM = (
    "Before emitting a technique, cite ≤1 short quote copied verbatim from the "
    "CVE description (not ATT&CK technique docs, not a paraphrase). "
    "High-prior false positives T1190, T1203, T1059, T1055, T1068 need a "
    "verbatim supporting quote ≥8 characters from the CVE description; "
    "otherwise omit. "
    "Put access/trigger IDs only in exploitation_techniques; put post-exploit "
    "impact (T1005, T1499.xxx, T1557, T1574, T1565, …) only in primary_impact. "
    "Prefer sub-techniques when docs name them. "
    "Emit ONLY technique IDs from the provided candidate set "
    "(parent↔sub of a candidate is OK); do not invent off-list T-IDs."
)

LlmChatFn = Callable[[list[dict[str, str]]], dict[str, Any]]


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text (raw or fenced)."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def format_few_shot(exemplars: list[dict[str, Any]]) -> str:
    """Format neighbor / leave-one-out exemplars as two-head soft-bias text."""
    if not exemplars:
        return ""
    lines = [
        "Few-shot exemplars from similar labeled CTID neighbors "
        "(other CVEs only — do not copy labels blindly; never the target id).",
        "Soft bias only: overlapping neighbor technique IDs are hints when "
        "evidence supports them — still emit ONLY from the candidate set.",
        "CTID splits exploitation (how access is gained) vs primary_impact "
        "(data access, DoS, AiTM, hijack, etc.):",
    ]
    for ex in exemplars:
        techs = ex.get("techniques") or {}
        exploitation = list(techs.get("exploitation") or [])
        primary = list(techs.get("primary") or [])
        if not exploitation and not primary:
            # CTID sometimes parks IDs in uncategorized/secondary; still emit
            # two-head lines so the agent sees the exploitation / impact schema.
            uncat = list(techs.get("uncategorized") or [])
            secondary = list(techs.get("secondary") or [])
            if uncat or secondary:
                exploitation = uncat
                primary = secondary
            else:
                flat = list(ex.get("attack_techniques") or [])
                flat_s = ", ".join(flat) if flat else "(none)"
                lines.append(
                    f"- {ex.get('cve_id')}: attack_techniques=[{flat_s}]; "
                    "exploitation=(none — unlabeled gold); "
                    "primary_impact=(none — unlabeled gold)"
                )
                continue
        ex_s = ", ".join(exploitation) if exploitation else "(none)"
        pi_s = ", ".join(primary) if primary else "(none)"
        lines.append(
            f"- {ex.get('cve_id')}: exploitation=[{ex_s}]; primary_impact=[{pi_s}]"
        )
    lines.append(
        "Prefer technique ID style like the exemplars (e.g. T1190, T1059). "
        "Predict for the target CVE only."
    )
    return "\n".join(lines)


def _enrichment_prompt_parts(enrichment: dict[str, Any] | None) -> list[str]:
    parts: list[str] = []
    if not enrichment:
        return parts
    desc = (enrichment.get("description") or "").strip()
    if desc:
        if len(desc) > 900:
            desc = desc[:899] + "…"
        parts.append(f"CVE description: {desc}")
    cwes = enrichment.get("cwes") or []
    if cwes:
        parts.append("CWEs: " + ", ".join(str(c) for c in cwes))
    if enrichment.get("cvss") is not None:
        parts.append(f"CVSS: {enrichment.get('cvss')}")
    return parts


def build_attack_prompt(
    inp: dict[str, Any],
    *,
    few_shot: list[dict[str, Any]] | None = None,
    rag_context: str | None = None,
    candidate_ids: list[str] | None = None,
    enrichment: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """System+user messages for two-head Live ATT&CK prediction."""
    cve_id = inp.get("cve_id", "")
    system = (
        "You map CVEs to MITRE ATT&CK technique IDs using CTID's two-head split. "
        "Reply with ONLY a JSON object of the form "
        '{"exploitation_techniques": ["T...."], '
        '"primary_impact": ["T...."], '
        '"attack_techniques": ["T...."], '
        '"rationale": "optional"}. '
        "exploitation_techniques = how the adversary gains access / triggers "
        "the vulnerability (Initial Access / Execution — pick the specific "
        "technique the CVE description supports; do not default to T1190). "
        "primary_impact = CTID-style impacts after exploitation "
        "(data access, DoS, AiTM, execution hijack, etc.). "
        "attack_techniques must be the union of the two lists (deduped) "
        "for backward compatibility. "
        "When a candidate set is provided, emit ONLY IDs from that set "
        "(or a parent/sub-technique of a candidate). "
        "Do NOT look up or invent CTID gold labels for the target CVE. "
        "Output only valid MITRE ATT&CK technique IDs (T#### or T####.###); "
        "do not invent non-T IDs. Up to 8 IDs total across heads. "
        "Optional: attach evidence per technique as "
        '{"id":"T....","evidence":"short quote"} '
        'or an evidence map {"T....": "quote"}. '
        + _EVIDENCE_GATE_SYSTEM
    )
    parts = [
        f"Predict MITRE ATT&CK techniques for {cve_id} (CTID exploitation + primary_impact).",
        "Ground the prediction in the CVE description and technique docs below.",
        "Separate exploitation (access / trigger) from primary_impact "
        "(data access, DoS, AiTM, hijack, etc.).",
    ]
    parts.extend(_enrichment_prompt_parts(enrichment))
    if candidate_ids:
        parts.append(
            "ONLY emit technique IDs from this candidate set "
            "(parent↔sub of a listed ID is OK): "
            + ", ".join(candidate_ids)
        )
    if rag_context and rag_context.strip():
        parts.append("")
        parts.append(rag_context.strip())
    fs = format_few_shot(few_shot or [])
    if fs:
        parts.append("")
        parts.append(fs)
    user = "\n".join(parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_exploitation_prompt(
    inp: dict[str, Any],
    *,
    rag_context: str | None = None,
    candidate_ids: list[str] | None = None,
    enrichment: dict[str, Any] | None = None,
    draft_primary_impact: list[str] | None = None,
    draft_exploitation: list[str] | None = None,
) -> list[dict[str, str]]:
    """Second LLM pass focused on the exploitation head (LLM-obligatory refine)."""
    cve_id = inp.get("cve_id", "")
    system = (
        "You specialize in CTID exploitation techniques for a CVE. "
        "Reply with ONLY a JSON object of the form "
        '{"exploitation_techniques": [{"id":"T....","evidence":"verbatim quote"}], '
        '"rationale": "optional"}. '
        "exploitation_techniques = how the adversary gains access / triggers "
        "the vulnerability (Initial Access / Execution). "
        "Do NOT emit primary_impact IDs here. "
        "When a candidate set is provided, emit ONLY IDs from that set "
        "(or a parent/sub of a candidate). "
        "Each ID needs a verbatim quote (≥8 chars) from the CVE description "
        "only — never from ATT&CK technique documentation. "
        "If a draft exploitation list is provided, you may ONLY keep or drop "
        "IDs from that draft (veto / filter); do not invent new exploitation "
        "IDs. If the draft is empty, you may propose up to "
        f"{MAX_EXPLOIT_REFINE} IDs with CVE-description quotes. "
        "Do NOT look up CTID gold for the target CVE. "
        f"At most {MAX_EXPLOIT_REFINE} exploitation IDs. "
        + _EVIDENCE_GATE_SYSTEM
    )
    parts = [
        f"Refine exploitation_techniques for {cve_id}.",
        "Focus only on access / trigger / initial execution — not post-exploit impact.",
        "Evidence quotes must be copied from the CVE description below.",
    ]
    parts.extend(_enrichment_prompt_parts(enrichment))
    if draft_exploitation:
        parts.append(
            "Draft exploitation_techniques (keep/drop only — do not add new IDs): "
            + ", ".join(draft_exploitation)
        )
    elif draft_exploitation is not None:
        parts.append(
            "Draft exploitation_techniques is empty — you may propose up to "
            f"{MAX_EXPLOIT_REFINE} CVE-grounded exploitation IDs."
        )
    if draft_primary_impact:
        parts.append(
            "Already-drafted primary_impact (do not repeat these as exploitation): "
            + ", ".join(draft_primary_impact)
        )
    if candidate_ids:
        parts.append(
            "ONLY emit exploitation IDs from this candidate set "
            "(parent↔sub OK): "
            + ", ".join(candidate_ids)
        )
    if rag_context and rag_context.strip():
        parts.append("")
        parts.append(
            "ATT&CK technique docs (context only — do NOT quote as evidence):\n"
            + rag_context.strip()
        )
    user = "\n".join(parts)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tech_id_from_item(item: Any) -> str | None:
    """Extract a normalized T-ID from a string or ``{"id": ...}`` dict."""
    if item is None:
        return None
    if isinstance(item, dict):
        for key in ("id", "technique", "tech_id", "technique_id"):
            if key in item and item[key] is not None:
                item = item[key]
                break
        else:
            item = str(item)
    s = str(item).strip().upper()
    if not s:
        return None
    if not _TECH_ID_RE.fullmatch(s):
        m2 = re.search(r"T\d{4}(?:\.\d{3})?", s, re.IGNORECASE)
        if not m2:
            return None
        s = m2.group(0).upper()
        if not _TECH_ID_RE.fullmatch(s):
            return None
    from evals.golden.attack_catalog import resolve_technique_id

    return resolve_technique_id(s)


def evidence_from_item(item: Any) -> str | None:
    """Optional short evidence quote attached to a technique dict."""
    if not isinstance(item, dict):
        return None
    for key in ("evidence", "quote", "citation", "support", "reason"):
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def collect_technique_evidence(parsed: dict[str, Any]) -> dict[str, str]:
    """Map technique ID → evidence quote from heads and optional evidence maps."""
    ev: dict[str, str] = {}
    for key in ("evidence", "technique_evidence", "evidence_by_technique"):
        blob = parsed.get(key)
        if not isinstance(blob, dict):
            continue
        for k, v in blob.items():
            tid = tech_id_from_item(k)
            if tid and v is not None and str(v).strip():
                ev[tid] = str(v).strip()
    for head in (
        "exploitation_techniques",
        "primary_impact",
        "attack_techniques",
        "techniques",
    ):
        raw = parsed.get(head)
        items: list[Any]
        if raw is None:
            continue
        if isinstance(raw, (list, tuple, set)):
            items = list(raw)
        else:
            items = [raw]
        for item in items:
            tid = tech_id_from_item(item)
            quote = evidence_from_item(item)
            if tid and quote:
                ev[tid] = quote
    return ev


def normalize_evidence_text(text: str) -> str:
    """Lowercase + collapse whitespace for grounded-quote checks."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def quote_is_grounded(
    quote: str,
    grounding_text: str,
    *,
    min_len: int = MIN_GROUNDED_QUOTE_LEN,
) -> bool:
    """True if ``quote`` is a verbatim (normalized) substring of grounding text.

    Rejects empty/punctuation-only quotes and short tokens like ``"."`` /
    ``"remote"`` that previously bypassed the evidence gate.
    """
    q = normalize_evidence_text(quote)
    g = normalize_evidence_text(grounding_text)
    if not q or not g:
        return False
    if len(q) < min_len:
        return False
    if sum(ch.isalnum() for ch in q) < max(6, min_len - 2):
        return False
    return q in g


def build_grounding_text(
    *,
    enrichment: dict[str, Any] | None = None,
    rag_context: str | None = None,
    extra: str | None = None,
    include_rag: bool = False,
) -> str:
    """Build text used for evidence substring checks.

    Default is CVE description + CWEs only. ATT&CK-doc RAG is excluded
    unless ``include_rag=True`` so the model cannot "prove" a technique by
    quoting the retrieval snippet that named it.
    """
    parts: list[str] = []
    if enrichment:
        desc = (enrichment.get("description") or "").strip()
        if desc:
            parts.append(desc)
        cwes = enrichment.get("cwes") or []
        if cwes:
            parts.append(" ".join(str(c) for c in cwes))
    if include_rag and rag_context and rag_context.strip():
        parts.append(rag_context.strip())
    if extra and extra.strip():
        parts.append(extra.strip())
    return "\n".join(parts)


def filter_enterprise_technique_ids(ids: list[str]) -> list[str]:
    """Drop mobile / ICS technique IDs from a candidate list (Enterprise gold)."""
    if not ids:
        return []
    from evals.golden.attack_catalog import load_current_techniques, resolve_technique_id

    by_id = {t["id"]: t for t in load_current_techniques()}
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        tid = resolve_technique_id(raw) or str(raw or "").strip().upper()
        if not tid or tid in seen:
            continue
        domain = str((by_id.get(tid) or {}).get("domain") or "").lower()
        # Unknown / missing domain: keep (catalog parents may resolve oddly)
        if domain in {"mobile", "ics"}:
            continue
        parent = technique_parent_id(tid)
        parent_domain = str((by_id.get(parent) or {}).get("domain") or "").lower()
        if parent_domain in {"mobile", "ics"}:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def apply_exploit_refine_veto(
    draft_exploitation: list[str],
    refined_exploitation: list[str],
    *,
    max_n: int = MAX_EXPLOIT_REFINE,
    refine_ok: bool = True,
) -> list[str]:
    """Merge refine into draft: veto-only when draft non-empty; else allow fill.

    When the first pass already emitted exploitation IDs, refine may only keep
    a subset (filter FPs). Empty refine output is an all-veto (return ``[]``).
    When ``refine_ok`` is False (parse/call failure), keep the full draft
    uncapped. When the draft is empty, refine may propose up to ``max_n``
    recovery IDs.
    """
    draft = list(draft_exploitation or [])
    refined = list(refined_exploitation or [])
    if not refine_ok:
        return draft
    if not refined:
        return []
    if draft:
        draft_set = set(draft)
        draft_parents = {technique_parent_id(t) for t in draft}
        kept: list[str] = []
        seen: set[str] = set()
        for tid in refined:
            if tid in seen:
                continue
            parent = technique_parent_id(tid)
            if tid in draft_set or parent in draft_set or tid in draft_parents:
                seen.add(tid)
                kept.append(tid)
        return kept[:max_n]
    return refined[:max_n]


def technique_parent_id(tid: str) -> str:
    """Parent technique id (``T1204.002`` → ``T1204``)."""
    return tid.split(".", 1)[0]


def _unique_ids(*groups: list[str], max_n: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for tid in group:
            if tid in seen:
                continue
            seen.add(tid)
            out.append(tid)
            if len(out) >= max_n:
                return out
    return out


def id_in_candidate_set(
    tid: str,
    candidates: set[str] | frozenset[str],
) -> bool:
    """True if ``tid`` is in ``candidates`` or a parent↔sub of a candidate."""
    if not tid or not candidates:
        return False
    if tid in candidates or technique_parent_id(tid) in candidates:
        return True
    return any("." in c and technique_parent_id(c) == tid for c in candidates)


def filter_to_candidates(
    ids: list[str],
    candidate_ids: list[str] | set[str] | frozenset[str] | None,
) -> list[str]:
    """Keep only IDs that match the candidate set (exact or parent↔sub).

    Empty / missing candidate set → pass through unchanged (no constraint).
    """
    if not candidate_ids:
        return list(ids)
    from evals.golden.attack_catalog import resolve_technique_id

    cand = {
        resolved
        for x in candidate_ids
        if x
        for resolved in [resolve_technique_id(x)]
        if resolved
    }
    if not cand:
        return list(ids)
    return [t for t in ids if id_in_candidate_set(t, cand)]


def normalize_attack_techniques(
    raw: Any,
    *,
    prefer_ids: list[str] | set[str] | None = None,
    max_n: int = 8,
) -> list[str]:
    """Keep valid ``T####`` / ``T####.###`` IDs; prefer retrieved candidates first."""
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = re.split(r"[;,\s]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]

    from evals.golden.attack_catalog import resolve_technique_id

    prefer = {
        resolved
        for x in (prefer_ids or [])
        if x
        for resolved in [resolve_technique_id(x)]
        if resolved
    }
    found: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = tech_id_from_item(item)
        if not s or s in seen:
            continue
        seen.add(s)
        found.append(s)

    if not prefer:
        return found[:max_n]

    preferred = [t for t in found if t in prefer]
    extras = [t for t in found if t not in prefer]
    return (preferred + extras)[:max_n]


def drop_ungated_high_prior_fps(
    ids: list[str],
    evidence: dict[str, str],
    *,
    grounding_text: str | None = None,
) -> list[str]:
    """Drop high-prior FP IDs when evidence is missing or not grounded.

    When ``grounding_text`` is provided, the quote must be a normalized
    substring of that text (and meet ``MIN_GROUNDED_QUOTE_LEN``). When
    ``grounding_text`` is ``None``, keep the legacy non-empty-quote rule
    for unit tests / callers that omit grounding.
    """
    out: list[str] = []
    for tid in ids:
        parent = technique_parent_id(tid)
        is_high_prior = tid in HIGH_PRIOR_FP_IDS or parent in HIGH_PRIOR_FP_IDS
        if not is_high_prior:
            out.append(tid)
            continue
        quote = (evidence.get(tid) or evidence.get(parent) or "").strip()
        if not quote:
            continue
        if grounding_text is None:
            out.append(tid)
            continue
        if quote_is_grounded(quote, grounding_text):
            out.append(tid)
    return out


def postprocess_attack_prediction(
    parsed: dict[str, Any],
    *,
    prefer_ids: list[str] | set[str] | None = None,
    max_n: int = 8,
    evidence_gate: bool = True,
    grounding_text: str | None = None,
    constrain_to_candidates: bool = False,
) -> dict[str, list[str]]:
    """Validate T-IDs on CTID heads; build ``attack_techniques`` from union if missing.

    When ``evidence_gate`` is True (default), drop high-prior FP IDs
    (T1190, T1203, T1059, T1055, T1068) that lack grounded evidence.

    When ``constrain_to_candidates`` is True and ``prefer_ids`` is non-empty,
    drop any predicted ID that is not in (or parent↔sub of) the candidate set.
    The LLM still chooses among candidates; we never invent IDs for it.
    """
    evidence = collect_technique_evidence(parsed)
    exploitation = normalize_attack_techniques(
        parsed.get("exploitation_techniques"),
        prefer_ids=prefer_ids,
        max_n=max_n,
    )
    primary_impact = normalize_attack_techniques(
        parsed.get("primary_impact"),
        prefer_ids=prefer_ids,
        max_n=max_n,
    )
    attack = normalize_attack_techniques(
        parsed.get("attack_techniques"),
        prefer_ids=prefer_ids,
        max_n=max_n,
    )
    if evidence_gate:
        exploitation = drop_ungated_high_prior_fps(
            exploitation, evidence, grounding_text=grounding_text
        )
        primary_impact = drop_ungated_high_prior_fps(
            primary_impact, evidence, grounding_text=grounding_text
        )
        attack = drop_ungated_high_prior_fps(
            attack, evidence, grounding_text=grounding_text
        )
    if constrain_to_candidates and prefer_ids:
        exploitation = filter_to_candidates(exploitation, prefer_ids)
        primary_impact = filter_to_candidates(primary_impact, prefer_ids)
        attack = filter_to_candidates(attack, prefer_ids)
    if not attack:
        attack = _unique_ids(exploitation, primary_impact, max_n=max_n)
    return {
        "exploitation_techniques": exploitation,
        "primary_impact": primary_impact,
        "attack_techniques": attack,
    }


def default_llm_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Chat-completion helper used by Live ATT&CK (ExperientialLabs client)."""
    from src.llm.client import chat_completion, get_model
    from src.llm.usage import extract_usage

    resp = chat_completion(messages, temperature=0.0, model=model)
    content = ""
    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        content = str(resp)
    parsed = extract_json_object(content)
    parsed["_raw"] = content
    usage = extract_usage(resp)
    if usage is not None:
        parsed["_usage"] = usage
    parsed["_model"] = model or get_model()
    return parsed


def _resolve_prep_context(
    cid: str,
    enrich: dict[str, Any],
    *,
    use_rag: bool,
    tech_k_eff: int,
    protocol: str | None,
    cache_on: bool,
) -> tuple[dict[str, Any], bool]:
    """Load cached RAG/neighbor prep, or build and cache it fresh.

    Returns ``(prep, prep_from_cache)`` where ``prep`` has keys ``rag_ctx``,
    ``hits``, ``enrich``, ``neighbors``, ``exemplars``, ``cand``.
    """
    from evals.golden.live_cache import cache_get, cache_key, cache_set

    attack_ver = None
    circl_rev = None
    try:
        from evals.golden.attack_catalog import load_revision

        attack_ver = (load_revision() or {}).get("attack_version")
    except Exception:  # noqa: BLE001
        pass
    try:
        from evals.golden.circl_import import HF_REVISION

        circl_rev = HF_REVISION
    except Exception:  # noqa: BLE001
        pass
    prep_key = cache_key(
        {
            "v": 3,  # enterprise filter + CVE-only grounding + parent high-prior
            "pipeline": LIVE_PIPELINE_VERSION,
            "cve_id": cid.strip().upper(),
            "protocol": (protocol or "").strip().lower(),
            "use_rag": bool(use_rag),
            "tech_k": tech_k_eff,
            "desc": (enrich.get("description") or "")[:1200],
            "cwes": list(enrich.get("cwes") or []),
            "cvss": enrich.get("cvss"),
            "enrich_source": enrich.get("source"),
            "attack_version": attack_ver,
            "circl_hf": circl_rev,
        }
    )
    prep_hit = cache_get(prep_key) if cache_on else None
    if isinstance(prep_hit, dict) and prep_hit.get("rag_ctx") is not None:
        return prep_hit, True

    prep = _build_attack_prep(
        cid, enrich, use_rag=use_rag, tech_k_eff=tech_k_eff, protocol=protocol
    )
    if cache_on:
        cache_set(prep_key, prep)
    return prep, False


def _build_attack_prep(
    cid: str,
    enrich: dict[str, Any],
    *,
    use_rag: bool,
    tech_k_eff: int,
    protocol: str | None,
) -> dict[str, Any]:
    """Build RAG context, candidate technique IDs, and neighbor few-shot exemplars."""
    hits: list[dict[str, Any]] = []
    rag_ctx: str | None = None
    if use_rag:
        rag_ctx, hits, enrich = build_attack_llm_context(
            cid,
            enrichment=enrich,
            tech_k=tech_k_eff,
        )
    else:
        from evals.golden.attack_docs import (
            format_technique_context,
            retrieve_techniques,
        )

        desc = (enrich.get("description") or "").strip()
        query = f"{desc} {cid}".strip() or cid
        hits = retrieve_techniques(query, k=tech_k_eff)
        parts = []
        if desc:
            parts.append(f"CVE description: {desc}")
        tech_block = format_technique_context(hits)
        if tech_block:
            parts.append(tech_block)
        rag_ctx = "\n\n".join(parts) if parts else None

    candidate_ids = [str(h.get("id")) for h in hits if h.get("id")]
    neighbors = []
    exemplars = []
    if use_rag:
        neighbors = retrieve_labeled_neighbors(
            cid,
            enrichment=enrich,
            k=3,
            protocol=protocol,
        )
        neighbors = [
            n
            for n in neighbors
            if str(n.get("cve_id") or "").upper() != cid.strip().upper()
        ]
        exemplars = neighbors_as_few_shot(neighbors)
        proto = (protocol or "").strip().lower()
        if not exemplars and proto not in {
            "circl_test",
            "circl",
            "circl_train",
            "circl_gold",
        }:
            exemplars = build_few_shot_exemplars(cid, n=3, seed=42)
    neighbor_ids = neighbor_technique_ids(neighbors) if neighbors else []
    # candidate_ids first (retrieval order), then neighbour ICL ids. A
    # fallback to candidate_ids alone would be dead: it is a subset of
    # what the filter just rejected.
    cand = filter_enterprise_technique_ids(
        list(dict.fromkeys([*candidate_ids, *neighbor_ids]))
    )
    return {
        "rag_ctx": rag_ctx,
        "hits": hits,
        "enrich": enrich,
        "neighbors": neighbors,
        "exemplars": exemplars,
        "cand": cand,
    }


def _refine_exploitation(
    chat: LlmChatFn,
    inp: dict[str, Any],
    heads: dict[str, list[str]],
    *,
    rag_ctx: str | None,
    cand: list[str],
    enrich: dict[str, Any],
    grounding: str,
    resolved_model: str | None,
    usage_acc: dict[str, Any] | None,
) -> tuple[dict[str, list[str]], int, str | None, str | None, dict[str, Any] | None]:
    """Second (exploitation-refine) LLM pass; merges into ``heads`` via veto.

    Mirrors the original inline block, including one quirk kept intentionally:
    if the refine LLM call succeeds but postprocessing then raises,
    ``llm_calls`` / ``usage_acc`` / ``resolved_model`` / ``refine_raw`` are
    already updated from the successful call, but ``heads`` is left unchanged
    (the mutation below never ran). Not "fixed" here, since that would change
    published eval numbers without a dedicated before/after comparison.

    Returns ``(heads, llm_calls, refine_raw, resolved_model, usage_acc)``.
    """
    from src.llm.usage import add_usage, usage_from_chat_payload

    draft_ex = list(heads.get("exploitation_techniques") or [])
    exploit_messages = build_exploitation_prompt(
        inp,
        rag_context=rag_ctx,
        candidate_ids=cand,
        enrichment=enrich,
        draft_primary_impact=heads.get("primary_impact") or [],
        draft_exploitation=draft_ex,
    )
    llm_calls = 1
    refine_raw = None
    try:
        exploit_parsed = chat(exploit_messages)
        llm_calls = 2
        refine_raw = exploit_parsed.get("_raw")
        usage_acc = add_usage(usage_acc, usage_from_chat_payload(exploit_parsed))
        if isinstance(exploit_parsed, dict) and exploit_parsed.get("_model"):
            resolved_model = str(exploit_parsed["_model"])
        refined = postprocess_attack_prediction(
            {
                "exploitation_techniques": exploit_parsed.get(
                    "exploitation_techniques"
                ),
                "primary_impact": [],
                "attack_techniques": [],
                "evidence": exploit_parsed.get("evidence"),
                "technique_evidence": exploit_parsed.get("technique_evidence"),
                "evidence_by_technique": exploit_parsed.get(
                    "evidence_by_technique"
                ),
            },
            prefer_ids=cand,
            max_n=MAX_EXPLOIT_REFINE,
            grounding_text=grounding,
            constrain_to_candidates=bool(cand),
        )
        refined_exploit = apply_exploit_refine_veto(
            draft_ex,
            refined.get("exploitation_techniques") or [],
            max_n=MAX_EXPLOIT_REFINE,
            refine_ok=(
                "exploitation_techniques" in exploit_parsed
                or "exploitation_techniques"
                in extract_json_object(str(exploit_parsed.get("_raw") or ""))
            ),
        )
        if refined_exploit or draft_ex:
            heads["exploitation_techniques"] = refined_exploit
            heads["attack_techniques"] = _unique_ids(
                heads["exploitation_techniques"],
                heads["primary_impact"],
                max_n=8,
            )
    except Exception:  # noqa: BLE001
        pass
    return heads, llm_calls, refine_raw, resolved_model, usage_acc


def predict_cve_to_attack(
    cve_id: str,
    *,
    description: str | None = None,
    use_rag: bool = True,
    llm_chat: LlmChatFn | None = None,
    protocol: str | None = None,
    model: str | None = None,
    refine_exploitation: bool = True,
    use_prep_cache: bool | None = None,
    tech_k: int = 12,
    extra_cwes: list[str] | None = None,
) -> dict[str, Any]:
    """Run the shared Live ATT&CK pipeline for one CVE.

    Same path as golden eval ``mode=agent`` for ``cve_to_attack``:
    enrich → ATT&CK-doc RAG → neighbor ICL → two-head LLM predict →
    optional exploitation-refine LLM call → candidate-constrain +
    grounded evidence-gate.

    The final heads always come from LLM output (filtered). Never looks up
    CTID gold for ``cve_id`` (target).

    When ``protocol`` is ``circl_test``, neighbor ICL uses the CIRCL train
    pool only and hard-excludes all CIRCL test CVE ids.

    Non-LLM prep (enrich context, technique hits, neighbors) is disk-cached
    by default to speed CIRCL re-runs; set ``use_prep_cache=False`` or
    ``LIVE_PREP_CACHE=0`` to disable.
    """
    from evals.golden.live_cache import prep_cache_enabled

    cid = (cve_id or "").strip()
    if not cid:
        return {
            "exploitation_techniques": [],
            "primary_impact": [],
            "attack_techniques": [],
            "mode": "agent_llm_failed",
            "error": "cve_id is required",
        }

    if llm_chat is not None:
        chat = llm_chat
    else:

        def chat(messages: list[dict[str, str]]) -> dict[str, Any]:
            return default_llm_chat(messages, model=model)

    enrich = get_cve_enrichment(cid)
    if extra_cwes and not (enrich.get("cwes") or []):
        enrich = {**enrich, "cwes": [str(x) for x in extra_cwes if x]}
    if not (enrich.get("description") or "").strip() and description:
        enrich = {
            **enrich,
            "description": str(description),
            "source": enrich.get("source") or "case_input",
        }

    tech_k_eff = int(tech_k) if use_rag else 8
    cache_on = prep_cache_enabled(use_prep_cache)
    prep, prep_from_cache = _resolve_prep_context(
        cid,
        enrich,
        use_rag=use_rag,
        tech_k_eff=tech_k_eff,
        protocol=protocol,
        cache_on=cache_on,
    )
    rag_ctx = prep.get("rag_ctx")
    enrich = dict(prep.get("enrich") or enrich)
    exemplars = list(prep.get("exemplars") or [])
    cand = list(prep.get("cand") or [])

    # CVE description only; never gate on ATT&CK-doc RAG snippets
    grounding = build_grounding_text(
        enrichment=enrich, rag_context=rag_ctx, include_rag=False
    )
    # Cached prep may predate the enterprise filter, so re-apply
    cand = filter_enterprise_technique_ids(list(cand or []))
    inp = {"cve_id": cid}
    if description:
        inp["description"] = description
    messages = build_attack_prompt(
        inp,
        few_shot=exemplars,
        rag_context=rag_ctx,
        candidate_ids=cand,
        enrichment=enrich,
    )
    from src.llm.client import get_model
    from src.llm.usage import add_usage, usage_from_chat_payload

    usage_acc = None
    resolved_model = model
    try:
        parsed = chat(messages)
    except Exception as exc:  # noqa: BLE001
        return {
            "exploitation_techniques": [],
            "primary_impact": [],
            "attack_techniques": [],
            "mode": "agent_llm_failed",
            "error": str(exc),
            "enrichment_source": enrich.get("source"),
            "cve_id": cid,
            "pipeline": "live_attack",
            "pipeline_version": LIVE_PIPELINE_VERSION,
            "prep_cache_hit": prep_from_cache,
            "model": resolved_model or get_model(),
        }
    usage_acc = add_usage(usage_acc, usage_from_chat_payload(parsed))
    if isinstance(parsed, dict) and parsed.get("_model"):
        resolved_model = str(parsed["_model"])
    heads = postprocess_attack_prediction(
        parsed,
        prefer_ids=cand,
        grounding_text=grounding,
        constrain_to_candidates=bool(cand),
    )

    llm_calls = 1
    refine_raw = None
    if refine_exploitation:
        heads, llm_calls, refine_raw, resolved_model, usage_acc = _refine_exploitation(
            chat,
            inp,
            heads,
            rag_ctx=rag_ctx,
            cand=cand,
            enrich=enrich,
            grounding=grounding,
            resolved_model=resolved_model,
            usage_acc=usage_acc,
        )

    out = {
        "cve_id": cid,
        "exploitation_techniques": heads["exploitation_techniques"],
        "primary_impact": heads["primary_impact"],
        "attack_techniques": heads["attack_techniques"],
        "mode": "agent_llm",
        "enrichment_source": enrich.get("source"),
        "pipeline": "live_attack",
        "pipeline_version": LIVE_PIPELINE_VERSION,
        "llm_calls": llm_calls,
        "refine_exploitation": bool(refine_exploitation),
        "prep_cache_hit": prep_from_cache,
        "model": resolved_model or get_model(),
        "_raw": parsed.get("_raw"),
    }
    if usage_acc is not None:
        out["usage"] = {
            "prompt_tokens": usage_acc.get("prompt_tokens", 0),
            "completion_tokens": usage_acc.get("completion_tokens", 0),
            "total_tokens": usage_acc.get("total_tokens", 0),
            "cost": usage_acc.get("cost"),
            "n_calls": usage_acc.get("n_calls", llm_calls),
            "n_calls_with_cost": usage_acc.get("n_calls_with_cost", 0),
        }
    if refine_raw is not None:
        out["_raw_exploitation"] = refine_raw
    if protocol:
        out["protocol"] = protocol
    return out
