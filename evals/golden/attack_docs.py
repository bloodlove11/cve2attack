"""Compact MITRE ATT&CK technique documentation for Live-agent RAG.

Primary context for ``mode=agent`` CVE→ATT&CK guesses: technique name +
description excerpts from the STIX v19.2 catalog (not the target CVE's CTID
gold mapping).
"""
from __future__ import annotations

import re
from typing import Any

from evals.golden.attack_catalog import (
    clear_catalog_caches,
    load_current_techniques,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 2}


# Lightweight query expansion so CVE prose maps onto ATT&CK vocabulary
_QUERY_SYNONYMS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("remote code", "arbitrary code", "rce", "code execution", "execute arbitrary"),
     ("exploit", "public", "facing", "application", "execution", "remote")),
    (("public facing", "internet facing", "web application", "web server", "unauthenticated"),
     ("public", "facing", "application", "external", "remote")),
    (("command injection", "os command", "arbitrary command", "execute commands"),
     ("command", "scripting", "interpreter", "execution")),
    (("privilege escalation", "elevated privileges", "local privilege"),
     ("privilege", "escalation", "exploitation")),
    (("sql injection", "sqli"),
     ("exploit", "public", "facing", "application", "injection")),
    (("authentication bypass", "broken authentication", "default credentials"),
     ("valid", "accounts", "external", "remote")),
    (("buffer overflow", "memory corruption", "use after free"),
     ("exploitation", "client", "execution", "privilege")),
    (("deserialization", "jndi", "ldap", "log4j", "log4shell"),
     ("exploit", "public", "facing", "application", "execution", "remote")),
    (("phishing", "spearphishing", "malicious email"),
     ("phishing", "user", "execution")),
    (("cross site", "xss", "csrf"),
     ("exploit", "public", "facing", "application")),
]


def _expand_query_tokens(query: str) -> set[str]:
    q = (query or "").lower()
    toks = _tokens(q)
    for needles, extras in _QUERY_SYNONYMS:
        if any(n in q for n in needles):
            toks.update(extras)
            toks.update(_tokens(" ".join(extras)))
    return toks


def load_techniques() -> tuple[dict[str, Any], ...]:
    """Load the vendored STIX technique catalog."""
    return load_current_techniques()


def clear_techniques_cache() -> None:
    """Drop in-memory technique list (tests)."""
    clear_catalog_caches()


def _score(query_tokens: set[str], tech: dict[str, Any]) -> float:
    if not query_tokens:
        return 0.0
    blob_tokens = _tokens(f"{tech.get('id', '')} {tech.get('name', '')} {tech.get('description', '')}")
    if not blob_tokens:
        return 0.0
    inter = len(query_tokens & blob_tokens)
    if not inter:
        return 0.0
    # Jaccard-ish with a slight boost for name token hits
    name_tokens = _tokens(str(tech.get("name") or ""))
    name_hits = len(query_tokens & name_tokens)
    jacc = inter / len(query_tokens | blob_tokens)
    score = jacc + 0.15 * name_hits
    # Prefer well-known initial-access / execution techniques when signals match.
    # Do NOT boost T1210 here: query expansion injects "public/facing/exploit"
    # for generic RCE, which wrongly promotes "Exploitation of Remote Services".
    tid = str(tech.get("id") or "")
    if {"exploit", "public", "facing"} <= query_tokens or (
        "public" in query_tokens and "facing" in query_tokens
    ):
        if tid == "T1190":
            score += 0.35
        elif tid in {"T1203", "T1059", "T1068"}:
            score += 0.12
    if "command" in query_tokens and "scripting" in query_tokens and tid.startswith("T1059"):
        score += 0.2
    if "privilege" in query_tokens and "escalation" in query_tokens and tid == "T1068":
        score += 0.25
    if "valid" in query_tokens and "accounts" in query_tokens and tid == "T1078":
        score += 0.25
    if "phishing" in query_tokens and tid.startswith("T1566"):
        score += 0.25
    return score


def retrieve_techniques(query: str, k: int = 12) -> list[dict[str, Any]]:
    """Lexical retrieve of ATT&CK technique docs by name+description."""
    techs = load_techniques()
    # Prefer Enterprise techniques for CVE→ATT&CK (CIRCL / CTID gold)
    techs = tuple(
        t for t in techs if str(t.get("domain") or "").lower() in {"", "enterprise"}
    )
    q_tokens = _expand_query_tokens(query or "")
    if not q_tokens:
        # Fall back to a few well-known initial-access / execution techniques
        defaults = ("T1190", "T1059", "T1068", "T1078", "T1133", "T1566", "T1203")
        by_id = {t["id"]: t for t in techs}
        out: list[dict[str, Any]] = []
        for tid in defaults:
            if tid in by_id and len(out) < k:
                hit = dict(by_id[tid])
                hit["score"] = 0.0
                out.append(hit)
        return out

    scored: list[tuple[float, dict[str, Any]]] = []
    for tech in techs:
        s = _score(q_tokens, tech)
        if s > 0:
            scored.append((s, tech))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    results: list[dict[str, Any]] = []
    for s, tech in scored[: max(1, k)]:
        row = dict(tech)
        row["score"] = round(s, 4)
        results.append(row)
    return results


def format_technique_context(hits: list[dict[str, Any]]) -> str:
    """Format retrieved technique docs for an LLM prompt."""
    if not hits:
        return ""
    lines = [
        "### Retrieved ATT&CK technique documentation (candidate set)",
        "Use these as the primary reference. Prefer IDs from this set when they fit.",
    ]
    for h in hits:
        tid = h.get("id") or ""
        name = h.get("name") or ""
        desc = (h.get("description") or "").strip()
        if len(desc) > 220:
            desc = desc[:217] + "…"
        lines.append(f"- {tid} — {name}: {desc}")
    ids = [str(h.get("id")) for h in hits if h.get("id")]
    if ids:
        lines.append("")
        lines.append(
            "Prefer from this candidate set when possible: " + ", ".join(ids)
        )
    return "\n".join(lines).strip()
