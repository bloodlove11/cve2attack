"""Detect CVE→ATT&CK intent in Chat / product queries."""

from __future__ import annotations

import re

_CVE_RE = re.compile(r"\b(CVE-\d{4}-\d{4,})\b", re.IGNORECASE)
# Mapping / ATT&CK language; triage-only questions should not match.
_ATTACK_HINT_RE = re.compile(
    r"(?:"
    r"att\s*&\s*ck|att&ck|attack\s+techniques?|mitre(?:\s+att(?:ack|&ck))?|"
    r"technique\s+ids?|map(?:ping)?\s+(?:to\s+)?(?:att|mitre|techniques?)|"
    r"cve\s*[→\-]+\s*att|ctid|primary[_\s-]?impact|exploitation_techniques|"
    r"exploitation\s+techniques?|primary\s+impact|"
    r"(?:for|about)\s+that\s+(?:cve|one)|those\s+techniques|"
    r"same\s+(?:cve|mapping)|follow[\s-]?up.*(?:att|techniq|map)"
    r")",
    re.IGNORECASE,
)
_TRIAGE_ONLY_RE = re.compile(
    r"\b(?:triage|prioritiz|severity|kev|epss|cvss)\b",
    re.IGNORECASE,
)
_STRONG_MAP_RE = re.compile(
    r"(?:att\s*&\s*ck|att&ck|mitre|technique|map(?:ping)?|ctid|"
    r"primary[_\s-]?impact|exploitation)",
    re.IGNORECASE,
)


def extract_cve_id(text: str) -> str | None:
    """Return the first CVE id in ``text``, or ``None``."""
    m = _CVE_RE.search(text or "")
    return m.group(1).upper() if m else None


def extract_cve_to_attack_intent(
    query: str,
    *,
    prior_text: str | None = None,
) -> str | None:
    """Return a CVE id when the query asks for CVE→ATT&CK mapping.

    Requires ATT&CK / mapping language on the current query, plus a CVE id
    in the query or (for multi-turn follow-ups) in ``prior_text``. Pure triage
    questions (severity / KEV / CVSS without ATT&CK hints) return ``None``.
    """
    text = (query or "").strip()
    if not text or not _ATTACK_HINT_RE.search(text):
        return None
    cve = extract_cve_id(text) or (extract_cve_id(prior_text) if prior_text else None)
    if cve is None:
        return None
    # Weak follow-up hints ("for that CVE") plus triage words → generic chat.
    if _TRIAGE_ONLY_RE.search(text) and not _STRONG_MAP_RE.search(text):
        return None
    return cve
