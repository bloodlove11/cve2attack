"""Product-facing ATT&CK helpers (Live pipeline shared with golden eval)."""

from src.attack.detect import extract_cve_to_attack_intent
from src.attack.chat_live import run_live_attack_answer

__all__ = [
    "extract_cve_to_attack_intent",
    "run_live_attack_answer",
]
