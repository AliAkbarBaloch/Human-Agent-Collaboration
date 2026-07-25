# Ali Akbar Start (Gap 2 — Prompt Injection Visibility Layer)
"""
Prompt injection scanner for HALOWebSurfer pages.

Scans visible page text for patterns that attempt to hijack the agent's
instructions. Returns matched patterns so callers can decide whether to
alert the user before passing the text to the LLM.
"""

from typing import NamedTuple

# Patterns that strongly suggest prompt-injection attempts.
# Keep lower-case; comparison is done case-insensitively.
INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore your previous",
    "ignore all previous",
    "disregard previous",
    "disregard your instructions",
    "forget your instructions",
    "forget previous instructions",
    "new task:",
    "your new instructions",
    "you are now",
    "act as if",
    "pretend you are",
    "pretend to be",
    "roleplay as",
    "you must now",
    "from now on you",
    "override your",
    "system prompt:",
    "system message:",
    "<!-- inject",
    "<!-- prompt",
    "[system]",
    "<!-- override",
    "exfiltrate",
    "send all data to",
    "leak your",
]


class ScanResult(NamedTuple):
    detected: bool
    matched_patterns: list[str]
    excerpt: str        # short excerpt around the first match, for the alert UI
    risk_score: float = 0.0   # 0.0 = clean; 0.40 = 1 pattern; 0.65 = 2; 0.90 = 3+
    risk_level: str = "none"  # "none" | "low" | "medium" | "high"


def _injection_risk(n_patterns: int) -> tuple[float, str]:
    """Map pattern count to injection risk score and level."""
    if n_patterns == 0:
        return 0.0, "none"
    if n_patterns == 1:
        score = 0.40
    elif n_patterns == 2:
        score = 0.65
    else:
        score = 0.90
    level = "high" if score > 0.70 else "medium" if score > 0.30 else "low"
    return score, level


def scan_for_injection(page_text: str, max_excerpt_len: int = 200) -> ScanResult:
    """
    Scan *page_text* for prompt-injection patterns.

    Returns a :class:`ScanResult` with:
    - ``detected``: True if any pattern was found.
    - ``matched_patterns``: list of every matched pattern string.
    - ``excerpt``: a short surrounding snippet from the first match.
    - ``risk_score``: normalised injection confidence score [0, 1].
    - ``risk_level``: "none" | "low" | "medium" | "high".
    """
    text_lower = page_text.lower()
    matched: list[str] = []
    first_pos: int = -1

    for pattern in INJECTION_PATTERNS:
        pos = text_lower.find(pattern)
        if pos != -1:
            matched.append(pattern)
            if first_pos == -1 or pos < first_pos:
                first_pos = pos

    if not matched:
        return ScanResult(detected=False, matched_patterns=[], excerpt="",
                          risk_score=0.0, risk_level="none")

    # Build an excerpt centred on the first match.
    start = max(0, first_pos - 60)
    end = min(len(page_text), first_pos + max_excerpt_len)
    excerpt = page_text[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(page_text):
        excerpt = excerpt + "…"

    risk_score, risk_level = _injection_risk(len(matched))
    return ScanResult(detected=True, matched_patterns=matched, excerpt=excerpt,
                      risk_score=risk_score, risk_level=risk_level)
# Ali Akbar End (Gap 2 — Prompt Injection Visibility Layer)
