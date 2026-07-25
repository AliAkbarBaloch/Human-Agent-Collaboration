# Ali Akbar Start (Gap 1 — Adaptive Action Guard Threshold)
import re
from typing import Literal

ApprovalPolicy = Literal["always", "never", "auto-conservative", "auto-permissive"]

RESEARCH_KEYWORDS = [
    "find", "search", "what is", "who is", "when was", "how many",
    "list", "explain", "tell me", "show me", "summarize", "describe",
    "read", "visit", "browse", "check", "look up", "get information",
]

TRANSACTIONAL_KEYWORDS = [
    "book", "order", "buy", "purchase", "sign up", "register",
    "pay", "checkout", "subscribe", "add to cart", "submit",
    "fill in", "fill out", "send", "upload", "post", "create",
    "transfer", "schedule", "reserve",
]

DESTRUCTIVE_KEYWORDS = [
    "delete", "remove", "uninstall", "cancel", "terminate",
    "wipe", "format", "drop", "destroy", "erase", "purge",
    "overwrite", "reset", "clear",
]

# Maps task type → valid ApprovalGuard approval_policy string
POLICY_MAP: dict[str, ApprovalPolicy] = {
    "research": "auto-permissive",        # LLM decides; defaults to no interruption
    "transactional": "auto-conservative", # LLM decides; conservative default
    "destructive": "always",              # always require explicit user approval
}


def _matches(text: str, keyword: str) -> bool:
    """Word-boundary match so 'format' does not match inside 'information'."""
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def classify_task(user_prompt: str) -> str:
    """
    Classify a user task as 'research', 'transactional', or 'destructive'
    based on keyword matching. Priority: destructive > transactional > research.
    """
    prompt_lower = user_prompt.lower()
    for kw in DESTRUCTIVE_KEYWORDS:
        if _matches(prompt_lower, kw):
            return "destructive"
    for kw in TRANSACTIONAL_KEYWORDS:
        if _matches(prompt_lower, kw):
            return "transactional"
    return "research"


def classify_task_with_keywords(user_prompt: str) -> tuple[str, list[str]]:
    """
    Classify a user task and return (task_type, matched_keywords).

    Returns all keywords that triggered the winning category.
    Priority: destructive > transactional > research.
    Used by the hybrid Gap 1 estimator to expose the rule layer's evidence.
    """
    prompt_lower = user_prompt.lower()

    dest_matched = [kw for kw in DESTRUCTIVE_KEYWORDS if _matches(prompt_lower, kw)]
    if dest_matched:
        return "destructive", dest_matched

    trans_matched = [kw for kw in TRANSACTIONAL_KEYWORDS if _matches(prompt_lower, kw)]
    if trans_matched:
        return "transactional", trans_matched

    res_matched = [kw for kw in RESEARCH_KEYWORDS if _matches(prompt_lower, kw)]
    return "research", res_matched


# --- Risk Estimation Layer (HALO Adaptive Oversight Framework — Layer 1) ---
# Normalized risk scores in [0, 1] for each task type.
# These encode inherent task danger for the Risk-Aware Supervision narrative.
RISK_SCORES: dict[str, float] = {
    "research":     0.15,
    "transactional": 0.55,
    "destructive":  0.95,
}


def get_risk_score(task_type: str) -> float:
    """Return the normalized risk score [0, 1] for the given task type."""
    return RISK_SCORES.get(task_type, 0.15)
# Ali Akbar End (Gap 1 — Adaptive Action Guard Threshold)
