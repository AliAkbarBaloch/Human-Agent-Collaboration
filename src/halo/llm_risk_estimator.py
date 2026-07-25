# Ali Akbar Start (Hybrid Gap 1 — LLM-based Risk Estimation)
"""
LLM-based risk estimator for the HALO Adaptive Oversight Framework.

Provides semantic risk estimation as the second layer of the hybrid Gap 1 architecture:

  Rule-Based Risk Estimator  →  LLM-Based Risk Estimator  →  Risk Fusion

Falls back gracefully to rule-based results on any failure (timeout, invalid JSON,
model error). The caller must always handle available=False.

Research alignment:
  LLM-based components provide contextual reasoning and flexibility beyond exact
  keyword matching, enabling detection of ambiguous tasks that rule-based classifiers
  would miss or misclassify.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VALID_TASK_TYPES = frozenset({"research", "transactional", "destructive"})
_VALID_POLICIES   = frozenset({"auto-permissive", "auto-conservative", "always"})

# Maximum user task characters sent to the LLM (cost and safety limit).
_MAX_TASK_CHARS = 2000

_LLM_RISK_PROMPT = """\
You are a risk estimation module for a human-agent collaboration system.

Your task is to estimate the risk of a user task before an autonomous agent acts.

Classify the task into one of:

1. research
   Low-risk tasks involving reading, searching, summarizing, browsing, or answering factual questions.

2. transactional
   Medium-risk tasks involving form submission, sign-up, booking, purchasing, uploading, sending messages, creating accounts, or interacting with external services.

3. destructive
   High-risk tasks involving deletion, cancellation, overwriting, uninstalling, wiping, irreversible changes, account termination, or modification of important user data.

Return JSON only. Do not include markdown.

Schema:
{{
  "task_type": "research | transactional | destructive",
  "risk_score": 0.0,
  "confidence": 0.0,
  "reason": "short explanation",
  "possible_harms": ["..."],
  "recommended_policy": "auto-permissive | auto-conservative | always"
}}

User task:
{user_task}"""


@dataclass
class LLMRiskResult:
    """Structured result from the LLM-based risk estimator."""
    task_type:          str        = "research"
    risk_score:         float      = 0.15
    confidence:         float      = 0.0
    reason:             str        = ""
    possible_harms:     list[str]  = field(default_factory=list)
    recommended_policy: str        = "auto-permissive"
    available:          bool       = False   # False = LLM unavailable; caller must fall back


def _parse_llm_risk_response(raw: str) -> LLMRiskResult:
    """Parse and validate the LLM's JSON response. Raises on failure."""
    raw = raw.strip()
    # Strip markdown code fences anywhere in the response (Ollama often adds preamble text)
    raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip()

    # If the model added preamble text before the JSON object, extract the first {...}
    if not raw.startswith("{"):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

    data = json.loads(raw)

    task_type = str(data.get("task_type", "research")).strip().lower()
    if task_type not in _VALID_TASK_TYPES:
        task_type = "research"

    risk_score = float(data.get("risk_score", 0.15))
    risk_score = max(0.0, min(1.0, risk_score))

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    recommended_policy = str(data.get("recommended_policy", "auto-permissive")).strip()
    if recommended_policy not in _VALID_POLICIES:
        recommended_policy = "auto-permissive"

    return LLMRiskResult(
        task_type=task_type,
        risk_score=risk_score,
        confidence=confidence,
        reason=str(data.get("reason", ""))[:300],
        possible_harms=[str(h)[:150] for h in data.get("possible_harms", [])[:5]],
        recommended_policy=recommended_policy,
        available=True,
    )


async def estimate_risk_with_llm(
    user_task: str,
    model_client: Any,
    timeout: float = 90.0,
) -> LLMRiskResult:
    """
    Estimate task risk using the LLM model client.

    Returns an ``LLMRiskResult`` with ``available=False`` on any failure.
    The caller is responsible for applying conservative fusion with the rule-based result.

    Args:
        user_task:     The raw user prompt / task text.
        model_client:  An AutoGen ``ChatCompletionClient`` instance.
        timeout:       Maximum seconds to wait for the LLM response.
    """
    if model_client is None:
        return LLMRiskResult(available=False, reason="No model client provided")

    try:
        from autogen_core.models import UserMessage  # imported lazily to avoid hard dep at module level

        prompt   = _LLM_RISK_PROMPT.format(user_task=user_task[:_MAX_TASK_CHARS])
        messages = [UserMessage(content=prompt, source="halo_risk_estimator")]

        response = await asyncio.wait_for(
            model_client.create(messages),
            timeout=timeout,
        )

        # Extract string content from various response shapes
        if isinstance(response.content, str):
            raw = response.content
        elif isinstance(response.content, list):
            raw = " ".join(
                part if isinstance(part, str) else getattr(part, "text", "")
                for part in response.content
            )
        else:
            raw = str(response.content)

        return _parse_llm_risk_response(raw)

    except asyncio.TimeoutError:
        logger.warning("[HybridGap1] LLM risk estimation timed out — falling back to rule-based")
        return LLMRiskResult(available=False, reason="LLM timeout")
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(f"[HybridGap1] Invalid LLM risk response: {exc} — falling back")
        return LLMRiskResult(available=False, reason=f"Invalid response: {exc}")
    except Exception as exc:
        logger.warning(f"[HybridGap1] LLM risk call failed: {exc} — falling back")
        return LLMRiskResult(available=False, reason=f"Error: {exc}")
# Ali Akbar End (Hybrid Gap 1 — LLM-based Risk Estimation)
