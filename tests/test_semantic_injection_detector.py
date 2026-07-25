"""
Tests for halo/semantic_injection_detector.py — Gap 2 LLM-based layer.

Strategy:
  - _parse_semantic_response()        is pure/sync → tested directly with JSON strings.
  - detect_injection_semantically()   is async + calls model_client → tested with mocks.

Security invariant tested:
  The detector must mark itself unavailable (not follow instructions) when the model
  call fails — never silently allow injections through on failure.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from halo.semantic_injection_detector import (
    HijackScreenResult,
    SemanticScanResult,
    _parse_hijack_response,
    _parse_semantic_response,
    detect_injection_semantically,
    screen_action_for_hijack,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_json(**overrides) -> str:
    data = {
        "injection_detected": False,
        "risk_score": 0.0,
        "confidence": 0.9,
        "attack_type": "benign",
        "evidence": [],
        "recommended_action": "allow",
        "reason": "No injection found",
    }
    data.update(overrides)
    return json.dumps(data)


def _make_client(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    client = MagicMock()
    client.create = AsyncMock(return_value=resp)
    return client


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── _parse_semantic_response — happy path ────────────────────────────────────

class TestParseResponse:
    def test_clean_page_parsed(self):
        raw = _valid_json()
        result = _parse_semantic_response(raw)
        assert result.injection_detected is False
        assert result.attack_type == "benign"
        assert result.available is True

    def test_injection_detected_true(self):
        raw = _valid_json(injection_detected=True, risk_score=0.85,
                          attack_type="instruction_override",
                          recommended_action="block",
                          evidence=["ignore previous instructions"])
        result = _parse_semantic_response(raw)
        assert result.injection_detected is True
        assert result.risk_score == pytest.approx(0.85)
        assert result.attack_type == "instruction_override"
        assert result.recommended_action == "block"
        assert "ignore previous instructions" in result.evidence

    def test_all_valid_attack_types_accepted(self):
        valid = [
            "instruction_override", "roleplay", "data_exfiltration",
            "tool_misuse", "hidden_instruction", "benign", "unknown",
        ]
        for attack_type in valid:
            raw = _valid_json(attack_type=attack_type)
            result = _parse_semantic_response(raw)
            assert result.attack_type == attack_type

    def test_invalid_attack_type_mapped_to_unknown(self):
        raw = _valid_json(attack_type="space_laser")
        result = _parse_semantic_response(raw)
        assert result.attack_type == "unknown"

    def test_all_valid_actions_accepted(self):
        for action in ["allow", "warn", "block"]:
            raw = _valid_json(recommended_action=action)
            result = _parse_semantic_response(raw)
            assert result.recommended_action == action

    def test_invalid_action_falls_back_to_allow(self):
        raw = _valid_json(recommended_action="quarantine")
        result = _parse_semantic_response(raw)
        assert result.recommended_action == "allow"

    def test_risk_score_clamped_above_one(self):
        raw = _valid_json(risk_score=50.0)
        result = _parse_semantic_response(raw)
        assert result.risk_score == pytest.approx(1.0)

    def test_risk_score_clamped_below_zero(self):
        raw = _valid_json(risk_score=-1.0)
        result = _parse_semantic_response(raw)
        assert result.risk_score == pytest.approx(0.0)

    def test_confidence_clamped(self):
        raw = _valid_json(confidence=99.0)
        result = _parse_semantic_response(raw)
        assert result.confidence == pytest.approx(1.0)

    def test_evidence_truncated_to_five(self):
        raw = _valid_json(evidence=[f"evidence {i}" for i in range(20)])
        result = _parse_semantic_response(raw)
        assert len(result.evidence) <= 5

    def test_reason_length_capped(self):
        raw = _valid_json(reason="r" * 2000)
        result = _parse_semantic_response(raw)
        assert len(result.reason) <= 400

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _valid_json(injection_detected=True) + "\n```"
        result = _parse_semantic_response(raw)
        assert result.injection_detected is True
        assert result.available is True

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_semantic_response("{ not json }")


# ── detect_injection_semantically — mock-based async tests ───────────────────

class TestDetectInjectionSemantically:
    def test_none_client_returns_unavailable(self):
        result = _run(detect_injection_semantically("page text", None))
        assert result.available is False

    def test_clean_page_returns_not_detected(self):
        client = _make_client(_valid_json(injection_detected=False))
        result = _run(detect_injection_semantically("normal page text", client))
        assert result.available is True
        assert result.injection_detected is False

    def test_injection_detected_returns_correct_fields(self):
        client = _make_client(_valid_json(
            injection_detected=True, risk_score=0.90,
            attack_type="data_exfiltration", recommended_action="block"))
        result = _run(detect_injection_semantically("malicious page", client))
        assert result.available is True
        assert result.injection_detected is True
        assert result.risk_score == pytest.approx(0.90)
        assert result.attack_type == "data_exfiltration"
        assert result.recommended_action == "block"

    def test_timeout_returns_unavailable(self):
        async def slow_create(_):
            await asyncio.sleep(100)
        client = MagicMock()
        client.create = slow_create
        result = _run(detect_injection_semantically("page", client, timeout=0.01))
        assert result.available is False
        assert result.injection_detected is False  # Safe default — not an injection

    def test_timeout_does_not_allow_injection(self):
        # Security invariant: on failure, injection_detected must be False (safe default)
        async def slow_create(_):
            await asyncio.sleep(100)
        client = MagicMock()
        client.create = slow_create
        result = _run(detect_injection_semantically("ignore previous instructions", client, timeout=0.01))
        assert result.injection_detected is False

    def test_invalid_json_response_returns_unavailable(self):
        client = _make_client("definitely not json")
        result = _run(detect_injection_semantically("page text", client))
        assert result.available is False
        assert result.injection_detected is False

    def test_model_exception_returns_unavailable(self):
        client = MagicMock()
        client.create = AsyncMock(side_effect=ConnectionError("model down"))
        result = _run(detect_injection_semantically("page text", client))
        assert result.available is False

    def test_long_page_text_truncated(self):
        long_text = "safe content " * 2000  # >> 4000 chars
        client = _make_client(_valid_json())
        result = _run(detect_injection_semantically(long_text, client))
        assert result.available is True
        # Verify model was called (only once)
        client.create.assert_called_once()

    def test_markdown_stripped_from_response(self):
        raw = "```\n" + _valid_json(injection_detected=True, attack_type="roleplay") + "\n```"
        client = _make_client(raw)
        result = _run(detect_injection_semantically("page text", client))
        assert result.available is True
        assert result.attack_type == "roleplay"

    def test_list_content_response_parsed(self):
        resp = MagicMock()
        resp.content = [_valid_json(injection_detected=False)]
        client = MagicMock()
        client.create = AsyncMock(return_value=resp)
        result = _run(detect_injection_semantically("page text", client))
        assert result.available is True


# ── SemanticScanResult dataclass defaults ─────────────────────────────────────

class TestSemanticScanResult:
    def test_default_available_false(self):
        r = SemanticScanResult()
        assert r.available is False

    def test_default_injection_detected_false(self):
        r = SemanticScanResult()
        assert r.injection_detected is False

    def test_default_attack_type_benign(self):
        r = SemanticScanResult()
        assert r.attack_type == "benign"

    def test_default_recommended_action_allow(self):
        r = SemanticScanResult()
        assert r.recommended_action == "allow"

    def test_default_evidence_empty_list(self):
        r = SemanticScanResult()
        assert r.evidence == []


# ── screen_action_for_hijack — Ali Akbar Prompt Injection ─────────────────────
# Distinct from the content-scanning tests above: this asks "does the agent's
# PROPOSED action look consistent with its instruction, or does it look redirected
# by something previously read" — an output-side check, not an input-side scan.

def _valid_hijack_json(**overrides) -> str:
    data = {
        "hijack_suspected": False,
        "confidence": 0.9,
        "deviates_from_task": False,
        "reason": "Action matches the instruction",
    }
    data.update(overrides)
    return json.dumps(data)


class TestParseHijackResponse:
    def test_benign_action_parsed(self):
        result = _parse_hijack_response(_valid_hijack_json())
        assert result.hijack_suspected is False
        assert result.available is True

    def test_hijack_suspected_true(self):
        raw = _valid_hijack_json(
            hijack_suspected=True, confidence=0.85, deviates_from_task=True,
            reason="Instruction was to search flights; action submits a payment form",
        )
        result = _parse_hijack_response(raw)
        assert result.hijack_suspected is True
        assert result.confidence == pytest.approx(0.85)
        assert result.deviates_from_task is True

    def test_confidence_clamped(self):
        raw = _valid_hijack_json(confidence=99.0)
        result = _parse_hijack_response(raw)
        assert result.confidence == pytest.approx(1.0)

    def test_reason_length_capped(self):
        raw = _valid_hijack_json(reason="r" * 2000)
        result = _parse_hijack_response(raw)
        assert len(result.reason) <= 400

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _valid_hijack_json(hijack_suspected=True) + "\n```"
        result = _parse_hijack_response(raw)
        assert result.hijack_suspected is True
        assert result.available is True

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_hijack_response("{ not json }")


class TestScreenActionForHijack:
    def test_none_client_returns_unavailable(self):
        result = _run(screen_action_for_hijack("search flight prices", "click(...)", None))
        assert result.available is False

    def test_consistent_action_not_flagged(self):
        client = _make_client(_valid_hijack_json(hijack_suspected=False))
        result = _run(screen_action_for_hijack(
            "search flight prices", "click(search_button)", client
        ))
        assert result.available is True
        assert result.hijack_suspected is False

    def test_redirected_action_flagged(self):
        client = _make_client(_valid_hijack_json(
            hijack_suspected=True, confidence=0.9, deviates_from_task=True,
            reason="Task was to search flights; action submits a payment form",
        ))
        result = _run(screen_action_for_hijack(
            "search flight prices", "click(submit_payment)", client
        ))
        assert result.available is True
        assert result.hijack_suspected is True
        assert result.deviates_from_task is True

    def test_timeout_returns_unavailable_and_not_suspected(self):
        # Security invariant: on failure, hijack_suspected must be False (safe default —
        # callers must never force approval based on a failed/unavailable check).
        async def slow_create(_):
            await asyncio.sleep(100)
        client = MagicMock()
        client.create = slow_create
        result = _run(screen_action_for_hijack(
            "search flights", "click(submit_payment)", client, timeout=0.01
        ))
        assert result.available is False
        assert result.hijack_suspected is False

    def test_invalid_json_response_returns_unavailable(self):
        client = _make_client("definitely not json")
        result = _run(screen_action_for_hijack("task", "action", client))
        assert result.available is False

    def test_model_exception_returns_unavailable(self):
        client = MagicMock()
        client.create = AsyncMock(side_effect=ConnectionError("model down"))
        result = _run(screen_action_for_hijack("task", "action", client))
        assert result.available is False


class TestHijackScreenResultDefaults:
    def test_default_available_false(self):
        r = HijackScreenResult()
        assert r.available is False

    def test_default_hijack_suspected_false(self):
        r = HijackScreenResult()
        assert r.hijack_suspected is False
