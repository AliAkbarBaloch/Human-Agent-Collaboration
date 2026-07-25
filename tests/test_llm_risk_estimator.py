"""
Tests for halo/llm_risk_estimator.py — Gap 1 LLM-based layer.

Strategy:
  - _parse_llm_risk_response() is pure/sync → tested directly with JSON strings.
  - estimate_risk_with_llm()   is async + calls model_client → tested with mocks.

Mock shape: model_client.create() is an async coroutine returning an object
with a .content attribute (str).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from halo.llm_risk_estimator import (
    LLMRiskResult,
    _parse_llm_risk_response,
    estimate_risk_with_llm,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


def _make_client(content: str) -> MagicMock:
    client = MagicMock()
    client.create = AsyncMock(return_value=_make_response(content))
    return client


def _valid_json(**overrides) -> str:
    data = {
        "task_type": "research",
        "risk_score": 0.15,
        "confidence": 0.80,
        "reason": "Low risk task",
        "possible_harms": [],
        "recommended_policy": "auto-permissive",
    }
    data.update(overrides)
    return json.dumps(data)


# ── _parse_llm_risk_response — happy path ─────────────────────────────────────

class TestParseResponse:
    def test_valid_research(self):
        raw = _valid_json(task_type="research", risk_score=0.15, confidence=0.9)
        result = _parse_llm_risk_response(raw)
        assert result.task_type == "research"
        assert result.risk_score == pytest.approx(0.15)
        assert result.confidence == pytest.approx(0.9)
        assert result.available is True

    def test_valid_transactional(self):
        raw = _valid_json(task_type="transactional", risk_score=0.55, confidence=0.75,
                          recommended_policy="auto-conservative")
        result = _parse_llm_risk_response(raw)
        assert result.task_type == "transactional"
        assert result.recommended_policy == "auto-conservative"

    def test_valid_destructive(self):
        raw = _valid_json(task_type="destructive", risk_score=0.95, confidence=0.85,
                          recommended_policy="always",
                          possible_harms=["data loss"])
        result = _parse_llm_risk_response(raw)
        assert result.task_type == "destructive"
        assert result.recommended_policy == "always"
        assert "data loss" in result.possible_harms

    def test_strips_markdown_fences(self):
        raw = "```json\n" + _valid_json() + "\n```"
        result = _parse_llm_risk_response(raw)
        assert result.task_type == "research"
        assert result.available is True

    def test_risk_score_clamped_above_one(self):
        raw = _valid_json(risk_score=99.0)
        result = _parse_llm_risk_response(raw)
        assert result.risk_score == pytest.approx(1.0)

    def test_risk_score_clamped_below_zero(self):
        raw = _valid_json(risk_score=-5.0)
        result = _parse_llm_risk_response(raw)
        assert result.risk_score == pytest.approx(0.0)

    def test_confidence_clamped(self):
        raw = _valid_json(confidence=1.5)
        result = _parse_llm_risk_response(raw)
        assert result.confidence == pytest.approx(1.0)

    def test_invalid_task_type_falls_back_to_research(self):
        raw = _valid_json(task_type="unknown_garbage")
        result = _parse_llm_risk_response(raw)
        assert result.task_type == "research"

    def test_invalid_policy_falls_back_to_permissive(self):
        raw = _valid_json(recommended_policy="bad-policy")
        result = _parse_llm_risk_response(raw)
        assert result.recommended_policy == "auto-permissive"

    def test_possible_harms_truncated_to_five(self):
        harms = [f"harm {i}" for i in range(10)]
        raw = _valid_json(possible_harms=harms)
        result = _parse_llm_risk_response(raw)
        assert len(result.possible_harms) <= 5

    def test_reason_length_capped(self):
        long_reason = "x" * 1000
        raw = _valid_json(reason=long_reason)
        result = _parse_llm_risk_response(raw)
        assert len(result.reason) <= 300

    def test_raises_on_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_llm_risk_response("not json at all")


# ── estimate_risk_with_llm — mock-based async tests ──────────────────────────

class TestEstimateRiskWithLlm:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_none_client_returns_unavailable(self):
        result = self._run(estimate_risk_with_llm("some task", None))
        assert result.available is False

    def test_successful_call_returns_available(self):
        client = _make_client(_valid_json(task_type="research"))
        result = self._run(estimate_risk_with_llm("find some data", client))
        assert result.available is True
        assert result.task_type == "research"

    def test_successful_transactional(self):
        client = _make_client(_valid_json(
            task_type="transactional", risk_score=0.55,
            recommended_policy="auto-conservative"))
        result = self._run(estimate_risk_with_llm("book a flight", client))
        assert result.task_type == "transactional"
        assert result.available is True

    def test_timeout_returns_unavailable(self):
        async def slow_create(_):
            await asyncio.sleep(100)
        client = MagicMock()
        client.create = slow_create
        result = self._run(estimate_risk_with_llm("task", client, timeout=0.01))
        assert result.available is False
        assert "timeout" in result.reason.lower()

    def test_invalid_json_response_returns_unavailable(self):
        client = _make_client("not valid json")
        result = self._run(estimate_risk_with_llm("task", client))
        assert result.available is False

    def test_model_exception_returns_unavailable(self):
        client = MagicMock()
        client.create = AsyncMock(side_effect=RuntimeError("model error"))
        result = self._run(estimate_risk_with_llm("task", client))
        assert result.available is False

    def test_list_content_response_parsed(self):
        resp = MagicMock()
        resp.content = [_valid_json(task_type="research")]
        client = MagicMock()
        client.create = AsyncMock(return_value=resp)
        result = self._run(estimate_risk_with_llm("task", client))
        assert result.available is True

    def test_task_truncated_to_max_chars(self):
        long_task = "find data " * 500  # >> 2000 chars
        client = _make_client(_valid_json())
        # Should not raise
        result = self._run(estimate_risk_with_llm(long_task, client))
        assert result.available is True
        # Verify model was called (once)
        client.create.assert_called_once()

    def test_markdown_stripped_from_response(self):
        raw = "```json\n" + _valid_json(task_type="destructive") + "\n```"
        client = _make_client(raw)
        result = self._run(estimate_risk_with_llm("delete everything", client))
        assert result.available is True
        assert result.task_type == "destructive"

    def test_default_return_values_on_failure(self):
        client = _make_client("bad json")
        result = self._run(estimate_risk_with_llm("task", client))
        assert result.task_type == "research"
        assert result.risk_score == pytest.approx(0.15)
        assert result.available is False


# ── LLMRiskResult dataclass defaults ─────────────────────────────────────────

class TestLLMRiskResult:
    def test_default_available_false(self):
        r = LLMRiskResult()
        assert r.available is False

    def test_default_task_type_research(self):
        r = LLMRiskResult()
        assert r.task_type == "research"

    def test_default_possible_harms_empty_list(self):
        r = LLMRiskResult()
        assert r.possible_harms == []
