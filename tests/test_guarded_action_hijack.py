"""
Tests for the action-hijack screening hook in guarded_action.py's
BaseGuardedAction.invoke_with_approval() — Ali Akbar Prompt Injection.

This is the ONE shared hook file_surfer and coder both route through (via
GuardedAction/TrivialGuardedAction), so testing it here covers both agents without
duplicating the test per agent.

Security invariant tested:
  A hijack-suspected action must be forced through approval even when the normal
  risk check (requires_approval) would have allowed it through — the two checks are
  independent and hijack detection must never be silently overridden by a permissive
  risk baseline/policy.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from autogen_agentchat.messages import TextMessage
from autogen_core.models import UserMessage

from halo.approval_guard import ApprovalConfig, ApprovalGuard
from halo.guarded_action import ApprovalDeniedError, TrivialGuardedAction


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeGuard:
    """An ApprovalGuard-like object with a scriptable requires_approval() and a
    model_client the hijack-screen call can key off of."""

    def __init__(self, requires_approval_result: bool, model_client=None, policy="auto-conservative"):
        self._requires_approval_result = requires_approval_result
        self.model_client = model_client
        self.config = ApprovalConfig(approval_policy=policy)
        self.approval_calls: list[str] = []
        self.approve_response = True

    async def requires_approval(self, baseline, llm_guess, action_context):
        return self._requires_approval_result

    async def get_approval(self, action_description, input_type="approval"):
        content = getattr(action_description, "content", "")
        self.approval_calls.append(content if isinstance(content, str) else str(content))
        return self.approve_response


def _make_hijack_client(suspected: bool, confidence: float = 0.9) -> AsyncMock:
    import json
    from unittest.mock import MagicMock

    resp = MagicMock()
    resp.content = json.dumps({
        "hijack_suspected": suspected,
        "confidence": confidence,
        "deviates_from_task": suspected,
        "reason": "Action does not match the instruction" if suspected else "Consistent",
    })
    client = MagicMock()
    client.create = AsyncMock(return_value=resp)
    return client


class TestHijackScreeningInGuardedAction:
    def test_no_hijack_no_forced_approval(self):
        guard = _FakeGuard(requires_approval_result=False, model_client=_make_hijack_client(False))
        action_called = {"count": 0}

        async def action():
            action_called["count"] += 1
            return "ok"

        ta = TrivialGuardedAction("read_file", baseline_override="never")
        # Override the action callable directly since TrivialGuardedAction's default
        # action is a no-op lambda.
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="do the thing", source="agent"),
            [UserMessage(content="read this file", source="user")],
            guard,
            hybrid_injection_detection=True,
        ))
        assert result == "ok"
        assert guard.approval_calls == []  # never asked — baseline said no approval needed

    def test_hijack_suspected_forces_approval_even_when_baseline_allows(self):
        guard = _FakeGuard(requires_approval_result=False, model_client=_make_hijack_client(True))
        guard.approve_response = True

        async def action():
            return "executed"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
            [UserMessage(content="summarize this document", source="user")],
            guard,
            hybrid_injection_detection=True,
        ))
        assert result == "executed"
        # Forced approval despite baseline_override="never" (normally never asks)
        assert len(guard.approval_calls) == 1
        assert "hijack" in guard.approval_calls[0].lower()

    def test_hijack_suspected_and_user_denies_raises(self):
        guard = _FakeGuard(requires_approval_result=False, model_client=_make_hijack_client(True))
        guard.approve_response = False

        async def action():
            return "should not run"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        with pytest.raises(ApprovalDeniedError):
            _run(ta.invoke_with_approval(
                {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
                [UserMessage(content="summarize this document", source="user")],
                guard,
                hybrid_injection_detection=True,
            ))

    def test_low_confidence_hijack_does_not_force_approval(self):
        guard = _FakeGuard(
            requires_approval_result=False,
            model_client=_make_hijack_client(True, confidence=0.3),
        )

        async def action():
            return "ok"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
            [UserMessage(content="summarize this document", source="user")],
            guard,
            hybrid_injection_detection=True,
        ))
        assert result == "ok"
        assert guard.approval_calls == []  # confidence below 0.6 threshold — not forced

    def test_hybrid_injection_detection_false_disables_screening(self):
        guard = _FakeGuard(requires_approval_result=False, model_client=_make_hijack_client(True))

        async def action():
            return "ok"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
            [UserMessage(content="summarize this document", source="user")],
            guard,
            hybrid_injection_detection=False,
        ))
        assert result == "ok"
        assert guard.approval_calls == []  # screening disabled entirely

    def test_never_policy_disables_screening(self):
        # is_scan_active()'s "never" bypass only applies to genuine ApprovalGuard
        # instances (same isinstance check the pre-existing page scanner used) — use
        # a real ApprovalGuard here, not the duck-typed _FakeGuard.
        guard = ApprovalGuard(
            model_client=_make_hijack_client(True),
            config=ApprovalConfig(approval_policy="never"),
        )

        async def action():
            return "ok"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
            [UserMessage(content="summarize this document", source="user")],
            guard,
            hybrid_injection_detection=True,
        ))
        assert result == "ok"

    def test_real_approval_guard_no_model_client_skips_screening_safely(self):
        # A real ApprovalGuard with no model_client must not error — hijack screening
        # should just no-op (no model client to call).
        guard = ApprovalGuard(config=ApprovalConfig(approval_policy="auto-permissive"))

        async def action():
            return "ok"

        ta = TrivialGuardedAction("run_code", baseline_override="never")
        ta.action = action  # type: ignore[assignment]

        result = _run(ta.invoke_with_approval(
            {}, TextMessage(content="Do you want to execute the code above?", source="agent"),
            [UserMessage(content="summarize this document", source="user")],
            guard,
            hybrid_injection_detection=True,
        ))
        assert result == "ok"
