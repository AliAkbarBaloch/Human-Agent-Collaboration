import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Union

from autogen_agentchat.base._task import TaskResult
from autogen_agentchat.messages import (
    AgentEvent,
    BaseTextChatMessage,
    ChatMessage,
    HandoffMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from ....input_func import InputFuncType, InputRequestType
from ....halo_state import HALOState
from ....feedback_loop import BayesianFeedbackLoop
from autogen_core import CancellationToken
from fastapi import WebSocket, WebSocketDisconnect
from pathlib import Path
from ....types import CheckpointEvent
from ...database import DatabaseManager
from ...datamodel import (
    LLMCallEventMessage,
    Message,
    MessageConfig,
    Run,
    RunStatus,
    Settings,
    TeamResult,
    TrustProfile,
)
from ...teammanager import TeamManager

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and message streaming for team task execution

    Args:
        db_manager (DatabaseManager): Database manager instance for database operations
        internal_workspace_root (Path): Path to the internal root directory
        external_workspace_root (Path): Path to the external root directory
        inside_docker (bool): Flag indicating if the application is running inside Docker
        run_without_docker (bool): Flag indicating if the application is running without docker
        config (dict): Configuration for HALO
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        internal_workspace_root: Path,
        external_workspace_root: Path,
        inside_docker: bool,
        config: Dict[str, Any],
        run_without_docker: bool,
    ):
        self.db_manager = db_manager
        self.internal_workspace_root = internal_workspace_root
        self.external_workspace_root = external_workspace_root
        self.inside_docker = inside_docker
        self.config = config
        self.run_without_docker = run_without_docker
        self._connections: Dict[int, WebSocket] = {}
        self._cancellation_tokens: Dict[int, CancellationToken] = {}
        # Track explicitly closed connections
        self._closed_connections: set[int] = set()
        self._input_responses: Dict[int, asyncio.Queue[str]] = {}
        self._team_managers: Dict[int, TeamManager] = {}
        self._halo_states: Dict[int, HALOState] = {}
        self._cancel_message = TeamResult(
            task_result=TaskResult(
                messages=[TextMessage(source="user", content="Run cancelled by user")],
                stop_reason="cancelled by user",
            ),
            usage="",
            duration=0,
        ).model_dump()

    def _get_stop_message(self, reason: str) -> dict[str, Any]:
        return TeamResult(
            task_result=TaskResult(
                messages=[TextMessage(source="user", content=reason)],
                stop_reason=reason,
            ),
            usage="",
            duration=0,
        ).model_dump()

    async def _save_halo_state(self, run_id: int) -> None:
        """Persist the current HALOState for run_id to the Run.halo_state DB column."""
        hs = self._halo_states.get(run_id)
        if hs is None:
            return
        try:
            run = await self._get_run(run_id)
            if run is not None:
                run.halo_state = hs.to_dict()
                self.db_manager.upsert(run)
                # Ali Akbar Start (Gap 3 — also persist trust at the user level when
                # closed-loop enforcement is enabled, so it survives across runs)
                if self._trust_feedback_enabled(run_id) and run.user_id:
                    await self._save_trust_profile(run.user_id, hs.feedback_loop)
                # Ali Akbar End (Gap 3)
        except Exception as exc:
            logger.warning(f"HALO: failed to save state for run {run_id}: {exc}")

    # Ali Akbar Start (Gap 3 — closed-loop trust enforcement helpers)
    def _trust_feedback_enabled(self, run_id: int) -> bool:
        """Whether HALOAppConfig.enable_trust_feedback is set for this run."""
        team_manager = self._team_managers.get(run_id)
        halo_cfg = getattr(team_manager, "halo_config", None) if team_manager else None
        return bool(getattr(halo_cfg, "enable_trust_feedback", False))

    def _get_approval_guard(self, run_id: int) -> Optional[Any]:
        """The live ApprovalGuard shared by the orchestrator and web_surfer for this run."""
        team_manager = self._team_managers.get(run_id)
        return getattr(team_manager, "approval_guard", None) if team_manager else None

    def _apply_trust_policy(self, run_id: int, policy: str) -> None:
        """Write a trust-derived policy into the run's live ApprovalGuard, unless the
        user has explicitly turned approvals off ("never")."""
        guard = self._get_approval_guard(run_id)
        if guard is None:
            return
        try:
            if getattr(guard.config, "approval_policy", None) != "never":
                guard.config.approval_policy = policy
        except Exception as exc:
            logger.warning(f"HALO: failed to apply trust policy for run {run_id}: {exc}")

    async def _load_trust_profile(self, user_id: str) -> Optional[TrustProfile]:
        try:
            response = self.db_manager.get(
                filters={"user_id": user_id}, model_class=TrustProfile, return_json=False
            )
            return response.data[0] if response.status and response.data else None
        except Exception as exc:
            logger.warning(f"HALO: failed to load trust profile for user {user_id}: {exc}")
            return None

    async def _save_trust_profile(
        self, user_id: str, feedback_loop: BayesianFeedbackLoop
    ) -> None:
        try:
            state = feedback_loop.get_state()
            profile = await self._load_trust_profile(user_id)
            if profile is None:
                profile = TrustProfile(user_id=user_id)
            profile.alpha = state["alpha"]
            profile.beta = state["beta"]
            self.db_manager.upsert(profile)
        except Exception as exc:
            logger.warning(f"HALO: failed to persist trust profile for user {user_id}: {exc}")
    # Ali Akbar End (Gap 3)

    async def connect(self, websocket: WebSocket, run_id: int) -> bool:
        try:
            await websocket.accept()
            self._connections[run_id] = websocket
            self._closed_connections.discard(run_id)
            # Initialize input queue for this connection
            self._input_responses[run_id] = asyncio.Queue()

            await self._send_message(
                run_id,
                {
                    "type": "system",
                    "status": "connected",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Ali Akbar — restore persisted HALO state so the panel repopulates
            # immediately when a completed/reloaded session is opened.
            try:
                run = await self._get_run(run_id)
                if run is not None and run.halo_state:
                    hs = HALOState.from_dict(run.halo_state, task_id=str(run_id))

                    async def _halo_send_connect(msg: dict) -> None:
                        await self._send_message(run_id, msg)

                    hs.send_update = _halo_send_connect
                    self._halo_states[run_id] = hs

                    # Re-send classification so Gap 1 panel fills in
                    if hs.risk_estimation or hs.task_type != "research":
                        await self._send_message(run_id, {
                            "type": "halo_classification",
                            "task_type": hs.task_type,
                            "policy": hs.policy,
                            "risk_score": hs.risk_score,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "risk_estimation": hs.risk_estimation,
                        })

                    # Re-send full state so Gap 2 + Gap 3 panels fill in
                    await self._send_message(run_id, hs.state_snapshot())
            except Exception as _re:
                logger.warning(f"HALO: failed to restore state for run {run_id}: {_re}")
            # ------------------------------------------------------------------

            return True
        except Exception as e:
            logger.error(f"Connection error for run {run_id}: {e}")
            return False

    async def start_stream(
        self,
        run_id: int,
        task: str | ChatMessage | Sequence[ChatMessage] | None,
        team_config: Dict[str, Any],
        settings_config: Dict[str, Any],
        user_settings: Settings | None = None,
    ) -> None:
        """
        Start streaming task execution with proper run management

        Args:
            run_id (int): ID of the run
            task (str | ChatMessage | Sequence[ChatMessage] | None): Task to execute
            team_config (Dict[str, Any]): Configuration for the team
            settings_config (Dict[str, Any]): Configuration for settings
            user_settings (Settings, optional): User settings for the run
        """
        if run_id not in self._connections or run_id in self._closed_connections:
            raise ValueError(f"No active connection for run {run_id}")

        # do not create a new team manager if one already exists
        if run_id not in self._team_managers:
            team_manager = TeamManager(
                internal_workspace_root=self.internal_workspace_root,
                external_workspace_root=self.external_workspace_root,
                inside_docker=self.inside_docker,
                config=self.config,
                run_without_docker=self.run_without_docker,
            )
            self._team_managers[run_id] = team_manager

        else:
            team_manager = self._team_managers[run_id]
        cancellation_token = CancellationToken()
        self._cancellation_tokens[run_id] = cancellation_token
        final_result = None

        try:
            # Update run with task and status
            run = await self._get_run(run_id)
            assert run is not None, f"Run {run_id} not found in database"
            assert run.user_id is not None, f"Run {run_id} has no user ID"

            env_vars = None

            settings_config["memory_controller_key"] = run.user_id

            state = None
            if run:
                run.task = MessageConfig(content=task, source="user").model_dump()
                run.status = RunStatus.ACTIVE
                state = run.state
                self.db_manager.upsert(run)
                await self._update_run_status(run_id, RunStatus.ACTIVE)

            # add task as message
            if isinstance(task, str):
                await self._send_message(
                    run_id,
                    self._format_message(TextMessage(source="user_proxy", content=task))
                    or {},
                )
                await self._save_message(
                    run_id, TextMessage(source="user_proxy", content=task)
                )

            elif isinstance(task, Sequence):
                for task_message in task:
                    if isinstance(task_message, TextMessage) or isinstance(
                        task_message, MultiModalMessage
                    ):
                        if (
                            hasattr(task_message, "metadata")
                            and task_message.metadata.get("internal") == "yes"
                        ):
                            continue

                        await self._send_message(
                            run_id, self._format_message(task_message) or {}
                        )
                        await self._save_message(run_id, task_message)

            # --- HALO session state setup (Features 7–11) ---
            _hs = HALOState(task_id=str(run_id))
            _run_id_cap = run_id

            async def _halo_send(msg: dict) -> None:
                await self._send_message(_run_id_cap, msg)

            _hs.send_update = _halo_send
            self._halo_states[run_id] = _hs
            # -------------------------------------------------

            input_func: InputFuncType = self.create_input_func(run_id)

            # Ali Akbar Start (Gap 3 — one-time per-user trust profile restore)
            # team_manager.halo_config/approval_guard only exist once _create_team()
            # has run inside team_manager.run_stream(), which happens before the
            # first message is yielded — so this is checked on the first loop
            # iteration below rather than here.
            _trust_profile_loaded = False
            # Ali Akbar End (Gap 3)

            message: ChatMessage | AgentEvent | TeamResult | LLMCallEventMessage
            async for message in team_manager.run_stream(
                task=task,
                team_config=team_config,
                state=state,
                input_func=input_func,
                cancellation_token=cancellation_token,
                env_vars=env_vars,
                settings_config=settings_config,
                run=run,
            ):
                if (
                    cancellation_token.is_cancelled()
                    or run_id in self._closed_connections
                ):
                    logger.info(
                        f"Stream cancelled or connection closed for run {run_id}"
                    )
                    break

                # Ali Akbar Start (Gap 3 — restore persisted per-user trust once
                # the run's ApprovalGuard/config are available)
                if not _trust_profile_loaded:
                    _trust_profile_loaded = True
                    if self._trust_feedback_enabled(run_id) and run.user_id:
                        _profile = await self._load_trust_profile(run.user_id)
                        if _profile is not None:
                            _hs.feedback_loop.restore_from(
                                alpha=_profile.alpha or {}, beta=_profile.beta or {}
                            )
                # Ali Akbar End (Gap 3)

                if isinstance(message, CheckpointEvent):
                    # Save state to run
                    run = await self._get_run(run_id)
                    if run:
                        # Store state as JSON string
                        run.state = message.state
                        self.db_manager.upsert(run)
                    continue

                # do not show internal messages
                if (
                    hasattr(message, "metadata")
                    and message.metadata.get("internal") == "yes"  # type: ignore
                ):
                    continue

                formatted_message = self._format_message(message)
                if formatted_message:
                    await self._send_message(run_id, formatted_message)

                    # Gap 1 (hybrid) — intercept classification metadata to push badge update
                    # Gap 3 — use Bayesian-derived policy, not the raw classifier policy
                    _meta = getattr(message, "metadata", None)
                    if isinstance(_meta, dict) and _meta.get("_halo_classification") == "true":
                        _task_type = str(_meta.get("task_type", "research"))
                        _hs2 = self._halo_states.get(run_id)
                        if _hs2:
                            _hs2.task_type = _task_type
                            # Layer 1 — store final fused risk score
                            try:
                                _hs2.risk_score = float(_meta.get("risk_score", "0.15"))
                            except (ValueError, TypeError):
                                _hs2.risk_score = 0.15
                            # Layer 3 — Bayesian policy overrides raw classifier policy
                            _hs2.policy = _hs2.feedback_loop.get_policy(_task_type)

                            # Ali Akbar Start (Gap 3 — closed loop: the trust-adjusted
                            # policy now actually gates enforcement, not just display)
                            if self._trust_feedback_enabled(run_id):
                                self._apply_trust_policy(run_id, _hs2.policy)
                            # Ali Akbar End (Gap 3)

                            # Hybrid Gap 1 — build and store full risk_estimation detail
                            import json as _cmeta_json
                            try:
                                _hs2.risk_estimation = {
                                    "rule_task_type":       _meta.get("rule_task_type", _task_type),
                                    "rule_risk_score":      float(_meta.get("rule_risk_score", _hs2.risk_score)),
                                    "rule_policy":          _meta.get("rule_policy", "auto-permissive"),
                                    "rule_reason":          _meta.get("rule_reason", ""),
                                    "rule_matched_keywords": _cmeta_json.loads(_meta.get("rule_matched_keywords", "[]")),
                                    "llm_task_type":          _meta.get("llm_task_type", ""),
                                    "llm_risk_score":         float(_meta.get("llm_risk_score", 0.0)),
                                    "llm_confidence":         float(_meta.get("llm_confidence", 0.0)),
                                    "llm_reason":             _meta.get("llm_reason", ""),
                                    "llm_possible_harms":     _cmeta_json.loads(_meta.get("llm_possible_harms", "[]")),
                                    "llm_recommended_policy": _meta.get("llm_recommended_policy", ""),
                                    "llm_available":          _meta.get("llm_available", "false") == "true",
                                    "final_task_type":  _task_type,
                                    "final_risk_score": _hs2.risk_score,
                                    "final_policy":     str(_meta.get("policy", _hs2.policy)),
                                    "fusion_mode":      _meta.get("fusion_mode", "rule-only fallback"),
                                }
                            except Exception as _re:
                                logger.warning(f"HALO risk_estimation build error: {_re}")

                        _bayesian_policy = _hs2.policy if _hs2 else str(_meta.get("policy", "auto-permissive"))
                        _risk_val = _hs2.risk_score if _hs2 else 0.15
                        await self._send_message(run_id, {
                            "type": "halo_classification",
                            "task_type": _task_type,
                            "policy": _bayesian_policy,
                            "risk_score": _risk_val,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "risk_estimation": _hs2.risk_estimation if _hs2 else {},
                        })
                        # Persist Gap 1 classification to DB
                        await self._save_halo_state(run_id)

                    # Save messages by concrete type
                    if isinstance(
                        message,
                        (
                            TextMessage,
                            MultiModalMessage,
                            StopMessage,
                            HandoffMessage,
                            ToolCallRequestEvent,
                            ToolCallExecutionEvent,
                            LLMCallEventMessage,
                        ),
                    ):
                        await self._save_message(run_id, message)
                    # Capture final result if it's a TeamResult
                    elif isinstance(message, TeamResult):
                        final_result = message.model_dump()
                    self._team_managers[run_id] = team_manager  # Track the team manager
            if (
                not cancellation_token.is_cancelled()
                and run_id not in self._closed_connections
            ):
                if final_result:
                    await self._update_run(
                        run_id, RunStatus.COMPLETE, team_result=final_result
                    )
                else:
                    logger.warning(
                        f"No final result captured for completed run {run_id}"
                    )
                    await self._update_run_status(run_id, RunStatus.COMPLETE)
            else:
                await self._send_message(
                    run_id,
                    {
                        "type": "completion",
                        "status": "cancelled",
                        "data": self._cancel_message,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                # Update run with cancellation result
                await self._update_run(
                    run_id, RunStatus.STOPPED, team_result=self._cancel_message
                )

        except Exception as e:
            logger.error(f"Stream error for run {run_id}: {e}")
            traceback.print_exc()
            await self._handle_stream_error(run_id, e)
        finally:
            self._cancellation_tokens.pop(run_id, None)
            self._team_managers.pop(run_id, None)  # Remove the team manager when done

    async def _save_message(
        self, run_id: int, message: Union[AgentEvent | ChatMessage, LLMCallEventMessage]
    ) -> None:
        """
        Save a message to the database

        Args:
            run_id (int): ID of the run
            message (Union[AgentEvent | ChatMessage, LLMCallEventMessage]): Message to save
        """

        run = await self._get_run(run_id)
        if run:
            db_message = Message(
                created_at=datetime.now(),
                session_id=run.session_id,
                run_id=run_id,
                config=message.model_dump(),
                user_id=run.user_id,  # Pass the user_id from the run object
            )
            self.db_manager.upsert(db_message)

    async def _update_run(
        self,
        run_id: int,
        status: RunStatus,
        team_result: Optional[TeamResult | Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update run status and result

        Args:
            run_id (int): ID of the run
            status (RunStatus): New status to set
            team_result (TeamResult | dict[str, Any], optional): Optional team result to set
            error (str, optional): Optional error message
        """
        run = await self._get_run(run_id)
        if run:
            run.status = status
            if team_result:
                run.team_result = team_result
            if error:
                run.error_message = error
            self.db_manager.upsert(run)

    def create_input_func(self, run_id: int, timeout: int = 3600) -> InputFuncType:
        """
        Creates an input function for a specific run

        Args:
            run_id (int): ID of the run
            timeout (int, optional): Timeout for input response in seconds. Default: 3600
        Returns:
            InputFuncType: Input function for the run
        """
        # Snapshot the HALOState reference at creation time (already set by start_stream)
        _halo_state = self._halo_states.get(run_id)

        async def input_handler(
            prompt: str = "",
            cancellation_token: Optional[CancellationToken] = None,
            input_type: InputRequestType = "text_input",
        ) -> str:
            try:
                # --- halo_scan_report: silent state update, no user prompt ---
                # Ali Akbar Start (Gap 2 — always report scan result regardless of detection)
                if input_type == "halo_scan_report" and _halo_state:
                    try:
                        import json as _srjson
                        _SCAN_SENTINEL = "<!--HALO_RISK:"
                        if _SCAN_SENTINEL in prompt:
                            _pre2, _rest2 = prompt.split(_SCAN_SENTINEL, 1)
                            _json_str2, _ = _rest2.split("-->", 1)
                            _scan_data = _srjson.loads(_json_str2)

                            _halo_state.injection_count += (
                                1 if _scan_data.get("final_injection_detected") else 0
                            )
                            _halo_state.injection_risk_score = float(
                                _scan_data.get("final_injection_risk_score",
                                               _scan_data.get("risk_score", 0.0))
                            )
                            _halo_state.injection_risk_level = str(
                                _scan_data.get("final_risk_level",
                                               _scan_data.get("risk_level", "none"))
                            )
                            _halo_state.injection_matched_patterns = list(
                                _scan_data.get("matched_patterns",
                                               _scan_data.get("patterns", []))
                            )
                            _halo_state.injection_detection = _scan_data
                            await _halo_state.push(_halo_state.state_snapshot())
                            await self._save_halo_state(run_id)
                    except Exception as _sre:
                        logger.warning(f"HALO scan_report parse error: {_sre}")
                    return "yes"
                # Ali Akbar End (Gap 2)

                # resume run if it is paused
                await self.resume_run(run_id)

                # update run status to awaiting_input
                await self._update_run_status(run_id, RunStatus.AWAITING_INPUT)

                # --- Hybrid Layer 2+3: parse injection risk before forwarding ---
                _cleaned_prompt = prompt
                if input_type == "injection_alert" and _halo_state:
                    try:
                        import json as _rjson
                        _SENTINEL = "<!--HALO_RISK:"
                        if _SENTINEL in prompt:
                            _pre, _rest = prompt.split(_SENTINEL, 1)
                            _json_str, _ = _rest.split("-->", 1)
                            _risk_data = _rjson.loads(_json_str)
                            _cleaned_prompt = _pre.rstrip()

                            # Prefer new hybrid fields; fall back to legacy single-layer fields
                            _final_risk  = float(_risk_data.get("final_injection_risk_score",
                                                                 _risk_data.get("risk_score", 0.0)))
                            _final_level = str(_risk_data.get("final_risk_level",
                                                              _risk_data.get("risk_level", "none")))
                            _final_pats  = list(_risk_data.get("matched_patterns",
                                                               _risk_data.get("patterns", [])))

                            _halo_state.injection_risk_score    = _final_risk
                            _halo_state.injection_risk_level    = _final_level
                            _halo_state.injection_matched_patterns = _final_pats

                            # Store full hybrid injection detection dict for UI
                            _halo_state.injection_detection = _risk_data

                            # Hybrid trust calibration: injection event contributes negative evidence.
                            # Updated once per unique page URL per session.
                            # Trust penalty is weighted by current task risk:
                            #   research→1.0x  transactional→1.25x  destructive→1.50x
                            _url_key = str(_risk_data.get("url", "_halo_no_url_"))
                            if _url_key not in _halo_state._injection_adjusted_urls:
                                _base_delta = {
                                    "low": 0.10, "medium": 0.25, "high": 0.50
                                }.get(_final_level, 0.0)
                                _task_mult = {
                                    "research": 1.0,
                                    "transactional": 1.25,
                                    "destructive": 1.50,
                                }.get(_halo_state.task_type, 1.0)
                                _beta_delta = _base_delta * _task_mult
                                if _beta_delta > 0 and _halo_state.task_type in _halo_state.feedback_loop._beta:
                                    _halo_state.feedback_loop._beta[_halo_state.task_type] += _beta_delta
                                    # Ali Akbar Start (Gap 3 — closed loop: an injection
                                    # event immediately tightens live enforcement, before
                                    # the user even answers the alert)
                                    if self._trust_feedback_enabled(run_id):
                                        self._apply_trust_policy(
                                            run_id,
                                            _halo_state.feedback_loop.get_policy(
                                                _halo_state.task_type
                                            ),
                                        )
                                    # Ali Akbar End (Gap 3)
                                _halo_state._injection_adjusted_urls.add(_url_key)

                            # Push sidebar update so panel reflects injection before user responds
                            await _halo_state.push(_halo_state.state_snapshot())
                            # Persist Gap 2 injection state to DB
                            await self._save_halo_state(run_id)
                    except Exception as _risk_exc:
                        logger.warning(f"HALO injection risk parse error: {_risk_exc}")
                # -----------------------------------------------------------------------

                # Send input request to client
                logger.info(
                    f"Sending input request for run {run_id}: ({input_type}) {_cleaned_prompt[:80]!r}"
                )
                _input_msg: dict = {
                    "type": "input_request",
                    "input_type": input_type,
                    "prompt": _cleaned_prompt,
                    "data": {"source": "system", "content": _cleaned_prompt},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if input_type == "injection_alert" and _halo_state:
                    _input_msg["injection_risk_score"]      = _halo_state.injection_risk_score
                    _input_msg["injection_risk_level"]      = _halo_state.injection_risk_level
                    _input_msg["injection_matched_patterns"] = _halo_state.injection_matched_patterns
                    _input_msg["injection_detection"]       = _halo_state.injection_detection
                await self._send_message(run_id, _input_msg)

                # Store input_request in the Run object
                run = await self._get_run(run_id)
                if run:
                    run.input_request = {"prompt": _cleaned_prompt, "input_type": input_type}
                    self.db_manager.upsert(run)

                # Wait for response with timeout
                if run_id in self._input_responses:
                    try:

                        async def poll_for_response():
                            while True:
                                # Check if run was closed/cancelled
                                if run_id in self._closed_connections:
                                    raise ValueError("Run was closed")

                                # Try to get response with short timeout
                                try:
                                    response = await asyncio.wait_for(
                                        self._input_responses[run_id].get(),
                                        timeout=min(timeout, 5),
                                    )
                                    await self._update_run_status(
                                        run_id, RunStatus.ACTIVE
                                    )
                                    return response
                                except asyncio.TimeoutError:
                                    continue  # Keep checking for closed status

                        response = await asyncio.wait_for(
                            poll_for_response(), timeout=timeout
                        )

                        # --- HALO Features 7–9: record decision in session state ---
                        if _halo_state and input_type in ("approval", "injection_alert"):
                            try:
                                import json as _json
                                try:
                                    _parsed = _json.loads(response)
                                    _accepted = bool(_parsed.get("accepted", True))
                                except Exception:
                                    _accepted = response.strip().lower() in ("accept", "yes", "y")

                                # Gap 2 — track injection count for the UI badge
                                if input_type == "injection_alert":
                                    _halo_state.injection_count += 1

                                # Gap 3 — Bayesian Trust Adaptation
                                _decision = "approve" if _accepted else "reject"
                                _old_pol = _halo_state.feedback_loop.get_policy(_halo_state.task_type)
                                _halo_state.feedback_loop.record_feedback(_halo_state.task_type, _decision)
                                _new_pol = _halo_state.feedback_loop.get_policy(_halo_state.task_type)
                                _halo_state.policy = _new_pol

                                # Ali Akbar Start (Gap 3 — closed loop: this decision's
                                # updated trust immediately governs the NEXT action in
                                # this run, not just the displayed badge)
                                if self._trust_feedback_enabled(run_id):
                                    self._apply_trust_policy(run_id, _new_pol)
                                # Ali Akbar End (Gap 3)

                                _escalation_payload = None
                                if _old_pol != _new_pol:
                                    _tt = _halo_state.task_type
                                    _escalation_payload = {
                                        "task_type": _tt,
                                        "old_policy": _old_pol,
                                        "new_policy": _new_pol,
                                        "trust_mean": round(_halo_state.feedback_loop.get_trust_mean(_tt), 4),
                                        "uncertainty": round(_halo_state.feedback_loop.get_uncertainty(_tt), 4),
                                        "confidence": _halo_state.feedback_loop.get_confidence(_tt),
                                    }

                                await _halo_state.push(
                                    _halo_state.state_snapshot(escalation=_escalation_payload)
                                )
                                # Persist Gap 3 Bayesian state to DB
                                await self._save_halo_state(run_id)
                            except Exception as _he:
                                logger.warning(f"HALO state recording error: {_he}")
                        # ----------------------------------------------------------

                        return response

                    except asyncio.TimeoutError:
                        # Stop the run if timeout occurs
                        logger.warning(f"Input response timeout for run {run_id}")
                        await self.stop_run(
                            run_id,
                            "HALO timed out while waiting for your input. To resume, please enter a follow-up message in the input box or you can simply type 'continue'.",
                        )
                        raise
                else:
                    raise ValueError(f"No input queue for run {run_id}")

            except Exception as e:
                logger.error(f"Error handling input for run {run_id}: {e}")
                raise

        return input_handler

    async def handle_input_response(self, run_id: int, response: str) -> None:
        """Handle input response from client"""
        if run_id in self._input_responses:
            await self._input_responses[run_id].put(response)
        else:
            logger.warning(f"Received input response for inactive run {run_id}")

    async def stop_run(self, run_id: int, reason: str) -> None:
        if run_id in self._cancellation_tokens:
            logger.info(f"Stopping run {run_id}")

            stop_message = self._get_stop_message(reason)

            try:
                # Update run record first
                await self._update_run(
                    run_id, status=RunStatus.STOPPED, team_result=stop_message
                )

                # Then handle websocket communication if connection is active
                if (
                    run_id in self._connections
                    and run_id not in self._closed_connections
                ):
                    await self._send_message(
                        run_id,
                        {
                            "type": "completion",
                            "status": "cancelled",
                            "data": stop_message,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

                # Finally cancel the token
                self._cancellation_tokens[run_id].cancel()
                # remove team manager
                team_manager = self._team_managers.pop(run_id, None)
                if team_manager:
                    await team_manager.close()
            except Exception as e:
                logger.error(f"Error stopping run {run_id}: {e}")
                # We might want to force disconnect here if db update failed
                # await self.disconnect(run_id)  # Optional

    async def disconnect(self, run_id: int) -> None:
        """
        Clean up connection and associated resources

        Args:
            run_id (int): ID of the run to disconnect
        """
        logger.info(f"Disconnecting run {run_id}")

        # Mark as closed before cleanup to prevent any new messages
        self._closed_connections.add(run_id)

        # Cancel any running tasks
        await self.stop_run(run_id, "Connection closed")

        # Clean up resources
        self._connections.pop(run_id, None)
        self._cancellation_tokens.pop(run_id, None)
        self._input_responses.pop(run_id, None)
        self._halo_states.pop(run_id, None)

    async def _send_message(self, run_id: int, message: Dict[str, Any]) -> None:
        """Send a message through the WebSocket with connection state checking

        Args:
            run_id (int): int of the run
            message (Dict[str, Any]): Message dictionary to send
        """
        if run_id in self._closed_connections:
            logger.warning(
                f"Attempted to send message to closed connection for run {run_id}"
            )
            return

        try:
            if run_id in self._connections:
                websocket = self._connections[run_id]
                await websocket.send_json(message)
        except WebSocketDisconnect:
            logger.warning(
                f"WebSocket disconnected while sending message for run {run_id}"
            )
            await self.disconnect(run_id)
        except Exception as e:
            logger.error(f"Error sending message for run {run_id}: {e}, {message}")
            # Don't try to send error message here to avoid potential recursive loop
            await self._update_run_status(run_id, RunStatus.ERROR, str(e))
            await self.disconnect(run_id)

    async def _handle_stream_error(self, run_id: int, error: Exception) -> None:
        """
        Handle stream errors with proper run updates

        Args:
            run_id (int): ID of the run
            error (Exception): Exception that occurred
        """
        if run_id not in self._closed_connections:
            error_result = TeamResult(
                task_result=TaskResult(
                    messages=[TextMessage(source="system", content=str(error))],
                    stop_reason="An error occurred while processing this run",
                ),
                usage="",
                duration=0,
            ).model_dump()

            await self._send_message(
                run_id,
                {
                    "type": "completion",
                    "status": "error",
                    "data": error_result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            await self._update_run(
                run_id, RunStatus.ERROR, team_result=error_result, error=str(error)
            )

    def _format_message(self, message: Any) -> Optional[Dict[str, Any]]:
        """Format message for WebSocket transmission

        Args:
            message (Any): Message to format

        Returns:
            Optional[Dict[str, Any]]: Formatted message or None if formatting fails
        """

        try:
            if isinstance(message, MultiModalMessage):
                message_dump = message.model_dump()

                message_content: list[dict[str, Any]] = []
                for row in message_dump["content"]:
                    if "data" in row:
                        message_content.append(
                            {
                                "url": f"data:image/png;base64,{row['data']}",
                                "alt": "HALOWebSurfer Screenshot",
                            }
                        )
                    else:
                        message_content.append(row)
                message_dump["content"] = message_content

                return {"type": "message", "data": message_dump}

            elif isinstance(message, TeamResult):
                return {
                    "type": "result",
                    "data": message.model_dump(),
                    "status": "complete",
                }
            elif isinstance(message, ModelClientStreamingChunkEvent):
                return {"type": "message_chunk", "data": message.model_dump()}

            elif isinstance(
                message,
                (
                    BaseTextChatMessage,
                    ToolCallRequestEvent,
                    ToolCallExecutionEvent,
                ),
            ):
                return {"type": "message", "data": message.model_dump()}
            elif isinstance(message, str):
                return {
                    "type": "message",
                    "data": {"source": "user", "content": message},
                }
            else:
                logger.warning(
                    f"Cannot format unrecognized message type: {type(message)}"
                )

            return None

        except Exception as e:
            logger.error(f"Message formatting error: {e}")
            return None

    async def _get_run(self, run_id: int) -> Optional[Run]:
        """Get run from database

        Args:
            run_id (int): int of the run to retrieve

        Returns:
            Optional[Run]: Run object if found, None otherwise
        """
        response = self.db_manager.get(Run, filters={"id": run_id}, return_json=False)
        return response.data[0] if response.status and response.data else None

    async def _get_settings(self, user_id: str) -> Optional[Settings]:
        """Get user settings from database
        Args:
            user_id (str): User ID to retrieve settings for
        Returns:
            Optional[Settings]: User settings if found, None otherwise
        """
        response = self.db_manager.get(
            filters={"user_id": user_id}, model_class=Settings, return_json=False
        )
        return response.data[0] if response.status and response.data else None

    async def _update_run_status(
        self, run_id: int, status: RunStatus, error: Optional[str] = None
    ) -> None:
        """Update run status in database

        Args:
            run_id (int): int of the run to update
            status (RunStatus): New status to set
            error (str, optional): Optional error message
        """
        run = await self._get_run(run_id)
        if run:
            run.status = status
            run.error_message = error
            self.db_manager.upsert(run)
        # send system message to client with status
        await self._send_message(
            run_id,
            {
                "type": "system",
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def cleanup(self) -> None:
        """Clean up all active connections and resources when server is shutting down"""
        logger.info(f"Cleaning up {len(self.active_connections)} active connections")

        try:
            # First cancel all running tasks
            for run_id in self.active_runs.copy():
                if run_id in self._cancellation_tokens:
                    self._cancellation_tokens[run_id].cancel()
                run = await self._get_run(run_id)
                if run and run.status == RunStatus.ACTIVE:
                    interrupted_result = TeamResult(
                        task_result=TaskResult(
                            messages=[
                                TextMessage(
                                    source="system",
                                    content="Run interrupted by server shutdown",
                                )
                            ],
                            stop_reason="server_shutdown",
                        ),
                        usage="",
                        duration=0,
                    ).model_dump()

                    run.status = RunStatus.STOPPED
                    run.team_result = interrupted_result
                    self.db_manager.upsert(run)

            # Then disconnect all websockets with timeout
            # 10 second timeout for entire cleanup
            async def disconnect_all():
                for run_id in self.active_connections.copy():
                    try:
                        await asyncio.wait_for(self.disconnect(run_id), timeout=2)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout disconnecting run {run_id}")
                    except Exception as e:
                        logger.error(f"Error disconnecting run {run_id}: {e}")

            await asyncio.wait_for(disconnect_all(), timeout=10)

        except asyncio.TimeoutError:
            logger.warning("WebSocketManager cleanup timed out")
        except Exception as e:
            logger.error(f"Error during WebSocketManager cleanup: {e}")
        finally:
            # Always clear internal state, even if cleanup had errors
            self._connections.clear()
            self._cancellation_tokens.clear()
            self._closed_connections.clear()
            self._input_responses.clear()

    @property
    def active_connections(self) -> set[int]:
        """Get set of active run IDs"""
        return set(self._connections.keys()) - self._closed_connections

    @property
    def active_runs(self) -> set[int]:
        """Get set of runs with active cancellation tokens"""
        return set(self._cancellation_tokens.keys())

    async def pause_run(self, run_id: int) -> None:
        """Pause the run"""
        if (
            run_id in self._connections
            and run_id not in self._closed_connections
            and run_id in self._team_managers
        ):
            team_manager = self._team_managers.get(run_id)
            if team_manager:
                await team_manager.pause_run()
                # await self._send_message(
                #     run_id,
                #     {
                #         "type": "system",
                #         "status": "paused",
                #         "timestamp": datetime.now(timezone.utc).isoformat(),
                #     },
                # )
                # await self._update_run_status(run_id, RunStatus.PAUSED)

    async def resume_run(self, run_id: int) -> None:
        """Resume the run"""
        if (
            run_id in self._connections
            and run_id not in self._closed_connections
            and run_id in self._team_managers
        ):
            team_manager = self._team_managers.get(run_id)
            if team_manager:
                await team_manager.resume_run()
                await self._update_run_status(run_id, RunStatus.ACTIVE)
