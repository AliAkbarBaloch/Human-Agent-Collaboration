from .approval_guard import (
    BaseApprovalGuard,
    MaybeRequiresApproval,
    DEFAULT_REQUIRES_APPROVAL,
)
from .injection_gateway import is_scan_active
from .semantic_injection_detector import screen_action_for_hijack
from .tools.tool_metadata import get_tool_metadata, REQUIRE_APPROVAL_KEY
from autogen_core.tools import ToolSchema
from autogen_core.models import LLMMessage
from autogen_agentchat.messages import (
    MultiModalMessage,
    TextMessage,
)

from typing import (
    Any,
    Dict,
    List,
    Optional,
    cast,
    Protocol,
    TypeVar,
    Union,
    Callable,
    Generic,
)
from dataclasses import dataclass

from inspect import iscoroutinefunction

from abc import ABC, abstractmethod

TReturn = TypeVar("TReturn", covariant=True)
TStream = TypeVar("TStream", covariant=True)


class SyncActionCallable(Protocol[TReturn]):
    def __call__(self, *args: Any, **kwargs: Any) -> TReturn: ...


class AsyncActionCallable(Protocol[TReturn]):
    async def __call__(self, *args: Any, **kwargs: Any) -> TReturn: ...


ActionCallable = SyncActionCallable[TReturn] | AsyncActionCallable[TReturn]


class CallableInvoker(Generic[TReturn]):
    def __init__(self, callable: ActionCallable[TReturn]) -> None:
        self._callable = callable
        self._is_async = iscoroutinefunction(callable)

    async def __call__(self, *args: Any, **kwargs: Any) -> TReturn:
        if self._is_async:
            return await cast(AsyncActionCallable[TReturn], self._callable)(
                *args, **kwargs
            )
        else:
            return cast(SyncActionCallable[TReturn], self._callable)(*args, **kwargs)


class ApprovalDeniedError(Exception):
    """Exception raised when an action is denied by the approval guard."""

    ...


DescriptionGenerator = Callable[[Dict[str, Any]], Union[TextMessage, MultiModalMessage]]


# Ali Akbar Prompt Injection Start
def _append_hijack_warning(
    msg: Union[TextMessage, MultiModalMessage], reason: str
) -> Union[TextMessage, MultiModalMessage]:
    """Prepend a hijack-screen warning onto an approval-prompt message so the user
    sees *why* they're being asked, not just that they're being asked."""
    warning = f"\n\n⚠️ Possible instruction hijack detected: {reason}"
    if isinstance(msg, TextMessage):
        return TextMessage(content=msg.content + warning, source=msg.source)
    if isinstance(msg, MultiModalMessage):
        return MultiModalMessage(content=list(msg.content) + [warning], source=msg.source)
    return msg
# Ali Akbar Prompt Injection End


@dataclass
class BaseGuardedAction(Generic[TReturn], ABC):
    name: str
    action: CallableInvoker[TReturn]
    prepare: Optional[CallableInvoker[None]] = None
    cleanup: Optional[CallableInvoker[None]] = None

    @abstractmethod
    def _get_baseline(self) -> MaybeRequiresApproval: ...

    async def invoke_with_approval(
        self,
        call_arguments: Dict[str, Any],
        action_description: Union[
            TextMessage,
            MultiModalMessage,
            ActionCallable[TextMessage]
            | ActionCallable[MultiModalMessage]
            | ActionCallable[TextMessage | MultiModalMessage],
        ],
        action_context: List[LLMMessage],
        action_guard: Optional[BaseApprovalGuard],
        action_description_for_user: Optional[
            Union[TextMessage, MultiModalMessage]
        ] = None,
        *,
        instruction: Optional[str] = None,
        hybrid_injection_detection: bool = True,
    ) -> TReturn:
        """
        Invokes the action with approval if the action guard is provided.
        Args:
            call_arguments (Dict[str, Any]): The arguments to pass to the action.
            action_description (TextMessage | MultiModalMessage): The description of the action to be approved in it's raw form.
            action_context (List[LLMMessage]): The context of the action to be approved.
            action_guard (ApprovalGuard, optional): The action guard to use to approve the action.
            action_description_for_user (TextMessage | MultiModalMessage, optional): The description of the action for the user.
            instruction (str, optional): What the agent was told to do this turn — used for
                action-hijack screening. Falls back to the last message in action_context.
            hybrid_injection_detection (bool): Whether to run action-hijack screening
                (Ali Akbar Prompt Injection — see below). Default True.
        """
        needs_approval: bool = False
        # Ali Akbar Prompt Injection Start
        _hijack_reason: Optional[str] = None
        # Ali Akbar Prompt Injection End
        if action_guard is not None:
            baseline: MaybeRequiresApproval = self._get_baseline()
            llm_guess: MaybeRequiresApproval = baseline

            if REQUIRE_APPROVAL_KEY in call_arguments:
                if call_arguments[REQUIRE_APPROVAL_KEY]:
                    llm_guess = "always"
                else:
                    llm_guess = "never"

            # Check if the action needs approval
            needs_approval = await action_guard.requires_approval(
                baseline,
                llm_guess,
                action_context,
            )

            # Ali Akbar Prompt Injection Start
            # Action-hijack screening — independent of the risk check above. Asks
            # whether this proposed action looks like it was redirected by something
            # previously read (a file, code output, a tool result), rather than
            # serving the instruction this turn. Forces approval if so, even for
            # baselines/policies that would otherwise skip it. Shared by every
            # GuardedAction/TrivialGuardedAction caller (file_surfer, coder) — one
            # hook covers both.
            if hybrid_injection_detection and is_scan_active(action_guard):
                _model_client = getattr(action_guard, "model_client", None)
                if _model_client is not None:
                    _instruction = instruction
                    if _instruction is None and action_context:
                        _last_content = getattr(action_context[-1], "content", "")
                        _instruction = (
                            _last_content
                            if isinstance(_last_content, str)
                            else str(_last_content)
                        )
                    _proposed_action = f"{self.name}({call_arguments})"
                    _hijack = await screen_action_for_hijack(
                        _instruction or "", _proposed_action, _model_client
                    )
                    if (
                        _hijack.available
                        and _hijack.hijack_suspected
                        and _hijack.confidence >= 0.6
                    ):
                        needs_approval = True
                        _hijack_reason = _hijack.reason
            # Ali Akbar Prompt Injection End

        if self.prepare is not None:
            await self.prepare()

        try:
            if needs_approval:
                assert action_guard is not None

                if callable(action_description):
                    # If action_description is a callable, convert it to an Invoker and call it
                    invoker = CallableInvoker(action_description)
                    action_description = await invoker(**call_arguments)

                # Get approval for the action
                final_description = (
                    action_description_for_user
                    if action_description_for_user is not None
                    else action_description
                )

                # Ali Akbar Prompt Injection Start
                if _hijack_reason:
                    final_description = _append_hijack_warning(
                        final_description, _hijack_reason
                    )
                # Ali Akbar Prompt Injection End

                approved = await action_guard.get_approval(final_description)

                if not approved:
                    raise ApprovalDeniedError(
                        "Action was denied by the approval guard."
                    )

            # Invoke the action
            result = await self.action(**call_arguments)

            if self.cleanup is not None:
                await self.cleanup()

            return result
        except Exception:
            if self.cleanup is not None:
                await self.cleanup()

            raise


class GuardedAction(BaseGuardedAction[TReturn], Generic[TReturn]):
    def __init__(
        self,
        name: str,
        action: ActionCallable[TReturn],
        prepare: Optional[ActionCallable[None]] = None,
        cleanup: Optional[ActionCallable[None]] = None,
    ) -> None:
        super().__init__(
            name,
            CallableInvoker(action),
            CallableInvoker(prepare) if prepare else None,
            CallableInvoker(cleanup) if cleanup else None,
        )

    @staticmethod
    def from_schema(
        tool_schema: ToolSchema,
        action: ActionCallable[TReturn],
        prepare: Optional[ActionCallable[None]] = None,
        cleanup: Optional[ActionCallable[None]] = None,
    ) -> "GuardedAction[TReturn]":
        return GuardedAction(
            name=tool_schema.get("name"),
            action=CallableInvoker(action),
            prepare=CallableInvoker(prepare) if prepare else None,
            cleanup=CallableInvoker(cleanup) if cleanup else None,
        )

    def _get_baseline(self) -> MaybeRequiresApproval:
        metadata = get_tool_metadata(self.name)

        return metadata.get("requires_approval", DEFAULT_REQUIRES_APPROVAL)


class TrivialGuardedAction(BaseGuardedAction[None]):
    def __init__(
        self, name: str, baseline_override: Optional[MaybeRequiresApproval] = None
    ) -> None:
        super().__init__(name, CallableInvoker(lambda: None), None, None)
        self._baseline_override: MaybeRequiresApproval | None = baseline_override

    def _get_baseline(self) -> MaybeRequiresApproval:
        return (
            self._baseline_override
            if self._baseline_override is not None
            else DEFAULT_REQUIRES_APPROVAL
        )
