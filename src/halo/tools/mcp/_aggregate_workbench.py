import asyncio
import warnings
from typing import Any, Dict, List, Literal, Mapping

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken, Component
from autogen_core.tools import (
    TextResultContent,
    ToolResult,
    ToolSchema,
    Workbench,
)
from pydantic import BaseModel

from autogen_ext.tools.mcp import (
    McpWorkbench,
    McpServerParams,
)

# Ali Akbar Prompt Injection Start
from ...approval_guard import BaseApprovalGuard
from ...injection_gateway import is_scan_active, scan_and_gate
from ...semantic_injection_detector import screen_action_for_hijack
# Ali Akbar Prompt Injection End

# According to the OpenAI API tool names can contain "Only letters, numbers, '_' and '-' are allowed."
NAMESPACE_SEPARATOR = "-"
# Kept for backwards compatibility with any external caller importing this name;
# no longer used internally now that escape_tool_name/unescape_tool_name hex-escape
# every non-alphanumeric character instead of special-casing just NAMESPACE_SEPARATOR.
NAMESPACE_ESCAPE = ":"


def escape_tool_name(name: str) -> str:
    """
    Escapes a tool name so the result contains only ASCII letters, numbers, and
    underscores, which is always safe to join with NAMESPACE_SEPARATOR ('-')
    without ambiguity and always valid under OpenAI's function-name charset
    (letters, numbers, '_', '-').

    The previous version only replaced NAMESPACE_SEPARATOR ('-') with a fixed
    ':' character, on the assumption that MCP tool names never contain any
    other character outside [a-zA-Z0-9_-]. Real servers break that assumption
    (e.g. the '@modelcontextprotocol/server-everything' reference server ships
    a tool literally named 'get:annotated:message'), and ':' itself isn't a
    valid character for OpenAI tool names either, so any tool name containing
    '-' or a stray non-alphanumeric character crashed tool-schema conversion
    outright. Every character outside [a-zA-Z0-9] (including '-', '_', and
    ':') is now replaced with '_XX', its two-digit lowercase hex ordinal, so
    the mapping stays fully reversible and the escaped name is always safe.
    """
    return "".join(c if c.isalnum() and c.isascii() else f"_{ord(c):02x}" for c in name)


def unescape_tool_name(name: str) -> str:
    """
    Reverses escape_tool_name: each '_XX' run is decoded back to the
    single character it encodes.
    """
    result = []
    i = 0
    while i < len(name):
        if name[i] == "_" and i + 2 < len(name):
            result.append(chr(int(name[i + 1 : i + 3], 16)))
            i += 3
        else:
            result.append(name[i])
            i += 1
    return "".join(result)


class NamedMcpServerParams(BaseModel):
    """A 'namespaced' McpServer"""

    server_name: str
    """The unique name of the server"""
    server_params: McpServerParams
    """The SseServerParams or StdioServerParams for the server."""


class AggregateMcpWorkbenchConfig(BaseModel):
    named_server_params: List[NamedMcpServerParams]


class AggregateMcpWorkbenchState(BaseModel):
    type: Literal["AggregateMcpWorkbenchState"] = "AggregateMcpWorkbenchState"


class AggregateMcpWorkbench(Workbench, Component[AggregateMcpWorkbenchConfig]):
    """
    A workbench that aggregates multiple named MCP servers, providing a unified interface
    to list and call tools from all servers. Each server is given a unique name, and tools
    from each server are namespaced using this name (e.g., "server1.tool_name").

    Args:
        named_server_params (List[NamedMcpServerParams]):
            A list of server configurations, each with a unique name and corresponding MCP server parameters.

    Examples:

        Here is an example of how to use the aggregate workbench with two MCP servers:

        .. code-block:: python

            import asyncio

            from autogen_ext.tools.mcp import StdioServerParams
            from halo.tools.mcp import AggregateMcpWorkbench, NamedMcpServerParams

            async def main() -> None:
                server1 = NamedMcpServerParams(
                    server_name="fetch",
                    server_params=StdioServerParams(command="uvx", args=["mcp-server-fetch"]),
                )
                server2 = NamedMcpServerParams(
                    server_name="github",
                    server_params=StdioServerParams(command="docker", args=["run", "ghcr.io/github/github-mcp-server"]),
                )
                async with AggregateMcpWorkbench([server1, server2]) as workbench:
                    tools = await workbench.list_tools()
                    print(tools)  # Tool names will be namespaced, e.g., 'fetch.tool1', 'github.tool2'
                    result = await workbench.call_tool("fetch.some_tool", {"url": "https://github.com/"})
                    print(result)

            asyncio.run(main())

    Notes:
        - Tool names are automatically namespaced with their server name (e.g., 'server_name.tool_name').
        - Use the namespaced tool name when calling tools.
        - All server names must be unique.
    """

    component_provider_override = "halo.tools.mcp.AggregateMcpWorkbench"
    component_config_schema = AggregateMcpWorkbenchConfig

    def __init__(self, named_server_params: List[NamedMcpServerParams]) -> None:
        # Create a copy of server_params
        self._workbenches: Dict[str, McpWorkbench] = {}
        # Ali Akbar Prompt Injection Start
        # Not constructor params — kept out of AggregateMcpWorkbenchConfig/serialization;
        # set post-construction via configure_injection_guard(), same pattern used
        # elsewhere in HALO for wiring a live ApprovalGuard onto an already-built agent.
        self._approval_guard: BaseApprovalGuard | None = None
        self._hybrid_injection_detection: bool = True
        self._current_instruction: str = ""
        self._blocked_mcp_ids: set[str] = set()
        # Ali Akbar Prompt Injection End
        for params in named_server_params:
            # Check if valid
            if escape_tool_name(params.server_name) != params.server_name:
                raise ValueError(
                    f"Invalid server_name '{params.server_name}'. Server names must not include {NAMESPACE_SEPARATOR} characters."
                )

            if params.server_name in self._workbenches:
                raise ValueError(
                    f"Each server_name in named_server_params must be unique. Encountered duplicate server_name: '{params.server_name}'"
                )
            else:
                self._workbenches[params.server_name] = McpWorkbench(
                    server_params=params.server_params
                )

    # Ali Akbar Prompt Injection Start
    def configure_injection_guard(
        self,
        approval_guard: BaseApprovalGuard | None,
        hybrid_injection_detection: bool = True,
    ) -> None:
        """Wire a live ApprovalGuard onto this workbench so call_tool() can scan tool
        results and hijack-screen tool calls the same way web_surfer/file_surfer/coder
        already do. Called post-construction from task_team.py, since McpAgent is built
        via the Component `_from_config` path which only sees serializable config."""
        self._approval_guard = approval_guard
        self._hybrid_injection_detection = hybrid_injection_detection

    def set_current_instruction(self, instruction: str) -> None:
        """Record what the agent was just told to do this turn, so a subsequent
        call_tool() can hijack-screen its proposed tool call against it. Set by
        McpAgent.on_messages_stream() before delegating to AssistantAgent's tool loop —
        call_tool() itself has no conversation context of its own."""
        self._current_instruction = instruction
    # Ali Akbar Prompt Injection End

    @property
    def server_params(self) -> List[NamedMcpServerParams]:
        return [
            NamedMcpServerParams(
                server_name=server_name, server_params=workbench.server_params
            )
            for server_name, workbench in self._workbenches.items()
        ]

    async def list_tools(self) -> List[ToolSchema]:
        schema: List[ToolSchema] = []
        for server_name, workbench in self._workbenches.items():
            workbench_tools = await workbench.list_tools()
            for tool in workbench_tools:
                # Make a copy of the tool updating the name to be escaped and within this server's 'namespace'
                tool_name = escape_tool_name(tool["name"])
                namespaced_tool_name = f"{server_name}{NAMESPACE_SEPARATOR}{tool_name}"
                namespaced_tool = ToolSchema({**tool, "name": namespaced_tool_name})
                schema.append(namespaced_tool)

        return schema

    async def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ToolResult:
        try:
            # Split the server name from teh tool name
            server_name, tool_name = name.split(NAMESPACE_SEPARATOR, 1)
            # Unescape before sending to teh workbench
            tool_name = unescape_tool_name(tool_name)
        except ValueError:
            raise ValueError(
                f"Cannot call tool named '{name}' with AggregateMcpWorkbench. Expected format: '{{server_name}}{NAMESPACE_SEPARATOR}{{tool_name}}'."
            )

        # Get the workbench for that server
        workbench = self._workbenches.get(server_name, None)
        if not workbench:
            raise KeyError(
                f"Cannot call tool named '{tool_name}' on server '{server_name}'. No known servers with that name."
            )

        # Ali Akbar Prompt Injection Start
        # Hijack screening — does this proposed tool call look consistent with what the
        # agent was just told to do this turn, or does it look like it was redirected by
        # an earlier tool result? Uses whatever instruction McpAgent.on_messages_stream
        # last recorded, since call_tool() has no conversation context of its own.
        _model_client = getattr(self._approval_guard, "model_client", None)
        if (
            is_scan_active(self._approval_guard)
            and self._hybrid_injection_detection
            and _model_client is not None
        ):
            assert self._approval_guard is not None
            _proposed = f"{name}({dict(arguments) if arguments else {}})"
            _hijack = await screen_action_for_hijack(
                self._current_instruction, _proposed, _model_client
            )
            if _hijack.available and _hijack.hijack_suspected and _hijack.confidence >= 0.6:
                approved = await self._approval_guard.get_approval(
                    TextMessage(
                        content=(
                            f"The agent proposes to call MCP tool '{name}' with arguments "
                            f"{dict(arguments) if arguments else {}}.\n\n"
                            f"⚠️ Possible instruction hijack detected: {_hijack.reason}"
                        ),
                        source="system",
                    )
                )
                if not approved:
                    return ToolResult(
                        name=name,
                        result=[
                            TextResultContent(
                                content=(
                                    "[SECURITY BLOCK] This tool call was blocked by the user "
                                    "because it may have been redirected by earlier content "
                                    "rather than the current instruction. No tool result is "
                                    "available."
                                )
                            )
                        ],
                        is_error=True,
                    )
        # Ali Akbar Prompt Injection End

        # Invoke the tool within that namespace
        result = await workbench.call_tool(
            tool_name, arguments=arguments, cancellation_token=cancellation_token
        )

        # Ali Akbar Prompt Injection Start
        # Scan the tool result for injected instructions, same shared gateway every
        # other content source in HALO routes through.
        if self._approval_guard is not None:
            _result_text = result.to_text()
            if _result_text:
                _gate = await scan_and_gate(
                    _result_text,
                    model_client=_model_client,
                    hybrid_enabled=self._hybrid_injection_detection,
                    action_guard=self._approval_guard,
                    source_label="mcp tool result",
                    source_id=f"{name}:{hash(str(arguments))}",
                    blocked_ids=self._blocked_mcp_ids,
                    block_variant="continue",
                )
                if _gate.blocked:
                    assert _gate.replacement_text is not None
                    return ToolResult(
                        name=result.name,
                        result=[TextResultContent(content=_gate.replacement_text)],
                        is_error=result.is_error,
                    )
        # Ali Akbar Prompt Injection End

        return result

    async def start(self) -> None:
        await asyncio.gather(
            *[workbench.start() for workbench in self._workbenches.values()]
        )

    async def stop(self) -> None:
        await asyncio.gather(
            *[workbench.stop() for workbench in self._workbenches.values()]
        )

    async def reset(self) -> None:
        await asyncio.gather(
            *[workbench.reset() for workbench in self._workbenches.values()]
        )

    async def save_state(self) -> Mapping[str, Any]:
        # TODO: McpWorkbenchState is a 'dummy' class so this is okay for now but we will eventually need to aggregate the sub-workbench states
        return AggregateMcpWorkbenchState().model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        # TODO: No state to save, so nothing to do. Again will need to fix this if it ever changes in the base McpWorkbench
        pass

    def _to_config(self) -> AggregateMcpWorkbenchConfig:
        named_server_params: List[NamedMcpServerParams] = []
        for server_name, workbench in self._workbenches.items():
            params = NamedMcpServerParams(
                server_name=server_name, server_params=workbench.server_params
            )
            named_server_params.append(params)

        return AggregateMcpWorkbenchConfig(named_server_params=named_server_params)

    @classmethod
    def _from_config(cls, config: AggregateMcpWorkbenchConfig):
        return cls(named_server_params=config.named_server_params)

    def __del__(self) -> None:
        for name, workbench in self._workbenches.items():
            try:
                del workbench
            except Exception as ex:
                msg = f"Caught exception deleting workbench for server named '{name}'. {type(ex).__name__}: {ex}"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
