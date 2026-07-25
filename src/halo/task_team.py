from typing import Any, Dict, List, Optional, Union

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import ChatAgent
from autogen_core import ComponentModel
from autogen_core.models import ChatCompletionClient

from .agents import (
    USER_PROXY_DESCRIPTION,
    HALOCoderAgent,
    HALOFileSurfer,
    FaraWebSurfer,
    HALOWebSurfer,
)
from .agents.mcp import McpAgent
from .agents.users import DummyUserProxy, MetadataUserProxy
from .agents.web_surfer import WebSurferConfig
from .approval_guard import (
    ApprovalConfig,
    ApprovalGuard,
    ApprovalGuardContext,
    BaseApprovalGuard,
)
from .input_func import InputFuncType, make_agentchat_input_func
from .learning.memory_provider import MemoryControllerProvider
from .halo_config import HALOAppConfig, ModelClientConfigs
from .teams import GroupChat, RoundRobinGroupChat
from .teams.orchestrator.orchestrator_config import OrchestratorConfig
from .tools.playwright.browser import get_browser_resource_config
from .types import RunPaths
from .utils import get_internal_urls


async def get_task_team(
    halo_config: Optional[HALOAppConfig] = None,
    input_func: Optional[InputFuncType] = None,
    *,
    paths: RunPaths,
) -> GroupChat | RoundRobinGroupChat:
    """
    Creates and returns a GroupChat team with specified configuration.

    Args:
        halo_config (HALOAppConfig, optional): HALO configuration for team. Default: None.
        paths (RunPaths): Paths for internal and external run directories.

    Returns:
        GroupChat | RoundRobinGroupChat: An instance of GroupChat or RoundRobinGroupChat with the specified agents and configuration.
    """
    if halo_config is None:
        halo_config = HALOAppConfig()

    def get_model_client(
        model_client_config: Union[ComponentModel, Dict[str, Any], None],
        is_action_guard: bool = False,
    ) -> ChatCompletionClient:
        if model_client_config is None:
            return ChatCompletionClient.load_component(
                ModelClientConfigs.get_default_client_config()
                if not is_action_guard
                else ModelClientConfigs.get_default_action_guard_config()
            )
        return ChatCompletionClient.load_component(model_client_config)

    if not halo_config.inside_docker:
        assert (
            paths.external_run_dir == paths.internal_run_dir
        ), "External and internal run dirs must be the same in non-docker mode"

    model_client_orch = get_model_client(
        halo_config.model_client_configs.orchestrator
    )
    approval_guard: BaseApprovalGuard | None = None

    approval_policy = (
        halo_config.approval_policy
        if halo_config.approval_policy
        else "never"
    )

    websurfer_loop_team: bool = (
        halo_config.websurfer_loop if halo_config else False
    )

    model_client_coder = get_model_client(halo_config.model_client_configs.coder)
    model_client_file_surfer = get_model_client(
        halo_config.model_client_configs.file_surfer
    )
    browser_resource_config, _novnc_port, _playwright_port = (
        get_browser_resource_config(
            paths.external_run_dir,
            halo_config.novnc_port,
            halo_config.playwright_port,
            halo_config.inside_docker,
            headless=halo_config.browser_headless,
            local=halo_config.browser_local
            or halo_config.run_without_docker,
            network_name=halo_config.network_name,
        )
    )

    orchestrator_config = OrchestratorConfig(
        cooperative_planning=halo_config.cooperative_planning,
        autonomous_execution=halo_config.autonomous_execution,
        allowed_websites=halo_config.allowed_websites,
        plan=halo_config.plan,
        model_context_token_limit=halo_config.model_context_token_limit,
        do_bing_search=halo_config.do_bing_search,
        retrieve_relevant_plans=halo_config.retrieve_relevant_plans,
        memory_controller_key=halo_config.memory_controller_key,
        allow_follow_up_input=halo_config.allow_follow_up_input,
        final_answer_prompt=halo_config.final_answer_prompt,
        sentinel_plan=halo_config.sentinel_plan,
        # Ali Akbar Start (Gap 1 — pass adaptive_approval flag to orchestrator config)
        adaptive_approval=halo_config.adaptive_approval,
        # Ali Akbar End (Gap 1)
        # Ali Akbar Start (Hybrid Gap 1 — LLM risk estimation flag)
        hybrid_risk_estimation=halo_config.hybrid_risk_estimation,
        # Ali Akbar End (Hybrid Gap 1)
        # Ali Akbar Prompt Injection Start
        hybrid_injection_detection=halo_config.hybrid_injection_detection,
        # Ali Akbar Prompt Injection End
    )
    websurfer_model_client = halo_config.model_client_configs.web_surfer
    if websurfer_model_client is None:
        websurfer_model_client = ModelClientConfigs.get_default_client_config()
    websurfer_config = WebSurferConfig(
        name="web_surfer",
        model_client=websurfer_model_client,
        browser=browser_resource_config,
        single_tab_mode=False,
        max_actions_per_step=halo_config.max_actions_per_step,
        url_statuses={key: "allowed" for key in orchestrator_config.allowed_websites}
        if orchestrator_config.allowed_websites
        else None,
        url_block_list=get_internal_urls(halo_config.inside_docker, paths),
        multiple_tools_per_call=halo_config.multiple_tools_per_call,
        downloads_folder=str(paths.internal_run_dir),
        debug_dir=str(paths.internal_run_dir),
        animate_actions=True,
        start_page=None,
        use_action_guard=True,
        to_save_screenshots=False,
        # Ali Akbar Start (Hybrid Gap 2 — semantic injection detection flag)
        hybrid_injection_detection=halo_config.hybrid_injection_detection,
        # Ali Akbar End (Hybrid Gap 2)
    )

    user_proxy: DummyUserProxy | MetadataUserProxy | UserProxyAgent

    if halo_config.user_proxy_type == "dummy":
        user_proxy = DummyUserProxy(name="user_proxy")
    elif halo_config.user_proxy_type == "metadata":
        assert (
            halo_config.task is not None
        ), "Task must be provided for metadata user proxy"
        assert (
            halo_config.hints is not None
        ), "Hints must be provided for metadata user proxy"
        assert (
            halo_config.answer is not None
        ), "Answer must be provided for metadata user proxy"
        user_proxy = MetadataUserProxy(
            name="user_proxy",
            description="Metadata User Proxy Agent",
            task=halo_config.task,
            helpful_task_hints=halo_config.hints,
            task_answer=halo_config.answer,
            model_client=model_client_orch,
        )
    else:
        user_proxy_input_func = make_agentchat_input_func(input_func)
        user_proxy = UserProxyAgent(
            description=USER_PROXY_DESCRIPTION,
            name="user_proxy",
            input_func=user_proxy_input_func,
        )

    if halo_config.user_proxy_type in ["dummy", "metadata"]:
        model_client_action_guard = get_model_client(
            halo_config.model_client_configs.action_guard,
            is_action_guard=True,
        )

        # Simple approval function that always returns yes
        def always_yes_input(prompt: str, input_type: str = "text_input") -> str:
            return "yes"

        approval_guard = ApprovalGuard(
            input_func=always_yes_input,
            default_approval=False,
            model_client=model_client_action_guard,
            config=ApprovalConfig(
                approval_policy=approval_policy,
            ),
        )
    elif input_func is not None:
        model_client_action_guard = get_model_client(
            halo_config.model_client_configs.action_guard
        )
        approval_guard = ApprovalGuard(
            input_func=input_func,
            default_approval=False,
            model_client=model_client_action_guard,
            config=ApprovalConfig(
                approval_policy=approval_policy,
            ),
        )
    with ApprovalGuardContext.populate_context(approval_guard):
        if halo_config.use_fara_agent:
            web_surfer = FaraWebSurfer.from_config(websurfer_config)
        else:
            web_surfer = HALOWebSurfer.from_config(websurfer_config)
    # Ali Akbar Start (Gap 2 — guarantee injection-scan guard is set on HALOWebSurfer)
    # Guarantee the injection-scan guard is set even if the ContextVar was not
    # readable at __init__ time (e.g. async-task boundary edge-cases).
    if web_surfer.action_guard is None and approval_guard is not None:
        web_surfer.action_guard = approval_guard
    # Ali Akbar End (Gap 2)
    if websurfer_loop_team:
        # simplified team of only the web surfer
        team = RoundRobinGroupChat(
            participants=[web_surfer, user_proxy],
            max_turns=10000,
        )
        # Ali Akbar Start (Gap 3 — expose the shared ApprovalGuard so the
        # WebSocketManager can write trust-adjusted policy back into it)
        team.halo_approval_guard = approval_guard  # type: ignore[attr-defined]
        # Ali Akbar End (Gap 3)
        return team
    coder_agent: HALOCoderAgent | None = None
    file_surfer: HALOFileSurfer | None = None
    if not halo_config.run_without_docker:
        coder_agent = HALOCoderAgent(
            name="coder_agent",
            model_client=model_client_coder,
            work_dir=paths.internal_run_dir,
            bind_dir=paths.external_run_dir,
            model_context_token_limit=halo_config.model_context_token_limit,
            approval_guard=approval_guard,
            # Ali Akbar Prompt Injection Start
            hybrid_injection_detection=halo_config.hybrid_injection_detection,
            # Ali Akbar Prompt Injection End
        )

        file_surfer = HALOFileSurfer(
            name="file_surfer",
            model_client=model_client_file_surfer,
            work_dir=paths.internal_run_dir,
            bind_dir=paths.external_run_dir,
            model_context_token_limit=halo_config.model_context_token_limit,
            approval_guard=approval_guard,
            # Ali Akbar Prompt Injection Start
            hybrid_injection_detection=halo_config.hybrid_injection_detection,
            # Ali Akbar Prompt Injection End
        )

    # Setup any mcp_agents
    mcp_agents: List[McpAgent] = [
        # TODO: Init from constructor?
        McpAgent._from_config(config)  # type: ignore
        for config in halo_config.mcp_agent_configs
    ]
    # Ali Akbar Prompt Injection Start
    # Wire the shared ApprovalGuard onto each MCP agent's workbench so tool calls/
    # results get the same injection-scan and hijack-screening coverage as
    # web_surfer/file_surfer/coder. Post-construction because McpAgent is built via
    # the Component `_from_config` path, which only sees serializable config.
    for _mcp_agent in mcp_agents:
        _mcp_agent.configure_injection_guard(
            approval_guard, halo_config.hybrid_injection_detection
        )
    # Ali Akbar Prompt Injection End

    if (
        orchestrator_config.memory_controller_key is not None
        and orchestrator_config.retrieve_relevant_plans in ["reuse", "hint"]
    ):
        memory_provider = MemoryControllerProvider(
            internal_workspace_root=paths.internal_root_dir,
            external_workspace_root=paths.external_root_dir,
            inside_docker=halo_config.inside_docker,
        )
    else:
        memory_provider = None

    team_participants: List[ChatAgent] = [
        web_surfer,
        user_proxy,
    ]
    if not halo_config.run_without_docker:
        assert coder_agent is not None
        assert file_surfer is not None
        team_participants.extend([coder_agent, file_surfer])
    team_participants.extend(mcp_agents)

    team = GroupChat(
        participants=team_participants,
        orchestrator_config=orchestrator_config,
        model_client=model_client_orch,
        memory_provider=memory_provider,
        # Ali Akbar Start (Gap 1 — pass approval_guard so orchestrator can update policy adaptively)
        approval_guard=approval_guard,
        # Ali Akbar End (Gap 1)
    )

    # Ali Akbar Start (Gap 3 — expose the shared ApprovalGuard so the
    # WebSocketManager can write trust-adjusted policy back into it)
    team.halo_approval_guard = approval_guard  # type: ignore[attr-defined]
    # Ali Akbar End (Gap 3)

    return team
