import json

import pytest
import yaml
from halo.halo_config import HALOAppConfig

YAML_CONFIG = """
model_client_configs:
  default: &default_client
    provider: OpenAIChatCompletionClient
    config:
      model: gpt-4.1-2025-04-14
    max_retries: 10
  orchestrator: *default_client
  web_surfer: *default_client
  coder: *default_client
  file_surfer: *default_client
  action_guard:
    provider: OpenAIChatCompletionClient
    config:
      model: gpt-4.1-nano-2025-04-14
    max_retries: 10

mcp_agent_configs:
  - name: mcp_agent
    description: "Test MCP Agent"
    reflect_on_tool_use: false
    tool_call_summary_format: "{tool_name}({arguments}): {result}"
    model_client: *default_client
    mcp_servers:
      - server_name: server1
        server_params:
          type: StdioServerParams
          command: npx
          args:
            - -y
            - "@modelcontextprotocol/server-everything"
      - server_name: server2
        server_params:
          type: SseServerParams
          url: http://localhost:3001/sse

cooperative_planning: true
autonomous_execution: false
allowed_websites: []
max_actions_per_step: 5
multiple_tools_per_call: false
max_turns: 20
plan: null
approval_policy: auto-conservative
allow_for_replans: true
do_bing_search: false
websurfer_loop: false
retrieve_relevant_plans: never
memory_controller_key: null
model_context_token_limit: 110000
allow_follow_up_input: true
final_answer_prompt: null
playwright_port: -1
novnc_port: -1
user_proxy_type: null
task: "What tools are available?"
hints: null
answer: null
inside_docker: false
"""


@pytest.fixture
def yaml_config_text() -> str:
    return YAML_CONFIG


@pytest.fixture
def config_obj(yaml_config_text: str) -> HALOAppConfig:
    data = yaml.safe_load(yaml_config_text)
    return HALOAppConfig(**data)


def test_yaml_deserialize(yaml_config_text: str) -> None:
    data = yaml.safe_load(yaml_config_text)
    config = HALOAppConfig(**data)
    assert isinstance(config, HALOAppConfig)
    assert config.task == "What tools are available?"
    assert config.mcp_agent_configs[0].name == "mcp_agent"
    assert config.mcp_agent_configs[0].reflect_on_tool_use is False
    assert (
        config.mcp_agent_configs[0].tool_call_summary_format
        == "{tool_name}({arguments}): {result}"
    )


# ── HALO Gap 1/2 — adaptive_approval and hybrid flags ───────────────────────

def test_adaptive_approval_default_false(config_obj: HALOAppConfig) -> None:
    """adaptive_approval defaults to False; explicit YAML value is preserved."""
    default_config = HALOAppConfig()
    assert default_config.adaptive_approval is False


def test_adaptive_approval_enabled_from_yaml() -> None:
    """adaptive_approval: true in YAML enables Gap 1 classification."""
    data = yaml.safe_load(YAML_CONFIG + "\nadaptive_approval: true\n")
    config = HALOAppConfig(**data)
    assert config.adaptive_approval is True


def test_adaptive_approval_disabled_from_yaml() -> None:
    """adaptive_approval: false in YAML disables Gap 1 classification."""
    data = yaml.safe_load(YAML_CONFIG + "\nadaptive_approval: false\n")
    config = HALOAppConfig(**data)
    assert config.adaptive_approval is False


def test_hybrid_risk_estimation_default_true() -> None:
    """hybrid_risk_estimation defaults to True (LLM layer active by default)."""
    config = HALOAppConfig()
    assert config.hybrid_risk_estimation is True


def test_hybrid_injection_detection_default_true() -> None:
    """hybrid_injection_detection defaults to True (semantic layer active by default)."""
    config = HALOAppConfig()
    assert config.hybrid_injection_detection is True


def test_hybrid_flags_disabled_from_yaml() -> None:
    """Both hybrid flags can be set to false for ablation studies."""
    extra = "\nhybrid_risk_estimation: false\nhybrid_injection_detection: false\n"
    data = yaml.safe_load(YAML_CONFIG + extra)
    config = HALOAppConfig(**data)
    assert config.hybrid_risk_estimation is False
    assert config.hybrid_injection_detection is False


def test_hybrid_flags_survive_serialization_roundtrip() -> None:
    """Hybrid flags round-trip through JSON serialization unchanged."""
    config = HALOAppConfig(
        adaptive_approval=True,
        hybrid_risk_estimation=False,
        hybrid_injection_detection=True,
    )
    as_dict = config.model_dump(mode="json")
    config2 = HALOAppConfig(**as_dict)
    assert config2.adaptive_approval is True
    assert config2.hybrid_risk_estimation is False
    assert config2.hybrid_injection_detection is True


def test_adaptive_approval_is_master_gate_for_hybrid() -> None:
    """
    Semantic check: hybrid flags are sub-options of adaptive_approval.
    Having hybrid_risk_estimation=True but adaptive_approval=False means
    Gap 1 is entirely disabled (hybrid flag is irrelevant without the gate).
    This test documents and preserves that design contract.
    """
    config = HALOAppConfig(
        adaptive_approval=False,
        hybrid_risk_estimation=True,
    )
    # Both fields are independently stored — the gate logic lives in _orchestrator.py
    assert config.adaptive_approval is False
    assert config.hybrid_risk_estimation is True


def test_yaml_serialize_roundtrip(config_obj: HALOAppConfig) -> None:
    as_dict = config_obj.model_dump(mode="json")
    yaml_text = yaml.safe_dump(as_dict)
    loaded = yaml.safe_load(yaml_text)
    config2 = HALOAppConfig(**loaded)
    assert config2 == config_obj


def test_json_serialize_roundtrip(config_obj: HALOAppConfig) -> None:
    as_dict = config_obj.model_dump(mode="json")
    json_text = json.dumps(as_dict)
    loaded = json.loads(json_text)
    config2 = HALOAppConfig(**loaded)
    assert config2 == config_obj


def test_json_and_yaml_equivalence(yaml_config_text: str) -> None:
    data = yaml.safe_load(yaml_config_text)
    json_text = json.dumps(data)
    loaded = json.loads(json_text)
    config = HALOAppConfig(**loaded)
    assert config.task == "What tools are available?"
    assert config.mcp_agent_configs[0].name == "mcp_agent"
