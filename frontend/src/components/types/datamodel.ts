export interface RequestUsage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface ImageContent {
  url: string;
  alt?: string;
  data?: string;
}

export interface FunctionCall {
  id: string;
  arguments: string; // JSON string
  name: string;
}

export interface FunctionExecutionResult {
  call_id: string;
  content: string;
}

// Base message configuration (maps to Python BaseMessage)
export interface BaseMessageConfig {
  source: string;
  models_usage?: RequestUsage;
  metadata?: Record<string, string>;
  version?: number;
}

// Message configurations (mapping directly to Python classes)
export interface TextMessageConfig extends BaseMessageConfig {
  content: string;
}

export interface MultiModalMessageConfig extends BaseMessageConfig {
  content: (string | ImageContent)[];
}

export interface StopMessageConfig extends BaseMessageConfig {
  content: string;
}

export interface HandoffMessageConfig extends BaseMessageConfig {
  content: string;
  target: string;
}

export interface ToolCallMessageConfig extends BaseMessageConfig {
  content: FunctionCall[];
}

export interface ToolCallResultMessageConfig extends BaseMessageConfig {
  content: FunctionExecutionResult[];
}

// Message type unions (matching Python type aliases)
export type InnerMessageConfig =
  | ToolCallMessageConfig
  | ToolCallResultMessageConfig;

export type ChatMessageConfig =
  | TextMessageConfig
  | MultiModalMessageConfig
  | StopMessageConfig
  | HandoffMessageConfig;

export type AgentMessageConfig =
  | TextMessageConfig
  | MultiModalMessageConfig
  | StopMessageConfig
  | HandoffMessageConfig
  | ToolCallMessageConfig
  | ToolCallResultMessageConfig;

// Database model
export interface DBModel {
  id?: number;
  user_id?: string;
  created_at?: string;
  updated_at?: string;
  version?: number;
}

export interface Message extends DBModel {
  config: AgentMessageConfig;
  session_id: number;
  run_id: string;
}

export interface Session extends DBModel {
  name: string;
  team_id?: number;
  selected_mcp_configs?: any[];
}

export interface SessionRuns {
  runs: Run[];
}

export interface BaseConfig {
  component_type: string;
  version?: string;
  description?: string;
}

export interface WebSocketMessage {
  type:
    | "message"
    | "result"
    | "completion"
    | "input_request"
    | "error"
    | "system"
    | "halo_classification"
    | "halo_state_update"
    | "message_chunk";
  data?: AgentMessageConfig | TaskResult;
  input_type?: InputType;
  status?: RunStatus;
  error?: string;
  timestamp?: string;
  // HALO Adaptive Oversight Framework — Layer 1 (Hybrid Risk-Aware Supervision)
  task_type?: string;
  policy?: string;
  risk_score?: number;
  task_classification?: { task_type: string; risk_score?: number; policy: string };
  risk_estimation?: HaloRiskEstimation;
  // HALO Adaptive Oversight Framework — Layer 2 (Hybrid Security Transparency)
  injection_count?: number;
  injection_risk_score?: number;
  injection_risk_level?: string;
  injection_matched_patterns?: string[];
  injection_detection?: HaloInjectionDetection;
  // HALO Adaptive Oversight Framework — Layer 3 (Trust Calibration)
  feedback_loop?: HaloFeedbackLoopState;
  escalation?: HaloEscalation | null;
}

// --- HALO Feature Types (Features 7–9) ---

// Gap 3 — Bayesian Trust Adaptation state (one entry per task type)
export interface HaloFeedbackLoopState {
  trust_means:   Record<string, number>;
  uncertainties: Record<string, number>;
  confidences:   Record<string, string>;
  policies:      Record<string, string>;
  alpha:         Record<string, number>;
  beta:          Record<string, number>;
}

// Gap 1 — Hybrid Risk Estimation state (rule + LLM + fusion)
export interface HaloRiskEstimation {
  // Rule layer
  rule_task_type: string;
  rule_risk_score: number;
  rule_policy: string;
  rule_reason: string;
  rule_matched_keywords: string[];
  // LLM layer
  llm_task_type: string;
  llm_risk_score: number;
  llm_confidence: number;
  llm_reason: string;
  llm_possible_harms: string[];
  llm_recommended_policy: string;
  llm_available: boolean;
  // Final (fused)
  final_task_type: string;
  final_risk_score: number;
  final_policy: string;
  fusion_mode: string;
}

// Gap 2 — Hybrid Injection Detection state (pattern + semantic + fusion)
export interface HaloInjectionDetection {
  // Pattern layer
  pattern_detected: boolean;
  pattern_risk_score: number;
  pattern_risk_level: string;
  matched_patterns: string[];
  excerpt: string;
  // Semantic layer
  semantic_detected: boolean;
  semantic_risk_score: number;
  semantic_confidence: number;
  semantic_attack_type: string;
  semantic_evidence: string[];
  semantic_recommended_action: string;
  semantic_reason: string;
  semantic_available: boolean;
  // Final (fused)
  final_injection_detected: boolean;
  final_injection_risk_score: number;
  final_risk_level: string;
  final_recommended_action: string;
  fusion_mode: string;
  url?: string;
}

// Policy-change notification (emitted when Bayesian policy transitions)
export interface HaloEscalation {
  task_type:   string;
  old_policy:  string;
  new_policy:  string;
  trust_mean?: number;
  uncertainty?: number;
  confidence?: string;
}

export interface InputRequestMessage extends WebSocketMessage {
  type: "input_request";
  input_type: InputType;
  prompt: string;
}

export interface TaskResult {
  messages: AgentMessageConfig[];
  stop_reason?: string;
}

export type ModelTypes =
  | "OpenAIChatCompletionClient"
  | "AzureOpenAIChatCompletionClient";

export type AgentTypes =
  | "AssistantAgent"
  | "UserProxyAgent"
  | "MultimodalWebSurfer"
  | "HALOFileSurfer"
  | "HALOOneCoderAgent";

export type ToolTypes = "PythonFunction";

export type TeamTypes =
  | "RoundRobinGroupChat"
  | "SelectorGroupChat"
  | "HALOOneGroupChat";

export type TerminationTypes =
  | "MaxMessageTermination"
  | "StopMessageTermination"
  | "TextMentionTermination"
  | "TimeoutTermination"
  | "CombinationTermination";

export type ComponentTypes =
  | "team"
  | "agent"
  | "model"
  | "tool"
  | "termination";

export type ComponentConfigTypes =
  | TeamConfig
  | AgentConfig
  | ModelConfig
  | ToolConfig
  | TerminationConfig;

export interface BaseModelConfig extends BaseConfig {
  model: string;
  model_type: ModelTypes;
  api_key?: string;
  base_url?: string;
}

export interface AzureOpenAIModelConfig extends BaseModelConfig {
  model_type: "AzureOpenAIChatCompletionClient";
  azure_deployment: string;
  api_version: string;
  azure_endpoint: string;
  azure_ad_token_provider: string;
}

export interface OpenAIModelConfig extends BaseModelConfig {
  model_type: "OpenAIChatCompletionClient";
}

export type ModelConfig = AzureOpenAIModelConfig | OpenAIModelConfig;

export interface BaseToolConfig extends BaseConfig {
  name: string;
  description: string;
  content: string;
  tool_type: ToolTypes;
}

export interface PythonFunctionToolConfig extends BaseToolConfig {
  tool_type: "PythonFunction";
}

export type ToolConfig = PythonFunctionToolConfig;

export interface BaseAgentConfig extends BaseConfig {
  name: string;
  agent_type: AgentTypes;
  system_message?: string;
  model_client?: ModelConfig;
  tools?: ToolConfig[];
  description?: string;
}

export interface AssistantAgentConfig extends BaseAgentConfig {
  agent_type: "AssistantAgent";
}

export interface UserProxyAgentConfig extends BaseAgentConfig {
  agent_type: "UserProxyAgent";
}

export interface MultimodalWebSurferAgentConfig extends BaseAgentConfig {
  agent_type: "MultimodalWebSurfer";
}

export interface FileSurferAgentConfig extends BaseAgentConfig {
  agent_type: "HALOFileSurfer";
}

export interface HALOOneCoderAgentConfig extends BaseAgentConfig {
  agent_type: "HALOOneCoderAgent";
}

export type AgentConfig =
  | AssistantAgentConfig
  | UserProxyAgentConfig
  | MultimodalWebSurferAgentConfig
  | FileSurferAgentConfig
  | HALOOneCoderAgentConfig;

export interface BaseTerminationConfig extends BaseConfig {
  termination_type: TerminationTypes;
}

export interface MaxMessageTerminationConfig extends BaseTerminationConfig {
  termination_type: "MaxMessageTermination";
  max_messages: number;
}

export interface TextMentionTerminationConfig extends BaseTerminationConfig {
  termination_type: "TextMentionTermination";
  text: string;
}

export interface CombinationTerminationConfig extends BaseTerminationConfig {
  termination_type: "CombinationTermination";
  operator: "and" | "or";
  conditions: TerminationConfig[];
}

export type TerminationConfig =
  | MaxMessageTerminationConfig
  | TextMentionTerminationConfig
  | CombinationTerminationConfig;

export interface BaseTeamConfig extends BaseConfig {
  name: string;
  participants: AgentConfig[];
  team_type: TeamTypes;
  termination_condition?: TerminationConfig;
}

export interface RoundRobinGroupChatConfig extends BaseTeamConfig {
  team_type: "RoundRobinGroupChat";
}

export interface SelectorGroupChatConfig extends BaseTeamConfig {
  team_type: "SelectorGroupChat";
  selector_prompt: string;
  model_client: ModelConfig;
}

export interface DefaultTeamConfig extends BaseTeamConfig {
  team_type: "RoundRobinGroupChat";
}

export type TeamConfig = RoundRobinGroupChatConfig | SelectorGroupChatConfig;

export interface Team extends DBModel {
  config: TeamConfig;
}

export interface TeamResult {
  task_result: TaskResult;
  usage: string;
  duration: number;
}

export interface Run {
  id: string;
  created_at: string;
  updated_at?: string;
  status: RunStatus;
  input_request?: InputRequest;
  task: AgentMessageConfig;
  team_result: TeamResult | null;
  messages: Message[]; // Change to Message[]
  error_message?: string;
}

export interface InputRequest {
  input_type: InputType;
  prompt?: string;
}

export interface ApprovalInputRequest extends InputRequest {
  input_type: "approval";
  prompt: string;
}

// Ali Akbar Start (Gap 2 — injection alert input request type)
export interface InjectionAlertInputRequest extends InputRequest {
  input_type: "injection_alert";
  prompt: string;
}
// Ali Akbar End (Gap 2)

export type RunStatus =
  | "created"
  | "active" // covers 'streaming'
  | "awaiting_input"
  | "timeout"
  | "complete"
  | "error"
  | "stopped"
  | "paused"
  | "pausing"
  | "resuming"
  | "connected";

// Ali Akbar Start (Gap 2 — added "injection_alert" to InputType union)
export type InputType = "text_input" | "approval" | "injection_alert";
// Ali Akbar End (Gap 2)
