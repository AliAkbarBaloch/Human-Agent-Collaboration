import * as React from "react";
import {
  HaloEscalation,
  HaloFeedbackLoopState,
  HaloRiskEstimation,
  HaloInjectionDetection,
} from "../../types/datamodel";

interface Props {
  taskClassification: { task_type: string; risk_score?: number; policy: string } | null;
  riskEstimation: HaloRiskEstimation | null;
  feedbackLoop: HaloFeedbackLoopState | null;
  injectionCount: number;
  injectionRiskScore: number;
  injectionRiskLevel: string;
  injectionMatchedPatterns: string[];
  injectionDetection: HaloInjectionDetection | null;
  escalation: HaloEscalation | null;
  onDismissEscalation: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

// ── Colour maps ───────────────────────────────────────────────────────────────

const POLICY_BG: Record<string, string> = {
  "auto-permissive":  "#22c55e",
  "auto-conservative":"#f97316",
  always:             "#ef4444",
  never:              "#6b7280",
};
const POLICY_LABEL: Record<string, string> = {
  "auto-permissive":  "Permissive",
  "auto-conservative":"Conservative",
  always:             "Always",
  never:              "Never",
};
const TASK_LABEL: Record<string, string> = {
  research:     "Research",
  transactional:"Transactional",
  destructive:  "Destructive",
};
const RISK_LEVEL_BG: Record<string, string> = {
  none:   "#22c55e",
  low:    "#facc15",
  medium: "#f97316",
  high:   "#ef4444",
};
const ACTION_BG: Record<string, string> = {
  allow: "#22c55e",
  warn:  "#f97316",
  block: "#ef4444",
};

// ── Small helpers ─────────────────────────────────────────────────────────────

function PolicyBadge({ policy }: { policy: string }) {
  return (
    <span style={{
      background: POLICY_BG[policy] ?? "#6b7280",
      color: "#fff", fontSize: "10px", fontWeight: 700,
      padding: "1px 6px", borderRadius: "999px", whiteSpace: "nowrap",
    }}>
      {POLICY_LABEL[policy] ?? policy}
    </span>
  );
}

function RiskLevelBadge({ level }: { level: string }) {
  return (
    <span style={{
      background: RISK_LEVEL_BG[level] ?? "#6b7280",
      color: level === "none" || level === "high" ? "#fff" : "#000",
      fontSize: "10px", fontWeight: 700,
      padding: "1px 6px", borderRadius: "999px",
      whiteSpace: "nowrap", textTransform: "uppercase",
    }}>
      {level === "none" ? "None" : level.toUpperCase()}
    </span>
  );
}

function ActionBadge({ action }: { action: string }) {
  return (
    <span style={{
      background: ACTION_BG[action] ?? "#6b7280",
      color: "#fff", fontSize: "10px", fontWeight: 700,
      padding: "1px 6px", borderRadius: "999px",
      whiteSpace: "nowrap", textTransform: "uppercase",
    }}>
      {action.toUpperCase()}
    </span>
  );
}

function AvailBadge({ available, labelYes = "LLM OK", labelNo = "Fallback" }: {
  available: boolean; labelYes?: string; labelNo?: string
}) {
  return (
    <span style={{
      background: available ? "#22c55e" : "#6b7280",
      color: "#fff", fontSize: "9px", fontWeight: 700,
      padding: "1px 5px", borderRadius: "999px",
    }}>
      {available ? labelYes : labelNo}
    </span>
  );
}

function RiskBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const colour = score < 0.31 ? "#22c55e" : score < 0.71 ? "#f97316" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <div style={{ flex: 1, height: "6px", borderRadius: "3px",
        background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: colour,
          borderRadius: "3px", transition: "width 0.4s ease" }} />
      </div>
      <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.7)", minWidth: "28px" }}>
        {score.toFixed(2)}
      </span>
    </div>
  );
}

function ConfBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <div style={{ flex: 1, height: "4px", borderRadius: "3px",
        background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%",
          background: "#60a5fa", borderRadius: "3px" }} />
      </div>
      <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.7)", minWidth: "28px" }}>
        {pct}%
      </span>
    </div>
  );
}

function Section({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.1)", borderRadius: "6px",
      padding: "8px", marginBottom: "8px" }}>
      <div style={{ fontSize: "11px", fontWeight: 700, color: "#FFB300", marginBottom: "6px" }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function SubSection({ title, badge, children }: {
  title: string; badge?: React.ReactNode; children: React.ReactNode
}) {
  return (
    <div style={{ marginBottom: "6px", padding: "5px 6px",
      background: "rgba(255,255,255,0.03)", borderRadius: "4px",
      border: "1px solid rgba(255,255,255,0.07)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
        <span style={{ fontSize: "10px", fontWeight: 700,
          color: "rgba(255,255,255,0.6)", textTransform: "uppercase",
          letterSpacing: "0.05em" }}>{title}</span>
        {badge}
      </div>
      {children}
    </div>
  );
}

function KV({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr",
      columnGap: "6px", alignItems: "center", marginBottom: "2px" }}>
      <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.45)",
        whiteSpace: "nowrap" }}>{label}:</span>
      <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.85)" }}>{children}</span>
    </div>
  );
}

function FusionChip({ mode }: { mode: string }) {
  const isHybrid = mode === "hybrid";
  return (
    <span style={{
      background: isHybrid ? "rgba(96,165,250,0.2)" : "rgba(255,255,255,0.08)",
      border: `1px solid ${isHybrid ? "#60a5fa" : "rgba(255,255,255,0.15)"}`,
      color: isHybrid ? "#93c5fd" : "rgba(255,255,255,0.5)",
      fontSize: "9px", fontWeight: 700, padding: "1px 6px",
      borderRadius: "999px", textTransform: "uppercase", letterSpacing: "0.06em",
    }}>
      {isHybrid ? "⚡ Hybrid" : "⚙ Rule-only"}
    </span>
  );
}

function LayerCard({ label, title, children }: {
  label: string; title: string; children: React.ReactNode
}) {
  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.15)", borderRadius: "6px",
      padding: "6px 8px", background: "rgba(255,255,255,0.03)" }}>
      <div style={{ fontSize: "9px", color: "#FFB300", fontWeight: 700,
        letterSpacing: "0.06em", marginBottom: "2px", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: "10px", fontWeight: 700, color: "rgba(255,255,255,0.9)",
        marginBottom: "4px" }}>{title}</div>
      {children}
    </div>
  );
}

function Arrow() {
  return (
    <div style={{ textAlign: "center", color: "rgba(255,255,255,0.25)",
      fontSize: "14px", margin: "3px 0" }}>↓</div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export default function HaloFeaturesPanel({
  taskClassification,
  riskEstimation,
  feedbackLoop,
  injectionCount,
  injectionRiskScore,
  injectionRiskLevel,
  injectionMatchedPatterns,
  injectionDetection,
  escalation,
  onDismissEscalation,
  isOpen,
  onToggle,
}: Props) {
  const panelWidth = isOpen ? "290px" : "36px";

  const currentTT   = taskClassification?.task_type ?? "research";
  const finalRisk   = riskEstimation?.final_risk_score ?? taskClassification?.risk_score ?? 0.15;
  const finalPolicy = riskEstimation?.final_policy ?? taskClassification?.policy ?? "auto-permissive";
  const fusionMode1 = riskEstimation?.fusion_mode ?? "rule-only fallback";

  const finalInjRisk  = injectionDetection?.final_injection_risk_score ?? injectionRiskScore;
  const finalInjLevel = injectionDetection?.final_risk_level ?? injectionRiskLevel;
  const fusionMode2   = injectionDetection?.fusion_mode ?? "pattern-only fallback";

  const fwTrustMean   = feedbackLoop?.trust_means[currentTT] ?? null;
  const fwUncertainty = feedbackLoop?.uncertainties[currentTT] ?? null;
  const fwConfidence  = feedbackLoop?.confidences[currentTT] ?? null;
  const fwPolicy      = feedbackLoop?.policies[currentTT] ?? finalPolicy;

  return (
    <div style={{
      width: panelWidth, minWidth: panelWidth, maxWidth: panelWidth,
      transition: "width 0.2s ease, min-width 0.2s ease",
      borderLeft: "1px solid rgba(255,255,255,0.08)",
      display: "flex", flexDirection: "column", overflow: "hidden",
      background: "var(--color-bg-secondary, #0f1b35)", height: "100%",
    }}>
      {/* Toggle */}
      <button onClick={onToggle}
        title={isOpen ? "Collapse HALO Features" : "Expand HALO Features"}
        style={{
          display: "flex", alignItems: "center",
          justifyContent: isOpen ? "space-between" : "center",
          padding: "8px", fontSize: "11px", fontWeight: 700, color: "#FFB300",
          background: "transparent", border: "none",
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          cursor: "pointer", whiteSpace: "nowrap", flexShrink: 0,
        }}>
        {isOpen ? (
          <><span>HALO Features</span><span style={{ fontSize: "10px", color: "rgba(255,255,255,0.4)" }}>◀</span></>
        ) : (
          <span style={{ fontSize: "14px" }}>⚙</span>
        )}
      </button>

      {!isOpen && (
        <div style={{
          writingMode: "vertical-rl", transform: "rotate(180deg)",
          fontSize: "10px", color: "rgba(255,255,255,0.35)",
          padding: "12px 0", textAlign: "center", letterSpacing: "0.1em",
        }}>
          HALO Features
        </div>
      )}

      {isOpen && (
        <div style={{ flex: 1, overflowY: "auto", padding: "8px" }}>

          {/* Policy-change banner */}
          {escalation && (
            <div style={{
              background: "rgba(234,179,8,0.15)", border: "1px solid #ca8a04",
              borderRadius: "6px", padding: "8px", marginBottom: "8px", fontSize: "11px",
            }}>
              <div style={{ fontWeight: 700, color: "#fbbf24", marginBottom: "4px" }}>
                ⚡ HALO Bayesian Feedback Loop — Policy Adapted
              </div>
              <div style={{ color: "rgba(255,255,255,0.75)", lineHeight: "1.5" }}>
                [{TASK_LABEL[escalation.task_type] ?? escalation.task_type}]{" "}
                {escalation.old_policy} → {escalation.new_policy}
                {escalation.trust_mean !== undefined && (
                  <><br />Trust: {escalation.trust_mean.toFixed(2)}, σ:{" "}
                  {(escalation.uncertainty ?? 0).toFixed(2)}, conf: {escalation.confidence ?? "—"}</>
                )}
              </div>
              <button onClick={onDismissEscalation}
                style={{ marginTop: "4px", fontSize: "10px", color: "#fbbf24",
                  background: "transparent", border: "none", cursor: "pointer", padding: 0 }}>
                Dismiss ×
              </button>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════
              HALO ADAPTIVE OVERSIGHT FRAMEWORK — overview
          ══════════════════════════════════════════════════════════════════ */}
          <Section title="🏛 HALO Adaptive Oversight Framework">
            <div style={{ fontSize: "10px" }}>

              <LayerCard label="Layer 1" title="Risk-Aware Supervision">
                {taskClassification ? (
                  <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr",
                    columnGap: "6px", rowGap: "2px" }}>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Task:</span>
                    <span style={{ color: "rgba(255,255,255,0.85)", fontWeight: 600,
                      textTransform: "capitalize" }}>{TASK_LABEL[currentTT] ?? currentTT}</span>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Risk:</span>
                    <RiskBar score={finalRisk} />
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Policy:</span>
                    <PolicyBadge policy={finalPolicy} />
                    {riskEstimation && (
                      <>
                        <span style={{ color: "rgba(255,255,255,0.5)" }}>Fusion:</span>
                        <FusionChip mode={fusionMode1} />
                      </>
                    )}
                  </div>
                ) : (
                  <div style={{ color: "rgba(255,255,255,0.35)", fontStyle: "italic" }}>
                    Awaiting classification…
                  </div>
                )}
              </LayerCard>

              <Arrow />

              <LayerCard label="Layer 2" title="Security Transparency">
                <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr",
                  columnGap: "6px", rowGap: "2px" }}>
                  <span style={{ color: "rgba(255,255,255,0.5)" }}>Inj. Risk:</span>
                  {finalInjLevel === "none" ? (
                    <span style={{ color: "#4ade80" }}>0.00</span>
                  ) : (
                    <RiskBar score={finalInjRisk} />
                  )}
                  <span style={{ color: "rgba(255,255,255,0.5)" }}>Status:</span>
                  {finalInjLevel === "none" ? (
                    <span style={{ color: "#4ade80" }}>✓ Clean</span>
                  ) : (
                    <RiskLevelBadge level={finalInjLevel} />
                  )}
                  {injectionDetection && (
                    <>
                      <span style={{ color: "rgba(255,255,255,0.5)" }}>Fusion:</span>
                      <FusionChip mode={fusionMode2} />
                    </>
                  )}
                </div>
              </LayerCard>

              <Arrow />

              <LayerCard label="Layer 3" title="Trust Calibration">
                {fwTrustMean !== null ? (
                  <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr",
                    columnGap: "6px", rowGap: "2px" }}>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Trust:</span>
                    <span style={{ color: "rgba(255,255,255,0.85)" }}>{fwTrustMean.toFixed(2)}</span>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>σ:</span>
                    <span style={{ color: "rgba(255,255,255,0.85)" }}>{(fwUncertainty ?? 0).toFixed(2)}</span>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Conf:</span>
                    <span style={{ fontWeight: 600,
                      color: fwConfidence === "high" ? "#4ade80"
                           : fwConfidence === "medium" ? "#fbbf24" : "#f87171" }}>
                      {fwConfidence ?? "—"}
                    </span>
                    <span style={{ color: "rgba(255,255,255,0.5)" }}>Policy:</span>
                    <PolicyBadge policy={fwPolicy} />
                  </div>
                ) : (
                  <div style={{ color: "rgba(255,255,255,0.35)", fontStyle: "italic" }}>
                    Prior beliefs active
                  </div>
                )}
              </LayerCard>

              <div style={{ marginTop: "8px", fontSize: "9px",
                color: "rgba(255,255,255,0.3)", lineHeight: "1.5", textAlign: "center" }}>
                Layer 1 determines initial oversight →<br />
                Layer 2 exposes attacks before LLM processing →<br />
                Layer 3 adapts future oversight via feedback
              </div>
            </div>
          </Section>

          {/* ══════════════════════════════════════════════════════════════════
              GAP 1 — Adaptive Action Guard: Hybrid Risk-Aware Supervision
          ══════════════════════════════════════════════════════════════════ */}
          <Section title="🛡 Adaptive Action Guard — Hybrid Risk-Aware Supervision">
            {taskClassification ? (
              <div style={{ fontSize: "11px" }}>

                {/* Final Decision */}
                <SubSection title="Final Decision">
                  <KV label="Task Type">
                    <span style={{ fontWeight: 600, textTransform: "capitalize" }}>
                      {TASK_LABEL[riskEstimation?.final_task_type ?? currentTT] ?? currentTT}
                    </span>
                  </KV>
                  <KV label="Risk Score"><RiskBar score={finalRisk} /></KV>
                  <KV label="Policy"><PolicyBadge policy={finalPolicy} /></KV>
                  <KV label="Fusion"><FusionChip mode={fusionMode1} /></KV>
                </SubSection>

                {/* Rule Layer */}
                {riskEstimation && (
                  <SubSection title="Rule-Based Layer">
                    <KV label="Task">{TASK_LABEL[riskEstimation.rule_task_type] ?? riskEstimation.rule_task_type}</KV>
                    <KV label="Risk"><RiskBar score={riskEstimation.rule_risk_score} /></KV>
                    {riskEstimation.rule_matched_keywords?.length > 0 && (
                      <KV label="Keywords">
                        <span style={{ fontSize: "9px", color: "#fbbf24" }}>
                          {riskEstimation.rule_matched_keywords.join(", ")}
                        </span>
                      </KV>
                    )}
                    {riskEstimation.rule_reason && (
                      <KV label="Reason">
                        <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.6)" }}>
                          {riskEstimation.rule_reason}
                        </span>
                      </KV>
                    )}
                  </SubSection>
                )}

                {/* LLM Layer */}
                {riskEstimation && (
                  <SubSection
                    title="LLM-Based Layer"
                    badge={<AvailBadge available={riskEstimation.llm_available} />}
                  >
                    {riskEstimation.llm_available ? (
                      <>
                        <KV label="Task">{TASK_LABEL[riskEstimation.llm_task_type] ?? riskEstimation.llm_task_type}</KV>
                        <KV label="Risk"><RiskBar score={riskEstimation.llm_risk_score} /></KV>
                        <KV label="Confidence"><ConfBar score={riskEstimation.llm_confidence} /></KV>
                        {riskEstimation.llm_reason && (
                          <KV label="Reason">
                            <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.6)" }}>
                              {riskEstimation.llm_reason}
                            </span>
                          </KV>
                        )}
                        {riskEstimation.llm_possible_harms?.length > 0 && (
                          <div style={{ marginTop: "3px" }}>
                            <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.4)" }}>Possible harms:</div>
                            {riskEstimation.llm_possible_harms.map((h, i) => (
                              <div key={i} style={{ fontSize: "9px", color: "#f87171" }}>• {h}</div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
                        LLM unavailable — rule-based fallback active
                      </div>
                    )}
                  </SubSection>
                )}

                <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)",
                  marginTop: "4px", lineHeight: "1.4" }}>
                  Fusion: <span style={{ color: "rgba(255,255,255,0.5)" }}>max(rule_risk, llm_risk)</span> —
                  higher-risk signal dominates. Destructive classification cannot be downgraded.
                </div>
              </div>
            ) : (
              <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
                Waiting for task classification…
              </div>
            )}
          </Section>

          {/* ══════════════════════════════════════════════════════════════════
              GAP 2 — Prompt Injection: Hybrid Security Transparency
          ══════════════════════════════════════════════════════════════════ */}
          <Section title="🔍 Prompt Injection — Hybrid Security Transparency">
            <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.55)", marginBottom: "6px" }}>
              Active — scans pages before LLM context assembly.
            </div>

            {/* Final Detection */}
            <SubSection title="Final Detection">
              {injectionDetection?.final_injection_detected ? (
                <>
                  <KV label="Detected"><span style={{ color: "#f87171", fontWeight: 600 }}>Yes</span></KV>
                  <KV label="Risk Score"><RiskBar score={finalInjRisk} /></KV>
                  <KV label="Risk Level"><RiskLevelBadge level={finalInjLevel} /></KV>
                  {injectionDetection?.final_recommended_action && (
                    <KV label="Action"><ActionBadge action={injectionDetection.final_recommended_action} /></KV>
                  )}
                  <KV label="Fusion"><FusionChip mode={fusionMode2} /></KV>
                </>
              ) : (
                <>
                  <div style={{ color: "#4ade80" }}>✓ No injection detected</div>
                  {finalInjRisk > 0 && (
                    <KV label="Semantic Risk"><RiskBar score={finalInjRisk} /></KV>
                  )}
                </>
              )}
            </SubSection>

            {/* Pattern Layer */}
            {injectionDetection ? (
              <SubSection title="Pattern-Based Layer">
                <KV label="Detected">
                  <span style={{ color: injectionDetection.pattern_detected ? "#f87171" : "#4ade80" }}>
                    {injectionDetection.pattern_detected ? "Yes" : "No"}
                  </span>
                </KV>
                {injectionDetection.pattern_detected && (
                  <>
                    <KV label="Risk"><RiskBar score={injectionDetection.pattern_risk_score} /></KV>
                    {injectionDetection.matched_patterns?.length > 0 && (
                      <div style={{ marginTop: "3px" }}>
                        <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.4)" }}>Patterns:</div>
                        {injectionDetection.matched_patterns.map((p) => (
                          <div key={p} style={{ fontSize: "9px", color: "#f87171" }}>✓ {p}</div>
                        ))}
                      </div>
                    )}
                    {injectionDetection.excerpt && (
                      <KV label="Excerpt">
                        <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.5)",
                          fontStyle: "italic", wordBreak: "break-word" }}>
                          {injectionDetection.excerpt.slice(0, 120)}
                          {injectionDetection.excerpt.length > 120 ? "…" : ""}
                        </span>
                      </KV>
                    )}
                  </>
                )}
              </SubSection>
            ) : injectionRiskLevel !== "none" && (
              /* Fallback: no hybrid data but legacy fields present */
              <SubSection title="Pattern-Based Layer">
                <KV label="Risk"><RiskBar score={injectionRiskScore} /></KV>
                <KV label="Level"><RiskLevelBadge level={injectionRiskLevel} /></KV>
                {injectionMatchedPatterns.length > 0 && (
                  <div style={{ marginTop: "3px" }}>
                    {injectionMatchedPatterns.map((p) => (
                      <div key={p} style={{ fontSize: "9px", color: "#f87171" }}>✓ {p}</div>
                    ))}
                  </div>
                )}
              </SubSection>
            )}

            {/* Semantic Layer */}
            {injectionDetection && (
              <SubSection
                title="Semantic LLM Layer"
                badge={<AvailBadge available={injectionDetection.semantic_available} />}
              >
                {injectionDetection.semantic_available ? (
                  <>
                    <KV label="Detected">
                      <span style={{ color: injectionDetection.semantic_detected ? "#f87171" : "#4ade80" }}>
                        {injectionDetection.semantic_detected ? "Yes" : "No"}
                      </span>
                    </KV>
                    {injectionDetection.semantic_detected && (
                      <>
                        <KV label="Risk"><RiskBar score={injectionDetection.semantic_risk_score} /></KV>
                        <KV label="Confidence"><ConfBar score={injectionDetection.semantic_confidence} /></KV>
                        <KV label="Attack Type">
                          <span style={{ fontSize: "9px", color: "#fbbf24", textTransform: "capitalize" }}>
                            {injectionDetection.semantic_attack_type?.replace(/_/g, " ")}
                          </span>
                        </KV>
                        {injectionDetection.semantic_evidence?.length > 0 && (
                          <div style={{ marginTop: "3px" }}>
                            <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.4)" }}>Evidence:</div>
                            {injectionDetection.semantic_evidence.slice(0, 3).map((e, i) => (
                              <div key={i} style={{ fontSize: "9px", color: "#f87171", wordBreak: "break-word" }}>
                                • {e.slice(0, 100)}{e.length > 100 ? "…" : ""}
                              </div>
                            ))}
                          </div>
                        )}
                        {injectionDetection.semantic_reason && (
                          <KV label="Reason">
                            <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.6)" }}>
                              {injectionDetection.semantic_reason.slice(0, 150)}
                            </span>
                          </KV>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
                    LLM unavailable — pattern-based fallback active
                  </div>
                )}
              </SubSection>
            )}

            {/* Session counter */}
            <div style={{ marginTop: "4px" }}>
              {injectionCount > 0 ? (
                <div style={{ fontSize: "11px", color: "#f87171", fontWeight: 600 }}>
                  ⚠ {injectionCount} injection{injectionCount !== 1 ? "s" : ""} this session
                </div>
              ) : (
                <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.3)" }}>
                  0 injections this session
                </div>
              )}
            </div>

            <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)",
              marginTop: "4px", lineHeight: "1.4" }}>
              Fusion: <span style={{ color: "rgba(255,255,255,0.5)" }}>max(pattern_risk, semantic_risk)</span> —
              higher-risk signal dominates.
            </div>
          </Section>

          {/* ══════════════════════════════════════════════════════════════════
              GAP 3 — Bayesian Trust Adaptation
          ══════════════════════════════════════════════════════════════════ */}
          <Section title="🔄 User Feedback Loop : Bayesian Trust Adaptation">
            {feedbackLoop ? (
              <div style={{ fontSize: "11px" }}>
                {(["research", "transactional", "destructive"] as const).map((tt) => (
                  <div key={tt} style={{ marginBottom: "8px", paddingBottom: "6px",
                    borderBottom: tt !== "destructive" ? "1px solid rgba(255,255,255,0.06)" : "none" }}>
                    <div style={{ display: "flex", alignItems: "center",
                      justifyContent: "space-between", marginBottom: "4px" }}>
                      <span style={{ color: "rgba(255,255,255,0.85)", fontWeight: 600,
                        textTransform: "capitalize" }}>{TASK_LABEL[tt]}</span>
                      <PolicyBadge policy={feedbackLoop.policies[tt] ?? "auto-permissive"} />
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "max-content 1fr",
                      columnGap: "6px", rowGap: "1px", color: "rgba(255,255,255,0.45)" }}>
                      <span>trust:</span>
                      <span style={{ color: "rgba(255,255,255,0.8)" }}>
                        {(feedbackLoop.trust_means[tt] ?? 0).toFixed(2)}
                      </span>
                      <span>uncertainty:</span>
                      <span style={{ color: "rgba(255,255,255,0.8)" }}>
                        {(feedbackLoop.uncertainties[tt] ?? 0).toFixed(2)}
                      </span>
                      <span>confidence:</span>
                      <span style={{ fontWeight: 600,
                        color: feedbackLoop.confidences[tt] === "high" ? "#4ade80"
                             : feedbackLoop.confidences[tt] === "medium" ? "#fbbf24" : "#f87171" }}>
                        {feedbackLoop.confidences[tt] ?? "—"}
                      </span>
                    </div>
                  </div>
                ))}
                <div style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)", lineHeight: "1.4" }}>
                  Injection penalty: low→0.10× · medium→0.25× · high→0.50×<br />
                  Task multiplier: research 1.0× · transactional 1.25× · destructive 1.50×
                </div>
              </div>
            ) : (
              <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
                No decisions recorded yet — prior beliefs active
              </div>
            )}
          </Section>

        </div>
      )}
    </div>
  );
}
