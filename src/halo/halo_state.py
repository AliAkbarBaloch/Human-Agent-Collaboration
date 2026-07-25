"""
Per-run session state container for the HALO Adaptive Oversight Framework.
Stored in WebSocketManager._halo_states[run_id].

Layers (hybrid architecture):
  Layer 1 — Risk-Aware Supervision
    Rule-Based Risk Estimator → LLM-Based Risk Estimator → Conservative Risk Fusion
    Fields: task_type, risk_score, policy, risk_estimation (full hybrid detail)

  Layer 2 — Security Transparency
    Pattern-Based Scanner → Semantic/LLM Detector → Conservative Injection Risk Fusion
    Fields: injection_count, injection_risk_score, injection_risk_level,
            injection_matched_patterns, injection_detection (full hybrid detail)

  Layer 3 — Trust Calibration
    BayesianFeedbackLoop updated by user decisions and injection security events.
    Injection trust penalty is weighted by task risk (higher-risk tasks → stronger penalty).
"""

from typing import Callable, Awaitable

from .feedback_loop import BayesianFeedbackLoop


class HALOState:
    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id

        # Layer 3 — Trust Calibration (Bayesian model, session-scoped)
        self.feedback_loop = BayesianFeedbackLoop()

        # Layer 1 — Risk-Aware Supervision (final fused values)
        self.task_type: str   = "research"
        self.risk_score: float = 0.15
        self.policy: str      = "auto-permissive"

        # Layer 1 — Full hybrid risk estimation detail (populated by connection.py)
        # Keys match the HaloRiskEstimation TypeScript interface.
        self.risk_estimation: dict = {}

        # Layer 2 — Security Transparency (final fused values)
        self.injection_count: int          = 0
        self.injection_risk_score: float   = 0.0
        self.injection_risk_level: str     = "none"   # "none"|"low"|"medium"|"high"
        self.injection_matched_patterns: list[str] = []

        # Layer 2 — Full hybrid injection detection detail (populated by connection.py)
        # Keys match the HaloInjectionDetection TypeScript interface.
        self.injection_detection: dict = {}

        # Per-URL set: tracks which page URLs have already contributed a trust beta update.
        # Prevents repeated penalization when the agent revisits the same page.
        # (Replaces the old boolean _injection_trust_adjusted flag.)
        self._injection_adjusted_urls: set[str] = set()

        # Injected by WebSocketManager so state updates can be pushed over WS
        self.send_update: Callable[[dict], Awaitable[None]] | None = None

    def to_dict(self) -> dict:
        """Serialize full HALO state to a JSON-safe dict for DB persistence."""
        fl_state = self.feedback_loop.get_state()
        return {
            "task_type":  self.task_type,
            "risk_score": self.risk_score,
            "policy":     self.policy,
            "risk_estimation": self.risk_estimation,
            "injection_count": self.injection_count,
            "injection_risk_score": self.injection_risk_score,
            "injection_risk_level": self.injection_risk_level,
            "injection_matched_patterns": self.injection_matched_patterns,
            "injection_detection": self.injection_detection,
            "injection_adjusted_urls": list(self._injection_adjusted_urls),
            "feedback_alpha": fl_state["alpha"],
            "feedback_beta":  fl_state["beta"],
        }

    @classmethod
    def from_dict(cls, data: dict, task_id: str = "") -> "HALOState":
        """Reconstruct HALOState from a previously serialized dict."""
        hs = cls(task_id=task_id)
        hs.task_type  = str(data.get("task_type", "research"))
        hs.risk_score = float(data.get("risk_score", 0.15))
        hs.policy     = str(data.get("policy", "auto-permissive"))
        hs.risk_estimation = data.get("risk_estimation") or {}
        hs.injection_count = int(data.get("injection_count", 0))
        hs.injection_risk_score = float(data.get("injection_risk_score", 0.0))
        hs.injection_risk_level = str(data.get("injection_risk_level", "none"))
        hs.injection_matched_patterns = list(data.get("injection_matched_patterns") or [])
        hs.injection_detection = data.get("injection_detection") or {}
        hs._injection_adjusted_urls = set(data.get("injection_adjusted_urls") or [])
        hs.feedback_loop.restore_from(
            alpha=data.get("feedback_alpha") or {},
            beta=data.get("feedback_beta") or {},
        )
        return hs

    async def push(self, payload: dict) -> None:
        if self.send_update:
            try:
                await self.send_update(payload)
            except Exception:
                pass

    def state_snapshot(self, escalation: dict | None = None) -> dict:
        """Full halo_state_update payload for the frontend (all three layers)."""
        return {
            "type": "halo_state_update",
            # Layer 1 — final fused values
            "task_classification": {
                "task_type": self.task_type,
                "risk_score": self.risk_score,
                "policy": self.policy,
            },
            # Layer 1 — hybrid detail (empty dict until hybrid classification runs)
            "risk_estimation": self.risk_estimation,
            # Layer 2 — final fused values
            "injection_count": self.injection_count,
            "injection_risk_score": self.injection_risk_score,
            "injection_risk_level": self.injection_risk_level,
            "injection_matched_patterns": self.injection_matched_patterns,
            # Layer 2 — hybrid detail (empty dict until hybrid detection runs)
            "injection_detection": self.injection_detection,
            # Layer 3 — Bayesian trust state
            "feedback_loop": self.feedback_loop.get_state(),
            # Policy-change notification (None unless policy changed this round)
            "escalation": escalation,
        }
