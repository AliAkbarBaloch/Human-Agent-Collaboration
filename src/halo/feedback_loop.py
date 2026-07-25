"""
Gap 3 — User Feedback Loop: Bayesian Trust Adaptation
Session-scoped per-task-type Beta distribution that adapts approval policy
using approvals, rejections, and corrections as evidence.
"""

import math

TASK_TYPES = ("research", "transactional", "destructive")

# Initial Beta(alpha, beta) priors per task type
_PRIORS: dict[str, tuple[float, float]] = {
    "research":     (8.0, 2.0),
    "transactional":(5.0, 5.0),
    "destructive":  (2.0, 8.0),
}

# Ali Akbar Start (Gap 3 — asymmetric trust weighting)
# A reject costs more trust than an approve earns. This bounds how fast
# repeated approvals (e.g. approval fatigue, or a page/task steering the user
# toward rubber-stamping) can loosen the enforced policy for closed-loop mode
# (enable_trust_feedback: true, halo_config.py) — one bad decision now takes
# more approvals to earn back than it would under a symmetric +1/+1 model.
# "correct" is a pre-existing softer negative signal; it keeps its original
# "half of a reject" ratio.
APPROVE_WEIGHT = 1.0
REJECT_WEIGHT = 1.5
CORRECT_WEIGHT = REJECT_WEIGHT / 2  # 0.75
# Ali Akbar End (Gap 3)


class BayesianFeedbackLoop:
    """Per-task-type Beta distribution model for trust-adaptive policy.

    Alpha/beta parameters are updated from user decisions:
      approve   → alpha += APPROVE_WEIGHT (1.0)
      reject    → beta  += REJECT_WEIGHT  (1.5)
      correct   → beta  += CORRECT_WEIGHT (0.75)
    Reject outweighs approve so trust erodes faster than it builds.
    """

    def __init__(self) -> None:
        self._alpha: dict[str, float] = {t: _PRIORS[t][0] for t in TASK_TYPES}
        self._beta:  dict[str, float] = {t: _PRIORS[t][1] for t in TASK_TYPES}

    def _t(self, task_type: str) -> str:
        return task_type if task_type in self._alpha else "research"

    def record_feedback(self, task_type: str, decision: str) -> None:
        """Update Beta parameters from one user decision.

        Args:
            task_type: "research" | "transactional" | "destructive"
            decision:  "approve" | "reject" | "correct"
        """
        t = self._t(task_type)
        if decision == "approve":
            self._alpha[t] += APPROVE_WEIGHT
        elif decision == "reject":
            self._beta[t] += REJECT_WEIGHT
        elif decision == "correct":
            self._beta[t] += CORRECT_WEIGHT

    def get_trust_mean(self, task_type: str) -> float:
        t = self._t(task_type)
        a, b = self._alpha[t], self._beta[t]
        return a / (a + b)

    def get_uncertainty(self, task_type: str) -> float:
        t = self._t(task_type)
        a, b = self._alpha[t], self._beta[t]
        s = a + b
        variance = (a * b) / (s * s * (s + 1.0))
        return math.sqrt(variance)

    def get_confidence(self, task_type: str) -> str:
        u = self.get_uncertainty(task_type)
        if u < 0.08:
            return "high"
        elif u < 0.15:
            return "medium"
        return "low"

    def get_policy(self, task_type: str) -> str:
        """Derive approval policy from current trust mean and confidence."""
        if task_type == "destructive":
            return "always"
        mean = self.get_trust_mean(task_type)
        conf = self.get_confidence(task_type)
        if mean >= 0.75 and conf != "low":
            return "auto-permissive"
        elif mean >= 0.40:
            return "auto-conservative"
        return "always"

    def get_state(self) -> dict:
        return {
            "trust_means":   {t: round(self.get_trust_mean(t), 4) for t in TASK_TYPES},
            "uncertainties": {t: round(self.get_uncertainty(t), 4) for t in TASK_TYPES},
            "confidences":   {t: self.get_confidence(t) for t in TASK_TYPES},
            "policies":      {t: self.get_policy(t) for t in TASK_TYPES},
            "alpha":         dict(self._alpha),
            "beta":          dict(self._beta),
        }

    def restore_from(self, alpha: dict, beta: dict) -> None:
        """Restore Beta parameters from persisted values (e.g. loaded from DB)."""
        for t in TASK_TYPES:
            if t in alpha:
                self._alpha[t] = float(alpha[t])
            if t in beta:
                self._beta[t] = float(beta[t])
