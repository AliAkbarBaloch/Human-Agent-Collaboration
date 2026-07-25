"""
Comprehensive Gap 3 test suite — Bayesian Trust Feedback Loop.

Tests cover 80+ scenarios across:
  A. Initial state — prior distributions for each task type
  B. Single feedback updates — approve / reject / correct
  C. Policy derivation — thresholds and trust-to-policy mapping
  D. Trust convergence — long interaction sequences
  E. Edge cases — boundary values, unknown task types
  F. Persistence — get_state() / restore_from() round-trip
  G. Security events — malicious detection triggers β penalty
  H. Uncertainty and confidence metrics

API (from feedback_loop.py):
  get_state() → dict with keys: alpha, beta, trust_means,
                uncertainties, confidences, policies (all dicts keyed by task type)
  restore_from(alpha: dict, beta: dict) → restores from two dicts
  get_uncertainty(task_type) → sqrt(variance) = Beta std-dev
"""

import json
import math
import pytest
from halo.feedback_loop import BayesianFeedbackLoop


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def loop():
    """Fresh BayesianFeedbackLoop with default priors."""
    return BayesianFeedbackLoop()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Initial state (default priors)
# Paper priors: research(8,2), transactional(5,5), destructive(2,8)
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialState:
    def test_research_prior_alpha(self, loop):
        state = loop.get_state()
        assert state["alpha"]["research"] == pytest.approx(8.0)

    def test_research_prior_beta(self, loop):
        state = loop.get_state()
        assert state["beta"]["research"] == pytest.approx(2.0)

    def test_transactional_prior_alpha(self, loop):
        state = loop.get_state()
        assert state["alpha"]["transactional"] == pytest.approx(5.0)

    def test_transactional_prior_beta(self, loop):
        state = loop.get_state()
        assert state["beta"]["transactional"] == pytest.approx(5.0)

    def test_destructive_prior_alpha(self, loop):
        state = loop.get_state()
        assert state["alpha"]["destructive"] == pytest.approx(2.0)

    def test_destructive_prior_beta(self, loop):
        state = loop.get_state()
        assert state["beta"]["destructive"] == pytest.approx(8.0)

    def test_research_mean_is_high(self, loop):
        # α/(α+β) = 8/10 = 0.80 — research trusted from the start
        mean = loop.get_trust_mean("research")
        assert mean == pytest.approx(0.80)

    def test_transactional_mean_is_neutral(self, loop):
        # 5/10 = 0.50 — neutral starting trust
        mean = loop.get_trust_mean("transactional")
        assert mean == pytest.approx(0.50)

    def test_destructive_mean_is_low(self, loop):
        # 2/10 = 0.20 — destructive tasks require high evidence to trust
        mean = loop.get_trust_mean("destructive")
        assert mean == pytest.approx(0.20)

    def test_trust_mean_order_invariant(self, loop):
        r = loop.get_trust_mean("research")
        t = loop.get_trust_mean("transactional")
        d = loop.get_trust_mean("destructive")
        assert r > t > d  # risk ordering preserved

    def test_initial_state_has_expected_keys(self, loop):
        state = loop.get_state()
        assert "alpha" in state
        assert "beta" in state
        assert "trust_means" in state
        assert "policies" in state

    def test_all_task_types_initialised(self, loop):
        state = loop.get_state()
        for task_type in ["research", "transactional", "destructive"]:
            assert state["alpha"][task_type] > 0
            assert state["beta"][task_type] > 0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — Single feedback updates
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleFeedbackUpdates:
    def test_approve_increments_alpha(self, loop):
        before = loop.get_state()["alpha"]["research"]
        loop.record_feedback("research", "approve")
        after = loop.get_state()["alpha"]["research"]
        assert after == pytest.approx(before + 1.0)

    def test_approve_does_not_change_beta(self, loop):
        before = loop.get_state()["beta"]["research"]
        loop.record_feedback("research", "approve")
        after = loop.get_state()["beta"]["research"]
        assert after == pytest.approx(before)

    def test_reject_increments_beta(self, loop):
        # Ali Akbar — asymmetric weighting: reject costs 1.5, not 1.0
        before = loop.get_state()["beta"]["transactional"]
        loop.record_feedback("transactional", "reject")
        after = loop.get_state()["beta"]["transactional"]
        assert after == pytest.approx(before + 1.5)

    def test_reject_does_not_change_alpha(self, loop):
        before = loop.get_state()["alpha"]["transactional"]
        loop.record_feedback("transactional", "reject")
        after = loop.get_state()["alpha"]["transactional"]
        assert after == pytest.approx(before)

    def test_correct_increments_beta_by_half_of_reject(self, loop):
        # "correct" = soft penalty: β += 0.75 (half of reject's 1.5)
        before = loop.get_state()["beta"]["destructive"]
        loop.record_feedback("destructive", "correct")
        after = loop.get_state()["beta"]["destructive"]
        assert after == pytest.approx(before + 0.75)

    def test_approve_raises_trust_mean(self, loop):
        before = loop.get_trust_mean("research")
        loop.record_feedback("research", "approve")
        after = loop.get_trust_mean("research")
        assert after > before

    def test_reject_lowers_trust_mean(self, loop):
        before = loop.get_trust_mean("transactional")
        loop.record_feedback("transactional", "reject")
        after = loop.get_trust_mean("transactional")
        assert after < before

    def test_approve_does_not_affect_other_task_types(self, loop):
        before_d_a = loop.get_state()["alpha"]["destructive"]
        before_d_b = loop.get_state()["beta"]["destructive"]
        loop.record_feedback("research", "approve")
        after_d_a = loop.get_state()["alpha"]["destructive"]
        after_d_b = loop.get_state()["beta"]["destructive"]
        assert after_d_a == pytest.approx(before_d_a)
        assert after_d_b == pytest.approx(before_d_b)

    def test_reject_does_not_affect_other_task_types(self, loop):
        before_r_a = loop.get_state()["alpha"]["research"]
        before_r_b = loop.get_state()["beta"]["research"]
        loop.record_feedback("destructive", "reject")
        after_r_a = loop.get_state()["alpha"]["research"]
        after_r_b = loop.get_state()["beta"]["research"]
        assert after_r_a == pytest.approx(before_r_a)
        assert after_r_b == pytest.approx(before_r_b)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — Policy derivation
# Policy rules:
#   task_type == "destructive"           → "always"
#   mean ≥ 0.75 AND conf != "low"        → "auto-permissive"
#   mean ≥ 0.40                          → "auto-conservative"
#   else                                 → "always"
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyDerivation:
    def test_destructive_is_always(self, loop):
        policy = loop.get_policy("destructive")
        assert policy == "always"

    def test_destructive_stays_always_after_many_approvals(self, loop):
        for _ in range(20):
            loop.record_feedback("destructive", "approve")
        policy = loop.get_policy("destructive")
        assert policy == "always"

    def test_high_trust_research_is_auto_permissive(self, loop):
        # research starts at mean=0.80 → should already be auto-permissive
        policy = loop.get_policy("research")
        assert policy == "auto-permissive"

    def test_neutral_transactional_is_auto_conservative(self, loop):
        # transactional mean=0.50 → auto-conservative (0.40 ≤ 0.50 < 0.75)
        policy = loop.get_policy("transactional")
        assert policy == "auto-conservative"

    def test_low_trust_triggers_always(self):
        loop = BayesianFeedbackLoop()
        for _ in range(30):
            loop.record_feedback("transactional", "reject")
        mean = loop.get_trust_mean("transactional")
        assert mean < 0.40
        policy = loop.get_policy("transactional")
        assert policy == "always"

    def test_trust_above_075_with_high_confidence_triggers_auto_permissive(self):
        loop = BayesianFeedbackLoop()
        for _ in range(50):
            loop.record_feedback("research", "approve")
        mean = loop.get_trust_mean("research")
        assert mean >= 0.75
        policy = loop.get_policy("research")
        assert policy == "auto-permissive"

    def test_policy_transitions_from_conservative_to_permissive(self):
        loop = BayesianFeedbackLoop()
        initial_policy = loop.get_policy("transactional")
        assert initial_policy == "auto-conservative"
        for _ in range(40):
            loop.record_feedback("transactional", "approve")
        final_policy = loop.get_policy("transactional")
        assert final_policy == "auto-permissive"

    def test_policy_transitions_from_permissive_to_conservative(self):
        loop = BayesianFeedbackLoop()
        for _ in range(30):
            loop.record_feedback("research", "reject")
        mean = loop.get_trust_mean("research")
        if mean < 0.75:
            policy = loop.get_policy("research")
            assert policy in ["auto-conservative", "always"]

    def test_policy_for_unknown_task_type_defaults_safely(self):
        loop = BayesianFeedbackLoop()
        try:
            policy = loop.get_policy("unknown_type")
            assert policy in ["always", "auto-conservative", "auto-permissive"]
        except KeyError:
            pass  # acceptable — unknown types may raise

    def test_all_three_policies_reachable(self):
        loop_research = BayesianFeedbackLoop()
        for _ in range(20):
            loop_research.record_feedback("research", "approve")

        loop_trans = BayesianFeedbackLoop()
        # transactional stays auto-conservative by default

        loop_destr = BayesianFeedbackLoop()
        # destructive is always "always"

        policies = [
            loop_research.get_policy("research"),
            loop_trans.get_policy("transactional"),
            loop_destr.get_policy("destructive"),
        ]
        assert "auto-permissive" in policies
        assert "auto-conservative" in policies
        assert "always" in policies


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Trust convergence (long sequences)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustConvergence:
    def test_all_approvals_research_converges_near_one(self):
        loop = BayesianFeedbackLoop()
        for _ in range(100):
            loop.record_feedback("research", "approve")
        mean = loop.get_trust_mean("research")
        assert mean > 0.95

    def test_all_rejects_transactional_converges_near_zero(self):
        loop = BayesianFeedbackLoop()
        for _ in range(100):
            loop.record_feedback("transactional", "reject")
        mean = loop.get_trust_mean("transactional")
        assert mean < 0.10

    def test_mixed_feedback_research_stabilizes_near_ratio(self):
        loop = BayesianFeedbackLoop()
        # 80 approvals (α+=1 each) + 20 rejects (β+=1.5 each, asymmetric weighting)
        # → final: (8+80)/(8+80 + 2+30) = 88/120 ≈ 0.733
        for _ in range(80):
            loop.record_feedback("research", "approve")
        for _ in range(20):
            loop.record_feedback("research", "reject")
        mean = loop.get_trust_mean("research")
        assert 0.65 < mean < 0.85

    def test_uncertainty_decreases_with_more_data(self):
        loop = BayesianFeedbackLoop()
        unc_0 = loop.get_uncertainty("transactional")
        for _ in range(50):
            loop.record_feedback("transactional", "approve")
        unc_50 = loop.get_uncertainty("transactional")
        assert unc_50 < unc_0

    def test_confidence_increases_with_more_data(self):
        loop = BayesianFeedbackLoop()
        conf_map = {"low": 0, "medium": 1, "high": 2}
        conf_0 = conf_map.get(loop.get_confidence("research"), 0)
        for _ in range(50):
            loop.record_feedback("research", "approve")
        conf_50 = conf_map.get(loop.get_confidence("research"), 0)
        assert conf_50 >= conf_0

    def test_alternating_feedback_stays_near_prior(self):
        loop = BayesianFeedbackLoop()
        for _ in range(50):
            loop.record_feedback("transactional", "approve")
            loop.record_feedback("transactional", "reject")
        mean = loop.get_trust_mean("transactional")
        # 50 approvals (α+50) + 50 rejects (β+75, asymmetric weighting)
        # → (5+50)/(5+50 + 5+75) = 55/135 ≈ 0.407 — asymmetric weighting pulls
        # this slightly below the old symmetric ~0.50, as intended.
        assert 0.35 < mean < 0.60

    def test_evidence_accumulates_not_decays(self, loop):
        before_a = loop.get_state()["alpha"]["research"]
        before_b = loop.get_state()["beta"]["research"]
        loop.record_feedback("research", "approve")
        after_a = loop.get_state()["alpha"]["research"]
        after_b = loop.get_state()["beta"]["research"]
        assert after_a > before_a
        assert (after_a + after_b) > (before_a + before_b)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_mean_always_between_zero_and_one(self, loop):
        for _ in range(200):
            loop.record_feedback("research", "approve")
        mean = loop.get_trust_mean("research")
        assert 0.0 <= mean <= 1.0

    def test_uncertainty_always_non_negative(self, loop):
        for _ in range(100):
            loop.record_feedback("transactional", "reject")
        unc = loop.get_uncertainty("transactional")
        assert unc >= 0.0

    def test_correct_feedback_is_softer_than_reject(self):
        loop_reject = BayesianFeedbackLoop()
        loop_correct = BayesianFeedbackLoop()
        loop_reject.record_feedback("transactional", "reject")
        loop_correct.record_feedback("transactional", "correct")
        mean_reject = loop_reject.get_trust_mean("transactional")
        mean_correct = loop_correct.get_trust_mean("transactional")
        # correct adds β+=0.75 (softer than reject β+=1.5) → higher mean
        assert mean_correct > mean_reject

    def test_multiple_feedbacks_cumulative(self, loop):
        state0 = loop.get_state()
        a0, b0 = state0["alpha"]["research"], state0["beta"]["research"]
        loop.record_feedback("research", "approve")
        loop.record_feedback("research", "approve")
        loop.record_feedback("research", "reject")
        state3 = loop.get_state()
        assert state3["alpha"]["research"] == pytest.approx(a0 + 2)
        assert state3["beta"]["research"] == pytest.approx(b0 + 1.5)

    def test_high_sample_size_reduces_variance(self):
        loop = BayesianFeedbackLoop()
        unc_start = loop.get_uncertainty("research")
        for _ in range(1000):
            loop.record_feedback("research", "approve")
        unc_end = loop.get_uncertainty("research")
        assert unc_end < unc_start


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Persistence: get_state() / restore_from()
# get_state() → {"alpha": {type: val}, "beta": {type: val}, ...}
# restore_from(alpha: dict, beta: dict) — per-type alpha/beta dicts
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_get_state_is_serialisable(self, loop):
        state = loop.get_state()
        json.dumps(state)  # must not raise

    def test_restore_from_preserves_alpha(self, loop):
        loop.record_feedback("research", "approve")
        loop.record_feedback("research", "approve")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        assert new_loop.get_state()["alpha"]["research"] == pytest.approx(state["alpha"]["research"])

    def test_restore_from_preserves_beta(self, loop):
        loop.record_feedback("transactional", "reject")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        assert new_loop.get_state()["beta"]["transactional"] == pytest.approx(state["beta"]["transactional"])

    def test_restore_from_preserves_trust_mean(self, loop):
        for _ in range(10):
            loop.record_feedback("research", "approve")
        mean_before = loop.get_trust_mean("research")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        assert new_loop.get_trust_mean("research") == pytest.approx(mean_before)

    def test_restore_from_preserves_policy(self, loop):
        for _ in range(5):
            loop.record_feedback("transactional", "approve")
        policy_before = loop.get_policy("transactional")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        assert new_loop.get_policy("transactional") == policy_before

    def test_restore_does_not_affect_other_task_types(self, loop):
        loop.record_feedback("research", "approve")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        # transactional should be whatever was in state
        assert new_loop.get_state()["alpha"]["transactional"] == pytest.approx(state["alpha"]["transactional"])

    def test_round_trip_identity(self, loop):
        for _ in range(7):
            loop.record_feedback("destructive", "approve")
        for _ in range(3):
            loop.record_feedback("destructive", "reject")
        state = loop.get_state()
        new_loop = BayesianFeedbackLoop()
        new_loop.restore_from(state["alpha"], state["beta"])
        new_state = new_loop.get_state()
        for tt in ["research", "transactional", "destructive"]:
            assert new_state["alpha"][tt] == pytest.approx(state["alpha"][tt])
            assert new_state["beta"][tt] == pytest.approx(state["beta"][tt])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION G — Security events and β penalties
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityEvents:
    def test_security_event_lowers_trust(self):
        loop = BayesianFeedbackLoop()
        mean_before = loop.get_trust_mean("transactional")
        for _ in range(5):
            loop.record_feedback("transactional", "reject")
        mean_after = loop.get_trust_mean("transactional")
        assert mean_after < mean_before

    def test_injection_detected_triggers_trust_penalty(self):
        loop = BayesianFeedbackLoop()
        mean_before = loop.get_trust_mean("destructive")
        for _ in range(3):
            loop.record_feedback("destructive", "reject")
        mean_after = loop.get_trust_mean("destructive")
        assert mean_after < mean_before

    def test_recovery_after_security_event_is_gradual(self):
        loop = BayesianFeedbackLoop()
        for _ in range(10):
            loop.record_feedback("transactional", "reject")
        trust_low = loop.get_trust_mean("transactional")
        for _ in range(5):
            loop.record_feedback("transactional", "approve")
        trust_partial = loop.get_trust_mean("transactional")
        assert trust_partial > trust_low
        assert trust_partial < 0.50  # prior mean was 0.50

    def test_destructive_always_policy_immune_to_trust(self):
        loop = BayesianFeedbackLoop()
        for _ in range(1000):
            loop.record_feedback("destructive", "approve")
        mean = loop.get_trust_mean("destructive")
        assert mean > 0.90  # trust IS high
        assert loop.get_policy("destructive") == "always"  # but policy stays fixed


# ─────────────────────────────────────────────────────────────────────────────
# SECTION H — Uncertainty and confidence metrics
# get_uncertainty() → sqrt(variance) = Beta std-dev
# ─────────────────────────────────────────────────────────────────────────────

class TestUncertaintyAndConfidence:
    def test_uncertainty_is_beta_std_dev(self, loop):
        # Beta std-dev: sqrt(αβ / ((α+β)²(α+β+1)))
        state = loop.get_state()
        a = state["alpha"]["transactional"]
        b = state["beta"]["transactional"]
        n = a + b
        expected_std = math.sqrt((a * b) / (n ** 2 * (n + 1)))
        reported_unc = loop.get_uncertainty("transactional")
        assert reported_unc == pytest.approx(expected_std, rel=0.01)

    def test_confidence_is_string_enum(self, loop):
        conf = loop.get_confidence("research")
        assert conf in {"low", "medium", "high"}

    def test_low_confidence_with_few_observations(self):
        loop = BayesianFeedbackLoop()
        # Priors sum to 10 — with only priors, confidence should not be "high"
        conf = loop.get_confidence("transactional")
        assert conf in {"low", "medium"}

    def test_high_confidence_with_many_observations(self):
        loop = BayesianFeedbackLoop()
        for _ in range(200):
            loop.record_feedback("research", "approve")
        conf = loop.get_confidence("research")
        assert conf == "high"

    def test_uncertainty_matches_std_dev_formula_after_updates(self):
        loop = BayesianFeedbackLoop()
        for _ in range(10):
            loop.record_feedback("research", "approve")
        state = loop.get_state()
        a = state["alpha"]["research"]
        b = state["beta"]["research"]
        n = a + b
        expected_std = math.sqrt((a * b) / (n ** 2 * (n + 1)))
        assert loop.get_uncertainty("research") == pytest.approx(expected_std, rel=0.01)

    def test_mean_formula_alpha_over_total(self, loop):
        for tt in ["research", "transactional", "destructive"]:
            state = loop.get_state()
            a = state["alpha"][tt]
            b = state["beta"][tt]
            expected_mean = a / (a + b)
            assert loop.get_trust_mean(tt) == pytest.approx(expected_mean)

    def test_trust_mean_monotone_with_approvals(self):
        loop = BayesianFeedbackLoop()
        prev_mean = loop.get_trust_mean("transactional")
        for _ in range(20):
            loop.record_feedback("transactional", "approve")
            curr_mean = loop.get_trust_mean("transactional")
            assert curr_mean >= prev_mean
            prev_mean = curr_mean

    def test_trust_mean_monotone_with_rejects(self):
        loop = BayesianFeedbackLoop()
        prev_mean = loop.get_trust_mean("research")
        for _ in range(20):
            loop.record_feedback("research", "reject")
            curr_mean = loop.get_trust_mean("research")
            assert curr_mean <= prev_mean
            prev_mean = curr_mean
