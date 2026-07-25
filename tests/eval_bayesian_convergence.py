"""
eval_bayesian_convergence.py
----------------------------
Standalone evaluation script for Bayesian trust convergence in the HALO project.
No pytest required.  Run from the HALO directory:

    PYTHONPATH=src python tests/eval_bayesian_convergence.py

Sections
--------
Part 1 : Mathematical convergence theorem (analytical derivation)
Part 2 : Numerical simulation (50 interactions per tier)
Part 3 : Cross-tier isolation test
Part 4 : Cold-start comparison (informative vs. flat priors)
Part 5 : Report-ready summary table
"""

import math
import random
import importlib.util
import sys
import os

# Import feedback_loop directly by file path to avoid pulling in the full
# halo package __init__.py, which requires optional autogen_agentchat.
_HALO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "halo", "feedback_loop.py",
)
_spec = importlib.util.spec_from_file_location("halo.feedback_loop", _HALO_SRC)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BayesianFeedbackLoop = _mod.BayesianFeedbackLoop
_PRIORS    = _mod._PRIORS
TASK_TYPES = _mod.TASK_TYPES
APPROVE_WEIGHT = _mod.APPROVE_WEIGHT
REJECT_WEIGHT  = _mod.REJECT_WEIGHT

random.seed(42)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_std(a: float, b: float) -> float:
    s = a + b
    variance = (a * b) / (s * s * (s + 1.0))
    return math.sqrt(variance)


def analytical_n_converge(a0: float, b0: float, p_approve: float,
                           sigma_threshold: float = 0.05) -> int:
    """
    Find the smallest N such that sigma_N < sigma_threshold, given that
    each step adds APPROVE_WEIGHT to alpha with probability p_approve and
    REJECT_WEIGHT to beta with probability (1 - p_approve), matching the
    asymmetric weights actually used by BayesianFeedbackLoop.record_feedback().
    The expected increments are used (fractional steps), so the returned
    value is the ceiling.
    """
    a, b = a0, b0
    n = 0
    while True:
        if beta_std(a, b) < sigma_threshold:
            return n
        # advance one expected step, weighted the same way record_feedback() is
        a += p_approve * APPROVE_WEIGHT
        b += (1.0 - p_approve) * REJECT_WEIGHT
        n += 1
        if n > 100_000:
            return -1   # should never happen for reasonable inputs


def _sep(char: str = "-", width: int = 72) -> str:
    return char * width


# ---------------------------------------------------------------------------
# PART 1 — Mathematical Convergence Theorem
# ---------------------------------------------------------------------------

print(_sep("="))
print("PART 1 — MATHEMATICAL CONVERGENCE THEOREM")
print(_sep("="))
print()
print("Theorem (Bayesian update for a Beta-distributed trust parameter)")
print(_sep())
print()
print("Let θ be the latent approval probability for a task tier, with")
print("conjugate prior  θ ~ Beta(α₀, β₀).")
print()
print("After observing n_a approvals and n_r rejections the posterior is:")
print()
print("  θ | data  ~  Beta(α_n, β_n)")
print("  α_n = α₀ + n_a")
print("  β_n = β₀ + n_r")
print()
print("Posterior mean:")
print()
print("  θ̂_n = α_n / (α_n + β_n)")
print("       = (α₀ + n_a) / (α₀ + β₀ + n_a + n_r)")
print()
print("Posterior standard deviation:")
print()
print("  σ_n = sqrt( α_n · β_n / ((α_n + β_n)² · (α_n + β_n + 1)) )")
print()
print("Convergence proof  (σ_n → 0 monotonically as n = n_a + n_r → ∞):")
print()
print("  Let n = n_a + n_r  and  p = n_a / n  (the empirical approve rate).")
print("  Then  α_n ≈ α₀ + p·n  and  β_n ≈ β₀ + (1−p)·n  for large n.")
print()
print("  Numerator:  α_n · β_n  ≈  [p(1−p)] · n²   →  O(n²)")
print("  Denominator: (α_n+β_n)² · (α_n+β_n+1)")
print("               ≈  n²  ·  (n+1)              →  O(n³)")
print()
print("  Therefore  σ²_n  ~  O(n²)/O(n³)  =  O(1/n)  → 0 as n → ∞.")
print()
print("  Monotonicity: adding any observation (approve or reject) strictly")
print("  increases α_n+β_n, making (α_n+β_n)²(α_n+β_n+1) grow faster than")
print("  α_n·β_n (the AM-GM bound α·β ≤ (α+β)²/4 ensures the ratio falls).")
print()

# --- Analytical N_converge per tier ---
tier_configs = {
    "research":      {"a0": 8.0, "b0": 2.0, "p": 0.80},
    "transactional": {"a0": 5.0, "b0": 5.0, "p": 0.50},
    "destructive":   {"a0": 2.0, "b0": 8.0, "p": 0.20},
}

print("Analytical N such that σ_N < 0.05 (expected-value path):")
print()
print(f"  {'Tier':<14}  {'α₀':>4}  {'β₀':>4}  {'θ̂₀':>6}  {'σ₀':>7}  {'N_converge':>10}")
print(f"  {'-'*14}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*10}")

analytical_n = {}
initial_sigma = {}
initial_mean = {}

for tier, cfg in tier_configs.items():
    a0, b0, p = cfg["a0"], cfg["b0"], cfg["p"]
    s0 = beta_std(a0, b0)
    m0 = beta_mean(a0, b0)
    N = analytical_n_converge(a0, b0, p)
    analytical_n[tier] = N
    initial_sigma[tier] = s0
    initial_mean[tier] = m0
    print(f"  {tier:<14}  {a0:>4.1f}  {b0:>4.1f}  {m0:>6.4f}  {s0:>7.4f}  {N:>10d}")

print()

# ---------------------------------------------------------------------------
# PART 2 — Numerical Simulation (50 interactions per tier)
# ---------------------------------------------------------------------------

print(_sep("="))
print("PART 2 — NUMERICAL SIMULATION (50 INTERACTIONS PER TIER)")
print(_sep("="))
print()

# Approval probabilities mirror the prior means
SIM_PROBS = {
    "research":      0.80,
    "transactional": 0.50,
    "destructive":   0.20,   # low approve; corrections also update beta+=0.5
}

sim_results = {}  # tier -> list of dicts per step

for tier in TASK_TYPES:
    loop = BayesianFeedbackLoop()
    p_approve = SIM_PROBS[tier]
    steps = []
    n_converge = None

    for n in range(1, 51):
        r = random.random()
        if tier == "destructive":
            # policy is "always" → use a mix of approve/correct/reject
            # to reflect realistic low-trust interaction
            if r < p_approve:
                decision = "approve"
            elif r < p_approve + 0.40:
                decision = "correct"
            else:
                decision = "reject"
        else:
            decision = "approve" if r < p_approve else "reject"

        loop.record_feedback(tier, decision)

        mean = loop.get_trust_mean(tier)
        unc  = loop.get_uncertainty(tier)
        pol  = loop.get_policy(tier)

        steps.append({"n": n, "mean": mean, "uncertainty": unc, "policy": pol,
                      "decision": decision})

        if n_converge is None and unc < 0.05:
            n_converge = n

    sim_results[tier] = {"steps": steps, "n_converge": n_converge}

# Print step-by-step tables per tier
for tier in TASK_TYPES:
    cfg   = tier_configs[tier]
    data  = sim_results[tier]
    steps = data["steps"]
    nc    = data["n_converge"] if data["n_converge"] else ">50"

    print(f"Tier: {tier.upper()}  (prior Beta({cfg['a0']:.0f},{cfg['b0']:.0f}), "
          f"p_approve={SIM_PROBS[tier]:.0%})  N_converge={nc}")
    print()
    print(f"  {'n':>3}  {'decision':<8}  {'θ̂_n':>7}  {'σ_n':>7}  policy")
    print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*18}")

    # Print every 5th step to keep output readable, plus step 1 and final
    highlights = set([1] + list(range(5, 51, 5)))
    for s in steps:
        if s["n"] in highlights:
            print(f"  {s['n']:>3}  {s['decision']:<8}  "
                  f"{s['mean']:>7.4f}  {s['uncertainty']:>7.4f}  {s['policy']}")
    print()

# ---------------------------------------------------------------------------
# PART 3 — Cross-tier isolation test
# ---------------------------------------------------------------------------

print(_sep("="))
print("PART 3 — CROSS-TIER ISOLATION TEST")
print(_sep("="))
print()
print("Claim: updating one tier's Beta distribution does NOT alter others.")
print()

loop_iso = BayesianFeedbackLoop()

initial_unc = {t: loop_iso.get_uncertainty(t) for t in TASK_TYPES}
print("Initial uncertainties (all tiers, fresh instance):")
for t in TASK_TYPES:
    print(f"  {t:<14}  σ₀ = {initial_unc[t]:.6f}")
print()

# Apply 20 approvals to "research" only
for _ in range(20):
    loop_iso.record_feedback("research", "approve")

post_unc = {t: loop_iso.get_uncertainty(t) for t in TASK_TYPES}

print("After 20 approvals applied exclusively to 'research':")
for t in TASK_TYPES:
    changed = " ← CHANGED" if abs(post_unc[t] - initial_unc[t]) > 1e-12 else " (unchanged)"
    print(f"  {t:<14}  σ = {post_unc[t]:.6f}{changed}")

print()
print("Isolation verified:", (
    abs(post_unc["transactional"] - initial_unc["transactional"]) < 1e-12 and
    abs(post_unc["destructive"]   - initial_unc["destructive"])   < 1e-12
))
print()

# Confirm research did converge further
delta_research = initial_unc["research"] - post_unc["research"]
print(f"Research σ decreased by {delta_research:.6f}  "
      f"(from {initial_unc['research']:.6f} to {post_unc['research']:.6f})")
print()

# ---------------------------------------------------------------------------
# PART 4 — Cold-start comparison
# ---------------------------------------------------------------------------

print(_sep("="))
print("PART 4 — COLD-START COMPARISON")
print(_sep("="))
print()
print("Comparing convergence speed: informative priors (HALO) vs. flat Beta(1,1).")
print("Focus tier: TRANSACTIONAL  (p_approve = 0.50, threshold σ < 0.05)")
print()

# Informative prior for transactional: Beta(5,5)
N_informative = analytical_n_converge(5.0, 5.0, 0.50)

# Flat prior: Beta(1,1)
N_flat = analytical_n_converge(1.0, 1.0, 0.50)

sigma0_informative = beta_std(5.0, 5.0)
sigma0_flat        = beta_std(1.0, 1.0)
mean0_informative  = beta_mean(5.0, 5.0)
mean0_flat         = beta_mean(1.0, 1.0)

extra_steps = N_flat - N_informative

print(f"  {'Prior':<20}  {'θ̂₀':>6}  {'σ₀':>7}  {'N_converge':>10}")
print(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*10}")
print(f"  {'Beta(5,5) [HALO]':<20}  {mean0_informative:>6.4f}  "
      f"{sigma0_informative:>7.4f}  {N_informative:>10d}")
print(f"  {'Beta(1,1) [flat]':<20}  {mean0_flat:>6.4f}  "
      f"{sigma0_flat:>7.4f}  {N_flat:>10d}")
print()
print(f"  The flat prior requires {extra_steps} additional interactions to reach σ < 0.05.")
print()

# Show convergence trajectory for both priors at selected steps
print("  Trajectory comparison (expected-value path):")
print(f"  {'n':>4}  {'σ (Beta(5,5))':>14}  {'σ (Beta(1,1))':>14}")
print(f"  {'-'*4}  {'-'*14}  {'-'*14}")

a_inf, b_inf = 5.0, 5.0
a_flat, b_flat = 1.0, 1.0
p = 0.50

checkpoints = list(range(0, max(N_informative, N_flat) + 1, max(1, (max(N_informative, N_flat)) // 10)))
checkpoints = sorted(set(checkpoints + [0, N_informative, N_flat]))

ai, bi = a_inf, b_inf
af, bf = a_flat, b_flat
prev_chk = 0

# Walk step by step and capture at checkpoints
ai, bi = a_inf, b_inf
af, bf = a_flat, b_flat

max_n = max(N_informative, N_flat)
traj_inf  = [beta_std(ai, bi)]
traj_flat = [beta_std(af, bf)]
for _ in range(max_n):
    ai += p; bi += (1 - p)
    af += p; bf += (1 - p)
    traj_inf.append(beta_std(ai, bi))
    traj_flat.append(beta_std(af, bf))

for chk in checkpoints:
    if chk <= max_n:
        marker_inf  = " *" if chk == N_informative else "  "
        marker_flat = " *" if chk == N_flat        else "  "
        print(f"  {chk:>4}  {traj_inf[chk]:>12.6f}{marker_inf}  "
              f"{traj_flat[chk]:>12.6f}{marker_flat}")
print()
print("  (* = first step where σ < 0.05)")
print()

# Also show all three tiers
print("  All tiers — flat Beta(1,1) vs. HALO informative priors:")
print()
print(f"  {'Tier':<14}  {'N_converge HALO':>16}  {'N_converge flat':>16}  {'Extra steps':>11}")
print(f"  {'-'*14}  {'-'*16}  {'-'*16}  {'-'*11}")
for tier, cfg in tier_configs.items():
    Nh = analytical_n_converge(cfg["a0"], cfg["b0"], cfg["p"])
    Nf = analytical_n_converge(1.0,       1.0,       cfg["p"])
    print(f"  {tier:<14}  {Nh:>16d}  {Nf:>16d}  {Nf-Nh:>11d}")
print()

# ---------------------------------------------------------------------------
# PART 5 — Report-ready summary table
# ---------------------------------------------------------------------------

print(_sep("="))
print("PART 5 — REPORT-READY SUMMARY TABLE")
print(_sep("="))
print()

# N_converge from numerical simulation
sim_nc = {tier: sim_results[tier]["n_converge"] for tier in TASK_TYPES}

# Flat prior simulation (transactional, Beta(1,1))
loop_flat = BayesianFeedbackLoop()
# Patch alpha/beta directly using restore_from to simulate flat prior
loop_flat.restore_from(
    {"research": 1.0, "transactional": 1.0, "destructive": 1.0},
    {"research": 1.0, "transactional": 1.0, "destructive": 1.0},
)
nc_flat_sim = None
random.seed(42)
for step in range(1, 501):
    decision = "approve" if random.random() < 0.50 else "reject"
    loop_flat.record_feedback("transactional", decision)
    if nc_flat_sim is None and loop_flat.get_uncertainty("transactional") < 0.05:
        nc_flat_sim = step

# Gather initial (prior) values for display
prior_labels = {
    "research":      "Beta(8,2)",
    "transactional": "Beta(5,5)",
    "destructive":   "Beta(2,8)",
}

header = (
    f"{'Tier':<14} | {'Prior':<10} | {'θ̂₀':>5} | {'σ₀':>7} | "
    f"{'N_converge (σ<0.05)':>20}"
)
sep_row = "-" * 14 + "-+-" + "-" * 10 + "-+-" + "-" * 5 + "-+-" + "-" * 7 + "-+-" + "-" * 20

print(header)
print(sep_row)

rows = [
    ("Research",      "Beta(8,2)", 8.0, 2.0, sim_nc.get("research")),
    ("Transactional", "Beta(5,5)", 5.0, 5.0, sim_nc.get("transactional")),
    ("Destructive",   "Beta(2,8)", 2.0, 8.0, sim_nc.get("destructive")),
    ("Flat (comp.)",  "Beta(1,1)", 1.0, 1.0, nc_flat_sim),
]

for (name, prior_lbl, a0, b0, nc) in rows:
    m0 = beta_mean(a0, b0)
    s0 = beta_std(a0, b0)
    nc_str = str(nc) if nc is not None else ">500"
    print(
        f"{name:<14} | {prior_lbl:<10} | {m0:>5.2f} | {s0:>7.4f} | {nc_str:>20}"
    )

print()
print("Notes:")
print("  θ̂₀  = prior mean = α₀/(α₀+β₀)")
print("  σ₀   = prior std dev = sqrt(α₀β₀/((α₀+β₀)²(α₀+β₀+1)))")
print("  N_converge = first interaction step where σ_n < 0.05 in simulation")
print("  Flat (comp.) row uses Beta(1,1) for transactional tier only;")
print("  all other parameters as defined in HALO feedback_loop.py.")
print()
print(_sep("="))
print("END OF EVALUATION")
print(_sep("="))
