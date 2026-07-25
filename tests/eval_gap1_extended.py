"""
eval_gap1_extended.py — Gap 1 Extended Evaluation
===================================================
Evaluates classify_task() on a labeled dataset of 120 actions.

Dataset composition
-------------------
  30  research       — clearly non-transactional, non-destructive phrasing
  30  transactional  — contain booking/ordering/payment keywords
  30  destructive    — contain delete/remove/wipe/format/drop/destroy/erase/
                       purge/overwrite/reset/clear keywords
  15  boundary R→T   — phrasing that leans research but contains transactional
                       intent; ground truth = transactional
  15  boundary T→D   — contain words like "cancel"/"remove" which are in
                       DESTRUCTIVE_KEYWORDS; ground truth = destructive

Outputs
-------
  1. Overall accuracy + per-class precision / recall / F1
  2. 3×3 confusion matrix
  3. Power analysis: minimum detectable effect size at 80 % power, α = 0.05
"""

import os
import sys
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup — import task_classifier directly to avoid halo/__init__.py
# which depends on autogen_agentchat (not installed in eval environment)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.join(_HERE, "..", "src", "halo", "task_classifier.py")
_spec = _ilu.spec_from_file_location("task_classifier", _MOD_PATH)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
classify_task = _mod.classify_task

# ---------------------------------------------------------------------------
# Dataset — 120 labeled actions
# ---------------------------------------------------------------------------

# ── 30 research ──────────────────────────────────────────────────────────────
RESEARCH = [
    {"text": "What is the capital of France?",                                        "expected": "research"},
    {"text": "Who is the current CEO of OpenAI?",                                     "expected": "research"},
    {"text": "When was the Eiffel Tower built?",                                      "expected": "research"},
    {"text": "How many moons does Jupiter have?",                                     "expected": "research"},
    {"text": "Explain the concept of quantum entanglement.",                          "expected": "research"},
    {"text": "Tell me about the history of the Roman Empire.",                        "expected": "research"},
    {"text": "Summarize the plot of War and Peace.",                                  "expected": "research"},
    {"text": "Describe the water cycle.",                                             "expected": "research"},
    {"text": "Find the nearest public library.",                                      "expected": "research"},
    {"text": "Search for peer-reviewed papers on transformer architectures.",         "expected": "research"},
    {"text": "List all Nobel Prize winners in Physics since 2010.",                   "expected": "research"},
    {"text": "Show me the weather forecast for Berlin this weekend.",                 "expected": "research"},
    {"text": "Look up the opening hours of the Louvre.",                              "expected": "research"},
    {"text": "Read the Wikipedia article on machine learning.",                       "expected": "research"},
    {"text": "Browse the company's annual report for 2023.",                          "expected": "research"},
    {"text": "Visit the NASA website and get information on the Mars missions.",      "expected": "research"},
    {"text": "Check the current exchange rate between USD and EUR.",                  "expected": "research"},
    {"text": "Tell me the ingredients in a classic margherita pizza.",                "expected": "research"},
    {"text": "Show me a list of Python built-in functions.",                          "expected": "research"},
    {"text": "Explain how HTTPS encryption works.",                                   "expected": "research"},
    {"text": "Find the top ten universities in Germany.",                             "expected": "research"},
    {"text": "Search for the latest news on electric vehicles.",                      "expected": "research"},
    {"text": "Look up the definition of 'epistemology'.",                             "expected": "research"},
    {"text": "List the countries in the European Union.",                             "expected": "research"},
    {"text": "Describe the symptoms of vitamin D deficiency.",                        "expected": "research"},
    {"text": "Get information about internship programs at Google.",                  "expected": "research"},
    {"text": "Check whether the ISS is visible tonight from Munich.",                 "expected": "research"},
    {"text": "Tell me about TypeScript's type inference system.",                     "expected": "research"},
    {"text": "Show me the official Python documentation for asyncio.",                "expected": "research"},
    {"text": "Browse recent arXiv preprints on large language models.",               "expected": "research"},
]

# ── 30 transactional ─────────────────────────────────────────────────────────
TRANSACTIONAL = [
    {"text": "Book a round-trip flight from Munich to London.",                       "expected": "transactional"},
    {"text": "Order a pepperoni pizza from the nearest delivery app.",                "expected": "transactional"},
    {"text": "Buy a three-month gym membership.",                                     "expected": "transactional"},
    {"text": "Purchase the Pro plan on the SaaS dashboard.",                         "expected": "transactional"},
    {"text": "Sign up for the weekly newsletter.",                                    "expected": "transactional"},
    {"text": "Register for the machine learning conference in Vienna.",               "expected": "transactional"},
    {"text": "Pay the outstanding invoice #4521.",                                    "expected": "transactional"},
    {"text": "Checkout and place the items in my cart.",                              "expected": "transactional"},
    {"text": "Subscribe to the premium tier of the streaming service.",               "expected": "transactional"},
    {"text": "Add to cart the noise-cancelling headphones.",                          "expected": "transactional"},
    {"text": "Submit the travel reimbursement form to HR.",                           "expected": "transactional"},
    {"text": "Fill in the contact form and send it to the support team.",             "expected": "transactional"},
    {"text": "Upload my CV to the job application portal.",                           "expected": "transactional"},
    {"text": "Post the new blog entry on the company website.",                       "expected": "transactional"},
    {"text": "Create a new user account on the internal wiki.",                       "expected": "transactional"},
    {"text": "Transfer €500 to the savings account.",                                "expected": "transactional"},
    {"text": "Schedule a dentist appointment for next Tuesday.",                      "expected": "transactional"},
    {"text": "Reserve a table for four at the Italian restaurant.",                   "expected": "transactional"},
    {"text": "Fill out the visa application form.",                                   "expected": "transactional"},
    {"text": "Send the project proposal to the client via email.",                    "expected": "transactional"},
    {"text": "Order replacement ink cartridges for the office printer.",              "expected": "transactional"},
    {"text": "Pay the electricity bill online.",                                      "expected": "transactional"},
    {"text": "Book a hotel room in Amsterdam for three nights.",                      "expected": "transactional"},
    {"text": "Register my new laptop on the manufacturer warranty portal.",           "expected": "transactional"},
    {"text": "Subscribe me to the daily digest email.",                               "expected": "transactional"},
    {"text": "Submit the pull-request review form.",                                  "expected": "transactional"},
    {"text": "Purchase an annual software licence for the design tool.",              "expected": "transactional"},
    {"text": "Sign up for the free trial of the project management tool.",            "expected": "transactional"},
    {"text": "Create an event on the shared team calendar.",                          "expected": "transactional"},
    {"text": "Post my availability on the meeting scheduling page.",                  "expected": "transactional"},
]

# ── 30 destructive ───────────────────────────────────────────────────────────
DESTRUCTIVE = [
    {"text": "Delete all log files older than 30 days.",                              "expected": "destructive"},
    {"text": "Remove the duplicate rows from the database table.",                    "expected": "destructive"},
    {"text": "Wipe the SD card before giving the camera to a colleague.",             "expected": "destructive"},
    {"text": "Format the USB drive and reinstall the OS.",                            "expected": "destructive"},
    {"text": "Drop the 'sessions' table from the production database.",               "expected": "destructive"},
    {"text": "Destroy all cached artefacts in the CI pipeline.",                      "expected": "destructive"},
    {"text": "Erase the customer record with ID 8801.",                               "expected": "destructive"},
    {"text": "Purge the message queue before restarting the service.",                "expected": "destructive"},
    {"text": "Overwrite the existing configuration file with the new template.",      "expected": "destructive"},
    {"text": "Reset the device to factory settings.",                                 "expected": "destructive"},
    {"text": "Clear the application cache to free up disk space.",                    "expected": "destructive"},
    {"text": "Uninstall the legacy antivirus software.",                              "expected": "destructive"},
    {"text": "Cancel the active subscription and delete the associated data.",        "expected": "destructive"},
    {"text": "Terminate all idle database connections.",                              "expected": "destructive"},
    {"text": "Delete the GitHub branch after the PR is merged.",                      "expected": "destructive"},
    {"text": "Remove all temporary build artefacts from the workspace.",              "expected": "destructive"},
    {"text": "Wipe the development VM and rebuild from the base image.",              "expected": "destructive"},
    {"text": "Erase the user's browsing history and cookies.",                        "expected": "destructive"},
    {"text": "Drop all indexes on the analytics schema and recreate them.",           "expected": "destructive"},
    {"text": "Purge the CDN cache to force fresh content delivery.",                  "expected": "destructive"},
    {"text": "Format the hard drive partition before encryption.",                    "expected": "destructive"},
    {"text": "Reset all user passwords and invalidate active sessions.",              "expected": "destructive"},
    {"text": "Clear all notifications in the admin dashboard.",                       "expected": "destructive"},
    {"text": "Overwrite the backup file with today's snapshot.",                      "expected": "destructive"},
    {"text": "Destroy the Docker containers and volumes used for testing.",           "expected": "destructive"},
    {"text": "Delete the redundant microservice and its database schema.",            "expected": "destructive"},
    {"text": "Remove outdated dependencies from the requirements file.",              "expected": "destructive"},
    {"text": "Terminate the stale worker processes on the compute cluster.",          "expected": "destructive"},
    {"text": "Erase all draft messages in the inbox.",                                "expected": "destructive"},
    {"text": "Purge user activity logs from the analytics platform.",                 "expected": "destructive"},
]

# ── 15 boundary research→transactional (ground truth = transactional) ─────────
# These prompts have a research-style opening but contain a transactional
# keyword (book / order / pay / sign up / etc.) that should dominate.
BOUNDARY_R2T = [
    {"text": "Look up the cheapest flight and then book it for me.",                  "expected": "transactional"},
    {"text": "Find a good Italian restaurant and reserve a table for tonight.",       "expected": "transactional"},
    {"text": "Check available times and schedule a meeting with the client.",         "expected": "transactional"},
    {"text": "Search for the best insurance plan and sign up for the cheapest.",      "expected": "transactional"},
    {"text": "Show me the product page and add it to cart.",                          "expected": "transactional"},
    {"text": "Look up my energy usage and pay the electricity bill.",                 "expected": "transactional"},
    {"text": "Get information on the event and then register me as an attendee.",     "expected": "transactional"},
    {"text": "Find the invoice and submit the payment.",                              "expected": "transactional"},
    {"text": "Search for the subscription tiers and subscribe to the Pro plan.",      "expected": "transactional"},
    {"text": "Browse the hotel options and book the one with free breakfast.",        "expected": "transactional"},
    {"text": "Tell me what the library charges and then pay the overdue fine.",       "expected": "transactional"},
    {"text": "Check the ticket availability and order two seats for Saturday.",       "expected": "transactional"},
    {"text": "Describe the service and sign me up if there is a free trial.",         "expected": "transactional"},
    {"text": "List the available time slots and reserve the first open appointment.", "expected": "transactional"},
    {"text": "Get information about the shipping cost and purchase the item.",        "expected": "transactional"},
]

# ── 15 boundary transactional→destructive (ground truth = destructive) ────────
# These prompts might read as transactional actions but contain a destructive
# keyword (cancel / remove / reset / clear / terminate / uninstall / erase /
# purge / delete / wipe / drop / format / overwrite / destroy) which wins
# because destructive has highest priority in classify_task().
BOUNDARY_T2D = [
    {"text": "Cancel the order and remove the item from my purchase history.",        "expected": "destructive"},
    {"text": "Uninstall the app after transferring my data to the new device.",       "expected": "destructive"},
    {"text": "Reset my password and then log in to pay the bill.",                    "expected": "destructive"},
    {"text": "Clear the cart and start a fresh order.",                               "expected": "destructive"},
    {"text": "Terminate the active session before booking a new appointment.",        "expected": "destructive"},
    {"text": "Remove the old payment method and add a new credit card.",              "expected": "destructive"},
    {"text": "Delete the draft and resubmit the form with corrected details.",        "expected": "destructive"},
    {"text": "Cancel the subscription and sign up for a different plan.",             "expected": "destructive"},
    {"text": "Purge the failed transactions and retry the payment.",                  "expected": "destructive"},
    {"text": "Erase the saved address and enter a new shipping destination.",         "expected": "destructive"},
    {"text": "Wipe the test data and then schedule a new demo session.",              "expected": "destructive"},
    {"text": "Drop the staging table and recreate it before uploading the dataset.",  "expected": "destructive"},
    {"text": "Overwrite the old booking with the updated travel dates.",              "expected": "destructive"},
    {"text": "Destroy the temporary environment and provision a production one.",     "expected": "destructive"},
    {"text": "Format the report template and post the revised version online.",       "expected": "destructive"},
]

DATASET = RESEARCH + TRANSACTIONAL + DESTRUCTIVE + BOUNDARY_R2T + BOUNDARY_T2D

assert len(DATASET) == 120, f"Expected 120 items, got {len(DATASET)}"

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

CLASSES = ["research", "transactional", "destructive"]


def run_classifier(dataset):
    """Return list of (expected, predicted) pairs."""
    results = []
    for item in dataset:
        pred = classify_task(item["text"])
        results.append((item["expected"], pred))
    return results


def confusion_matrix(results):
    """Build a 3×3 confusion matrix as a dict-of-dicts: cm[actual][predicted]."""
    cm = {c: {p: 0 for p in CLASSES} for c in CLASSES}
    for actual, pred in results:
        cm[actual][pred] += 1
    return cm


def overall_accuracy(results):
    correct = sum(1 for a, p in results if a == p)
    return correct / len(results)


def per_class_metrics(cm):
    """Return dict of {class: {precision, recall, f1}} computed from cm."""
    metrics = {}
    for cls in CLASSES:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in CLASSES if other != cls)
        fn = sum(cm[cls][other] for other in CLASSES if other != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


# ---------------------------------------------------------------------------
# Power analysis — one-proportion z-test
# ---------------------------------------------------------------------------
# n = ((z_α/2 + z_β) / (p1 - p0))^2 * p0 * (1 - p0)
# We solve for the minimum detectable effect size (p1 - p0) given N = 120,
# α = 0.05, 1-β = 0.80, and baseline p0 = 0.75.
# Rearranging: (p1 - p0) = (z_α/2 + z_β) * sqrt(p0*(1-p0) / n)

def power_analysis(n=120, alpha=0.05, power=0.80, p0=0.75):
    z_alpha_half = 1.959964  # scipy.stats.norm.ppf(0.975)
    z_beta       = 0.841621  # scipy.stats.norm.ppf(0.80)
    mde = (z_alpha_half + z_beta) * math.sqrt(p0 * (1 - p0) / n)
    # Cross-check: minimum n to detect this MDE
    n_check = ((z_alpha_half + z_beta) / mde) ** 2 * p0 * (1 - p0)
    return {
        "n":            n,
        "alpha":        alpha,
        "power":        power,
        "p0_baseline":  p0,
        "z_alpha_half": z_alpha_half,
        "z_beta":       z_beta,
        "mde":          mde,
        "p1_min":       p0 + mde,
        "n_check":      n_check,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_confusion_matrix(cm):
    w = 14
    header = f"{'':>{w}}" + "".join(f"{p:>{w}}" for p in CLASSES) + f"{'(predicted)':>{w}}"
    print(header)
    print("-" * (w * (len(CLASSES) + 2)))
    for actual in CLASSES:
        row = f"{actual:>{w}}" + "".join(f"{cm[actual][p]:>{w}}" for p in CLASSES)
        print(row)
    print(f"\n  Rows = actual class, Columns = predicted class")


def main():
    print_section("GAP 1 EXTENDED EVALUATION — classify_task()")
    print(f"  Dataset: {len(DATASET)} labeled actions")
    print(f"  Composition: 30 research | 30 transactional | 30 destructive")
    print(f"               15 boundary R→T | 15 boundary T→D")

    results = run_classifier(DATASET)
    cm      = confusion_matrix(results)
    acc     = overall_accuracy(results)
    metrics = per_class_metrics(cm)

    # ── Section 1: Overall accuracy ───────────────────────────────────────────
    print_section("1. OVERALL ACCURACY")
    correct = sum(1 for a, p in results if a == p)
    print(f"  Correct   : {correct} / {len(results)}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")

    # ── Section 2: Per-class metrics ──────────────────────────────────────────
    print_section("2. PER-CLASS METRICS")
    col_w = 12
    hdr = (f"{'Class':>{col_w}}  {'Precision':>{col_w}}  "
           f"{'Recall':>{col_w}}  {'F1':>{col_w}}")
    print(hdr)
    print("-" * (col_w * 4 + 6))
    for cls in CLASSES:
        m = metrics[cls]
        print(f"  {cls:<{col_w-2}}  {m['precision']:>{col_w}.4f}  "
              f"{m['recall']:>{col_w}.4f}  {m['f1']:>{col_w}.4f}")

    # Macro averages
    macro_p  = sum(metrics[c]["precision"] for c in CLASSES) / len(CLASSES)
    macro_r  = sum(metrics[c]["recall"]    for c in CLASSES) / len(CLASSES)
    macro_f1 = sum(metrics[c]["f1"]        for c in CLASSES) / len(CLASSES)
    print("-" * (col_w * 4 + 6))
    print(f"  {'macro avg':<{col_w-2}}  {macro_p:>{col_w}.4f}  "
          f"{macro_r:>{col_w}.4f}  {macro_f1:>{col_w}.4f}")

    # ── Section 3: Confusion matrix ───────────────────────────────────────────
    print_section("3. CONFUSION MATRIX  (rows=actual, cols=predicted)")
    print_confusion_matrix(cm)

    # ── Section 4: Misclassifications ────────────────────────────────────────
    print_section("4. MISCLASSIFIED EXAMPLES")
    misses = [(item, pred)
              for item, (actual, pred) in zip(DATASET, results)
              if actual != pred]
    if not misses:
        print("  All 120 examples classified correctly.")
    else:
        print(f"  Total misclassified: {len(misses)}")
        for item, pred in misses:
            print(f"\n  Text     : {item['text'][:80]}")
            print(f"  Expected : {item['expected']}")
            print(f"  Got      : {pred}")

    # ── Section 5: Power analysis ─────────────────────────────────────────────
    print_section("5. POWER ANALYSIS  (one-proportion z-test)")
    pa = power_analysis(n=len(DATASET), alpha=0.05, power=0.80, p0=0.75)
    print(f"  Formula  : MDE = (z_α/2 + z_β) × √(p0·(1−p0)/n)")
    print(f"  N        : {pa['n']}")
    print(f"  α        : {pa['alpha']}")
    print(f"  Power    : {pa['power']} (1−β)")
    print(f"  p0       : {pa['p0_baseline']}  (baseline accuracy)")
    print(f"  z_α/2    : {pa['z_alpha_half']:.6f}")
    print(f"  z_β      : {pa['z_beta']:.6f}")
    print(f"  MDE      : {pa['mde']:.4f}  (minimum detectable effect size)")
    print(f"  p1 min   : {pa['p1_min']:.4f}  (smallest detectable accuracy)")
    print(f"  n check  : {pa['n_check']:.1f}  (should equal {pa['n']})")
    print(f"\n  Observed accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    if acc >= pa['p0_baseline'] + pa['mde']:
        print(f"  → Observed improvement ({acc - pa['p0_baseline']:.4f}) "
              f"exceeds MDE ({pa['mde']:.4f}).  SIGNIFICANT at α=0.05.")
    else:
        print(f"  → Observed improvement ({acc - pa['p0_baseline']:.4f}) "
              f"is below MDE ({pa['mde']:.4f}).  Not significant at α=0.05.")

    print_section("DONE")


if __name__ == "__main__":
    main()
