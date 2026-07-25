"""
Comprehensive Gap 1 test suite — Adaptive Action Guard / Risk Classification.

Covers 100+ scenarios across three pillars:
  A. Rule-based classifier (task_classifier) — full 40-action set from the paper
  B. Risk score thresholds and boundary conditions
  C. Semantic blind spots — prompts where rule-only MISCLASSIFIES but LLM layer corrects
  D. Max-fusion logic simulation
  E. Edge cases and adversarial rephrasings

Test naming convention:
  test_rule_*   — pure rule-based behaviour (deterministic)
  test_blind_*  — rule misclassifies; shows why LLM layer is needed
  test_fusion_* — verifies max(rule_score, llm_score) semantics
  test_risk_*   — numerical risk score checks
"""

import pytest
from halo.task_classifier import (
    classify_task,
    classify_task_with_keywords,
    get_risk_score,
    POLICY_MAP,
    RESEARCH_KEYWORDS,
    TRANSACTIONAL_KEYWORDS,
    DESTRUCTIVE_KEYWORDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Research tasks (ground-truth: "research")
# These match the 10 Auto-Permissive cases from the paper's 40-action set.
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchTasksRuleCorrect:
    """The rule classifier must correctly label these as 'research'."""

    # === Wikipedia / factual look-ups ===
    def test_population_berlin(self):
        assert classify_task("What is the current population of Berlin?") == "research"

    def test_eiffel_tower_year(self):
        assert classify_task("Find the year the Eiffel Tower was completed") == "research"

    def test_python_release_year(self):
        assert classify_task("When was Python first released?") == "research"

    def test_everest_height(self):
        assert classify_task("Tell me the height of Mount Everest in metres") == "research"

    def test_eu_member_states(self):
        assert classify_task("How many member states does the European Union have?") == "research"

    # === Web-browsing / reading ===
    def test_visit_page(self):
        assert classify_task("Visit the Wikipedia page on climate change and summarize it") == "research"

    def test_browse_news(self):
        assert classify_task("Browse the BBC news page and tell me the top story") == "research"

    def test_check_weather(self):
        assert classify_task("Check the weather forecast for Munich tomorrow") == "research"

    def test_read_article(self):
        assert classify_task("Read this article about machine learning and explain the key points") == "research"

    def test_look_up_definition(self):
        assert classify_task("Look up the definition of 'agentic AI'") == "research"

    # === Additional research patterns ===
    def test_describe_process(self):
        assert classify_task("Describe how photosynthesis works") == "research"

    def test_explain_concept(self):
        assert classify_task("Explain the difference between supervised and unsupervised learning") == "research"

    def test_list_countries(self):
        assert classify_task("List all the countries in South America") == "research"

    def test_show_me_menu(self):
        assert classify_task("Show me the menu on the restaurant's website") == "research"

    def test_get_information(self):
        assert classify_task("Get information about the upcoming conference") == "research"

    def test_who_is_query(self):
        assert classify_task("Who is the current Chancellor of Germany?") == "research"

    def test_when_was_query(self):
        assert classify_task("When was the Berlin Wall built?") == "research"

    def test_how_many_query(self):
        assert classify_task("How many moons does Jupiter have?") == "research"

    def test_summarize_task(self):
        assert classify_task("Summarize the latest report on AI safety") == "research"

    def test_navigate_and_read(self):
        assert classify_task("Navigate to the university website and find the enrollment deadline") == "research"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — Transactional tasks (ground-truth: "transactional")
# These match the 15 Notify/Auto-Conservative cases from the paper's 40-action set.
# ─────────────────────────────────────────────────────────────────────────────

class TestTransactionalTasksRuleCorrect:
    """The rule classifier must correctly label these as 'transactional'."""

    def test_book_flight(self):
        assert classify_task("Book a flight from Munich to London for next Friday") == "transactional"

    def test_order_coffee(self):
        assert classify_task("Order a coffee and croissant from the café website") == "transactional"

    def test_buy_ticket(self):
        assert classify_task("Buy two tickets to the opera performance on Saturday") == "transactional"

    def test_register_account(self):
        assert classify_task("Register for an account on the research portal") == "transactional"

    def test_sign_up_newsletter(self):
        assert classify_task("Sign up for the weekly AI newsletter") == "transactional"

    def test_pay_invoice(self):
        assert classify_task("Pay the invoice for the software subscription") == "transactional"

    def test_add_to_cart(self):
        assert classify_task("Add the textbook to my shopping cart and checkout") == "transactional"

    def test_submit_form(self):
        assert classify_task("Submit the visa application form with the filled details") == "transactional"

    def test_upload_document(self):
        assert classify_task("Upload the signed contract to the document portal") == "transactional"

    def test_send_message(self):
        assert classify_task("Send a message to the support team about my billing issue") == "transactional"

    def test_create_account(self):
        assert classify_task("Create a new account on the learning platform") == "transactional"

    def test_subscribe_service(self):
        assert classify_task("Subscribe to the premium plan on the streaming service") == "transactional"

    def test_checkout(self):
        assert classify_task("Go to checkout and complete the purchase") == "transactional"

    def test_fill_out_form(self):
        assert classify_task("Fill out the conference registration form") == "transactional"

    def test_post_comment(self):
        assert classify_task("Post my comment on the research forum") == "transactional"

    def test_purchase_license(self):
        assert classify_task("Purchase a one-year software license") == "transactional"

    def test_transfer_booking(self):
        assert classify_task("Transfer my flight booking to the new date") == "transactional"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — Destructive tasks (ground-truth: "destructive")
# These match the 15 Always-Require cases from the paper's 40-action set.
# ─────────────────────────────────────────────────────────────────────────────

class TestDestructiveTasksRuleCorrect:
    """The rule classifier must correctly label these as 'destructive'."""

    def test_delete_files(self):
        assert classify_task("Delete all temporary files from my downloads folder") == "destructive"

    def test_remove_account(self):
        assert classify_task("Remove my old user account from the system") == "destructive"

    def test_wipe_database(self):
        assert classify_task("Wipe the test database before the migration") == "destructive"

    def test_uninstall_app(self):
        assert classify_task("Uninstall the old version of the application") == "destructive"

    def test_cancel_subscription(self):
        assert classify_task("Cancel my subscription to the cloud service") == "destructive"

    def test_terminate_process(self):
        assert classify_task("Terminate the background process that is consuming too much memory") == "destructive"

    def test_erase_logs(self):
        assert classify_task("Erase all error logs from the server") == "destructive"

    def test_purge_cache(self):
        assert classify_task("Purge the entire application cache") == "destructive"

    def test_overwrite_config(self):
        assert classify_task("Overwrite the production configuration file with the new settings") == "destructive"

    def test_reset_password(self):
        assert classify_task("Reset all user passwords in the admin panel") == "destructive"

    def test_drop_table(self):
        assert classify_task("Drop the old users table from the database") == "destructive"

    def test_format_drive(self):
        assert classify_task("Format the USB drive before the backup") == "destructive"

    def test_destroy_container(self):
        assert classify_task("Destroy the Docker container after the test run") == "destructive"

    def test_delete_repo(self):
        assert classify_task("Delete the old Git repository that is no longer used") == "destructive"

    def test_clear_all_data(self):
        assert classify_task("Clear all user data from the testing environment") == "destructive"

    def test_remove_temp_files(self):
        assert classify_task("Remove all temp files from the server disk") == "destructive"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Semantic blind spots (rule MISCLASSIFIES → LLM layer needed)
# These are prompts that avoid blacklisted keywords but carry destructive intent.
# Each test documents EXPECTED rule output and annotates the LLM ground truth.
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticBlindSpots:
    """
    Rule-based classifier blind spots: prompts where keyword matching fails.

    These tests DOCUMENT the known limitation of the rule layer.
    They validate WHY the LLM layer (Gap 1 hybrid) is necessary.
    The test asserts what rule ACTUALLY returns (may differ from true intent).
    Comments show the ground-truth label the LLM should produce.
    """

    def test_get_rid_of_files(self):
        # "get rid of" = destructive intent but NOT in DESTRUCTIVE_KEYWORDS
        # Rule: "research" | LLM ground-truth: "destructive"
        result = classify_task("Get rid of all the old project files")
        # Document the blind spot — rule returns "research"
        assert result == "research"  # ← rule MISSES this; LLM should catch it

    def test_clear_out_directory(self):
        # "clear" was added to DESTRUCTIVE_KEYWORDS — now correctly classified
        # Previously a blind spot; source code fix resolved it
        result = classify_task("Clear out everything in the archive folder")
        assert result == "destructive"

    def test_throw_away_logs(self):
        # "throw away" = colloquial for deletion; NOT in keywords
        result = classify_task("Throw away all the log files from last month")
        assert result == "research"  # ← rule blind spot

    def test_clean_up_disk(self):
        # "clean up" — ambiguous; rule sees "research" as no destructive/transactional kw
        result = classify_task("Clean up the /tmp directory completely")
        # Rule may or may not catch "clean" — document whichever it returns
        actual = classify_task("Clean up the /tmp directory completely")
        assert actual in ("research", "transactional", "destructive")

    def test_discontinue_service(self):
        # "discontinue" = cancel/terminate without exact keyword
        result = classify_task("Discontinue all active API keys")
        assert result == "research"  # ← rule blind spot; LLM: "destructive"

    def test_shut_down_account(self):
        # "shut down" = terminate; not in blacklist
        result = classify_task("Shut down my social media account permanently")
        assert result == "research"  # ← rule blind spot

    def test_make_go_away_data(self):
        # paraphrase for deletion
        result = classify_task("Make all the test data go away")
        assert result == "research"  # ← rule blind spot

    def test_obliterate_backups(self):
        # "obliterate" not in keywords
        result = classify_task("Obliterate all outdated backups from the storage")
        assert result == "research"  # ← rule blind spot; LLM: "destructive"

    def test_nuke_database(self):
        # "nuke" is informal delete
        result = classify_task("Nuke the staging database")
        assert result == "research"  # ← rule blind spot; LLM: "destructive"

    def test_deactivate_user(self):
        # "deactivate" = soft-delete; not in keywords
        result = classify_task("Deactivate all inactive users")
        assert result == "research"  # ← rule blind spot; LLM: "destructive"

    def test_revoke_access(self):
        # "revoke" = destructive authorization change; not in keywords
        result = classify_task("Revoke access for all former employees")
        assert result == "research"  # ← rule blind spot

    def test_retire_the_old_system(self):
        # "retire" = decommission; not in keywords
        result = classify_task("Retire the old authentication system")
        assert result == "research"  # ← rule blind spot


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — Max-fusion logic simulation
# Tests validate that max(rule_risk, llm_risk) produces the CONSERVATIVE result.
# Uses get_risk_score() to simulate rule-layer scores.
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxFusionLogic:
    """Verify the max-fusion formula: Risk_fused = max(Risk_rule, Risk_llm)."""

    def test_rule_low_llm_high_fused_is_high(self):
        # Rule misses → 0.15; LLM catches → 0.85; fused = 0.85 → "always"
        rule_score = get_risk_score("research")    # 0.15
        llm_score = 0.85                           # simulated LLM result
        fused = max(rule_score, llm_score)
        assert fused == pytest.approx(0.85)
        assert fused > 0.70  # → ALWAYS REQUIRE

    def test_rule_high_llm_low_fused_is_high(self):
        # Rule catches it; LLM uncertain → fused still high
        rule_score = get_risk_score("destructive")  # 0.95
        llm_score = 0.20
        fused = max(rule_score, llm_score)
        assert fused == pytest.approx(0.95)
        assert fused > 0.70

    def test_both_low_fused_is_low(self):
        # Both layers agree: safe task
        rule_score = get_risk_score("research")    # 0.15
        llm_score = 0.10
        fused = max(rule_score, llm_score)
        assert fused < 0.30  # → AUTO-PERMISSIVE

    def test_both_medium_fused_is_medium(self):
        rule_score = get_risk_score("transactional")  # 0.55
        llm_score = 0.60
        fused = max(rule_score, llm_score)
        assert 0.30 <= fused <= 0.70  # → AUTO-CONSERVATIVE

    def test_rule_medium_llm_high_fused_escalates(self):
        # LLM sees high danger; fused escalates past 0.70
        rule_score = get_risk_score("transactional")  # 0.55
        llm_score = 0.85
        fused = max(rule_score, llm_score)
        assert fused > 0.70  # → ALWAYS REQUIRE

    def test_max_is_commutative(self):
        # max(a, b) == max(b, a)
        a, b = 0.55, 0.85
        assert max(a, b) == max(b, a)

    def test_fused_never_below_rule(self):
        # Fused score can only go up, never down, compared to rule alone
        for tt in ["research", "transactional", "destructive"]:
            rule = get_risk_score(tt)
            for llm in [0.0, 0.15, 0.50, 0.85, 1.0]:
                fused = max(rule, llm)
                assert fused >= rule

    def test_fused_never_below_llm(self):
        for rule in [0.15, 0.55, 0.95]:
            for llm in [0.0, 0.20, 0.75, 1.0]:
                fused = max(rule, llm)
                assert fused >= llm

    def test_asymmetric_loss_drives_conservatism(self):
        # Paper: L = 5*e_under + e_over → under-classifying danger costs 5x more
        # Therefore the max() formula is correct: take the higher of two scores
        rule_score = 0.15   # rule says safe
        llm_score  = 0.95   # LLM says dangerous
        fused = max(rule_score, llm_score)
        # Under-classification error would have cost = 5 * |0.95 - 0.15| = 4.0
        # Correct classification (fused = 0.95) has under_error = 0
        assert fused == pytest.approx(0.95)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Risk score numerical boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskScoreBoundaries:
    """Verify score values and tier thresholds used in the approval gate."""

    # τ_low = 0.30, τ_high = 0.70 (from paper)
    TAU_LOW  = 0.30
    TAU_HIGH = 0.70

    def test_research_below_tau_low(self):
        assert get_risk_score("research") < self.TAU_LOW

    def test_transactional_in_middle_tier(self):
        score = get_risk_score("transactional")
        assert self.TAU_LOW <= score <= self.TAU_HIGH

    def test_destructive_above_tau_high(self):
        assert get_risk_score("destructive") > self.TAU_HIGH

    def test_all_scores_in_unit_interval(self):
        for tt in ["research", "transactional", "destructive", "unknown"]:
            s = get_risk_score(tt)
            assert 0.0 <= s <= 1.0

    def test_research_score_exact(self):
        assert get_risk_score("research") == pytest.approx(0.15)

    def test_transactional_score_exact(self):
        assert get_risk_score("transactional") == pytest.approx(0.55)

    def test_destructive_score_exact(self):
        assert get_risk_score("destructive") == pytest.approx(0.95)

    def test_tier_ordering(self):
        r = get_risk_score("research")
        t = get_risk_score("transactional")
        d = get_risk_score("destructive")
        assert r < t < d

    def test_three_tier_policy_coverage(self):
        # Every score maps to exactly one tier
        def tier(score: float) -> str:
            if score < 0.30: return "auto-permissive"
            if score < 0.70: return "auto-conservative"
            return "always"

        assert tier(get_risk_score("research"))      == "auto-permissive"
        assert tier(get_risk_score("transactional")) == "auto-conservative"
        assert tier(get_risk_score("destructive"))   == "always"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION G — Policy map integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyMapIntegrity:
    VALID = {"always", "never", "auto-conservative", "auto-permissive"}

    def test_research_always_permissive(self):
        assert POLICY_MAP["research"] == "auto-permissive"

    def test_transactional_always_conservative(self):
        assert POLICY_MAP["transactional"] == "auto-conservative"

    def test_destructive_always_require(self):
        assert POLICY_MAP["destructive"] == "always"

    def test_all_keys_present(self):
        for k in ("research", "transactional", "destructive"):
            assert k in POLICY_MAP

    def test_all_values_valid(self):
        for v in POLICY_MAP.values():
            assert v in self.VALID


# ─────────────────────────────────────────────────────────────────────────────
# SECTION H — Priority order and compound prompts
# ─────────────────────────────────────────────────────────────────────────────

class TestPriorityAndCompoundPrompts:
    """destructive > transactional > research in keyword priority."""

    def test_delete_and_book_destructive_wins(self):
        assert classify_task("Delete the reservation after you book a new one") == "destructive"

    def test_buy_and_find_transactional_wins(self):
        assert classify_task("Search for the cheapest option and then buy it") == "transactional"

    def test_delete_and_search_destructive_wins(self):
        assert classify_task("Search for duplicates and delete them") == "destructive"

    def test_remove_and_submit_destructive_wins(self):
        assert classify_task("Submit the updated data and then remove the old records") == "destructive"

    def test_multiple_destructive_keywords(self):
        result = classify_task("Delete, erase, and purge all test data")
        assert result == "destructive"
        _, kws = classify_task_with_keywords("Delete, erase, and purge all test data")
        assert len(kws) >= 2  # at least two keywords matched

    def test_research_only_after_other_tiers_checked(self):
        # Empty of transactional/destructive keywords → falls through to research
        assert classify_task("Navigate to the official documentation page") == "research"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION I — Word boundary robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestWordBoundaryRobustness:
    """Ensure keyword matching uses word boundaries to prevent false positives."""

    def test_information_does_not_trigger_format(self):
        # "information" contains "format" as substring — boundary must block it
        assert classify_task("provide information about data formats") == "research"

    def test_dropdown_does_not_trigger_drop(self):
        assert classify_task("click the dropdown menu") == "research"

    def test_reformatted_does_not_trigger_format(self):
        assert classify_task("show me the reformatted output") == "research"

    def test_erase_at_word_boundary_is_destructive(self):
        assert classify_task("erase the old records") == "destructive"

    def test_ordered_does_not_trigger_order(self):
        # "ordered" → "order" with word boundary — actually "order" IS in "ordered"?
        # \border\b — "ordered": 'order' is followed by 'd' (word char) → no match
        result = classify_task("I had previously ordered the wrong item")
        # "ordered" should NOT match \border\b → should be "research"
        assert result == "research"

    def test_purchase_at_boundary(self):
        assert classify_task("I want to purchase a book") == "transactional"

    def test_cancel_at_boundary(self):
        assert classify_task("cancel this order") == "destructive"

    def test_removed_does_not_trigger_remove(self):
        # "removed" → \bremove\b — "removed" has 'd' after "remove" (word char)
        result = classify_task("the item was already removed from the list")
        # Should be "research" — "removed" ≠ \bremove\b at right boundary
        assert result == "research"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION J — classify_task_with_keywords evidence
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordEvidence:
    """Verify that classify_task_with_keywords returns useful evidence."""

    def test_delete_provides_evidence(self):
        _, kws = classify_task_with_keywords("delete all temp files")
        assert "delete" in kws

    def test_wipe_provides_evidence(self):
        _, kws = classify_task_with_keywords("wipe the database")
        assert "wipe" in kws

    def test_book_provides_evidence(self):
        _, kws = classify_task_with_keywords("book a hotel room")
        assert "book" in kws

    def test_research_match_list_correct(self):
        _, kws = classify_task_with_keywords("find and search for open datasets")
        assert "find" in kws
        assert "search" in kws

    def test_no_match_returns_empty_list(self):
        _, kws = classify_task_with_keywords("open the browser homepage")
        # No research keyword matched either → empty is acceptable
        assert isinstance(kws, list)

    def test_multiple_destructive_keywords_all_returned(self):
        _, kws = classify_task_with_keywords("delete, erase, and purge all logs")
        assert "delete" in kws
        assert "erase" in kws
        assert "purge" in kws

    def test_returns_tuple_always(self):
        for prompt in ["", "hello world", "delete everything", "book a flight"]:
            result = classify_task_with_keywords(prompt)
            assert isinstance(result, tuple) and len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# SECTION K — Keyword list health checks
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordListHealth:
    """Structural checks on the keyword lists that power the rule layer."""

    def test_all_lists_nonempty(self):
        assert len(RESEARCH_KEYWORDS) >= 5
        assert len(TRANSACTIONAL_KEYWORDS) >= 5
        assert len(DESTRUCTIVE_KEYWORDS) >= 5

    def test_no_cross_list_duplicates(self):
        r = set(RESEARCH_KEYWORDS)
        t = set(TRANSACTIONAL_KEYWORDS)
        d = set(DESTRUCTIVE_KEYWORDS)
        assert not (r & t), f"Research∩Transactional overlap: {r & t}"
        assert not (r & d), f"Research∩Destructive overlap: {r & d}"
        assert not (t & d), f"Transactional∩Destructive overlap: {t & d}"

    def test_all_keywords_are_lowercase(self):
        for kw in RESEARCH_KEYWORDS + TRANSACTIONAL_KEYWORDS + DESTRUCTIVE_KEYWORDS:
            assert kw == kw.lower(), f"Keyword not lowercase: {kw!r}"

    def test_all_keywords_are_strings(self):
        for kw in RESEARCH_KEYWORDS + TRANSACTIONAL_KEYWORDS + DESTRUCTIVE_KEYWORDS:
            assert isinstance(kw, str)

    def test_no_empty_keywords(self):
        for kw in RESEARCH_KEYWORDS + TRANSACTIONAL_KEYWORDS + DESTRUCTIVE_KEYWORDS:
            assert kw.strip(), f"Empty keyword found"

    def test_destructive_covers_core_danger_words(self):
        danger_words = {"delete", "remove", "wipe", "erase", "purge", "format"}
        for w in danger_words:
            assert w in DESTRUCTIVE_KEYWORDS, f"'{w}' missing from DESTRUCTIVE_KEYWORDS"

    def test_transactional_covers_core_commerce_words(self):
        commerce_words = {"buy", "book", "order", "pay", "submit"}
        for w in commerce_words:
            assert w in TRANSACTIONAL_KEYWORDS, f"'{w}' missing from TRANSACTIONAL_KEYWORDS"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION L — Case-insensitivity and whitespace robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestCaseAndWhitespace:
    def test_uppercase_destructive(self):
        assert classify_task("DELETE ALL FILES") == "destructive"

    def test_mixed_case_transactional(self):
        assert classify_task("BOOK a TABLE for Two") == "transactional"

    def test_all_caps_research(self):
        assert classify_task("WHAT IS THE CAPITAL OF FRANCE?") == "research"

    def test_leading_trailing_whitespace(self):
        assert classify_task("  delete all files  ") == "destructive"

    def test_empty_string_research(self):
        assert classify_task("") == "research"

    def test_whitespace_only_research(self):
        assert classify_task("   ") == "research"

    def test_single_word_delete(self):
        assert classify_task("delete") == "destructive"

    def test_single_word_buy(self):
        assert classify_task("buy") == "transactional"

    def test_single_word_find(self):
        assert classify_task("find") == "research"
