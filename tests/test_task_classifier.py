"""
Tests for halo/task_classifier.py — Gap 1 rule-based classification layer.

Covers:
  - classify_task()              : priority order, boundary words, no-match fallback
  - classify_task_with_keywords(): matched keyword evidence, research fallback list
  - get_risk_score()             : known and unknown task types
  - POLICY_MAP                   : every task type maps to a valid policy
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


# ── classify_task ─────────────────────────────────────────────────────────────

class TestClassifyTask:
    def test_research_default(self):
        assert classify_task("tell me about the weather in Berlin") == "research"

    def test_research_keyword_find(self):
        assert classify_task("find the best coffee shops nearby") == "research"

    def test_research_keyword_search(self):
        assert classify_task("search for open source LLM benchmarks") == "research"

    def test_transactional_buy(self):
        assert classify_task("buy a flight ticket to Munich") == "transactional"

    def test_transactional_register(self):
        assert classify_task("register for the conference") == "transactional"

    def test_transactional_submit(self):
        assert classify_task("submit the form on the contact page") == "transactional"

    def test_destructive_delete(self):
        assert classify_task("delete all files in the downloads folder") == "destructive"

    def test_destructive_remove(self):
        assert classify_task("remove the old account") == "destructive"

    def test_destructive_wipe(self):
        assert classify_task("wipe the database before the next experiment") == "destructive"

    def test_priority_destructive_over_transactional(self):
        # "delete" (destructive) and "buy" (transactional) in same prompt
        assert classify_task("delete the order after you buy it") == "destructive"

    def test_priority_destructive_over_research(self):
        assert classify_task("find and delete all temp files") == "destructive"

    def test_priority_transactional_over_research(self):
        assert classify_task("search and then book a hotel") == "transactional"

    def test_no_keywords_returns_research(self):
        assert classify_task("open the browser and navigate to the homepage") == "research"

    def test_word_boundary_format_not_information(self):
        # "information" contains the substring "format" but \bformat\b must NOT match it
        # (word boundary: the 'f' in "information" is preceded by 'n', not a non-word char)
        assert classify_task("please provide information about the system") == "research"

    def test_word_boundary_standalone_format_is_destructive(self):
        # standalone "format" (with spaces around it) IS destructive
        assert classify_task("format the USB drive") == "destructive"

    def test_word_boundary_exact_format(self):
        assert classify_task("format the USB drive") == "destructive"

    def test_word_boundary_drop_not_dropdown(self):
        # "drop" is destructive but must not match "dropdown"
        # "dropdown" contains "drop" — regex \b word boundary prevents it
        result = classify_task("click on the dropdown menu")
        # "dropdown" should NOT trigger destructive — word boundary check
        assert result == "research"

    def test_case_insensitive(self):
        assert classify_task("DELETE the entire table") == "destructive"
        assert classify_task("SEARCH for a good restaurant") == "research"
        assert classify_task("Book a table for two") == "transactional"

    def test_empty_prompt(self):
        assert classify_task("") == "research"


# ── classify_task_with_keywords ───────────────────────────────────────────────

class TestClassifyTaskWithKeywords:
    def test_returns_tuple(self):
        result = classify_task_with_keywords("find some data")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_destructive_returns_matched_keywords(self):
        task_type, kws = classify_task_with_keywords("delete and erase all logs")
        assert task_type == "destructive"
        assert "delete" in kws
        assert "erase" in kws

    def test_transactional_returns_matched_keywords(self):
        task_type, kws = classify_task_with_keywords("buy and order a new laptop")
        assert task_type == "transactional"
        assert "buy" in kws
        assert "order" in kws

    def test_research_returns_matched_keywords(self):
        task_type, kws = classify_task_with_keywords("find and search for recipes")
        assert task_type == "research"
        assert "find" in kws
        assert "search" in kws

    def test_no_match_returns_empty_keyword_list(self):
        task_type, kws = classify_task_with_keywords("navigate to the main page")
        assert task_type == "research"
        assert isinstance(kws, list)
        # No research keyword matched — empty list is OK

    def test_destructive_priority_over_transactional(self):
        task_type, kws = classify_task_with_keywords("cancel and delete the subscription")
        assert task_type == "destructive"
        assert "delete" in kws

    def test_keyword_list_type(self):
        _, kws = classify_task_with_keywords("remove the file")
        assert all(isinstance(k, str) for k in kws)

    def test_matches_all_keywords_in_category(self):
        task_type, kws = classify_task_with_keywords("delete, erase, and purge all data")
        assert task_type == "destructive"
        for word in ["delete", "erase", "purge"]:
            assert word in kws

    def test_word_boundary_respected(self):
        # "format" inside "information" must not match
        task_type, kws = classify_task_with_keywords("get information about formatting")
        # "format" should not be present — "formatting" has an -ting suffix but the
        # boundary is at the end of "format" in "formatting" — actually \b matches
        # between "format" and "ting" if "t" is a word char... let me think.
        # "formatting" → \bformat\b → "format" is followed by "t" (word char), so
        # \b does NOT match. Correct: "formatting" won't match \bformat\b.
        assert "format" not in kws


# ── get_risk_score ────────────────────────────────────────────────────────────

class TestGetRiskScore:
    def test_research_score(self):
        score = get_risk_score("research")
        assert 0.0 <= score < 0.31

    def test_transactional_score(self):
        score = get_risk_score("transactional")
        assert 0.31 <= score < 0.71

    def test_destructive_score(self):
        score = get_risk_score("destructive")
        assert 0.71 <= score <= 1.0

    def test_unknown_type_returns_research_default(self):
        score = get_risk_score("unknown_garbage")
        assert score == get_risk_score("research")

    def test_all_scores_are_floats_in_range(self):
        for tt in ["research", "transactional", "destructive"]:
            s = get_risk_score(tt)
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0


# ── POLICY_MAP ────────────────────────────────────────────────────────────────

class TestPolicyMap:
    VALID_POLICIES = {"always", "never", "auto-conservative", "auto-permissive"}

    def test_all_task_types_present(self):
        for tt in ["research", "transactional", "destructive"]:
            assert tt in POLICY_MAP

    def test_all_policies_are_valid(self):
        for tt, policy in POLICY_MAP.items():
            assert policy in self.VALID_POLICIES

    def test_research_maps_to_permissive(self):
        assert POLICY_MAP["research"] == "auto-permissive"

    def test_transactional_maps_to_conservative(self):
        assert POLICY_MAP["transactional"] == "auto-conservative"

    def test_destructive_maps_to_always(self):
        assert POLICY_MAP["destructive"] == "always"


# ── Keyword lists are populated ───────────────────────────────────────────────

class TestKeywordLists:
    def test_keyword_lists_nonempty(self):
        assert len(RESEARCH_KEYWORDS) > 0
        assert len(TRANSACTIONAL_KEYWORDS) > 0
        assert len(DESTRUCTIVE_KEYWORDS) > 0

    def test_no_overlap_between_lists(self):
        r = set(RESEARCH_KEYWORDS)
        t = set(TRANSACTIONAL_KEYWORDS)
        d = set(DESTRUCTIVE_KEYWORDS)
        assert len(r & t) == 0, "Research and transactional share keywords"
        assert len(r & d) == 0, "Research and destructive share keywords"
        assert len(t & d) == 0, "Transactional and destructive share keywords"
