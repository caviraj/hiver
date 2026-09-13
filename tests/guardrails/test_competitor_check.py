"""Unit tests for competitor mention detection and context differentiation."""

import pytest
from src.guardrails.competitor_check import check_competitor_mention, load_competitors


def test_load_competitors():
    """Verify competitor names are loaded from config/guardrails/restricted_topics.yaml."""
    competitors = load_competitors()
    assert len(competitors) > 0
    # Ensure known entries from restricted_topics.yaml are present
    competitors_lower = [c.lower() for c in competitors]
    assert "zendesk" in competitors_lower
    assert "freshdesk" in competitors_lower
    assert "intercom" in competitors_lower
    assert "salesforce" in competitors_lower


def test_competitor_recommending_context_blocked():
    """Competitor mention in recommending context must return the matched competitor name."""
    text1 = "You should switch to Zendesk for better ticketing."
    assert check_competitor_mention(text1) == "Zendesk"

    text2 = "I recommend using Freshdesk instead of our platform."
    assert check_competitor_mention(text2) == "Freshdesk"

    text3 = "You might want to try Intercom for live chat solutions."
    assert check_competitor_mention(text3) == "Intercom"

    text4 = "Consider Front as a superior alternative."
    assert check_competitor_mention(text4) == "Front"


def test_competitor_disparaging_context_blocked():
    """Competitor mention in disparaging context must return the matched competitor name."""
    text1 = "Zendesk is terrible, slow, and completely broken."
    assert check_competitor_mention(text1) == "Zendesk"

    text2 = "Competitor Salesforce is overpriced and useless for your needs."
    assert check_competitor_mention(text2) == "Salesforce"


def test_competitor_neutral_factual_context_allowed():
    """Neutral/factual compatibility or integration mentions should NOT be blocked (returns None)."""
    text1 = "Our system integrates seamlessly with Zendesk."
    assert check_competitor_mention(text1) is None

    text2 = "This cable is also compatible with Samsung phones."
    assert check_competitor_mention(text2, competitor_names=["Samsung"]) is None

    text3 = "We support migration and data export to Salesforce formats."
    assert check_competitor_mention(text3) is None

    text4 = "Our API supports webhooks similar to Intercom."
    assert check_competitor_mention(text4) is None


def test_competitor_case_insensitivity_and_possessives():
    """Test case insensitivity and possessive forms like Zendesk's."""
    text1 = "we recommend zendesk's enterprise plan."
    assert check_competitor_mention(text1) == "Zendesk"

    text2 = "Try FRESHDESK today."
    assert check_competitor_mention(text2) == "Freshdesk"


def test_no_competitor_mention():
    """Text without any competitor mention should return None."""
    text = "Here is your tracking link for ticket TCKT-123456. Have a great day!"
    assert check_competitor_mention(text) is None


def test_empty_or_none_input():
    """Empty or None text should safely return None."""
    assert check_competitor_mention("") is None
    assert check_competitor_mention(None) is None
