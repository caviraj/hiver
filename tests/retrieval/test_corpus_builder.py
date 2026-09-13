"""Tests for corpus builder (M3.P3.1.F1)."""

from datetime import datetime
import logging
import pytest

from src.data.schema import RawTweet
from src.data.thread_schema import Thread
from src.retrieval.corpus_builder import build_corpus


def _make_tweet(
    tweet_id: str,
    author_id: str,
    inbound: bool,
    text: str,
    in_response_to: str = None,
) -> RawTweet:
    return RawTweet(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=inbound,
        created_at=datetime(2023, 1, 1, 12, 0, 0),
        text=text,
        in_response_to_tweet_id=in_response_to,
    )


def test_build_corpus_valid_threads():
    """Test extracting RetrievalDocument from customer thread with brand reply."""
    t1 = _make_tweet("101", "cust1", True, "My battery on SM-T280 dies quickly.")
    t2 = _make_tweet("102", "brand_support", False, "Please try resetting your device settings.", "101")
    thread1 = Thread(thread_id="101", tweets=[t1, t2], terminal=True)

    t3 = _make_tweet("201", "cust2", True, "App crashing on iOS 11.1")
    t4 = _make_tweet("202", "brand_support", False, "Please reinstall the application.", "201")
    thread2 = Thread(thread_id="201", tweets=[t3, t4], terminal=True)

    docs = build_corpus([thread1, thread2])
    assert len(docs) == 2

    assert docs[0].doc_id == "101"
    assert docs[0].query_text == "My battery on SM-T280 dies quickly."
    assert docs[0].resolution_text == "Please try resetting your device settings."

    assert docs[1].doc_id == "201"
    assert docs[1].query_text == "App crashing on iOS 11.1"
    assert docs[1].resolution_text == "Please reinstall the application."


def test_build_corpus_excludes_unresolved_threads():
    """Test that threads lacking any brand reply are excluded."""
    # Thread with customer only
    t1 = _make_tweet("301", "cust3", True, "Hello? Is anyone there?")
    t2 = _make_tweet("302", "cust3", True, "Still waiting for help.", "301")
    thread = Thread(thread_id="301", tweets=[t1, t2], terminal=False)

    docs = build_corpus([thread])
    assert len(docs) == 0


def test_build_corpus_excludes_non_customer_initiated():
    """Test that threads initiated by brand (inbound=False) are excluded."""
    t1 = _make_tweet("401", "brand_support", False, "We are having scheduled maintenance.")
    t2 = _make_tweet("402", "cust4", True, "Thanks for the heads up!", "401")
    thread = Thread(thread_id="401", tweets=[t1, t2], terminal=False)

    docs = build_corpus([thread])
    assert len(docs) == 0


def test_build_corpus_retains_canned_duplicate_resolutions(caplog):
    """Test that duplicate resolution texts are NOT dropped, and are logged."""
    canned_reply = "We apologize for the inconvenience. Please DM us your account number."

    t1 = _make_tweet("501", "cust1", True, "WiFi not working.")
    t2 = _make_tweet("502", "brand_support", False, canned_reply, "501")
    thread1 = Thread(thread_id="501", tweets=[t1, t2], terminal=True)

    t3 = _make_tweet("503", "cust2", True, "Cable box not turning on.")
    t4 = _make_tweet("504", "brand_support", False, canned_reply, "503")
    thread2 = Thread(thread_id="503", tweets=[t3, t4], terminal=True)

    with caplog.at_level(logging.INFO):
        docs = build_corpus([thread1, thread2])

    assert len(docs) == 2
    assert docs[0].resolution_text == canned_reply
    assert docs[1].resolution_text == canned_reply
    assert docs[0].doc_id == "501"
    assert docs[1].doc_id == "503"

    # Verify that duplicate resolution text was logged
    assert any("canned responses" in record.message for record in caplog.records)


def test_build_corpus_empty():
    """Test building corpus on empty input list."""
    assert build_corpus([]) == []
