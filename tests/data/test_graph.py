"""Unit tests for conversation graph reconstruction and thread assembly.

Covers:
- parse_response_ids edge cases (comma-separated, single ID, whitespace, empty, NaN, None, deduplication).
- build_graph bidirectional agreement checking and disagreement logging/healing.
- Orphan dropping (replies pointing to non-existent parent IDs).
- Traversal cycle detection and breaking (e.g. A -> B -> A).
- Branching thread decomposition into prefix-sharing distinct Thread objects.
- Filtering of pure consumer chatter threads (0 brand turns).
- Terminal boolean resolution (brand-terminated vs. dangling consumer turn).
- Chronological sorting with timestamp collision tweet_id tiebreaking.
- Turn count monitoring and >20 turns threshold warning.
- Thread serialization and deserialization round-trip (JSON Lines).
- Graph-expanded brand filtering in ingest.py.
"""

from datetime import datetime, timezone, timedelta
import logging
from pathlib import Path
import pandas as pd
import pytest

from src.data.graph import (
    GraphNode,
    build_graph,
    load_threads_from_jsonl,
    parse_response_ids,
    reconstruct_threads,
    save_threads_to_jsonl,
)
from src.data.ingest import filter_brand
from src.data.schema import RawTweet
from src.data.thread_schema import Thread


# ---------------------------------------------------------------------------
# 1. parse_response_ids Tests
# ---------------------------------------------------------------------------

def test_parse_response_ids_comma_separated():
    """Test parsing multi-child comma-separated string."""
    raw = "101, 102, 103"
    result = parse_response_ids(raw)
    assert result == ["101", "102", "103"]


def test_parse_response_ids_single_id():
    """Test parsing single tweet ID."""
    assert parse_response_ids("555") == ["555"]
    assert parse_response_ids(" 555 ") == ["555"]


def test_parse_response_ids_whitespace_and_deduplication():
    """Test whitespace trimming and preservation of first-seen order without duplicates."""
    raw = " 101 , 102,  101 , 103 , 102 "
    result = parse_response_ids(raw)
    assert result == ["101", "102", "103"]


@pytest.mark.parametrize("empty_input", [None, "", "   ", float("nan"), "nan", "NaN"])
def test_parse_response_ids_empty_and_null_inputs(empty_input):
    """Test that None, NaN, and empty strings safely return empty lists."""
    assert parse_response_ids(empty_input) == []


# ---------------------------------------------------------------------------
# 2. Thread Schema & Validation Tests
# ---------------------------------------------------------------------------

def test_thread_schema_properties_and_validation():
    """Test Thread model properties, turn count, and root validation."""
    t1 = RawTweet(
        tweet_id="1",
        author_id="cust1",
        inbound=True,
        created_at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        text="Help me please",
        response_tweet_id="2",
        in_response_to_tweet_id=None,
    )
    t2 = RawTweet(
        tweet_id="2",
        author_id="AppleSupport",
        inbound=False,
        created_at=datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc),
        text="We are here to help!",
        response_tweet_id=None,
        in_response_to_tweet_id="1",
    )

    thread = Thread(thread_id="1", tweets=[t1, t2], terminal=True)
    assert thread.turn_count == 2
    assert thread.root_tweet.tweet_id == "1"
    assert thread.last_tweet.tweet_id == "2"
    assert thread.participant_ids == ["cust1", "AppleSupport"]
    assert thread.has_brand_tweet is True

    # Validation failure: thread_id does not match root tweet
    with pytest.raises(ValueError, match="does not match root tweet_id"):
        Thread(thread_id="999", tweets=[t1, t2], terminal=True)


# ---------------------------------------------------------------------------
# 3. build_graph & Bidirectional Disagreement Logging Tests
# ---------------------------------------------------------------------------

def test_build_graph_bidirectional_agreement(caplog):
    """Test normal consistent parent/child links without warnings."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Issue with phone",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Send DM",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1",
        },
    ])

    with caplog.at_level(logging.WARNING):
        graph = build_graph(df)

    assert "1" in graph and "2" in graph
    assert graph["1"].children_ids == ["2"]
    assert graph["2"].parent_id == "1"
    # No disagreement warning should be logged
    disagreement_logs = [r for r in caplog.records if "Bidirectional disagreement" in r.message]
    assert len(disagreement_logs) == 0


def test_build_graph_bidirectional_disagreement_logging_and_reconciliation(caplog):
    """Test that parent/child disagreements are logged and missing links are reconciled."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    # Tweet 2 claims parent is 1, but Tweet 1 did not list Tweet 2 in response_tweet_id
    # Tweet 3 claims parent is 1, but Tweet 1 lists child 4 which doesn't list 1 as parent
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Question",
            "response_tweet_id": "4",  # declares 4 as child
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Answer from Apple",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1",  # declares 1 as parent, but 1 didn't list 2
        },
        {
            "tweet_id": "4",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time + timedelta(minutes=6),
            "text": "Unrelated tweet",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "999",  # parent is not 1!
        },
    ])

    with caplog.at_level(logging.WARNING):
        graph = build_graph(df)

    # Reconciled: Tweet 2 was added to Tweet 1's children_ids
    assert "2" in graph["1"].children_ids

    # Logged warnings for both disagreements
    disagreement_logs = [r for r in caplog.records if "Bidirectional disagreement" in r.message]
    assert len(disagreement_logs) >= 2


# ---------------------------------------------------------------------------
# 4. Orphan Node Dropping Tests
# ---------------------------------------------------------------------------

def test_reconstruct_threads_drops_orphans(caplog):
    """Test that tweets pointing to non-existent parents are dropped and logged."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        # Valid thread: 1 -> 2
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Help",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "On it",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1",
        },
        # Orphan node: in_response_to_tweet_id points to missing 99999
        {
            "tweet_id": "3",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=10),
            "text": "Orphan reply",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "99999",
        },
    ])

    with caplog.at_level(logging.WARNING):
        graph = build_graph(df)
        threads = reconstruct_threads(graph)

    # Only 1 valid thread should be returned
    assert len(threads) == 1
    assert threads[0].thread_id == "1"
    assert [t.tweet_id for t in threads[0].tweets] == ["1", "2"]

    # Warning logged for dropped orphan
    orphan_warning = [r for r in caplog.records if "Dropped" in r.message and "orphaned" in r.message]
    assert len(orphan_warning) == 1
    assert "Dropped 1 orphaned nodes" in orphan_warning[0].message


# ---------------------------------------------------------------------------
# 5. Cycle Detection and Breaking Tests
# ---------------------------------------------------------------------------

def test_reconstruct_threads_detects_and_breaks_cycles(caplog):
    """Test that circular responses (A -> B -> A) are detected, broken, and logged."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    # Root 1 -> Child 2 -> Child 1 (loop!)
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Start",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Loop reply",
            "response_tweet_id": "1",  # pointing back to 1
            "in_response_to_tweet_id": "1",
        },
    ])

    with caplog.at_level(logging.WARNING):
        graph = build_graph(df)
        threads = reconstruct_threads(graph)

    # Thread reconstructed without infinite loop
    assert len(threads) == 1
    assert threads[0].thread_id == "1"
    assert [t.tweet_id for t in threads[0].tweets] == ["1", "2"]

    # Cycle warning logged
    cycle_logs = [r for r in caplog.records if "Cycle detected" in r.message]
    assert len(cycle_logs) >= 1


# ---------------------------------------------------------------------------
# 6. Branching Thread Splitting Tests
# ---------------------------------------------------------------------------

def test_reconstruct_threads_branching_splits_into_separate_threads():
    """Test that a tweet with multiple children produces multiple distinct Threads sharing prefix."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    # Tree structure:
    #         1 (Customer)
    #           |
    #         2 (AppleSupport)
    #        / \
    #       3   4 (Both customers replying to brand tweet 2)
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "My phone is broken",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "What model do you have?",
            "response_tweet_id": "3, 4",
            "in_response_to_tweet_id": "1",
        },
        {
            "tweet_id": "3",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time + timedelta(minutes=10),
            "text": "iPhone 13 Pro",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "2",
        },
        {
            "tweet_id": "4",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time + timedelta(minutes=15),
            "text": "I have the same problem with iPhone 14",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "2",
        },
    ])

    graph = build_graph(df)
    threads = reconstruct_threads(graph)

    # Should produce exactly 2 distinct threads
    assert len(threads) == 2

    # Both threads share root_id "1" and prefix [1, 2]
    thread_paths = [[t.tweet_id for t in th.tweets] for th in threads]
    assert ["1", "2", "3"] in thread_paths
    assert ["1", "2", "4"] in thread_paths

    # Both threads end with consumer tweet -> terminal = False
    for th in threads:
        assert th.terminal is False


# ---------------------------------------------------------------------------
# 7. Zero-Brand Thread Dropping & Terminal Flag Tests
# ---------------------------------------------------------------------------

def test_reconstruct_threads_drops_zero_brand_threads():
    """Test that pure consumer chatter threads (0 brand turns) are discarded."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        # Thread A: pure consumer chatter (cust1 -> cust2)
        {
            "tweet_id": "10",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Hey friend",
            "response_tweet_id": "11",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "11",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Hey back",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "10",
        },
        # Thread B: valid brand interaction (cust3 -> AppleSupport)
        {
            "tweet_id": "20",
            "author_id": "cust3",
            "inbound": True,
            "created_at": base_time,
            "text": "Need help with iPad",
            "response_tweet_id": "21",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "21",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Restart your iPad",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "20",
        },
    ])

    graph = build_graph(df)
    threads = reconstruct_threads(graph)

    # Pure consumer thread 10 -> 11 is dropped, only 20 -> 21 kept
    assert len(threads) == 1
    assert threads[0].thread_id == "20"
    assert [t.tweet_id for t in threads[0].tweets] == ["20", "21"]
    assert threads[0].terminal is True  # ends with AppleSupport


def test_terminal_flag_true_and_false():
    """Test terminal flag accuracy for brand termination vs dangling consumer."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    # Thread 1 ends with brand response -> terminal = True
    # Thread 2 ends with consumer reply -> terminal = False
    df = pd.DataFrame([
        # Thread 1: 1 (cust) -> 2 (brand)
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Bug report",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Fixed!",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1",
        },
        # Thread 3: 3 (cust) -> 4 (brand) -> 5 (cust)
        {
            "tweet_id": "3",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time,
            "text": "Question",
            "response_tweet_id": "4",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "4",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Try resetting",
            "response_tweet_id": "5",
            "in_response_to_tweet_id": "3",
        },
        {
            "tweet_id": "5",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time + timedelta(minutes=10),
            "text": "Still not working...",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "4",
        },
    ])

    graph = build_graph(df)
    threads = reconstruct_threads(graph)
    assert len(threads) == 2

    t1 = next(th for th in threads if th.thread_id == "1")
    t3 = next(th for th in threads if th.thread_id == "3")

    assert t1.terminal is True
    assert t3.terminal is False


# ---------------------------------------------------------------------------
# 8. Chronological Sorting & Tiebreaker Tests
# ---------------------------------------------------------------------------

def test_chronological_sorting_with_tweet_id_tiebreaker():
    """Test sorting by created_at and using tweet_id tiebreaker for identical timestamps."""
    same_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    later_time = datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc)

    # Node 1 -> Node 3 and Node 2 share exact same timestamp, but 2 arrives out of order
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": same_time,
            "text": "Start",
            "response_tweet_id": "3",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "3",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": same_time,  # Identical created_at to 1
            "text": "Brand response fast",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": "1",
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": later_time,
            "text": "Followup",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "3",
        },
    ])

    graph = build_graph(df)
    threads = reconstruct_threads(graph)

    assert len(threads) == 1
    # Both 1 and 3 have same_time, but '1' < '3', so order should be 1, then 3, then 2
    tweet_ids = [t.tweet_id for t in threads[0].tweets]
    assert tweet_ids == ["1", "3", "2"]


# ---------------------------------------------------------------------------
# 9. Over 20 Turns Warning Test
# ---------------------------------------------------------------------------

def test_over_20_turns_warning_logged(caplog):
    """Test that threads exceeding 20 turns log a warning."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    rows = []
    total_turns = 22

    for i in range(1, total_turns + 1):
        tid = str(i)
        parent = str(i - 1) if i > 1 else None
        child = str(i + 1) if i < total_turns else None
        # Alternate author: cust vs AppleSupport
        is_brand = (i % 2 == 0)
        rows.append({
            "tweet_id": tid,
            "author_id": "AppleSupport" if is_brand else f"cust{i}",
            "inbound": not is_brand,
            "created_at": base_time + timedelta(minutes=i),
            "text": f"Turn {i}",
            "response_tweet_id": child,
            "in_response_to_tweet_id": parent,
        })

    df = pd.DataFrame(rows)
    with caplog.at_level(logging.WARNING):
        graph = build_graph(df)
        threads = reconstruct_threads(graph)

    assert len(threads) == 1
    assert threads[0].turn_count == 22

    high_turn_warning = [r for r in caplog.records if "exceeding 20-turn threshold" in r.message]
    assert len(high_turn_warning) == 1
    assert "22 turns" in high_turn_warning[0].message


# ---------------------------------------------------------------------------
# 10. JSONL Persistence Round-Trip Tests
# ---------------------------------------------------------------------------

def test_save_and_load_threads_jsonl(tmp_path: Path):
    """Test serializing Thread objects to JSON Lines and reading back without loss."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    t1 = RawTweet(
        tweet_id="101",
        author_id="cust1",
        inbound=True,
        created_at=base_time,
        text="Need support",
        response_tweet_id="102",
        in_response_to_tweet_id=None,
    )
    t2 = RawTweet(
        tweet_id="102",
        author_id="AppleSupport",
        inbound=False,
        created_at=base_time + timedelta(minutes=5),
        text="Happy to assist!",
        response_tweet_id=None,
        in_response_to_tweet_id="101",
    )
    thread = Thread(thread_id="101", tweets=[t1, t2], terminal=True)

    jsonl_file = tmp_path / "threads.jsonl"
    save_threads_to_jsonl([thread], jsonl_file)

    assert jsonl_file.exists()
    loaded_threads = load_threads_from_jsonl(jsonl_file)

    assert len(loaded_threads) == 1
    loaded = loaded_threads[0]
    assert loaded.thread_id == "101"
    assert loaded.turn_count == 2
    assert loaded.terminal is True
    assert loaded.tweets[0].text == "Need support"
    assert loaded.tweets[1].author_id == "AppleSupport"


# ---------------------------------------------------------------------------
# 11. Graph-Expanded Brand Filter Tests in ingest.py
# ---------------------------------------------------------------------------

def test_filter_brand_graph_expansion():
    """Test that filter_brand with expand_threads=True includes inbound consumer tweets in brand threads."""
    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        # Brand thread: 1 (cust1) -> 2 (AppleSupport)
        {
            "tweet_id": "1",
            "author_id": "cust1",
            "inbound": True,
            "created_at": base_time,
            "text": "Phone broke",
            "response_tweet_id": "2",
            "in_response_to_tweet_id": None,
        },
        {
            "tweet_id": "2",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": base_time + timedelta(minutes=5),
            "text": "Let us check",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "1",
        },
        # Unrelated consumer tweet (no brand response)
        {
            "tweet_id": "3",
            "author_id": "cust2",
            "inbound": True,
            "created_at": base_time,
            "text": "Random tweet",
            "response_tweet_id": None,
            "in_response_to_tweet_id": None,
        },
    ])

    # With expand_threads=False: only tweet 2 (AppleSupport) is retained
    df_direct = filter_brand(df, brand_handle="AppleSupport", expand_threads=False)
    assert len(df_direct) == 1
    assert df_direct["tweet_id"].tolist() == ["2"]

    # With expand_threads=True: both tweet 1 (consumer) and tweet 2 (brand) are retained!
    df_expanded = filter_brand(df, brand_handle="AppleSupport", expand_threads=True)
    assert len(df_expanded) == 2
    assert set(df_expanded["tweet_id"].tolist()) == {"1", "2"}
    assert "3" not in df_expanded["tweet_id"].tolist()
