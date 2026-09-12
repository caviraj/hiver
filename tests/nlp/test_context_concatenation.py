"""Unit tests for M1.P1.2.F3: Context Window Concatenation.

Covers:
- format_turn() with default and custom speaker mappings
- Single-turn and multi-turn thread context construction
- Sliding window depth behavior:
    - max_history_turns is None (unbounded full history)
    - max_history_turns == 0 (no history, current turn only)
    - max_history_turns == k (sliding window of k preceding turns)
    - max_history_turns > available history (graceful full history)
- Error handling:
    - turn_idx out of bounds (< 0 or >= len(tweets)) raises IndexError
    - max_history_turns < 0 raises ValueError
- Inbound-only filtering:
    - Only inbound (customer) turns emitted as records
    - Inbound records retain prior brand (agent) turns in their history
- Turn separators (default '\\n', '\\n\\n', custom separators)
- Masked entity and device token preservation (<url>, __email__, iPhone X, iOS 11.1)
- Immutability of Thread and RawTweet instances
- ContextWindowRecord Pydantic serialization / validation
- Streaming run() pipeline with JSONL I/O
- CLI main() argument parsing and execution
- Package-level imports via src.nlp
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import List

from pydantic import ValidationError
import pytest

from src.data.thread_schema import RawTweet, Thread
from src.nlp.context_concatenation import (
    DEFAULT_SPEAKER_LABELS,
    ContextWindowRecord,
    build_thread_contexts,
    build_turn_context,
    concatenate_thread_text,
    format_turn,
    main,
    run,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed UTC base timestamp for reproducible tests."""
    return datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def single_turn_thread(base_timestamp: datetime) -> Thread:
    """Thread with a single root customer tweet."""
    tweet = RawTweet(
        tweet_id="101",
        author_id="cust_1",
        inbound=True,
        created_at=base_timestamp,
        text="My iPhone X battery is draining fast on iOS 11.1.",
        response_tweet_id="102",
        in_response_to_tweet_id=None,
    )
    return Thread(thread_id="101", tweets=[tweet], terminal=False)


@pytest.fixture
def four_turn_thread(base_timestamp: datetime) -> Thread:
    """Thread with 4 turns alternating: Customer -> Agent -> Customer -> Agent."""
    t0 = RawTweet(
        tweet_id="201",
        author_id="cust_2",
        inbound=True,
        created_at=base_timestamp,
        text="Help! My Apple Watch Series 3 screen is frozen.",
        response_tweet_id="202",
        in_response_to_tweet_id=None,
    )
    t1 = RawTweet(
        tweet_id="202",
        author_id="AppleSupport",
        inbound=False,
        created_at=datetime(2026, 9, 12, 12, 5, 0, tzinfo=timezone.utc),
        text="Have you tried force restarting it? See <url> for steps.",
        response_tweet_id="203",
        in_response_to_tweet_id="201",
    )
    t2 = RawTweet(
        tweet_id="203",
        author_id="cust_2",
        inbound=True,
        created_at=datetime(2026, 9, 12, 12, 10, 0, tzinfo=timezone.utc),
        text="Yes I did. Still having the issue. Contact me at __email__",
        response_tweet_id="204",
        in_response_to_tweet_id="202",
    )
    t3 = RawTweet(
        tweet_id="204",
        author_id="AppleSupport",
        inbound=False,
        created_at=datetime(2026, 9, 12, 12, 15, 0, tzinfo=timezone.utc),
        text="Please send us a DM so we can look into repair options.",
        response_tweet_id=None,
        in_response_to_tweet_id="203",
    )
    return Thread(thread_id="201", tweets=[t0, t1, t2, t3], terminal=True)


# =====================================================================
# Testformat_turn
# =====================================================================


class TestFormatTurn:
    """Test format_turn with default and custom speaker mappings."""

    def test_default_speaker_customer(self, base_timestamp: datetime) -> None:
        tweet = RawTweet(
            tweet_id="1",
            author_id="user_a",
            inbound=True,
            created_at=base_timestamp,
            text="Need help with my MacBook Pro",
        )
        assert format_turn(tweet) == "Customer: Need help with my MacBook Pro"

    def test_default_speaker_agent(self, base_timestamp: datetime) -> None:
        tweet = RawTweet(
            tweet_id="2",
            author_id="AppleSupport",
            inbound=False,
            created_at=base_timestamp,
            text="We would be glad to help! Which macOS version are you on?",
        )
        assert (
            format_turn(tweet)
            == "Agent: We would be glad to help! Which macOS version are you on?"
        )

    def test_custom_speaker_labels(self, base_timestamp: datetime) -> None:
        tweet_cust = RawTweet(
            tweet_id="1",
            author_id="user_a",
            inbound=True,
            created_at=base_timestamp,
            text="Hello",
        )
        tweet_agent = RawTweet(
            tweet_id="2",
            author_id="support",
            inbound=False,
            created_at=base_timestamp,
            text="Hi there",
        )
        custom_labels = {True: "User", False: "SupportRep"}

        assert format_turn(tweet_cust, custom_labels) == "User: Hello"
        assert format_turn(tweet_agent, custom_labels) == "SupportRep: Hi there"

    def test_empty_tweet_text(self, base_timestamp: datetime) -> None:
        tweet = RawTweet(
            tweet_id="3",
            author_id="user_b",
            inbound=True,
            created_at=base_timestamp,
            text="",
        )
        assert format_turn(tweet) == "Customer: "


# =====================================================================
# TestBuildTurnContext
# =====================================================================


class TestBuildTurnContext:
    """Test build_turn_context across window depths, separators, and edge cases."""

    def test_single_turn_thread_root(
        self, single_turn_thread: Thread, base_timestamp: datetime
    ) -> None:
        record = build_turn_context(single_turn_thread, turn_idx=0)

        assert record.thread_id == "101"
        assert record.turn_index == 0
        assert record.tweet_id == "101"
        assert record.author_id == "cust_1"
        assert record.speaker == "Customer"
        assert record.inbound is True
        assert record.created_at == base_timestamp
        assert (
            record.text == "My iPhone X battery is draining fast on iOS 11.1."
        )
        assert record.history == []
        assert (
            record.context_text
            == "Customer: My iPhone X battery is draining fast on iOS 11.1."
        )
        assert record.in_response_to_tweet_id is None
        assert record.response_tweet_id == "102"

    def test_unbounded_history(self, four_turn_thread: Thread) -> None:
        # Turn 0
        rec0 = build_turn_context(
            four_turn_thread, turn_idx=0, max_history_turns=None
        )
        assert rec0.history == []
        assert (
            rec0.context_text
            == "Customer: Help! My Apple Watch Series 3 screen is frozen."
        )

        # Turn 1
        rec1 = build_turn_context(
            four_turn_thread, turn_idx=1, max_history_turns=None
        )
        assert rec1.history == [
            "Customer: Help! My Apple Watch Series 3 screen is frozen."
        ]
        assert rec1.context_text == (
            "Customer: Help! My Apple Watch Series 3 screen is frozen.\n"
            "Agent: Have you tried force restarting it? See <url> for steps."
        )

        # Turn 2
        rec2 = build_turn_context(
            four_turn_thread, turn_idx=2, max_history_turns=None
        )
        assert rec2.history == [
            "Customer: Help! My Apple Watch Series 3 screen is frozen.",
            "Agent: Have you tried force restarting it? See <url> for steps.",
        ]
        assert rec2.context_text == (
            "Customer: Help! My Apple Watch Series 3 screen is frozen.\n"
            "Agent: Have you tried force restarting it? See <url> for steps.\n"
            "Customer: Yes I did. Still having the issue. Contact me at __email__"
        )

        # Turn 3
        rec3 = build_turn_context(
            four_turn_thread, turn_idx=3, max_history_turns=None
        )
        assert len(rec3.history) == 3
        assert rec3.context_text.endswith(
            "Agent: Please send us a DM so we can look into repair options."
        )

    def test_history_depth_zero(self, four_turn_thread: Thread) -> None:
        rec2 = build_turn_context(
            four_turn_thread, turn_idx=2, max_history_turns=0
        )
        assert rec2.history == []
        assert (
            rec2.context_text
            == "Customer: Yes I did. Still having the issue. Contact me at __email__"
        )

    def test_sliding_window_depth_one(self, four_turn_thread: Thread) -> None:
        # Turn 0: no prior history
        rec0 = build_turn_context(
            four_turn_thread, turn_idx=0, max_history_turns=1
        )
        assert rec0.history == []

        # Turn 1: history is turn 0
        rec1 = build_turn_context(
            four_turn_thread, turn_idx=1, max_history_turns=1
        )
        assert rec1.history == [
            "Customer: Help! My Apple Watch Series 3 screen is frozen."
        ]

        # Turn 2: history is turn 1 only
        rec2 = build_turn_context(
            four_turn_thread, turn_idx=2, max_history_turns=1
        )
        assert rec2.history == [
            "Agent: Have you tried force restarting it? See <url> for steps."
        ]
        assert rec2.context_text == (
            "Agent: Have you tried force restarting it? See <url> for steps.\n"
            "Customer: Yes I did. Still having the issue. Contact me at __email__"
        )

        # Turn 3: history is turn 2 only
        rec3 = build_turn_context(
            four_turn_thread, turn_idx=3, max_history_turns=1
        )
        assert rec3.history == [
            "Customer: Yes I did. Still having the issue. Contact me at __email__"
        ]

    def test_sliding_window_depth_two(self, four_turn_thread: Thread) -> None:
        # Turn 3 with max_history_turns=2: includes turns 1 and 2
        rec3 = build_turn_context(
            four_turn_thread, turn_idx=3, max_history_turns=2
        )
        assert rec3.history == [
            "Agent: Have you tried force restarting it? See <url> for steps.",
            "Customer: Yes I did. Still having the issue. Contact me at __email__",
        ]
        assert rec3.context_text == (
            "Agent: Have you tried force restarting it? See <url> for steps.\n"
            "Customer: Yes I did. Still having the issue. Contact me at __email__\n"
            "Agent: Please send us a DM so we can look into repair options."
        )

    def test_window_exceeds_available_history(
        self, four_turn_thread: Thread
    ) -> None:
        rec1 = build_turn_context(
            four_turn_thread, turn_idx=1, max_history_turns=10
        )
        # Only 1 prior turn available
        assert len(rec1.history) == 1
        assert rec1.history[0].startswith("Customer:")

    def test_negative_max_history_raises_value_error(
        self, four_turn_thread: Thread
    ) -> None:
        with pytest.raises(
            ValueError, match="max_history_turns must be non-negative"
        ):
            build_turn_context(
                four_turn_thread, turn_idx=1, max_history_turns=-1
            )

    def test_out_of_bounds_turn_idx_raises_index_error(
        self, four_turn_thread: Thread
    ) -> None:
        with pytest.raises(IndexError, match="out of range"):
            build_turn_context(four_turn_thread, turn_idx=-1)

        with pytest.raises(IndexError, match="out of range"):
            build_turn_context(four_turn_thread, turn_idx=4)

        with pytest.raises(IndexError, match="out of range"):
            build_turn_context(four_turn_thread, turn_idx=100)

    def test_custom_separator(self, four_turn_thread: Thread) -> None:
        rec2_double_nl = build_turn_context(
            four_turn_thread, turn_idx=2, separator="\n\n"
        )
        assert "\n\n" in rec2_double_nl.context_text
        assert rec2_double_nl.context_text.count("\n\n") == 2

        rec2_pipe = build_turn_context(
            four_turn_thread, turn_idx=2, separator=" | "
        )
        assert " | " in rec2_pipe.context_text
        assert rec2_pipe.context_text.count(" | ") == 2


# =====================================================================
# TestBuildThreadContexts
# =====================================================================


class TestBuildThreadContexts:
    """Test build_thread_contexts for complete threads and inbound filtering."""

    def test_all_turns_generated_by_default(
        self, four_turn_thread: Thread
    ) -> None:
        records = build_thread_contexts(four_turn_thread)
        assert len(records) == 4

        for idx, rec in enumerate(records):
            assert rec.turn_index == idx
            assert rec.thread_id == "201"
            assert rec.tweet_id == four_turn_thread.tweets[idx].tweet_id

        assert records[0].speaker == "Customer"
        assert records[1].speaker == "Agent"
        assert records[2].speaker == "Customer"
        assert records[3].speaker == "Agent"

    def test_inbound_only_filtering(self, four_turn_thread: Thread) -> None:
        records = build_thread_contexts(four_turn_thread, inbound_only=True)
        assert len(records) == 2

        rec0, rec2 = records[0], records[1]
        # First inbound turn is turn 0
        assert rec0.turn_index == 0
        assert rec0.inbound is True
        assert rec0.speaker == "Customer"
        assert rec0.history == []

        # Second inbound turn is turn 2
        assert rec2.turn_index == 2
        assert rec2.inbound is True
        assert rec2.speaker == "Customer"
        # Crucial: turn 2's history must include Agent's turn 1!
        assert len(rec2.history) == 2
        assert rec2.history[0].startswith("Customer:")
        assert rec2.history[1].startswith("Agent:")

    def test_inbound_only_with_sliding_window(
        self, four_turn_thread: Thread
    ) -> None:
        records = build_thread_contexts(
            four_turn_thread, max_history_turns=1, inbound_only=True
        )
        assert len(records) == 2
        rec2 = records[1]
        assert rec2.turn_index == 2
        # max_history=1 slices just preceding turn (Agent turn 1)
        assert len(rec2.history) == 1
        assert rec2.history[0].startswith("Agent:")

    def test_thread_immutability(self, four_turn_thread: Thread) -> None:
        dump_before = four_turn_thread.model_dump()
        _ = build_thread_contexts(
            four_turn_thread, max_history_turns=2, inbound_only=True
        )
        dump_after = four_turn_thread.model_dump()
        assert dump_before == dump_after


# =====================================================================
# TestConcatenateThreadText
# =====================================================================


class TestConcatenateThreadText:
    """Test concatenate_thread_text transcript generation."""

    def test_default_transcript(self, four_turn_thread: Thread) -> None:
        transcript = concatenate_thread_text(four_turn_thread)
        lines = transcript.split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("Customer:")
        assert lines[1].startswith("Agent:")
        assert lines[2].startswith("Customer:")
        assert lines[3].startswith("Agent:")

    def test_custom_separator_and_labels(
        self, single_turn_thread: Thread
    ) -> None:
        transcript = concatenate_thread_text(
            single_turn_thread,
            speaker_labels={True: "Client", False: "Support"},
            separator=" --- ",
        )
        assert (
            transcript
            == "Client: My iPhone X battery is draining fast on iOS 11.1."
        )


# =====================================================================
# TestEntityAndTokenPreservation
# =====================================================================


class TestEntityAndTokenPreservation:
    """Ensure normalization tokens (<url>, __email__, devices) survive concatenation verbatim."""

    def test_masked_tokens_and_entities_survive(
        self, four_turn_thread: Thread
    ) -> None:
        records = build_thread_contexts(four_turn_thread)

        # Check turn 1 for <url>
        rec1 = records[1]
        assert "<url>" in rec1.text
        assert "<url>" in rec1.context_text

        # Check turn 2 for __email__ and prior watch entity in history
        rec2 = records[2]
        assert "__email__" in rec2.text
        assert "__email__" in rec2.context_text
        assert "Apple Watch Series 3" in rec2.history[0]
        assert "Apple Watch Series 3" in rec2.context_text


# =====================================================================
# TestContextWindowRecordModel
# =====================================================================


class TestContextWindowRecordModel:
    """Test Pydantic model validation and JSON serialization."""

    def test_model_json_roundtrip(self, base_timestamp: datetime) -> None:
        rec = ContextWindowRecord(
            thread_id="101",
            turn_index=1,
            tweet_id="102",
            author_id="agent_1",
            speaker="Agent",
            inbound=False,
            created_at=base_timestamp,
            text="Hello",
            history=["Customer: Hi"],
            context_text="Customer: Hi\nAgent: Hello",
            in_response_to_tweet_id="101",
            response_tweet_id=None,
        )

        json_str = rec.model_dump_json()
        data = json.loads(json_str)
        assert data["thread_id"] == "101"
        assert data["turn_index"] == 1
        assert data["history"] == ["Customer: Hi"]
        assert data["context_text"] == "Customer: Hi\nAgent: Hello"

        restored = ContextWindowRecord.model_validate_json(json_str)
        assert restored.thread_id == rec.thread_id
        assert restored.turn_index == rec.turn_index
        assert restored.history == rec.history
        assert restored.context_text == rec.context_text

    def test_turn_index_negative_validation_error(
        self, base_timestamp: datetime
    ) -> None:
        with pytest.raises(ValidationError):
            ContextWindowRecord(
                thread_id="101",
                turn_index=-1,  # ge=0 violation
                tweet_id="102",
                author_id="agent_1",
                speaker="Agent",
                inbound=False,
                created_at=base_timestamp,
                text="Hello",
                history=[],
                context_text="Agent: Hello",
            )


# =====================================================================
# TestRunStreamingAndCli
# =====================================================================


class TestRunStreamingAndCli:
    """Test JSONL streaming processing and CLI entrypoint."""

    def test_run_file_not_found(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.jsonl"
        output_file = tmp_path / "output.jsonl"
        with pytest.raises(FileNotFoundError, match="Input thread file not found"):
            run(input_path=non_existent, output_path=output_file)

    def test_run_streaming_success(
        self,
        tmp_path: Path,
        single_turn_thread: Thread,
        four_turn_thread: Thread,
    ) -> None:
        input_file = tmp_path / "threads.jsonl"
        output_file = tmp_path / "windows.jsonl"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(single_turn_thread.model_dump_json() + "\n")
            f.write("\n")  # empty line should be ignored
            f.write("   \n")  # whitespace line should be ignored
            f.write(four_turn_thread.model_dump_json() + "\n")

        # 1 turn + 4 turns = 5 total records
        count = run(input_path=input_file, output_path=output_file)
        assert count == 5
        assert output_file.exists()

        records: List[ContextWindowRecord] = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(ContextWindowRecord.model_validate_json(line))

        assert len(records) == 5
        assert [r.thread_id for r in records] == [
            "101",
            "201",
            "201",
            "201",
            "201",
        ]
        assert [r.turn_index for r in records] == [0, 0, 1, 2, 3]

    def test_run_streaming_inbound_only(
        self,
        tmp_path: Path,
        single_turn_thread: Thread,
        four_turn_thread: Thread,
    ) -> None:
        input_file = tmp_path / "threads.jsonl"
        output_file = tmp_path / "inbound_windows.jsonl"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(single_turn_thread.model_dump_json() + "\n")
            f.write(four_turn_thread.model_dump_json() + "\n")

        # single_turn_thread has 1 inbound turn; four_turn_thread has 2 inbound turns -> 3 total
        count = run(
            input_path=input_file,
            output_path=output_file,
            inbound_only=True,
            max_history_turns=2,
        )
        assert count == 3

        records: List[ContextWindowRecord] = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                records.append(ContextWindowRecord.model_validate_json(line))

        assert len(records) == 3
        for r in records:
            assert r.inbound is True
            assert r.speaker == "Customer"

    def test_cli_main(
        self,
        tmp_path: Path,
        four_turn_thread: Thread,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        input_file = tmp_path / "in.jsonl"
        output_file = tmp_path / "out.jsonl"

        with open(input_file, "w", encoding="utf-8") as f:
            f.write(four_turn_thread.model_dump_json() + "\n")

        test_args = [
            "context_concatenation.py",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--max-history",
            "1",
            "--inbound-only",
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        main()

        assert output_file.exists()
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2  # 2 inbound turns in four_turn_thread


# =====================================================================
# TestPackageExports
# =====================================================================


class TestPackageExports:
    """Ensure context concatenation symbols are exported by src.nlp."""

    def test_package_exports(self) -> None:
        import src.nlp as nlp

        assert hasattr(nlp, "ContextWindowRecord")
        assert hasattr(nlp, "DEFAULT_SPEAKER_LABELS")
        assert hasattr(nlp, "format_turn")
        assert hasattr(nlp, "build_turn_context")
        assert hasattr(nlp, "build_thread_contexts")
        assert hasattr(nlp, "concatenate_thread_text")

        from src.nlp import (
            ContextWindowRecord as CWRecord,
            DEFAULT_SPEAKER_LABELS as DSLabels,
            build_thread_contexts as btc,
            build_turn_context as b_turn_ctx,
            concatenate_thread_text as ctt,
            format_turn as ft,
        )

        assert CWRecord is ContextWindowRecord
        assert DSLabels is DEFAULT_SPEAKER_LABELS
        assert btc is build_thread_contexts
        assert b_turn_ctx is build_turn_context
        assert ctt is concatenate_thread_text
        assert ft is format_turn
