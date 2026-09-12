"""Unit tests for M1.P1.2.F1: Lexical De-contraction.

Covers:
- Standard contractions expansion
- Social-media shorthand expansion
- Casing preservation (lowercase, TitleCase, ALL_CAPS)
- Possessive apostrophe preservation (e.g., "Apple's policy")
- Ambiguous contraction default ("it's" -> "it is")
- Defensive URL bypass (slugs containing contractions)
- Emoji and non-ASCII unicode adjacency
- Smart/curly apostrophes
- Empty and whitespace-only strings
- Idempotency on single and repeated applications
- Multiple contractions in a single string
- Functional thread immutability in expand_thread()
- CLI/run() file streaming execution
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from src.data.thread_schema import RawTweet, Thread
from src.nlp.contraction_map import CONTRACTIONS
from src.nlp.decontraction import (
    expand_contractions,
    expand_thread,
    run,
)


class TestStandardContractions:
    """Test standard English contractions."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("I can't update my iPhone", "I cannot update my iPhone"),
            ("It won't turn on", "It will not turn on"),
            ("I'm having trouble with iCloud", "I am having trouble with iCloud"),
            ("I didn't receive the verification code", "I did not receive the verification code"),
            ("We couldn't've known about the outage", "We could not have known about the outage"),
            ("They shouldn't do that", "They should not do that"),
            ("Let's figure this out", "Let us figure this out"),
            ("They're already on iOS 17", "They are already on iOS 17"),
            ("I've been waiting for hours", "I have been waiting for hours"),
            ("You'll get a prompt soon", "You will get a prompt soon"),
            ("We'd love to help you", "We would love to help you"),
        ],
    )
    def test_standard_expansion(self, input_text: str, expected_text: str) -> None:
        assert expand_contractions(input_text) == expected_text


class TestSocialMediaShorthand:
    """Test PRD-specified social media and informal shorthand contractions."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("'bout time Apple fixed this", "about time Apple fixed this"),
            ("I'm gonna reset my settings", "I am going to reset my settings"),
            ("I wanna transfer my data", "I want to transfer my data"),
            ("You gotta install the latest patch", "You got to install the latest patch"),
            ("Lemme check the serial number", "Let me check the serial number"),
            ("I dunno what happened to my battery", "I do not know what happened to my battery"),
            ("Hey y'all, can anyone help?", "Hey you all, can anyone help?"),
            ("Imma try restarting it now", "I am going to try restarting it now"),
            ("It is kinda slow today", "It is kind of slow today"),
            ("Sorta works after restarting", "Sort of works after restarting"),
            ("Calling 'cause my screen went black", "Calling because my screen went black"),
        ],
    )
    def test_social_shorthand_expansion(self, input_text: str, expected_text: str) -> None:
        assert expand_contractions(input_text) == expected_text


class TestCasingPreservation:
    """Test preservation of casing patterns: lowercase, leading capital, and ALL-CAPS."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            # Lowercase
            ("can't", "cannot"),
            ("won't", "will not"),
            ("don't", "do not"),
            ("'bout", "about"),
            ("gonna", "going to"),
            # Leading Capital (TitleCase)
            ("Can't", "Cannot"),
            ("Won't", "Will not"),
            ("Don't", "Do not"),
            ("'Bout", "About"),
            ("Gonna", "Going to"),
            ("Y'all", "You all"),
            ("I'll", "I will"),
            # ALL-CAPS
            ("CAN'T", "CANNOT"),
            ("WON'T", "WILL NOT"),
            ("DON'T", "DO NOT"),
            ("I'LL", "I WILL"),
            ("IT'S", "IT IS"),
            ("GONNA", "GOING TO"),
        ],
    )
    def test_casing_styles(self, input_text: str, expected_text: str) -> None:
        assert expand_contractions(input_text) == expected_text


class TestPossessivePreservation:
    """Test that possessive apostrophes are strictly preserved and not expanded."""

    @pytest.mark.parametrize(
        "text",
        [
            "Apple's policy on battery replacements",
            "The user's device is unresponsive",
            "tim's macbook is malfunctioning",
            "Spirit's customer service",
            "Where is John's iPhone?",
            "The company's terms and conditions",
        ],
    )
    def test_possessives_untouched(self, text: str) -> None:
        assert expand_contractions(text) == text


class TestAmbiguousContractionDefault:
    """Test documented heuristic behavior for ambiguous contractions like 'it's'."""

    def test_its_defaults_to_it_is(self) -> None:
        assert expand_contractions("It's broken") == "It is broken"
        assert expand_contractions("it's not charging") == "it is not charging"
        assert expand_contractions("I think it's an iOS issue") == "I think it is an iOS issue"

    def test_other_ambiguous_defaults(self) -> None:
        assert expand_contractions("there's an update") == "there is an update"
        assert expand_contractions("he's available") == "he is available"
        assert expand_contractions("she's calling now") == "she is calling now"


class TestUrlDefensiveBypass:
    """Test that contraction-like slugs inside URLs are not corrupted."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            (
                "Check https://support.apple.com/en-us/don't-panic for help",
                "Check https://support.apple.com/en-us/don't-panic for help",
            ),
            (
                "Visit http://example.com/won't-fix and tell me if it's working",
                "Visit http://example.com/won't-fix and tell me if it is working",
            ),
            (
                "I can't open www.apple.com/can't-boot",
                "I cannot open www.apple.com/can't-boot",
            ),
            (
                "https://apple.co/support?q=can't_login",
                "https://apple.co/support?q=can't_login",
            ),
        ],
    )
    def test_urls_preserved_while_surrounding_contractions_expand(
        self, input_text: str, expected_text: str
    ) -> None:
        assert expand_contractions(input_text) == expected_text


class TestEmojiAndUnicodeAdjacency:
    """Test contractions adjacent to emoji or non-ASCII characters."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("I can't 😤", "I cannot 😤"),
            ("I can't😤", "I cannot😤"),
            ("😤can't believe this", "😤cannot believe this"),
            ("it's🔥broken", "it is🔥broken"),
            ("won't 🍎 work", "will not 🍎 work"),
            ("can't... why?!", "cannot... why?!"),
            ("I'm—wait—it's okay", "I am—wait—it is okay"),
        ],
    )
    def test_emoji_and_unicode_boundary(self, input_text: str, expected_text: str) -> None:
        assert expand_contractions(input_text) == expected_text


class TestSmartAndCurlyApostrophes:
    """Test matching and expanding curly/smart apostrophes (’ and ‘)."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("I can’t hear anything", "I cannot hear anything"),
            ("It won’t turn on", "It will not turn on"),
            ("’bout time to upgrade", "about time to upgrade"),
            ("I’m on the latest iOS", "I am on the latest iOS"),
            ("It’s not charging", "It is not charging"),
            ("Don‘t click that", "Do not click that"),
        ],
    )
    def test_smart_apostrophes_expanded(self, input_text: str, expected_text: str) -> None:
        assert expand_contractions(input_text) == expected_text


class TestEdgeCasesAndIdempotency:
    """Test edge cases: empty strings, whitespace, idempotency, multiple matches."""

    def test_empty_and_whitespace_strings(self) -> None:
        assert expand_contractions("") == ""
        assert expand_contractions("   ") == "   "
        assert expand_contractions("\n\t") == "\n\t"

    def test_idempotency_single_and_already_expanded(self) -> None:
        sample = "I do not think it will work, I cannot tell."
        assert expand_contractions(sample) == sample

        contracted = "I don't think it'll work, I can't tell."
        first_pass = expand_contractions(contracted)
        second_pass = expand_contractions(first_pass)
        assert first_pass == "I do not think it will work, I cannot tell."
        assert second_pass == first_pass

    def test_multiple_contractions_in_one_sentence(self) -> None:
        text = "I don't think it'll work, I can't tell, 'bout ready to quit"
        expected = "I do not think it will work, I cannot tell, about ready to quit"
        assert expand_contractions(text) == expected


class TestExpandThread:
    """Test pure functional expand_thread() function."""

    @pytest.fixture
    def sample_thread(self) -> Thread:
        t1 = RawTweet(
            tweet_id="101",
            author_id="user_1",
            inbound=True,
            text="@AppleSupport my phone won't charge and it's overheating 😤",
            created_at=datetime(2017, 10, 11, 10, 0, 0, tzinfo=timezone.utc),
            in_response_to_tweet_id=None,
            response_tweet_id="102",
        )
        t2 = RawTweet(
            tweet_id="102",
            author_id="AppleSupport",
            inbound=False,
            text="@user_1 We'd like to help. Can't you see Apple's charging indicator?",
            created_at=datetime(2017, 10, 11, 10, 5, 0, tzinfo=timezone.utc),
            in_response_to_tweet_id="101",
            response_tweet_id=None,
        )
        return Thread(
            thread_id="101",
            tweets=[t1, t2],
            terminal=True,
        )

    def test_expand_thread_pure_transformation(self, sample_thread: Thread) -> None:
        original_t1_text = sample_thread.tweets[0].text
        original_t2_text = sample_thread.tweets[1].text

        transformed = expand_thread(sample_thread)

        # Verify new thread is a distinct object
        assert transformed is not sample_thread
        assert transformed.tweets[0] is not sample_thread.tweets[0]

        # Verify original thread was NOT mutated
        assert sample_thread.tweets[0].text == original_t1_text
        assert sample_thread.tweets[1].text == original_t2_text

        # Verify expansions in transformed thread
        assert (
            transformed.tweets[0].text
            == "@AppleSupport my phone will not charge and it is overheating 😤"
        )
        assert (
            transformed.tweets[1].text
            == "@user_1 We would like to help. Cannot you see Apple's charging indicator?"
        )

        # Metadata intact
        assert transformed.thread_id == sample_thread.thread_id
        assert transformed.turn_count == sample_thread.turn_count
        assert transformed.terminal == sample_thread.terminal


class TestPipelineRun:
    """Test file-based streaming run() execution."""

    def test_run_streaming(self, tmp_path: Path) -> None:
        in_file = tmp_path / "test_threads.jsonl"
        out_file = tmp_path / "test_threads_decontracted.jsonl"

        tweet = RawTweet(
            tweet_id="201",
            author_id="user_2",
            inbound=True,
            text="I'm gonna update 'cause I can't wait",
            created_at=datetime(2017, 10, 12, 8, 0, 0, tzinfo=timezone.utc),
            in_response_to_tweet_id=None,
            response_tweet_id=None,
        )
        thread = Thread(
            thread_id="201",
            tweets=[tweet],
            terminal=False,
        )

        with open(in_file, "w", encoding="utf-8") as f:
            f.write(thread.model_dump_json() + "\n")

        processed_count = run(input_path=in_file, output_path=out_file)
        assert processed_count == 1
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            loaded = Thread.model_validate_json(line)
            assert (
                loaded.tweets[0].text
                == "I am going to update because I cannot wait"
            )

    def test_run_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            run(input_path=tmp_path / "non_existent.jsonl", output_path=tmp_path / "out.jsonl")
