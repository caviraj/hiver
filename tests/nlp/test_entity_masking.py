"""Unit tests for M1.P1.2.F2: Entity Masking & URL Handling.

Covers:
- URL masking (http, https, www, bare domain paths, multiple URLs, truncated t.co links)
- Trailing sentence punctuation preservation outside URLs
- Device model and OS version preservation across case and spacing variations
- Original casing and spacing preservation for protected entities
- Pre-masked PII token (__email__, __phone__) survival adjacent to punctuation
- Protect-process-restore ordering:
    - Device strings inside URLs neutralized with the URL (not extracted)
    - Device strings adjacent to URLs preserved without corruption
- False positive avoidance (e.g. common English words like 'smart', 'small', 'smoke')
- Edge cases: empty strings, whitespace, text without URLs/entities, idempotency
- Functional thread immutability in mask_thread()
- CLI run() streaming execution and error handling
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from src.data.thread_schema import RawTweet, Thread
from src.nlp.entity_masking import (
    mask_entities,
    mask_thread,
    mask_urls,
    preserve_device_entities,
    run,
)


class TestUrlMasking:
    """Test URL neutralization to <url> tokens."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("Check out https://apple.com for details", "Check out <url> for details"),
            ("Visit http://support.apple.com today", "Visit <url> today"),
            ("Go to www.apple.com/support to fix it", "Go to <url> to fix it"),
            ("Direct link: support.apple.com/kb/HT201263", "Direct link: <url>"),
            ("Short link apple.co/help now available", "Short link <url> now available"),
        ],
    )
    def test_standard_url_masking(self, input_text: str, expected_text: str) -> None:
        assert mask_urls(input_text) == expected_text

    def test_multiple_urls_in_tweet(self) -> None:
        text = "Visit https://apple.com and www.google.com or check https://t.co/abc12345"
        expected = "Visit <url> and <url> or check <url>"
        assert mask_urls(text) == expected

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("See https://t.co/AbC123xYz for help", "See <url> for help"),
            ("Update at t.co/xyz123 right now", "Update at <url> right now"),
            ("Truncated link t.co/7a8b9c", "Truncated link <url>"),
        ],
    )
    def test_twitter_shortener_urls(self, input_text: str, expected_text: str) -> None:
        assert mask_urls(input_text) == expected_text

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("Visit https://apple.com.", "Visit <url>."),
            ("Check out http://apple.com, then restart!", "Check out <url>, then restart!"),
            ("Did you check www.apple.com/help?", "Did you check <url>?"),
            ("Link: [https://apple.com/support]", "Link: [<url>]"),
            ("See 'https://apple.com/kb'", "See '<url>'"),
            ('Read "https://apple.com/docs"', 'Read "<url>"'),
        ],
    )
    def test_trailing_punctuation_exclusion(self, input_text: str, expected_text: str) -> None:
        assert mask_urls(input_text) == expected_text

    def test_empty_and_whitespace(self) -> None:
        assert mask_urls("") == ""
        assert mask_urls("   ") == "   "


class TestDeviceAndOsPreservation:
    """Test device models and OS version preservation across case and spacing variations."""

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("My iPhone X has battery issues", "iPhone X"),
            ("Got an iphone 8 plus recently", "iphone 8 plus"),
            ("Testing on iPhone 7+", "iPhone 7+"),
            ("My iPhone 11 Pro Max rebooted", "iPhone 11 Pro Max"),
            ("I love my iPhone 12 mini", "iPhone 12 mini"),
            ("Still on iPhone SE here", "iPhone SE"),
            ("Using an iphone 6s", "iphone 6s"),
        ],
    )
    def test_iphone_models_preservation(self, input_text: str, expected_entity: str) -> None:
        masked = mask_entities(input_text)
        assert expected_entity in masked

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("My iPad Pro is not charging", "iPad Pro"),
            ("Updating iPad Air 2 today", "iPad Air 2"),
            ("My ipad mini 4 is fast", "ipad mini 4"),
            ("Screen cracked on iPad 2", "iPad 2"),
            ("Selling my iPad Pro 10.5", "iPad Pro 10.5"),
        ],
    )
    def test_ipad_models_preservation(self, input_text: str, expected_entity: str) -> None:
        masked = mask_entities(input_text)
        assert expected_entity in masked

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("Connecting Apple Watch Series 3", "Apple Watch Series 3"),
            ("Battery on Apple Watch Ultra died", "Apple Watch Ultra"),
            ("Pairing apple watch se with phone", "apple watch se"),
        ],
    )
    def test_apple_watch_preservation(self, input_text: str, expected_entity: str) -> None:
        masked = mask_entities(input_text)
        assert expected_entity in masked

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("Audio glitch on MacBook Pro", "MacBook Pro"),
            ("Trackpad on MacBook Air stuck", "MacBook Air"),
            ("Display on iMac Pro flickers", "iMac Pro"),
            ("Restarted my Mac mini", "Mac mini"),
            ("Installed on macOS High Sierra", "macOS High Sierra"),
            ("Upgraded to macOS Big Sur 11.2", "macOS Big Sur 11.2"),
            ("Running macOS 10.13 smoothly", "macOS 10.13"),
        ],
    )
    def test_mac_hardware_and_macos_preservation(self, input_text: str, expected_entity: str) -> None:
        masked = mask_entities(input_text)
        assert expected_entity in masked

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("Running ios 11.1 on my phone", "ios 11.1"),
            ("Updated to IOS 11.1 yesterday", "IOS 11.1"),
            ("Running iOS 11.1 right now", "iOS 11.1"),
            ("Stuck on iOS11.1 update", "iOS11.1"),
            ("Is ios11.1 released?", "ios11.1"),
            ("Patched in iOS 17.0.3", "iOS 17.0.3"),
            ("Version watchOS 4.1 available", "watchOS 4.1"),
            ("tvOS 11 update failed", "tvOS 11"),
            ("iPadOS 16 stage manager", "iPadOS 16"),
        ],
    )
    def test_os_case_and_spacing_variations(self, input_text: str, expected_entity: str) -> None:
        """Case variations must be space-tolerant, case-insensitive, and preserve original text."""
        masked = mask_entities(input_text)
        assert expected_entity in masked

    @pytest.mark.parametrize(
        ("input_text", "expected_entity"),
        [
            ("My Samsung SM-T280 tablet is slow", "SM-T280"),
            ("Model SM-G950F cannot connect to WiFi", "SM-G950F"),
            ("Check serial for SM-A520F", "SM-A520F"),
        ],
    )
    def test_hardware_model_codes_preservation(self, input_text: str, expected_entity: str) -> None:
        masked = mask_entities(input_text)
        assert expected_entity in masked


class TestPiiPreservation:
    """Test survival of raw dataset pre-masked PII tokens (__email__, __phone__)."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("Contact us at __email__ for help", "Contact us at __email__ for help"),
            ("Please call __phone__ to reach support", "Please call __phone__ to reach support"),
            ("Please call __phone__.", "Please call __phone__."),
            ("Sent receipt to __email__!", "Sent receipt to __email__!"),
            ("Account details: (__email__)", "Account details: (__email__)"),
            ("Reach out: __email__, or call __phone__.", "Reach out: __email__, or call __phone__."),
        ],
    )
    def test_pii_adjacent_to_punctuation(self, input_text: str, expected_text: str) -> None:
        assert mask_entities(input_text) == expected_text


class TestProtectProcessRestoreOrdering:
    """Test interactions between entity protection and URL masking."""

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("See support.apple.com/iphone-x for info", "See <url> for info"),
            ("Guide at https://apple.com/ios-11/update", "Guide at <url>"),
            ("Go to www.apple.com/ipad-pro/specs now", "Go to <url> now"),
            ("Read support.apple.com/kb/SM-T280-guide", "Read <url>"),
        ],
    )
    def test_device_inside_url_is_masked_not_extracted(
        self, input_text: str, expected_text: str
    ) -> None:
        """Device substrings inside URLs must be neutralized with the URL into <url>."""
        assert mask_entities(input_text) == expected_text

    @pytest.mark.parametrize(
        ("input_text", "expected_text"),
        [
            ("iPhone X:https://apple.com", "iPhone X:<url>"),
            ("Check iOS 11.1 at https://support.apple.com", "Check iOS 11.1 at <url>"),
            ("https://apple.com on iPhone 8", "<url> on iPhone 8"),
            (
                "SM-T280 firmware: www.samsung.com/support/firmware",
                "SM-T280 firmware: <url>",
            ),
            (
                "Email __email__ or visit https://apple.co/help with iPhone X",
                "Email __email__ or visit <url> with iPhone X",
            ),
        ],
    )
    def test_device_adjacent_to_url(self, input_text: str, expected_text: str) -> None:
        """Device entities sitting right next to URLs must not corrupt or be corrupted by URL masking."""
        assert mask_entities(input_text) == expected_text

    def test_complex_interspersed_entities_and_urls(self) -> None:
        text = (
            "User with iPhone 11 Pro Max on iOS 17.0.3 emailed __email__. "
            "Refer to support.apple.com/iphone-x or call __phone__ at https://t.co/abc123"
        )
        expected = (
            "User with iPhone 11 Pro Max on iOS 17.0.3 emailed __email__. "
            "Refer to <url> or call __phone__ at <url>"
        )
        assert mask_entities(text) == expected


class TestFalsePositiveAvoidance:
    """Test that regular English words are not mistaken for device codes or URLs."""

    @pytest.mark.parametrize(
        "word",
        [
            "smart",
            "small",
            "smoke",
            "SMART",
            "smile",
            "smack",
            "smooth",
            "smear",
            "smash",
        ],
    )
    def test_common_sm_words_not_altered(self, word: str) -> None:
        text = f"That was a {word} choice for a phone."
        assert mask_entities(text) == text

    def test_hyphenated_words_not_altered(self) -> None:
        text = "This is state-of-the-art tech and top-notch quality."
        assert mask_entities(text) == text


class TestEdgeCasesAndIdempotency:
    """Test edge cases: empty strings, no-op passes, and idempotency."""

    def test_empty_and_whitespace(self) -> None:
        assert mask_entities("") == ""
        assert mask_entities("   ") == "   "
        assert mask_entities("\t\n") == "\t\n"

    def test_text_without_urls_or_entities(self) -> None:
        text = "My screen went black and won't turn on even after charging for an hour."
        assert mask_entities(text) == text

    @pytest.mark.parametrize(
        "text",
        [
            "Visit https://apple.com with your iPhone X on iOS 11.1, or call __phone__.",
            "support.apple.com/iphone-x is a useful link.",
            "Normal text with nothing special.",
            "SM-T280 tablet running smoothly.",
        ],
    )
    def test_idempotency(self, text: str) -> None:
        first_pass = mask_entities(text)
        second_pass = mask_entities(first_pass)
        assert second_pass == first_pass

    def test_preserve_device_entities_with_custom_process_fn(self) -> None:
        """Verify preserve_device_entities correctly protects entities during custom transformation."""
        def custom_fn(s: str) -> str:
            return s.replace("problem", "issue")

        text = "My iPhone X has a battery problem"
        expected = "My iPhone X has a battery issue"
        assert preserve_device_entities(text, custom_fn) == expected


class TestMaskThread:
    """Test functional immutability and thread masking."""

    @pytest.fixture
    def sample_thread(self) -> Thread:
        t1 = RawTweet(
            tweet_id="101",
            author_id="user1",
            inbound=True,
            in_response_to_tweet_id=None,
            text="My iPhone X on iOS 11.1 broke. See https://apple.com/broken",
            created_at=datetime.now(timezone.utc),
            response_tweet_id="102",
        )
        t2 = RawTweet(
            tweet_id="102",
            author_id="AppleSupport",
            inbound=False,
            in_response_to_tweet_id="101",
            text="Please check support.apple.com/iphone-x or call __phone__.",
            created_at=datetime.now(timezone.utc),
            response_tweet_id=None,
        )
        return Thread(
            thread_id="101",
            tweets=[t1, t2],
            terminal=True,
        )

    def test_mask_thread_immutability(self, sample_thread: Thread) -> None:
        original_t1_text = sample_thread.tweets[0].text
        original_t2_text = sample_thread.tweets[1].text

        masked = mask_thread(sample_thread)

        # Ensure return is a distinct instance (immutability)
        assert masked is not sample_thread
        assert masked.tweets[0] is not sample_thread.tweets[0]
        assert masked.tweets[1] is not sample_thread.tweets[1]

        # Ensure original thread was not mutated
        assert sample_thread.tweets[0].text == original_t1_text
        assert sample_thread.tweets[1].text == original_t2_text

        # Verify transformations in new thread
        assert masked.tweets[0].text == "My iPhone X on iOS 11.1 broke. See <url>"
        assert masked.tweets[1].text == "Please check <url> or call __phone__."

        # Verify metadata preserved
        assert masked.thread_id == sample_thread.thread_id
        assert masked.terminal == sample_thread.terminal
        assert len(masked.tweets) == len(sample_thread.tweets)


class TestPipelineRun:
    """Test CLI run() JSONL streaming."""

    def test_run_success(self, tmp_path: Path) -> None:
        in_file = tmp_path / "test_input.jsonl"
        out_file = tmp_path / "test_output.jsonl"

        tweet = RawTweet(
            tweet_id="201",
            author_id="cust1",
            inbound=True,
            in_response_to_tweet_id=None,
            text="Need help with iPhone 11 at https://t.co/abc123",
            created_at=datetime.now(timezone.utc),
            response_tweet_id=None,
        )
        thread = Thread(
            thread_id="201",
            tweets=[tweet],
            terminal=False,
        )

        with open(in_file, "w", encoding="utf-8") as f:
            f.write(thread.model_dump_json() + "\n")
            f.write("\n")  # Blank line to verify skipping empty lines

        processed_count = run(input_path=in_file, output_path=out_file)
        assert processed_count == 1
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 1
            result_data = json.loads(lines[0])
            assert result_data["thread_id"] == "201"
            assert result_data["tweets"][0]["text"] == "Need help with iPhone 11 at <url>"

    def test_run_file_not_found(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "nonexistent.jsonl"
        out_file = tmp_path / "out.jsonl"

        with pytest.raises(FileNotFoundError):
            run(input_path=missing_file, output_path=out_file)
