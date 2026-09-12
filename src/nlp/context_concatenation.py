"""Context window concatenation module for Phase 1.2 (NLP Normalization Pipeline).

Because isolated customer queries (e.g. 'Yes I did', 'Still having the issue')
lack semantic meaning in isolation, conversational turns within reconstructed
threads are concatenated to preceding turns to provide a complete conversational
state for downstream intent classification (M2), BM25 & dense vector search (M3),
and LLM response generation (M4).
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from src.data.schema import RawTweet
from src.data.thread_schema import Thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SPEAKER_LABELS: Dict[bool, str] = {
    True: "Customer",
    False: "Agent",
}


class ContextWindowRecord(BaseModel):
    """Pydantic model representing a conversational turn enriched with prior context.

    Attributes
    ----------
    thread_id : str
        Root tweet_id of the conversational thread.
    turn_index : int
        Zero-indexed position of this turn within the thread's sequence of tweets.
    tweet_id : str
        Unique tweet_id of the current turn.
    author_id : str
        Author identifier of the current turn.
    speaker : str
        Speaker role label (e.g. 'Customer' or 'Agent').
    inbound : bool
        True if the message is inbound from a customer, False if from brand support.
    created_at : datetime
        Timestamp when the current turn was created.
    text : str
        Raw or preprocessed text of the current turn itself.
    history : List[str]
        List of formatted preceding turns included in the context window.
    context_text : str
        Concatenated text representation of preceding history turns and the current turn.
    in_response_to_tweet_id : Optional[str]
        ID of the tweet to which this turn is responding, if applicable.
    response_tweet_id : Optional[str]
        ID of response tweet(s), if applicable.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    thread_id: str = Field(description="Root tweet_id of the conversational thread")
    turn_index: int = Field(ge=0, description="0-indexed position in thread.tweets")
    tweet_id: str = Field(description="Tweet ID of the current turn")
    author_id: str = Field(description="Author ID of the current turn")
    speaker: str = Field(description="Speaker label (e.g., Customer, Agent)")
    inbound: bool = Field(description="Inbound flag (True=Customer, False=Agent)")
    created_at: datetime = Field(description="Creation timestamp of current turn")
    text: str = Field(description="Text of the current turn")
    history: List[str] = Field(
        default_factory=list,
        description="Formatted preceding conversational turns within context window",
    )
    context_text: str = Field(
        description="Full concatenated context string (history turns + current turn)"
    )
    in_response_to_tweet_id: Optional[str] = Field(
        default=None, description="Tweet ID this turn is in response to"
    )
    response_tweet_id: Optional[str] = Field(
        default=None, description="Raw response tweet ID string if any"
    )


def format_turn(
    tweet: RawTweet, speaker_labels: Optional[Dict[bool, str]] = None
) -> str:
    """Format a single tweet into a standardized speaker-labeled turn string.

    Parameters
    ----------
    tweet : RawTweet
        The tweet turn to format.
    speaker_labels : Optional[Dict[bool, str]]
        Mapping from boolean inbound flag to speaker string label.
        Defaults to {True: "Customer", False: "Agent"}.

    Returns
    -------
    str
        Formatted turn string in the format '{speaker}: {tweet.text}'.
    """
    labels = DEFAULT_SPEAKER_LABELS if speaker_labels is None else speaker_labels
    speaker = labels.get(tweet.inbound, "Customer" if tweet.inbound else "Agent")
    return f"{speaker}: {tweet.text}"


def build_turn_context(
    thread: Thread,
    turn_idx: int,
    max_history_turns: Optional[int] = None,
    speaker_labels: Optional[Dict[bool, str]] = None,
    separator: str = "\n",
) -> ContextWindowRecord:
    """Construct a ContextWindowRecord for a specific turn in a thread.

    Parameters
    ----------
    thread : Thread
        The reconstructed conversational thread containing chronologically ordered tweets.
    turn_idx : int
        Zero-indexed position of the target turn in thread.tweets.
    max_history_turns : Optional[int]
        Maximum number of preceding turns to include in history.
        - None: full history from root up to turn_idx.
        - 0: no history (history list is empty, context_text is current turn only).
        - k > 0: sliding window of up to k most recent preceding turns.
    speaker_labels : Optional[Dict[bool, str]]
        Mapping of inbound boolean to speaker label.
        Defaults to {True: "Customer", False: "Agent"}.
    separator : str
        String separator used to join history turns and the current turn.
        Defaults to '\\n'.

    Returns
    -------
    ContextWindowRecord
        Enriched record containing current turn metadata, formatted history list,
        and concatenated context_text.

    Raises
    ------
    IndexError
        If turn_idx is less than 0 or greater than or equal to the number of tweets in the thread.
    ValueError
        If max_history_turns is less than 0.
    """
    if turn_idx < 0 or turn_idx >= len(thread.tweets):
        raise IndexError(
            f"turn_idx {turn_idx} out of range for thread '{thread.thread_id}' "
            f"with {len(thread.tweets)} turns (valid: 0 to {len(thread.tweets) - 1})"
        )

    if max_history_turns is not None and max_history_turns < 0:
        raise ValueError("max_history_turns must be non-negative")

    curr_tweet = thread.tweets[turn_idx]
    labels = DEFAULT_SPEAKER_LABELS if speaker_labels is None else speaker_labels
    curr_speaker = labels.get(
        curr_tweet.inbound, "Customer" if curr_tweet.inbound else "Agent"
    )

    # Slice preceding history turns
    if max_history_turns == 0:
        history_tweets: List[RawTweet] = []
    elif max_history_turns is None:
        history_tweets = thread.tweets[:turn_idx]
    else:
        start_idx = max(0, turn_idx - max_history_turns)
        history_tweets = thread.tweets[start_idx:turn_idx]

    history = [format_turn(t, labels) for t in history_tweets]
    curr_turn_str = format_turn(curr_tweet, labels)

    if history:
        context_text = separator.join(history + [curr_turn_str])
    else:
        context_text = curr_turn_str

    return ContextWindowRecord(
        thread_id=thread.thread_id,
        turn_index=turn_idx,
        tweet_id=curr_tweet.tweet_id,
        author_id=curr_tweet.author_id,
        speaker=curr_speaker,
        inbound=curr_tweet.inbound,
        created_at=curr_tweet.created_at,
        text=curr_tweet.text,
        history=history,
        context_text=context_text,
        in_response_to_tweet_id=curr_tweet.in_response_to_tweet_id,
        response_tweet_id=curr_tweet.response_tweet_id,
    )


def build_thread_contexts(
    thread: Thread,
    max_history_turns: Optional[int] = None,
    speaker_labels: Optional[Dict[bool, str]] = None,
    separator: str = "\n",
    inbound_only: bool = False,
) -> List[ContextWindowRecord]:
    """Generate ContextWindowRecord entries for all or filtered turns in a thread.

    Parameters
    ----------
    thread : Thread
        The reconstructed conversational thread.
    max_history_turns : Optional[int]
        Maximum preceding turns to include in history window.
    speaker_labels : Optional[Dict[bool, str]]
        Mapping of inbound boolean to speaker label.
    separator : str
        String separator used to join turns in context_text.
    inbound_only : bool
        If True, only generate records for customer turns (inbound == True).
        Preceding history for each customer turn will still contain prior brand turns.
        Defaults to False.

    Returns
    -------
    List[ContextWindowRecord]
        List of generated context records in chronological turn order.
    """
    records: List[ContextWindowRecord] = []
    for idx, tweet in enumerate(thread.tweets):
        if inbound_only and not tweet.inbound:
            continue
        record = build_turn_context(
            thread=thread,
            turn_idx=idx,
            max_history_turns=max_history_turns,
            speaker_labels=speaker_labels,
            separator=separator,
        )
        records.append(record)
    return records


def concatenate_thread_text(
    thread: Thread,
    speaker_labels: Optional[Dict[bool, str]] = None,
    separator: str = "\n",
) -> str:
    """Concatenate all conversational turns in a thread into a formatted transcript.

    Parameters
    ----------
    thread : Thread
        The conversational thread.
    speaker_labels : Optional[Dict[bool, str]]
        Mapping of inbound boolean to speaker label.
    separator : str
        String separator between turns. Defaults to '\\n'.

    Returns
    -------
    str
        Complete transcript string with speaker labels.
    """
    labels = DEFAULT_SPEAKER_LABELS if speaker_labels is None else speaker_labels
    return separator.join(format_turn(t, labels) for t in thread.tweets)


def run(
    input_path: Union[
        str, Path
    ] = "data/processed/apple_support_threads_masked.jsonl",
    output_path: Union[
        str, Path
    ] = "data/processed/apple_support_context_windows.jsonl",
    max_history_turns: Optional[int] = None,
    speaker_labels: Optional[Dict[bool, str]] = None,
    separator: str = "\n",
    inbound_only: bool = False,
) -> int:
    """Stream masked threads JSONL, generate context window records, and write to output JSONL.

    Streams line-by-line to minimize peak memory consumption.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to input masked threads JSONL.
    output_path : Union[str, Path]
        Path to output context windows JSONL.
    max_history_turns : Optional[int]
        Max history turns to retain per record.
    speaker_labels : Optional[Dict[bool, str]]
        Custom speaker label mapping.
    separator : str
        Separator for context_text.
    inbound_only : bool
        Whether to emit only customer turns.

    Returns
    -------
    int
        Total number of ContextWindowRecord objects written.
    """
    src = Path(input_path)
    dest = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Input thread file not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    logger.info("Building context windows from %s -> %s", src, dest)
    with open(src, "r", encoding="utf-8") as in_f, open(
        dest, "w", encoding="utf-8"
    ) as out_f:
        for line_num, line in enumerate(in_f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                thread = Thread.model_validate_json(line_str)
                records = build_thread_contexts(
                    thread=thread,
                    max_history_turns=max_history_turns,
                    speaker_labels=speaker_labels,
                    separator=separator,
                    inbound_only=inbound_only,
                )
                for rec in records:
                    out_f.write(rec.model_dump_json() + "\n")
                    count += 1
            except Exception as err:
                logger.error(
                    "Failed processing thread at line %d: %s", line_num, err
                )
                raise

    logger.info(
        "Successfully generated %d context window records to %s", count, dest
    )
    return count


def main() -> None:
    """CLI entrypoint for context window concatenation feature."""
    parser = argparse.ArgumentParser(
        description="M1.P1.2.F3: Context Window Concatenation for conversation threads"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/apple_support_threads_masked.jsonl",
        help="Input JSONL masked threads file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/apple_support_context_windows.jsonl",
        help="Output JSONL context windows file path",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Maximum number of history turns to concatenate (default: unbounded)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n",
        help="Separator string for concatenating turns (default: \\n)",
    )
    parser.add_argument(
        "--inbound-only",
        action="store_true",
        default=False,
        help="Only output context window records for inbound (customer) turns",
    )
    args = parser.parse_args()

    try:
        run(
            input_path=args.input,
            output_path=args.output,
            max_history_turns=args.max_history,
            separator=args.separator,
            inbound_only=args.inbound_only,
        )
    except Exception as e:
        logger.error("Context concatenation pipeline failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
