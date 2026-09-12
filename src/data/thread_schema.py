"""Pydantic schema definitions for reconstructed conversation threads."""

from typing import List
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data.schema import RawTweet


class Thread(BaseModel):
    """Pydantic model representing a reconstructed conversational thread.

    A thread consists of a sequence of tweets in chronological order,
    starting from a root tweet and progressing through conversational turns.

    Attributes
    ----------
    thread_id : str
        The tweet_id of the root tweet that initiated the conversation.
    tweets : List[RawTweet]
        The list of RawTweet objects belonging to this thread, ordered chronologically.
    terminal : bool
        True if the thread concludes with a brand response (i.e. not inbound),
        False if it ends with a dangling consumer tweet (i.e. inbound == True).
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    thread_id: str = Field(description="Root tweet_id of the conversational thread")
    tweets: List[RawTweet] = Field(
        min_length=1,
        description="Chronologically sorted sequence of tweets in the thread",
    )
    terminal: bool = Field(
        description="True if thread ends with a brand response, False if dangling consumer message"
    )

    @model_validator(mode="after")
    def validate_thread_structure(self) -> "Thread":
        """Validate thread integrity between thread_id and root tweet."""
        if not self.tweets:
            raise ValueError("A thread must contain at least one tweet.")
        if self.thread_id != self.tweets[0].tweet_id:
            raise ValueError(
                f"Thread thread_id '{self.thread_id}' does not match root tweet_id '{self.tweets[0].tweet_id}'"
            )
        return self

    @property
    def turn_count(self) -> int:
        """Return the number of turns (tweets) in the thread."""
        return len(self.tweets)

    @property
    def root_tweet(self) -> RawTweet:
        """Return the root tweet of the thread."""
        return self.tweets[0]

    @property
    def last_tweet(self) -> RawTweet:
        """Return the final tweet in the thread."""
        return self.tweets[-1]

    @property
    def participant_ids(self) -> List[str]:
        """Return deduplicated list of author IDs participating in this thread."""
        seen = set()
        participants = []
        for tweet in self.tweets:
            if tweet.author_id not in seen:
                seen.add(tweet.author_id)
                participants.append(tweet.author_id)
        return participants

    @property
    def has_brand_tweet(self) -> bool:
        """Return True if at least one tweet in the thread was authored by a brand (inbound=False)."""
        return any(not t.inbound for t in self.tweets)

    @property
    def turns(self) -> List[RawTweet]:
        """Return list of conversational turns (tweets) in the thread."""
        return self.tweets

