"""Conversation graph construction and thread reconstruction module.

This module parses relational tweet links (in_response_to_tweet_id, response_tweet_id),
builds an in-memory adjacency graph, resolves orphans and cycles, and traverses the
graph to assemble chronologically-ordered, non-branching conversational threads.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import pandas as pd

from src.data.schema import RawTweet
from src.data.thread_schema import Thread

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the conversational graph.

    Attributes
    ----------
    tweet : RawTweet
        The tweet payload and metadata.
    parent_id : Optional[str]
        The tweet_id of the tweet this node responds to, if any.
    children_ids : List[str]
        List of tweet_ids representing direct replies to this tweet.
    """

    tweet: RawTweet
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


def parse_response_ids(raw: Optional[Union[str, float]]) -> List[str]:
    """Parse a raw response_tweet_id field into clean, stripped, deduplicated IDs.

    Handles comma-separated strings (e.g. '123, 456'), single IDs ('123'),
    whitespace padding, None, NaN, and empty strings. Preserves first-seen order.

    Parameters
    ----------
    raw : Optional[Union[str, float]]
        Raw string or NaN value from response_tweet_id column.

    Returns
    -------
    List[str]
        Clean list of stripped, unique tweet ID strings.
    """
    if raw is None or pd.isna(raw):
        return []

    raw_str = str(raw).strip()
    if not raw_str or raw_str.lower() == "nan":
        return []

    tokens = [part.strip() for part in raw_str.split(",") if part.strip()]
    seen: Set[str] = set()
    deduped: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)

    return deduped


def build_graph(df: pd.DataFrame) -> Dict[str, GraphNode]:
    """Build an in-memory conversation graph from a DataFrame of tweets.

    Instantiates GraphNode objects with validated parent and children links,
    and audits bidirectional consistency between parent and child references.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing tweet records with tweet_id, author_id, inbound,
        created_at, text, and optional response_tweet_id / in_response_to_tweet_id.

    Returns
    -------
    Dict[str, GraphNode]
        Adjacency mapping from tweet_id to GraphNode.
    """
    graph: Dict[str, GraphNode] = {}

    # 1. Populate initial nodes and explicit parent/child declarations
    for _, row in df.iterrows():
        try:
            tweet = RawTweet.model_validate(row.to_dict())
        except Exception as err:
            logger.warning("Skipping invalid tweet row during graph build: %s", err)
            continue

        tweet_id = tweet.tweet_id
        parent_id = str(tweet.in_response_to_tweet_id).strip() if tweet.in_response_to_tweet_id else None
        if parent_id and (parent_id.lower() == "nan" or not parent_id):
            parent_id = None

        declared_children = parse_response_ids(tweet.response_tweet_id)
        graph[tweet_id] = GraphNode(
            tweet=tweet,
            parent_id=parent_id,
            children_ids=declared_children,
        )

    # 2. Audit bidirectional agreement and reconcile child links
    # If A lists B as child, verify B lists A as parent
    for tweet_id, node in list(graph.items()):
        for child_id in list(node.children_ids):
            if child_id in graph:
                child_node = graph[child_id]
                if child_node.parent_id != tweet_id:
                    logger.warning(
                        "Bidirectional disagreement: Tweet %s declares child %s, "
                        "but Tweet %s has parent %s",
                        tweet_id,
                        child_id,
                        child_id,
                        child_node.parent_id,
                    )
            else:
                logger.debug(
                    "Tweet %s references child %s not present in current dataset.",
                    tweet_id,
                    child_id,
                )

    # If B lists A as parent, verify B is in A's children_ids
    for tweet_id, node in list(graph.items()):
        if node.parent_id:
            parent_id = node.parent_id
            if parent_id in graph:
                parent_node = graph[parent_id]
                if tweet_id not in parent_node.children_ids:
                    logger.warning(
                        "Bidirectional disagreement: Tweet %s declares parent %s, "
                        "but parent %s does not list %s in its children (%s). Adding link.",
                        tweet_id,
                        parent_id,
                        parent_id,
                        tweet_id,
                        parent_node.children_ids,
                    )
                    parent_node.children_ids.append(tweet_id)

    return graph


def reconstruct_threads(
    graph: Dict[str, GraphNode],
    brand_handle: Optional[str] = "AppleSupport",
) -> List[Thread]:
    """Reconstruct conversational threads by traversing from root nodes to leaves.

    Handles edge cases:
    - Drops orphaned nodes (replies whose parent is absent from graph).
    - Detects and breaks traversal cycles without infinite looping.
    - Splits branching threads into distinct prefix-sharing Thread objects per leaf.
    - Drops threads containing zero brand-authored nodes.
    - Warns on threads exceeding 20 turns.
    - Sorts each thread chronologically by created_at, using tweet_id as tiebreaker.
    - Computes terminal boolean (True if ending with brand turn, False if consumer).

    Parameters
    ----------
    graph : Dict[str, GraphNode]
        Adjacency mapping from tweet_id to GraphNode.
    brand_handle : Optional[str], default "AppleSupport"
        Brand handle to identify brand-authored nodes.

    Returns
    -------
    List[Thread]
        List of validated Thread objects.
    """
    if not graph:
        return []

    # 1. Identify roots and detect orphaned nodes
    roots: List[str] = []
    orphaned_count = 0

    for tweet_id, node in graph.items():
        if not node.parent_id:
            # True root: initiated conversation without in_response_to_tweet_id
            roots.append(tweet_id)
        elif node.parent_id not in graph:
            # Orphan: reply pointing to a tweet not in the dataset
            orphaned_count += 1
            logger.debug(
                "Orphaned tweet detected: %s (in_response_to_tweet_id=%s missing from dataset)",
                tweet_id,
                node.parent_id,
            )

    if orphaned_count > 0:
        logger.warning(
            "Dropped %d orphaned nodes whose parent tweet was not found in dataset.",
            orphaned_count,
        )

    # 2. Depth-First Traversal to collect paths from each root to all reachable leaves
    assembled_threads: List[Thread] = []

    def dfs_traverse(current_id: str, current_path: List[str], visited_set: Set[str]) -> None:
        if current_id in visited_set:
            logger.warning(
                "Cycle detected in conversation graph involving tweet %s (path: %s). Breaking cycle.",
                current_id,
                current_path,
            )
            # Finish thread at the current node before cyclic edge
            finalize_path(current_path)
            return

        node = graph.get(current_id)
        if not node:
            return

        new_path = current_path + [current_id]
        new_visited = visited_set | {current_id}

        # Filter valid children present in graph
        valid_children = [
            cid for cid in node.children_ids
            if cid in graph and cid != current_id  # self-loop prevention
        ]

        if not valid_children:
            # Leaf reached
            finalize_path(new_path)
            return

        # Branching: fork into separate paths for each child
        for child_id in valid_children:
            if child_id in new_visited:
                logger.warning(
                    "Cycle detected between tweet %s and child %s. Breaking cycle.",
                    current_id,
                    child_id,
                )
                finalize_path(new_path)
            else:
                dfs_traverse(child_id, new_path, new_visited)

    def finalize_path(path_ids: List[str]) -> None:
        if not path_ids:
            return

        # Fetch RawTweet objects
        thread_tweets = [graph[tid].tweet for tid in path_ids if tid in graph]
        if not thread_tweets:
            return

        # Filter out threads with zero brand-authored nodes
        has_brand = any(
            (not t.inbound) or (brand_handle and t.author_id.lower() == brand_handle.lower())
            for t in thread_tweets
        )
        if not has_brand:
            logger.debug("Dropping consumer-only thread with root %s (0 brand nodes)", path_ids[0])
            return

        # Chronological sort: created_at ascending, tweet_id ascending as tiebreaker
        sorted_tweets = sorted(
            thread_tweets,
            key=lambda t: (t.created_at, str(t.tweet_id)),
        )

        turn_count = len(sorted_tweets)
        if turn_count > 20:
            logger.warning(
                "Thread %s has high turn count (%d turns, exceeding 20-turn threshold)",
                path_ids[0],
                turn_count,
            )

        # Terminal status: True if final tweet is from brand, False if consumer
        last_tweet = sorted_tweets[-1]
        is_terminal = (not last_tweet.inbound) or (
            brand_handle is not None and last_tweet.author_id.lower() == brand_handle.lower()
        )

        try:
            thread_obj = Thread(
                thread_id=sorted_tweets[0].tweet_id,
                tweets=sorted_tweets,
                terminal=is_terminal,
            )
            assembled_threads.append(thread_obj)
        except Exception as err:
            logger.error("Failed to construct Thread object for root %s: %s", path_ids[0], err)

    # Run DFS traversal starting from each valid root
    for root_id in roots:
        dfs_traverse(root_id, [], set())

    logger.info("Reconstructed %d valid conversation threads.", len(assembled_threads))
    return assembled_threads


def save_threads_to_jsonl(threads: List[Thread], output_path: Union[str, Path]) -> None:
    """Serialize a list of Thread objects to a JSON Lines file.

    Parameters
    ----------
    threads : List[Thread]
        List of Thread models to persist.
    output_path : Union[str, Path]
        Destination .jsonl file path.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as f:
        for thread in threads:
            f.write(thread.model_dump_json() + "\n")

    logger.info("Saved %d threads to %s", len(threads), dest)


def load_threads_from_jsonl(input_path: Union[str, Path]) -> List[Thread]:
    """Deserialize a list of Thread objects from a JSON Lines file.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to existing .jsonl file.

    Returns
    -------
    List[Thread]
        List of deserialized and validated Thread objects.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Thread file not found: {src}")

    threads: List[Thread] = []
    with open(src, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                thread = Thread.model_validate_json(line_str)
                threads.append(thread)
            except Exception as err:
                logger.error("Error parsing JSONL line %d in %s: %s", line_num, src, err)
                raise

    logger.info("Loaded %d threads from %s", len(threads), src)
    return threads
