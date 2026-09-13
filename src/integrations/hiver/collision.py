"""Thread collision detection and draft locking for Hiver integration.

Per PRD: While the AI drafts a response, Hiver's collision mechanism locks
the thread to prevent human agents from duplicating effort or sending
conflicting responses.

This module provides:
- File-backed local draft-lock registry (data/state/active_locks.json) with atomic writes
- acquire_draft_lock / release_draft_lock functions with local pre-flight checks
- draft_lock_context context manager ensuring lock release even on LLM/runtime exceptions
- Stale lock detection and force-release utilities for background recovery
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from src.integrations.hiver.client import HiverClient

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/state/active_locks.json")
DEFAULT_STALE_TIMEOUT_MINUTES = 30.0


def _resolve_path(registry_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the registry path, defaulting to DEFAULT_REGISTRY_PATH."""
    if registry_path is None:
        return DEFAULT_REGISTRY_PATH
    return Path(registry_path)


def _read_registry(path: Path) -> Dict[str, Any]:
    """Read the local locks registry file.

    Returns an empty dict if the file does not exist or contains invalid JSON.
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            data = json.loads(content)
            if isinstance(data, dict):
                return data
            logger.warning("Lock registry %s is not a JSON object. Resetting to empty.", path)
            return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Corrupted or unreadable lock registry at %s (%s). Starting with empty state.",
            path,
            exc,
        )
        return {}


def _write_registry(path: Path, data: Dict[str, Any]) -> None:
    """Write the registry atomically via a temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        logger.error("Failed to write lock registry to %s: %s", path, exc)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def get_local_lock_age_minutes(
    thread_id: str,
    registry_path: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """Get the age in minutes of a local lock, or None if not locked."""
    path = _resolve_path(registry_path)
    data = _read_registry(path)
    entry = data.get(thread_id)
    if not entry or not isinstance(entry, dict):
        return None

    acquired_at_str = entry.get("acquired_at")
    if not acquired_at_str:
        return None

    try:
        acquired_at = datetime.fromisoformat(acquired_at_str)
        if acquired_at.tzinfo is None:
            acquired_at = acquired_at.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (now - acquired_at).total_seconds() / 60.0)
    except (ValueError, TypeError):
        return None


def register_local_lock(
    thread_id: str,
    acquired_at: Optional[datetime] = None,
    registry_path: Optional[Union[str, Path]] = None,
    agent_id: str = "agent_ai",
) -> None:
    """Record an acquired thread lock in the local registry."""
    path = _resolve_path(registry_path)
    data = _read_registry(path)

    dt = acquired_at or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    data[thread_id] = {
        "thread_id": thread_id,
        "acquired_at": dt.isoformat(),
        "agent_id": agent_id,
    }
    _write_registry(path, data)
    logger.debug("Registered local lock for thread %s (agent: %s)", thread_id, agent_id)


def deregister_local_lock(
    thread_id: str,
    registry_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Remove a thread lock from the local registry."""
    path = _resolve_path(registry_path)
    data = _read_registry(path)
    if thread_id in data:
        del data[thread_id]
        _write_registry(path, data)
        logger.debug("Deregistered local lock for thread %s", thread_id)
        return True
    return False


def acquire_draft_lock(
    thread_id: str,
    client: HiverClient,
    registry_path: Optional[Union[str, Path]] = None,
    agent_id: str = "agent_ai",
) -> bool:
    """Acquire a draft lock for a thread.

    Checks local registry first to avoid redundant remote calls.
    Then attempts remote lock on Hiver API via client.lock_thread().
    If acquired, persists to the local registry.
    Returns True if lock acquired, False if collision detected or failed.
    """
    path = _resolve_path(registry_path)
    data = _read_registry(path)

    # Local pre-flight check: thread already locked locally by this agent/process
    if thread_id in data:
        logger.info(
            "Draft lock for thread %s is already held locally (agent: %s). Avoiding collision.",
            thread_id,
            data[thread_id].get("agent_id", "unknown"),
        )
        return False

    # Attempt remote lock on Hiver
    try:
        remote_success = client.lock_thread(thread_id)
    except Exception as exc:
        logger.error("Exception occurred while acquiring lock for thread %s: %s", thread_id, exc)
        return False

    if not remote_success:
        logger.info(
            "Collision detected: Thread %s is currently locked by a human agent or another worker.",
            thread_id,
        )
        return False

    # Lock confirmed on remote API, record locally
    register_local_lock(thread_id, registry_path=path, agent_id=agent_id)
    logger.info("Successfully acquired draft lock for thread %s.", thread_id)
    return True


def release_draft_lock(
    thread_id: str,
    client: HiverClient,
    registry_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Release a draft lock for a thread.

    Calls client.unlock_thread(). If successful, removes the lock from the
    local registry. If unlock fails or raises, logs a CRITICAL alert and
    RETAINS the lock in the local registry so recovery processes can clean it up.
    """
    path = _resolve_path(registry_path)

    try:
        remote_success = client.unlock_thread(thread_id)
    except Exception as exc:
        logger.critical(
            "CRITICAL: Exception raised while releasing lock on Hiver API for thread %s: %s. "
            "Local lock entry retained at %s for recovery.",
            thread_id,
            exc,
            path,
        )
        return False

    if not remote_success:
        logger.critical(
            "CRITICAL: Failed to release draft lock on Hiver API for thread %s! "
            "Local lock entry retained at %s for recovery.",
            thread_id,
            path,
        )
        return False

    deregister_local_lock(thread_id, registry_path=path)
    logger.info("Successfully released draft lock for thread %s.", thread_id)
    return True


@contextmanager
def draft_lock_context(
    thread_id: str,
    client: HiverClient,
    registry_path: Optional[Union[str, Path]] = None,
    agent_id: str = "agent_ai",
) -> Generator[bool, None, None]:
    """Context manager for acquiring and safely releasing a draft lock.

    Usage:
        with draft_lock_context(thread_id, client) as acquired:
            if not acquired:
                logger.info("Skipping draft, collision detected.")
                return
            # Draft response safely...

    Guarantees that release_draft_lock is called in a finally block if the
    lock was successfully acquired, protecting against unhandled exceptions
    (e.g., LLM generation timeout or runtime crash).
    """
    acquired = False
    try:
        acquired = acquire_draft_lock(
            thread_id=thread_id,
            client=client,
            registry_path=registry_path,
            agent_id=agent_id,
        )
        yield acquired
    finally:
        if acquired:
            release_draft_lock(
                thread_id=thread_id,
                client=client,
                registry_path=registry_path,
            )


def list_stale_locks(
    max_age_minutes: float = DEFAULT_STALE_TIMEOUT_MINUTES,
    registry_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Return a list of thread_ids whose locks exceed the given age threshold in minutes."""
    path = _resolve_path(registry_path)
    data = _read_registry(path)
    stale_threads: List[str] = []

    now = datetime.now(timezone.utc)
    for thread_id, info in data.items():
        if not isinstance(info, dict):
            continue
        acquired_at_str = info.get("acquired_at")
        if not acquired_at_str:
            stale_threads.append(thread_id)
            continue
        try:
            acquired_at = datetime.fromisoformat(acquired_at_str)
            if acquired_at.tzinfo is None:
                acquired_at = acquired_at.replace(tzinfo=timezone.utc)
            age_minutes = (now - acquired_at).total_seconds() / 60.0
            if age_minutes >= max_age_minutes:
                stale_threads.append(thread_id)
        except (ValueError, TypeError):
            stale_threads.append(thread_id)

    return stale_threads


def force_release_stale_locks(
    max_age_minutes: float = DEFAULT_STALE_TIMEOUT_MINUTES,
    client: Optional[HiverClient] = None,
    registry_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Purge stale locks exceeding max_age_minutes.

    If a client is provided, attempts to unlock them on Hiver.
    Regardless of remote result, removes stale entries from the local registry.
    Returns the list of thread_ids that were force-released.
    """
    path = _resolve_path(registry_path)
    stale_threads = list_stale_locks(max_age_minutes=max_age_minutes, registry_path=path)
    if not stale_threads:
        return []

    released: List[str] = []
    for thread_id in stale_threads:
        if client is not None:
            try:
                client.unlock_thread(thread_id)
            except Exception as exc:
                logger.warning("Failed to unlock stale thread %s on remote client: %s", thread_id, exc)

        deregister_local_lock(thread_id, registry_path=path)
        released.append(thread_id)
        logger.warning(
            "Force-released stale draft lock for thread %s (exceeded %.1f min threshold)",
            thread_id,
            max_age_minutes,
        )

    return released
