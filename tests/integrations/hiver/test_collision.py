"""Tests for thread collision detection and draft locking (M6.P6.2.F1).

Covers:
- Local draft-lock file-backed registry (read, atomic write, corruption handling)
- Local lock age calculation and registration / deregistration
- acquire_draft_lock / release_draft_lock with remote MockHiverClient & local registry sync
- Collision avoidance (remote 409/423 and local pre-flight)
- Unlock failure handling with CRITICAL log retention
- draft_lock_context context manager (success, collision, exception in block)
- list_stale_locks & force_release_stale_locks
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.integrations.hiver.client import MockHiverClient
from src.integrations.hiver.collision import (
    _read_registry,
    _resolve_path,
    _write_registry,
    acquire_draft_lock,
    deregister_local_lock,
    draft_lock_context,
    force_release_stale_locks,
    get_local_lock_age_minutes,
    list_stale_locks,
    register_local_lock,
    release_draft_lock,
)


# ============================================================================
# Registry Storage & File I/O Tests
# ============================================================================


def test_resolve_path_defaults_and_custom(tmp_path: Path) -> None:
    """_resolve_path defaults to standard path or uses custom path."""
    default_p = _resolve_path(None)
    assert default_p == Path("data/state/active_locks.json")

    custom_p = tmp_path / "locks.json"
    assert _resolve_path(custom_p) == custom_p
    assert _resolve_path(str(custom_p)) == custom_p


def test_read_registry_empty_or_nonexistent(tmp_path: Path) -> None:
    """Non-existent or empty file returns an empty dictionary."""
    reg_file = tmp_path / "locks.json"
    assert _read_registry(reg_file) == {}

    reg_file.write_text("", encoding="utf-8")
    assert _read_registry(reg_file) == {}

    reg_file.write_text("   \n", encoding="utf-8")
    assert _read_registry(reg_file) == {}


def test_read_registry_corrupted_or_invalid_type(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Corrupted JSON or non-dict root returns an empty dictionary and logs a warning."""
    reg_file = tmp_path / "corrupt.json"

    # Corrupt JSON
    reg_file.write_text("{not: json}", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        data = _read_registry(reg_file)
    assert data == {}
    assert "Corrupted or unreadable lock registry" in caplog.text

    caplog.clear()

    # Valid JSON but a list, not a dict
    reg_file.write_text("[1, 2, 3]", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        data = _read_registry(reg_file)
    assert data == {}
    assert "is not a JSON object" in caplog.text


def test_write_registry_atomic_creation(tmp_path: Path) -> None:
    """_write_registry creates directories and writes valid JSON atomically."""
    reg_file = tmp_path / "nested" / "dir" / "locks.json"
    sample_data = {"th_1": {"thread_id": "th_1", "agent_id": "agent_ai"}}

    _write_registry(reg_file, sample_data)

    assert reg_file.exists()
    # Ensure temporary file is cleaned up
    temp_file = reg_file.with_suffix(reg_file.suffix + ".tmp")
    assert not temp_file.exists()

    with open(reg_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == sample_data


# ============================================================================
# Local Lock Registration & Age Tests
# ============================================================================


def test_register_and_deregister_local_lock(tmp_path: Path) -> None:
    """Can register lock and deregister it cleanly."""
    reg_file = tmp_path / "locks.json"

    fixed_time = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
    register_local_lock("th_123", acquired_at=fixed_time, registry_path=reg_file, agent_id="agent_1")

    data = _read_registry(reg_file)
    assert "th_123" in data
    assert data["th_123"]["agent_id"] == "agent_1"
    assert data["th_123"]["acquired_at"] == fixed_time.isoformat()

    # Deregister
    assert deregister_local_lock("th_123", registry_path=reg_file) is True
    assert _read_registry(reg_file) == {}

    # Deregister non-existent returns False
    assert deregister_local_lock("th_unknown", registry_path=reg_file) is False


def test_get_local_lock_age_minutes(tmp_path: Path) -> None:
    """get_local_lock_age_minutes calculates elapsed time accurately."""
    reg_file = tmp_path / "locks.json"

    # Not locked
    assert get_local_lock_age_minutes("th_none", registry_path=reg_file) is None

    # Locked 15 minutes ago
    ten_min_ago = datetime.now(timezone.utc) - timedelta(minutes=15)
    register_local_lock("th_1", acquired_at=ten_min_ago, registry_path=reg_file)

    age = get_local_lock_age_minutes("th_1", registry_path=reg_file)
    assert age is not None
    assert 14.8 <= age <= 15.5

    # Malformed timestamp
    _write_registry(reg_file, {"th_bad": {"acquired_at": "invalid_date"}})
    assert get_local_lock_age_minutes("th_bad", registry_path=reg_file) is None


# ============================================================================
# acquire_draft_lock & Collision Avoidance Tests
# ============================================================================


def test_acquire_draft_lock_success(tmp_path: Path) -> None:
    """Successfully acquires lock remotely and registers in local file."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    success = acquire_draft_lock("th_abc", client=client, registry_path=reg_file, agent_id="ai_worker_1")
    assert success is True

    # Client recorded the call
    assert "th_abc" in client.locked_threads
    assert any(c["method"] == "lock_thread" for c in client.calls)

    # Local registry holds the lock
    data = _read_registry(reg_file)
    assert "th_abc" in data
    assert data["th_abc"]["agent_id"] == "ai_worker_1"


def test_acquire_draft_lock_remote_collision(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Remote lock returns False (human collision) -> returns False and does not register locally."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient(already_locked_threads={"th_busy"})

    with caplog.at_level(logging.INFO):
        success = acquire_draft_lock("th_busy", client=client, registry_path=reg_file)

    assert success is False
    assert _read_registry(reg_file) == {}
    assert "Collision detected" in caplog.text


def test_acquire_draft_lock_remote_exception_handled(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Client raises exception during lock -> returns False and logs error."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient(lock_exception=ConnectionError("Hiver timeout"))

    with caplog.at_level(logging.ERROR):
        success = acquire_draft_lock("th_err", client=client, registry_path=reg_file)

    assert success is False
    assert _read_registry(reg_file) == {}
    assert "Exception occurred while acquiring lock" in caplog.text


def test_acquire_draft_lock_local_preflight_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """If thread is already registered in local file, returns False without remote call."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    # First acquire
    assert acquire_draft_lock("th_dup", client=client, registry_path=reg_file) is True
    assert len(client.calls) == 1

    # Second acquire for same thread
    with caplog.at_level(logging.INFO):
        assert acquire_draft_lock("th_dup", client=client, registry_path=reg_file) is False

    # Client was NOT called a second time
    assert len(client.calls) == 1
    assert "already held locally" in caplog.text


# ============================================================================
# release_draft_lock & Unlock Failure Tests
# ============================================================================


def test_release_draft_lock_success(tmp_path: Path) -> None:
    """Successfully releases lock remotely and removes from local registry."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    # Acquire first
    assert acquire_draft_lock("th_rel", client=client, registry_path=reg_file) is True
    assert "th_rel" in client.locked_threads

    # Release
    assert release_draft_lock("th_rel", client=client, registry_path=reg_file) is True
    assert "th_rel" not in client.locked_threads
    assert _read_registry(reg_file) == {}


def test_release_draft_lock_remote_failure_retains_local_and_logs_critical(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Remote unlock fails -> logs CRITICAL and retains local registry entry for recovery."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient(fail_unlock=True)

    # Register local lock directly
    register_local_lock("th_fail", registry_path=reg_file)

    with caplog.at_level(logging.CRITICAL):
        result = release_draft_lock("th_fail", client=client, registry_path=reg_file)

    assert result is False
    # CRITICAL log was emitted
    assert "CRITICAL: Failed to release draft lock on Hiver API" in caplog.text
    # Local lock entry MUST be retained for recovery
    data = _read_registry(reg_file)
    assert "th_fail" in data


def test_release_draft_lock_remote_exception_retains_local_and_logs_critical(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Remote unlock raises exception -> logs CRITICAL and retains local registry entry."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient(unlock_exception=RuntimeError("Gateway crash"))

    register_local_lock("th_exc", registry_path=reg_file)

    with caplog.at_level(logging.CRITICAL):
        result = release_draft_lock("th_exc", client=client, registry_path=reg_file)

    assert result is False
    assert "CRITICAL: Exception raised while releasing lock" in caplog.text
    data = _read_registry(reg_file)
    assert "th_exc" in data


# ============================================================================
# draft_lock_context Context Manager Tests
# ============================================================================


def test_draft_lock_context_normal_completion(tmp_path: Path) -> None:
    """draft_lock_context acquires on entry and automatically releases on exit."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    with draft_lock_context("th_ctx_1", client=client, registry_path=reg_file) as acquired:
        assert acquired is True
        assert "th_ctx_1" in client.locked_threads
        assert "th_ctx_1" in _read_registry(reg_file)

    # Automatically unlocked on exit
    assert "th_ctx_1" not in client.locked_threads
    assert _read_registry(reg_file) == {}


def test_draft_lock_context_collision_skips_unlock(tmp_path: Path) -> None:
    """When collision occurs on entry, context manager yields False and doesn't call unlock on exit."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient(already_locked_threads={"th_busy"})

    with draft_lock_context("th_busy", client=client, registry_path=reg_file) as acquired:
        assert acquired is False

    # Unlock was NEVER attempted
    unlock_calls = [c for c in client.calls if c["method"] == "unlock_thread"]
    assert len(unlock_calls) == 0


def test_draft_lock_context_releases_on_exception(tmp_path: Path) -> None:
    """draft_lock_context ensures release even when an exception is raised inside the block."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    with pytest.raises(ValueError, match="LLM draft timeout"):
        with draft_lock_context("th_exc_block", client=client, registry_path=reg_file) as acquired:
            assert acquired is True
            assert "th_exc_block" in client.locked_threads
            raise ValueError("LLM draft timeout")

    # Verified lock is released after exception propagated
    assert "th_exc_block" not in client.locked_threads
    assert _read_registry(reg_file) == {}


# ============================================================================
# Stale Lock Detection and Force Release Tests
# ============================================================================


def test_list_and_force_release_stale_locks(tmp_path: Path) -> None:
    """Detects stale locks older than threshold and purges them with optional remote client unlock."""
    reg_file = tmp_path / "locks.json"
    client = MockHiverClient()

    now = datetime.now(timezone.utc)
    # Active lock (5 min old)
    register_local_lock("th_active", acquired_at=now - timedelta(minutes=5), registry_path=reg_file)
    # Stale lock (45 min old)
    register_local_lock("th_stale_1", acquired_at=now - timedelta(minutes=45), registry_path=reg_file)
    # Stale lock (60 min old)
    register_local_lock("th_stale_2", acquired_at=now - timedelta(minutes=60), registry_path=reg_file)
    # Corrupt entry without timestamp
    data = _read_registry(reg_file)
    data["th_corrupt"] = {"thread_id": "th_corrupt"}
    _write_registry(reg_file, data)

    # Lock them in mock client too
    client.lock_thread("th_active")
    client.lock_thread("th_stale_1")
    client.lock_thread("th_stale_2")

    # List stale locks (> 30 min)
    stale = list_stale_locks(max_age_minutes=30.0, registry_path=reg_file)
    assert set(stale) == {"th_stale_1", "th_stale_2", "th_corrupt"}

    # Force release
    released = force_release_stale_locks(max_age_minutes=30.0, client=client, registry_path=reg_file)
    assert set(released) == {"th_stale_1", "th_stale_2", "th_corrupt"}

    # Active lock remains intact locally and on client
    remaining = _read_registry(reg_file)
    assert "th_active" in remaining
    assert "th_stale_1" not in remaining
    assert "th_stale_2" not in remaining
    assert "th_corrupt" not in remaining

    assert "th_active" in client.locked_threads
    assert "th_stale_1" not in client.locked_threads
    assert "th_stale_2" not in client.locked_threads


def test_force_release_stale_locks_without_client(tmp_path: Path) -> None:
    """force_release_stale_locks works when client is None (local cleanup only)."""
    reg_file = tmp_path / "locks.json"
    now = datetime.now(timezone.utc)
    register_local_lock("th_old", acquired_at=now - timedelta(minutes=40), registry_path=reg_file)

    released = force_release_stale_locks(max_age_minutes=30.0, client=None, registry_path=reg_file)
    assert released == ["th_old"]
    assert _read_registry(reg_file) == {}
