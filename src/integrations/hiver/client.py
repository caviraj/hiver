"""Hiver API Client Interface and Implementations.

Phase: M6.P6.1.F1
Provides:
- HiverClient: Abstract base class defining all 5 operations for Milestone 6.
- HiverAPIClient: Production HTTP client using httpx with exponential backoff retry.
- MockHiverClient: In-memory test client recording invocations and supporting failure simulation.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)


class HiverClient(ABC):
    """Abstract interface defining all Hiver operations required across Milestone 6."""

    @abstractmethod
    def apply_tags(self, thread_id: str, tags: Sequence[str]) -> bool:
        """Apply categorical tags to a Hiver email/ticket thread.

        Args:
            thread_id: Unique identifier for the Hiver thread.
            tags: Sequence of tag strings to apply.

        Returns:
            True if tagging succeeded, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def start_sla_timer(self, thread_id: str, duration_minutes: int) -> bool:
        """Initiate an SLA countdown timer for a thread based on severity.

        Args:
            thread_id: Unique identifier for the Hiver thread.
            duration_minutes: Duration of the SLA countdown in minutes.

        Returns:
            True if SLA timer initiation succeeded, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def lock_thread(self, thread_id: str) -> bool:
        """Acquire an agent collision detection lock on a thread.

        Args:
            thread_id: Unique identifier for the Hiver thread.

        Returns:
            True if lock was successfully acquired, False if already locked by another agent or failed.
        """
        raise NotImplementedError

    @abstractmethod
    def unlock_thread(self, thread_id: str) -> bool:
        """Release an agent collision detection lock on a thread.

        Args:
            thread_id: Unique identifier for the Hiver thread.

        Returns:
            True if lock was successfully released, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def assign_ticket(self, thread_id: str, assignee_or_tier: str) -> bool:
        """Assign a thread to an agent or functional tier queue.

        Args:
            thread_id: Unique identifier for the Hiver thread.
            assignee_or_tier: User ID, email, or functional tier identifier to assign to.

        Returns:
            True if assignment was successful, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def append_internal_note(self, thread_id: str, note: str) -> bool:
        """Append an internal note or handoff summary to a thread.

        Args:
            thread_id: Unique identifier for the Hiver thread.
            note: Plain-text content of the internal note.

        Returns:
            True if note was appended successfully, False otherwise.
        """
        raise NotImplementedError


class HiverAPIClient(HiverClient):
    """Production HTTP client for the Hiver REST API.

    ASSUMPTION NOTICE:
        Exact Hiver API request schemas and endpoint paths are assumed based on standard
        shared inbox REST conventions, as no live credentials/spec exist in this sandbox.
        Assumed endpoints:
          - POST {base_url}/threads/{thread_id}/tags -> {"tags": list(tags)}
          - POST {base_url}/threads/{thread_id}/sla -> {"duration_minutes": duration_minutes}
          - POST {base_url}/threads/{thread_id}/lock -> acquire lock
          - POST {base_url}/threads/{thread_id}/unlock -> release lock
          - POST {base_url}/threads/{thread_id}/assign -> {"assignee": assignee_or_tier}
          - POST {base_url}/threads/{thread_id}/notes -> {"note": note}
        These should be validated and adjusted against official Hiver API documentation
        when live API credentials are provisioned.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Initialize the Hiver API client.

        Args:
            api_key: Hiver API authentication token (falls back to HIVER_API_KEY env var).
            base_url: Base URL for Hiver REST endpoints (falls back to HIVER_BASE_URL or default).
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts for transient errors (5xx, 429, network).
            backoff_factor: Multiplier for exponential backoff delays.
            sleep_fn: Optional custom sleep callable (defaults to time.sleep) for unit testing.
        """
        self.api_key = api_key or os.getenv("HIVER_API_KEY", "mock_hiver_key")
        self.base_url = (base_url or os.getenv("HIVER_BASE_URL", "https://api.hiverhq.com/v1")).rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff_factor = backoff_factor
        self.sleep_fn = sleep_fn or time.sleep

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Hiver-Triage-Orchestrator/1.0",
        }

    def _execute_with_retry(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff for transient 5xx and 429 rate limits.

        Retry Schedule:
            Delay = backoff_factor * (2 ** attempt).
            HTTP 429 will attempt to inspect 'Retry-After' header if present once verified.
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        last_exception: Optional[Exception] = None

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.request(method=method, url=url, headers=headers, json=json_data)

                    # Success (2xx) or Idempotent / already tagged (e.g. 409)
                    if response.status_code in (200, 201, 204):
                        return response

                    if response.status_code in (409, 423):
                        logger.warning(
                            "Hiver returned %d for %s %s (conflict/locked/already processed): %s",
                            response.status_code,
                            method,
                            endpoint,
                            response.text,
                        )
                        return response

                    # Rate limiting (429)
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff_factor * (2**attempt)
                        logger.warning(
                            "Hiver API rate limited (HTTP 429) on attempt %d/%d for %s. Sleeping %.2fs",
                            attempt + 1,
                            self.max_retries,
                            url,
                            delay,
                        )
                        if attempt < self.max_retries - 1:
                            self.sleep_fn(delay)
                            continue
                        response.raise_for_status()

                    # Transient server errors (5xx)
                    if 500 <= response.status_code < 600:
                        delay = self.backoff_factor * (2**attempt)
                        logger.warning(
                            "Hiver API server error (HTTP %d) on attempt %d/%d for %s. Sleeping %.2fs",
                            response.status_code,
                            attempt + 1,
                            self.max_retries,
                            url,
                            delay,
                        )
                        if attempt < self.max_retries - 1:
                            self.sleep_fn(delay)
                            continue
                        response.raise_for_status()

                    # Client errors (4xx other than 409/423/429) should fail fast without retrying
                    response.raise_for_status()

                except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_exception = exc
                    delay = self.backoff_factor * (2**attempt)
                    logger.warning(
                        "Hiver API network error (%s) on attempt %d/%d for %s. Sleeping %.2fs",
                        type(exc).__name__,
                        attempt + 1,
                        self.max_retries,
                        url,
                        delay,
                    )
                    if attempt < self.max_retries - 1:
                        self.sleep_fn(delay)
                        continue
                    raise

                except httpx.HTTPStatusError as exc:
                    last_exception = exc
                    # Client errors (< 500 and != 429 and not 409/423) fail fast immediately without retrying
                    if exc.response.status_code < 500 and exc.response.status_code not in (409, 423, 429):
                        raise
                    if attempt >= self.max_retries - 1:
                        raise

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Unexpected termination in retry loop for {method} {url}")

    def apply_tags(self, thread_id: str, tags: Sequence[str]) -> bool:
        """Apply tags to a thread via assumed Hiver API endpoint."""
        endpoint = f"/threads/{thread_id}/tags"
        payload = {"tags": list(tags)}
        try:
            resp = self._execute_with_retry("POST", endpoint, json_data=payload)
            return resp.status_code in (200, 201, 204, 409)
        except Exception as exc:
            logger.error("Failed to apply tags to thread %s: %s", thread_id, exc)
            return False

    def start_sla_timer(self, thread_id: str, duration_minutes: int) -> bool:
        """Initiate SLA timer on a thread via assumed Hiver API endpoint."""
        endpoint = f"/threads/{thread_id}/sla"
        payload = {"duration_minutes": duration_minutes}
        try:
            resp = self._execute_with_retry("POST", endpoint, json_data=payload)
            return resp.status_code in (200, 201, 204)
        except Exception as exc:
            logger.error("Failed to start SLA timer for thread %s: %s", thread_id, exc)
            return False

    def lock_thread(self, thread_id: str) -> bool:
        """Acquire lock on a thread via assumed Hiver API endpoint.

        Returns:
            True if 200/201/204, False if 409/423 (collision detected) or failure.
        """
        endpoint = f"/threads/{thread_id}/lock"
        try:
            resp = self._execute_with_retry("POST", endpoint)
            return resp.status_code in (200, 201, 204)
        except Exception as exc:
            logger.error("Failed to lock thread %s: %s", thread_id, exc)
            return False

    def unlock_thread(self, thread_id: str) -> bool:
        """Release lock on a thread via assumed Hiver API endpoint.

        Returns:
            True if 200/201/204, False on failure.
        """
        endpoint = f"/threads/{thread_id}/unlock"
        try:
            resp = self._execute_with_retry("POST", endpoint)
            return resp.status_code in (200, 201, 204)
        except Exception as exc:
            logger.error("Failed to unlock thread %s: %s", thread_id, exc)
            return False

    def assign_ticket(self, thread_id: str, assignee_or_tier: str) -> bool:
        """Assign ticket to an agent or tier via Hiver API.

        Calls POST /threads/{thread_id}/assign with {"assignee": assignee_or_tier}.

        Args:
            thread_id: Unique identifier of the Hiver thread.
            assignee_or_tier: Agent identifier (email/username) or tier name.

        Returns:
            True if assignment succeeded (200/201/204), False otherwise.
        """
        endpoint = f"/threads/{thread_id}/assign"
        payload = {"assignee": assignee_or_tier}
        try:
            resp = self._execute_with_retry("POST", endpoint, json_data=payload)
            return resp.status_code in (200, 201, 204)
        except Exception as exc:
            logger.error("Failed to assign thread %s to %s: %s", thread_id, assignee_or_tier, exc)
            return False

    def append_internal_note(self, thread_id: str, note: str) -> bool:
        """Append an internal note or handoff summary to a Hiver thread via the REST API.

        Args:
            thread_id: Unique identifier of the Hiver thread.
            note: Plain-text note content to append.

        Returns:
            True if the note was appended successfully (200/201/204), False otherwise.
        """
        endpoint = f"/threads/{thread_id}/notes"
        payload = {"note": note}
        try:
            resp = self._execute_with_retry("POST", endpoint, json_data=payload)
            return resp.status_code in (200, 201, 204)
        except Exception as exc:
            logger.error("Failed to append internal note to thread %s: %s", thread_id, exc)
            return False


class MockHiverClient(HiverClient):
    """In-memory mock Hiver client recording all calls and supporting test-controlled outcomes."""

    def __init__(
        self,
        fail_tagging: bool = False,
        fail_sla: bool = False,
        fail_lock: bool = False,
        fail_unlock: bool = False,
        fail_assign: bool = False,
        fail_note: bool = False,
        tagging_exception: Optional[Exception] = None,
        sla_exception: Optional[Exception] = None,
        lock_exception: Optional[Exception] = None,
        unlock_exception: Optional[Exception] = None,
        assign_exception: Optional[Exception] = None,
        note_exception: Optional[Exception] = None,
        already_locked_threads: Optional[set[str]] = None,
    ) -> None:
        """Initialize MockHiverClient.

        Args:
            fail_tagging: If True, apply_tags returns False.
            fail_sla: If True, start_sla_timer returns False.
            fail_lock: If True, lock_thread returns False.
            fail_unlock: If True, unlock_thread returns False.
            fail_assign: If True, assign_ticket returns False.
            fail_note: If True, append_internal_note returns False.
            tagging_exception: Optional exception to raise when apply_tags is called.
            sla_exception: Optional exception to raise when start_sla_timer is called.
            lock_exception: Optional exception to raise when lock_thread is called.
            unlock_exception: Optional exception to raise when unlock_thread is called.
            assign_exception: Optional exception to raise when assign_ticket is called.
            note_exception: Optional exception to raise when append_internal_note is called.
            already_locked_threads: Optional initial set of thread_ids treated as already locked.
        """
        self.calls: List[Dict[str, Any]] = []
        self.fail_tagging = fail_tagging
        self.fail_sla = fail_sla
        self.fail_lock = fail_lock
        self.fail_unlock = fail_unlock
        self.fail_assign = fail_assign
        self.fail_note = fail_note
        self.tagging_exception = tagging_exception
        self.sla_exception = sla_exception
        self.lock_exception = lock_exception
        self.unlock_exception = unlock_exception
        self.assign_exception = assign_exception
        self.note_exception = note_exception
        self.locked_threads: set[str] = set(already_locked_threads) if already_locked_threads else set()
        self.assigned_tickets: Dict[str, str] = {}
        self.internal_notes: Dict[str, List[str]] = {}

    def apply_tags(self, thread_id: str, tags: Sequence[str]) -> bool:
        """Record call and return simulated outcome."""
        tag_list = list(tags)
        self.calls.append({
            "method": "apply_tags",
            "thread_id": thread_id,
            "tags": tag_list,
            "args": {"thread_id": thread_id, "tags": tag_list},
        })
        if self.tagging_exception:
            raise self.tagging_exception
        return not self.fail_tagging

    def start_sla_timer(self, thread_id: str, duration_minutes: int) -> bool:
        """Record call and return simulated outcome."""
        self.calls.append({
            "method": "start_sla_timer",
            "thread_id": thread_id,
            "duration_minutes": duration_minutes,
            "args": {"thread_id": thread_id, "duration_minutes": duration_minutes},
        })
        if self.sla_exception:
            raise self.sla_exception
        return not self.fail_sla

    def lock_thread(self, thread_id: str) -> bool:
        """Record call and return simulated outcome."""
        self.calls.append({
            "method": "lock_thread",
            "thread_id": thread_id,
            "args": {"thread_id": thread_id},
        })
        if self.lock_exception:
            raise self.lock_exception
        if self.fail_lock or thread_id in self.locked_threads:
            return False
        self.locked_threads.add(thread_id)
        return True

    def unlock_thread(self, thread_id: str) -> bool:
        """Record call and return simulated outcome."""
        self.calls.append({
            "method": "unlock_thread",
            "thread_id": thread_id,
            "args": {"thread_id": thread_id},
        })
        if self.unlock_exception:
            raise self.unlock_exception
        if self.fail_unlock:
            return False
        self.locked_threads.discard(thread_id)
        return True

    def assign_ticket(self, thread_id: str, assignee_or_tier: str) -> bool:
        """Record call and return simulated outcome."""
        self.calls.append({
            "method": "assign_ticket",
            "thread_id": thread_id,
            "assignee_or_tier": assignee_or_tier,
            "args": {"thread_id": thread_id, "assignee_or_tier": assignee_or_tier},
        })
        if self.assign_exception:
            raise self.assign_exception
        if self.fail_assign:
            return False
        self.assigned_tickets[thread_id] = assignee_or_tier
        return True

    def append_internal_note(self, thread_id: str, note: str) -> bool:
        """Record call and return simulated outcome."""
        self.calls.append({
            "method": "append_internal_note",
            "thread_id": thread_id,
            "note": note,
            "args": {"thread_id": thread_id, "note": note},
        })
        if self.note_exception:
            raise self.note_exception
        if self.fail_note:
            return False
        self.internal_notes.setdefault(thread_id, []).append(note)
        return True
