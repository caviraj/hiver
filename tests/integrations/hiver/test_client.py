"""Tests for HiverClient interface, HiverAPIClient, and MockHiverClient.

Phase: M6.P6.1.F1
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.integrations.hiver.client import HiverAPIClient, HiverClient, MockHiverClient


class ConcreteHiverClient(HiverClient):
    """Concrete implementation for testing default base class methods."""

    def apply_tags(self, thread_id: str, tags: list[str]) -> bool:
        return True

    def start_sla_timer(self, thread_id: str, duration_minutes: int) -> bool:
        return True

    def lock_thread(self, thread_id: str) -> bool:
        return True

    def unlock_thread(self, thread_id: str) -> bool:
        return True

    def assign_ticket(self, thread_id: str, assignee_or_tier: str) -> bool:
        return True

    def append_internal_note(self, thread_id: str, note: str) -> bool:
        return True


# ============================================================================
# MockHiverClient Tests
# ============================================================================


def test_mock_client_records_arguments() -> None:
    """MockHiverClient records call arguments accurately for verification."""
    client = MockHiverClient()

    res_tag = client.apply_tags("th_100", ["Urgent", "Billing"])
    res_sla = client.start_sla_timer("th_100", 15)

    assert res_tag is True
    assert res_sla is True
    assert len(client.calls) == 2

    assert client.calls[0] == {
        "method": "apply_tags",
        "thread_id": "th_100",
        "tags": ["Urgent", "Billing"],
        "args": {"thread_id": "th_100", "tags": ["Urgent", "Billing"]},
    }
    assert client.calls[1] == {
        "method": "start_sla_timer",
        "thread_id": "th_100",
        "duration_minutes": 15,
        "args": {"thread_id": "th_100", "duration_minutes": 15},
    }


def test_mock_client_simulated_failures() -> None:
    """MockHiverClient returns False when fail flags are set."""
    client = MockHiverClient(fail_tagging=True, fail_sla=True)

    assert client.apply_tags("th_101", ["tag1"]) is False
    assert client.start_sla_timer("th_101", 60) is False
    assert len(client.calls) == 2


def test_mock_client_exception_injection() -> None:
    """MockHiverClient raises configured exceptions."""
    client_tag_err = MockHiverClient(tagging_exception=ConnectionResetError("Socket reset"))
    with pytest.raises(ConnectionResetError, match="Socket reset"):
        client_tag_err.apply_tags("th_102", ["tag1"])

    client_sla_err = MockHiverClient(sla_exception=ValueError("Invalid SLA parameter"))
    with pytest.raises(ValueError, match="Invalid SLA parameter"):
        client_sla_err.start_sla_timer("th_102", -10)


# ============================================================================
# Interface Stub Tests (NotImplementedError for Future Milestones)
# ============================================================================


def test_assign_ticket_is_implemented_and_mock_tracks_it() -> None:
    """Assignment is a live M6.P6.3 feature rather than a future stub."""
    client = ConcreteHiverClient()
    assert client.assign_ticket("th_999", "tier_1") is True

    mock_client = MockHiverClient()
    assert mock_client.assign_ticket("th_1", "agent_1@company.com") is True
    assert mock_client.assigned_tickets["th_1"] == "agent_1@company.com"


def test_append_internal_note_is_implemented_and_mock_tracks_it() -> None:
    """Internal note appending is implemented on ConcreteHiverClient and MockHiverClient."""
    client = ConcreteHiverClient()
    assert client.append_internal_note("th_999", "Customer escalated") is True

    mock_client = MockHiverClient()
    assert mock_client.append_internal_note("th_1", "Escalation note content") is True
    assert mock_client.internal_notes["th_1"] == ["Escalation note content"]
    assert mock_client.calls[-1] == {
        "method": "append_internal_note",
        "thread_id": "th_1",
        "note": "Escalation note content",
        "args": {"thread_id": "th_1", "note": "Escalation note content"},
    }

    # Test failure simulation
    failing_mock = MockHiverClient(fail_note=True)
    assert failing_mock.append_internal_note("th_1", "note") is False

    # Test exception simulation
    error_mock = MockHiverClient(note_exception=RuntimeError("Note API failed"))
    with pytest.raises(RuntimeError, match="Note API failed"):
        error_mock.append_internal_note("th_1", "note")


# ============================================================================
# HiverAPIClient Tests
# ============================================================================


def test_hiver_api_client_init_and_headers() -> None:
    """HiverAPIClient sets up headers and base configuration."""
    client = HiverAPIClient(api_key="secret_test_key", base_url="https://api.example.com/v1/")
    assert client.base_url == "https://api.example.com/v1"
    headers = client._get_headers()
    assert headers["Authorization"] == "Bearer secret_test_key"
    assert headers["Content-Type"] == "application/json"


@patch.object(httpx.Client, "request")
def test_hiver_api_client_apply_tags_success(mock_request: MagicMock) -> None:
    """HiverAPIClient sends correct payload and parses 200 response as success."""
    mock_request.return_value = httpx.Response(
        200,
        json={"success": True},
        request=httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags"),
    )

    client = HiverAPIClient(api_key="test_key")
    result = client.apply_tags("th_123", ["Urgent", "Bug"])

    assert result is True
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "https://api.hiverhq.com/v1/threads/th_123/tags"
    assert kwargs["json"] == {"tags": ["Urgent", "Bug"]}


@patch.object(httpx.Client, "request")
def test_hiver_api_client_apply_tags_409_conflict_treated_as_idempotent(mock_request: MagicMock) -> None:
    """409 Conflict is treated as a success/idempotent response."""
    mock_request.return_value = httpx.Response(
        409,
        text="Thread already has tags",
        request=httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags"),
    )

    client = HiverAPIClient(api_key="test_key")
    assert client.apply_tags("th_123", ["Urgent"]) is True


@patch.object(httpx.Client, "request")
def test_hiver_api_client_start_sla_timer_success(mock_request: MagicMock) -> None:
    """HiverAPIClient sends correct payload for SLA timer."""
    mock_request.return_value = httpx.Response(
        201,
        json={"timer_id": "sla_456"},
        request=httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/sla"),
    )

    client = HiverAPIClient(api_key="test_key")
    result = client.start_sla_timer("th_123", 15)

    assert result is True
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "https://api.hiverhq.com/v1/threads/th_123/sla"
    assert kwargs["json"] == {"duration_minutes": 15}


@patch.object(httpx.Client, "request")
def test_hiver_api_client_retry_transient_failure_then_succeed(mock_request: MagicMock) -> None:
    """5xx transient errors trigger exponential backoff retry and succeed on subsequent attempt."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=3, backoff_factor=1.0, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/sla")
    mock_request.side_effect = [
        httpx.Response(500, text="Internal Server Error", request=req),
        httpx.Response(503, text="Service Unavailable", request=req),
        httpx.Response(200, json={"status": "ok"}, request=req),
    ]

    result = client.start_sla_timer("th_123", 60)

    assert result is True
    assert mock_request.call_count == 3
    # Retry 1: 1.0 * 2^0 = 1.0, Retry 2: 1.0 * 2^1 = 2.0
    assert sleeps == [1.0, 2.0]


@patch.object(httpx.Client, "request")
def test_hiver_api_client_retry_exhaustion_returns_false(mock_request: MagicMock) -> None:
    """When retries are exhausted on 5xx, method returns False rather than crashing."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=3, backoff_factor=0.5, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags")
    mock_request.return_value = httpx.Response(502, text="Bad Gateway", request=req)

    result = client.apply_tags("th_123", ["Urgent"])

    assert result is False
    assert mock_request.call_count == 3
    assert len(sleeps) == 2


@patch.object(httpx.Client, "request")
def test_hiver_api_client_rate_limiting_429_with_retry_after(mock_request: MagicMock) -> None:
    """HTTP 429 respects Retry-After header duration."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=2, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags")
    mock_request.side_effect = [
        httpx.Response(429, headers={"Retry-After": "7"}, text="Rate limited", request=req),
        httpx.Response(200, json={"success": True}, request=req),
    ]

    result = client.apply_tags("th_123", ["Tag1"])

    assert result is True
    assert sleeps == [7.0]


@patch.object(httpx.Client, "request")
def test_hiver_api_client_rate_limiting_429_fallback_exponential(mock_request: MagicMock) -> None:
    """HTTP 429 without valid Retry-After header falls back to exponential schedule."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=2, backoff_factor=0.5, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags")
    mock_request.side_effect = [
        httpx.Response(429, headers={}, text="Rate limited", request=req),
        httpx.Response(200, json={"success": True}, request=req),
    ]

    result = client.apply_tags("th_123", ["Tag1"])

    assert result is True
    assert sleeps == [0.5]


@patch.object(httpx.Client, "request")
def test_hiver_api_client_network_error_retry(mock_request: MagicMock) -> None:
    """Network connection errors trigger retry and backoff."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=2, backoff_factor=0.5, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags")
    mock_request.side_effect = [
        httpx.ConnectError("Network is unreachable"),
        httpx.Response(200, json={"success": True}, request=req),
    ]

    result = client.apply_tags("th_123", ["Tag1"])

    assert result is True
    assert sleeps == [0.5]


@patch.object(httpx.Client, "request")
def test_hiver_api_client_client_error_fails_fast(mock_request: MagicMock) -> None:
    """4xx client errors (e.g. 400 Bad Request) fail fast without retrying."""
    sleeps: list[float] = []
    client = HiverAPIClient(max_retries=3, sleep_fn=sleeps.append)

    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/tags")
    mock_request.return_value = httpx.Response(400, text="Bad Request: invalid tag name", request=req)

    result = client.apply_tags("th_123", ["BadTag"])

    assert result is False
    assert mock_request.call_count == 1
    assert len(sleeps) == 0


# ============================================================================
# Lock / Unlock Tests (M6.P6.2.F1)
# ============================================================================

def test_mock_hiver_client_lock_unlock_flow() -> None:
    """MockHiverClient correctly manages lock state and tracks calls."""
    client = MockHiverClient()
    assert client.lock_thread("th_1") is True
    assert "th_1" in client.locked_threads

    # Second lock on same thread indicates collision
    assert client.lock_thread("th_1") is False

    # Unlock releases thread
    assert client.unlock_thread("th_1") is True
    assert "th_1" not in client.locked_threads

    # Can re-lock after unlock
    assert client.lock_thread("th_1") is True

    # Check calls recorded
    methods = [c["method"] for c in client.calls]
    assert methods == ["lock_thread", "lock_thread", "unlock_thread", "lock_thread"]


def test_mock_hiver_client_already_locked_init() -> None:
    """MockHiverClient respects pre-locked threads passed at init."""
    client = MockHiverClient(already_locked_threads={"th_busy"})
    assert client.lock_thread("th_busy") is False
    assert client.lock_thread("th_free") is True


def test_mock_hiver_client_failure_flags_and_exceptions() -> None:
    """MockHiverClient failure flags and exceptions trigger appropriately."""
    failing_client = MockHiverClient(fail_lock=True, fail_unlock=True)
    assert failing_client.lock_thread("th_1") is False
    assert failing_client.unlock_thread("th_1") is False

    error_client = MockHiverClient(
        lock_exception=RuntimeError("Lock API down"),
        unlock_exception=ValueError("Unlock failed"),
    )
    with pytest.raises(RuntimeError, match="Lock API down"):
        error_client.lock_thread("th_1")
    with pytest.raises(ValueError, match="Unlock failed"):
        error_client.unlock_thread("th_1")


@pytest.mark.parametrize("status_code", [200, 201, 204])
@patch.object(httpx.Client, "request")
def test_hiver_api_client_lock_thread_success(mock_request: MagicMock, status_code: int) -> None:
    """HiverAPIClient lock_thread returns True on 200, 201, or 204."""
    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/lock")
    mock_request.return_value = httpx.Response(status_code, text="OK", request=req)

    client = HiverAPIClient()
    assert client.lock_thread("th_123") is True
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["url"] == "https://api.hiverhq.com/v1/threads/th_123/lock"


@pytest.mark.parametrize("status_code", [409, 423])
@patch.object(httpx.Client, "request")
def test_hiver_api_client_lock_thread_collision(mock_request: MagicMock, status_code: int) -> None:
    """HiverAPIClient lock_thread returns False on 409 Conflict or 423 Locked without retrying."""
    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/lock")
    mock_request.return_value = httpx.Response(status_code, text="Thread already locked", request=req)

    client = HiverAPIClient(max_retries=3)
    assert client.lock_thread("th_123") is False
    assert mock_request.call_count == 1


@patch.object(httpx.Client, "request")
def test_hiver_api_client_lock_thread_failure_returns_false(mock_request: MagicMock) -> None:
    """HiverAPIClient lock_thread returns False on connection error."""
    mock_request.side_effect = httpx.ConnectError("Connection refused")

    client = HiverAPIClient(max_retries=2)
    assert client.lock_thread("th_123") is False


@patch.object(httpx.Client, "request")
def test_hiver_api_client_unlock_thread_success(mock_request: MagicMock) -> None:
    """HiverAPIClient unlock_thread returns True on 200."""
    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/unlock")
    mock_request.return_value = httpx.Response(200, text="Unlocked", request=req)

    client = HiverAPIClient()
    assert client.unlock_thread("th_123") is True
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["url"] == "https://api.hiverhq.com/v1/threads/th_123/unlock"


@patch.object(httpx.Client, "request")
def test_hiver_api_client_unlock_thread_failure_returns_false(mock_request: MagicMock) -> None:
    """HiverAPIClient unlock_thread returns False on network error."""
    mock_request.side_effect = httpx.NetworkError("Network down")

    client = HiverAPIClient(max_retries=2)
    assert client.unlock_thread("th_123") is False


@pytest.mark.parametrize("status_code", [200, 201, 204])
@patch.object(httpx.Client, "request")
def test_hiver_api_client_append_internal_note_success(mock_request: MagicMock, status_code: int) -> None:
    """HiverAPIClient append_internal_note returns True on 200, 201, or 204."""
    req = httpx.Request("POST", "https://api.hiverhq.com/v1/threads/th_123/notes")
    mock_request.return_value = httpx.Response(status_code, text="Created", request=req)

    client = HiverAPIClient()
    assert client.append_internal_note("th_123", "Test note") is True
    mock_request.assert_called_once()
    _, kwargs = mock_request.call_args
    assert kwargs["url"] == "https://api.hiverhq.com/v1/threads/th_123/notes"
    assert kwargs["json"] == {"note": "Test note"}


@patch.object(httpx.Client, "request")
def test_hiver_api_client_append_internal_note_failure(mock_request: MagicMock) -> None:
    """HiverAPIClient append_internal_note returns False on network error."""
    mock_request.side_effect = httpx.NetworkError("Network down")

    client = HiverAPIClient(max_retries=2)
    assert client.append_internal_note("th_123", "Test note") is False

