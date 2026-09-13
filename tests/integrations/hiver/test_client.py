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


def test_not_implemented_methods_raise_with_clear_milestone_reference() -> None:
    """Future Hiver methods must raise NotImplementedError pointing to future phases."""
    client = ConcreteHiverClient()

    with pytest.raises(NotImplementedError) as exc_lock:
        client.lock_thread("th_999")
    assert "M6.P6.2" in str(exc_lock.value)
    assert "Collision Detection" in str(exc_lock.value)

    with pytest.raises(NotImplementedError) as exc_assign:
        client.assign_ticket("th_999", "tier_1")
    assert "M6.P6.3" in str(exc_assign.value)
    assert "Ticket Assignment" in str(exc_assign.value)

    with pytest.raises(NotImplementedError) as exc_note:
        client.append_internal_note("th_999", "Customer escalated")
    assert "M6.P6.4" in str(exc_note.value)
    assert "Internal Notes" in str(exc_note.value)


def test_mock_client_inherits_not_implemented_stubs() -> None:
    """MockHiverClient inherits the same NotImplementedError stubs."""
    client = MockHiverClient()

    with pytest.raises(NotImplementedError, match="M6.P6.2"):
        client.lock_thread("th_1")
    with pytest.raises(NotImplementedError, match="M6.P6.3"):
        client.assign_ticket("th_1", "tier_2")
    with pytest.raises(NotImplementedError, match="M6.P6.4"):
        client.append_internal_note("th_1", "note")


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
