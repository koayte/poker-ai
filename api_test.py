import logging
import time
from unittest.mock import Mock, patch

import pytest
import requests
from match import (
    TIME_LIMIT_SECONDS,
    AgentFailure,
    AgentFailureTracker,
    call_agent_api,
    get_match_result,
    run_api_match,
)
from match import GET_ACTION_ENDPOINT


@pytest.fixture
def mock_logger():
    return Mock(spec=logging.Logger)


@pytest.fixture
def reset_failure_tracker():
    """Use a fresh failure tracker for the test (patch the global)."""
    fresh = AgentFailureTracker()
    with patch("match.failure_tracker", fresh):
        yield fresh


@pytest.fixture(autouse=True)
def reduce_delays():
    """Temporarily reduce delays and timeouts for testing"""
    original_time_limit = TIME_LIMIT_SECONDS
    original_sleep = time.sleep

    # Patch time.sleep to be instant
    with patch("time.sleep", return_value=None):
        # Reduce the time limit for faster timeout tests
        with patch("match.TIME_LIMIT_SECONDS", 0.1):
            yield


def test_call_agent_api_success(mock_logger):
    """Test successful API call"""
    mock_response = Mock()
    mock_response.json.return_value = {"action": [0, 0]}

    with patch("requests.request", return_value=mock_response):
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result == {"action": [0, 0]}
        assert mock_logger.error.call_count == 0


def test_call_agent_api_both_players_failing(mock_logger, reset_failure_tracker):
    """Test when both players fail 3 times: 1st/2nd return None, 3rd raises; then both-failed raises."""
    with patch("requests.request", side_effect=requests.exceptions.ConnectionError):
        # Player 0: 1st and 2nd failure return None, 3rd raises AgentFailure
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result is None
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result is None
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert "Player 0 has failed" in str(exc_info.value)

        # Player 1: same
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert result is None
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert result is None
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        # Either "Player 1 has failed" or "Both players" (since 0 already at 3)
        assert "Player 1 has failed" in str(exc_info.value) or "Both players have failed" in str(exc_info.value)

        # Now both have 3 failures; next failure for either raises "Both players"
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert "Both players have failed" in str(exc_info.value)


def test_call_agent_api_single_player_failing(mock_logger, reset_failure_tracker):
    """Test when one player consistently fails: 1st/2nd return None, 3rd raises."""
    with patch("requests.request") as mock_request:
        # Make player 0's calls succeed
        mock_request.return_value = Mock(json=Mock(return_value={"action": [0, 0]}), status_code=200)
        mock_request.return_value.raise_for_status = Mock()
        call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)

        # Make player 1's calls fail (after retries)
        mock_request.side_effect = requests.exceptions.ConnectionError
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert result is None
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert result is None
        with pytest.raises(AgentFailure) as exc_info:
            call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 1)
        assert "Player 1 has failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_api_match_timeout(mock_logger):
    """Test match handling when a player exceeds time limit"""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = TimeoutError("Player 0 exceeded time limit")

        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "timeout"
        assert result["result"] == "loss"  # Player 0 timed out, so Player 1 wins


@pytest.mark.asyncio
async def test_run_api_match_both_failing(mock_logger):
    """Test match handling when both players fail"""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = AgentFailure("Both players have failed multiple times")
        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "error"
        assert "Both players have failed multiple times" in result["error"]


@pytest.mark.asyncio
async def test_run_api_match_single_failure(mock_logger):
    """Test match handling when one player fails 3 times: match ends, other team wins."""
    with patch("match.call_agent_api") as mock_call:
        mock_call.side_effect = AgentFailure("Player 1 has failed 3 times")

        result = run_api_match("http://test1", "http://test2", mock_logger)
        assert result["status"] == "timeout"
        assert result["result"] == "win"  # Player 1 failed, so Player 0 wins


def test_run_api_match_continues_on_first_failure_default_fold(mock_logger, reset_failure_tracker):
    """Test that one API failure (get_action) uses default fold and match continues to completion."""
    # CHECK = 2, valid action [type, amount, keep1, keep2]
    valid_action = {"action": [2, 0, 0, 0]}
    get_call_count = [0]

    def mock_call(method, base_url, endpoint, payload, logger, player_id):
        if endpoint == GET_ACTION_ENDPOINT:
            get_call_count[0] += 1
            if get_call_count[0] == 1:
                return None  # first get_action fails -> default fold
            return valid_action
        return None  # POST post_observation

    with patch("match.call_agent_api", side_effect=mock_call):
        with patch("match.bankrolls", [0, 0]):
            with patch("time.time", return_value=0):  # no time accumulation -> no timeout
                result = run_api_match(
                    "http://test1", "http://test2", mock_logger, num_hands=2
                )
    assert result["status"] == "completed"


def test_call_agent_api_first_failure_returns_none(mock_logger, reset_failure_tracker):
    """Test that first (and second) API failure returns None so caller can use default fold."""
    with patch("requests.request", side_effect=requests.exceptions.ConnectionError):
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result is None
        result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
        assert result is None


def test_call_agent_api_retry_success(mock_logger, reset_failure_tracker):
    """Test successful retry after temporary failures"""
    mock_response = Mock()
    mock_response.json.return_value = {"action": [0, 0]}
    mock_response.raise_for_status = Mock()

    with patch("requests.request") as mock_request:
        with patch("time.sleep", return_value=None):  # Make retries instant
            # Fail twice, then succeed
            mock_request.side_effect = [
                requests.exceptions.ConnectionError,
                requests.exceptions.ConnectionError,
                mock_response,
            ]

            result = call_agent_api("GET", "http://test", "/endpoint", {}, mock_logger, 0)
            assert result == {"action": [0, 0]}
            assert mock_logger.error.call_count == 0


def test_match_result_format():
    """Test that get_match_result produces correctly formatted results"""
    test_cases = [
        {
            "name": "normal completion with win",
            "inputs": {"status": "completed", "rewards": (100, 50)},
            "expected": {
                "status": "completed",
                "result": "win",  # Player 0 won
                "bot0_reward": 100,
                "bot0_time_used": 0,
                "bot1_reward": 50,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "normal completion with loss",
            "inputs": {"status": "completed", "rewards": (50, 100)},
            "expected": {
                "status": "completed",
                "result": "loss",  # Player 1 won
                "bot0_reward": 50,
                "bot0_time_used": 0,
                "bot1_reward": 100,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "normal completion with tie",
            "inputs": {"status": "completed", "rewards": (100, 100)},
            "expected": {
                "status": "completed",
                "result": "tie",
                "bot0_reward": 100,
                "bot0_time_used": 0,
                "bot1_reward": 100,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "timeout",
            "inputs": {"status": "timeout", "winner": 0},
            "expected": {
                "status": "timeout",
                "result": "win",  # Player 0 won due to timeout
                "bot0_time_used": 0,
                "bot1_time_used": 0,
            },
        },
        {
            "name": "error",
            "inputs": {"status": "error", "error": "Test error"},
            "expected": {
                "status": "error",
                "result": "invalid",
                "error": "Test error",
                "bot0_time_used": 0,
                "bot1_time_used": 0,
            },
        },
    ]

    for case in test_cases:
        result = get_match_result(**case["inputs"])
        assert result == case["expected"], f"Case '{case['name']}' failed: expected {case['expected']}, got {result}"
