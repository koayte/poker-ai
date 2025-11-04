"""
Script to run matches between agents.
"""

import csv
import json
import logging
import time
import traceback
from typing import Any, Dict, Optional, Tuple, List, TypedDict
from collections import defaultdict
from deprecated import deprecated

import numpy as np
import requests

from gym_env import PokerEnv

NUM_PLAYERS = 6
TIME_LIMIT_SECONDS = 1500
GET_ACTION_ENDPOINT = "/get_action"
SEND_OBS_ENDPOINT = "/post_observation"
MAX_AGENT_FAILURES = 3

class MatchResult(TypedDict):
    """Standardized match result structure"""
    status: str  # 'completed', 'timeout', or 'error'
    result: str  # 'win', 'loss', 'tie', or 'invalid'
    rewards: List[float]
    time_used: List[float]
    error: Optional[str] = None

class AgentFailure(Exception):
    """Custom exception for tracking agent failures"""

    pass

class AgentFailureTracker:
    def __init__(self):
        self.failed_attempts = {i: 0 for i in range(NUM_PLAYERS)}

    def record_failure(self, player_id: int):
        self.failed_attempts[player_id] += 1

        # Check for all players failing
        all_failed = all(attempts > MAX_AGENT_FAILURES for attempts in self.failed_attempts.values())
        if all_failed:
            raise AgentFailure("Both players have failed multiple times")

        # Check for single player persistent failure
        if self.failed_attempts[player_id] >= MAX_AGENT_FAILURES:
            raise AgentFailure(f"Player {player_id} has failed {self.MAX_FAILURES} times")

    def record_success(self, player_id: int):
        self.failed_attempts[player_id] = 0


class AgentAPIClient:
    """Handles API communication with agents"""
    def __init__(
        self,
        logger: logging.Logger,
        failure_tracker: AgentFailureTracker,
        max_retries: int = 5,
        base_delay: float = 1.0,
        timeout: float = 5.0
    ):
        self.logger = logger
        self.failure_tracker = failure_tracker
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout

    def call(
        self,
        method: str,
        base_url: str,
        endpoint: str,
        payload: Dict[str, Any],
        player_id: int,
    ) -> Dict[str, Any]:
        """Make an API call with retry logic and failure tracking"""
        for attempt in range(self.max_retries):
            try:
                response = requests.request(
                    method,
                    base_url + endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                self.failure_tracker.record_success(player_id)
                return response.json()
                
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as e:
                if attempt == self.max_retries - 1:
                    self.failure_tracker.record_failure(player_id)
                    self.logger.error(f"Player {player_id} failed API call after {self.max_retries} attempts: {str(e)}")
                    raise

                delay = self.base_delay * (2 ** attempt)
                self.logger.debug(f"Player {player_id} API call failed, retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)


class PayloadPreparer:
    """Prepares payloads for API calls"""
    
    @staticmethod
    def convert_numpy(value: Any) -> Any:
        """Recursively convert numpy types to Python native types"""
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: PayloadPreparer.convert_numpy(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [PayloadPreparer.convert_numpy(item) for item in value]
        return value

    @staticmethod
    def prepare(
        obs: Dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the payload for API calls by converting numpy arrays and values to Python native types.

        Args:
            obs (Dict[str, Any]): The observation dictionary.
            reward (float): The reward value.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (Dict[str, Any]): Additional information.

        Returns:
            Dict[str, Any]: The prepared payload.
        """
        return {
            "observation": {k: PayloadPreparer.convert_numpy(v) for k, v in obs.items()},
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": PayloadPreparer.convert_numpy(info),
        }


class PokerMatch:
    """Manages a complete poker match between multiple agents"""
    # TODO set up logging for errors and game state changes
    # TODO implement the csv writer path
    #   - define new headers for csv file
    # TODO Handle when agenet disconnects and should continue gamne
    
    def __init__(
        self,
        base_urls: List[str],
        logger: logging.Logger,
        team_names: Optional[List[str]] = None,
        csv_path: str = "./match.csv"
    ):
        
        self.base_urls = base_urls
        self.num_players = NUM_PLAYERS # NOTE might change later
        self.logger = logger
        self.team_names = team_names or [f"Player {i}" for i in range(self.num_players)]
        self.csv_path = csv_path
        
        self.env = PokerEnv()
        
        # Initialize trackers
        self.bankrolls = [0.0] * self.num_players
        self.time_used = [0.0] * self.num_players
        self.failure_tracker = AgentFailureTracker(self.num_players)
        
        # Initialize helpers
        self.api_client = AgentAPIClient(logger, self.failure_tracker)
    
    def run(self, num_hands = 1000) -> MatchResult:
        
        for hand_number in num_hands:
            self._play_hand(hand_number)
        
        return self._create_result('completed')
            
    def _play_hand(self, hand_number: int):
        """Play a single hand of poker"""
        
        small_blind_player = hand_number % self.num_players

        # Initialize hand
        observations, info = self.env.reset(options={"small_blind_player": small_blind_player})
        info["hand_number"] = hand_number
        rewards = [0.0] * self.num_players
        terminated = truncated = False
        
        # Add time info to observations
        self._update_time_info(observations)
        
        # Main game loop
        while not terminated:
            action = self._get_action_from_active_player(observations, rewards, terminated, truncated, info)
            self._broadcast_to_inactive_players(observations, rewards, terminated, truncated, info)
            
            # Step environment
            observations, rewards, terminated, truncated, info = self.env.step(action=action["action"])
            info["hand_number"] = hand_number
            self._update_time_info(observations)
        
        # Send final observations
        self._send_final_observations(observations, rewards, terminated, truncated, info)

        # Update bankrolls
        for i, reward in enumerate(rewards):
            self.bankrolls[i] += reward
            
    def _get_action_from_active_player(
        self,
        observations: List[Dict[str, Any]],
        rewards: List[float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get action from the currently active player"""
        acting_player = observations[0]["acting_agent"]
        obs = observations[acting_player]
        url = self.base_urls[acting_player]
        
        payload = self.payload_preparer.prepare(obs, rewards[acting_player], terminated, truncated, info)
        
        start_time = time.time()
        action = self.api_client.call("GET", url, GET_ACTION_ENDPOINT, payload, acting_player)
        elapsed = time.time() - start_time
        
        self.time_used[acting_player] += elapsed
        
        if self.time_used[acting_player] > TIME_LIMIT_SECONDS:
            raise TimeoutError(f"Player {acting_player} exceeded time limit")
        
        return action
        
    def _broadcast_to_inactive_players(
        self,
        observations: List[Dict[str, Any]],
        rewards: List[float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """Send observations to all inactive players"""
        acting_player = observations[0]["acting_agent"]
        
        for player_id in range(self.num_players):
            if player_id != acting_player:
                obs = observations[player_id]
                url = self.base_urls[player_id]
                payload = self.payload_preparer.prepare(obs, rewards[player_id], terminated, truncated, info)
                self.api_client.call("POST", url, SEND_OBS_ENDPOINT, payload, player_id)

    def _send_final_observations(
        self,
        observations: List[Dict[str, Any]],
        rewards: List[float],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """Send final observations to all players after hand completes"""
        for player_id in range(self.num_players):
            obs = observations[player_id]
            url = self.base_urls[player_id]
            payload = self.payload_preparer.prepare(obs, rewards[player_id], terminated, truncated, info)
            self.api_client.call("POST", url, SEND_OBS_ENDPOINT, payload, player_id)
    
    def _update_time_info(self, observations: List[Dict[str, Any]]) -> None:
        """Add time usage information to observations"""
        for i, obs in enumerate(observations):
            obs["time_used"] = self.time_used[i]
            obs["time_left"] = TIME_LIMIT_SECONDS - self.time_used[i]
    
    def _create_result(
        self,
        status: str,
        disqualified_player: Optional[int] = None,
        error: Optional[str] = None
    ) -> MatchResult:
        """Create a standardized match result"""
        # TODO Handle disconnected players
        # should be ranked based on who disconnected first?
        raise NotImplementedError()




# Create a global instance
failure_tracker = AgentFailureTracker()


def get_street_name(street_num: int) -> str:
    """Convert numeric street value to human-readable name"""
    street_names = {0: "Pre-Flop", 1: "Flop", 2: "Turn", 3: "River"}
    return street_names.get(street_num, f"Unknown-{street_num}")

@deprecated
def prepare_payload(
    obs: Dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Prepare the payload for API calls by converting numpy arrays and values to Python native types.

    Args:
        obs (Dict[str, Any]): The observation dictionary.
        reward (float): The reward value.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode has been truncated.
        info (Dict[str, Any]): Additional information.

    Returns:
        Dict[str, Any]: The prepared payload.
    """

    def _convert_numpy(v):
        if isinstance(v, np.integer):
            return int(v)
        elif isinstance(v, np.floating):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: _convert_numpy(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [_convert_numpy(item) for item in v]
        return v

    def _prepare_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
        return {k: _convert_numpy(v) for k, v in observation.items()}

    return {
        "observation": _prepare_observation(obs),
        "reward": float(reward),
        "terminated": terminated,
        "truncated": truncated,
        "info": _convert_numpy(info),
    }


def call_agent_api(
    method: str,
    base_url: str,
    endpoint: str,
    payload: Dict[str, Any],
    logger: logging.Logger,
    player_id: int,
) -> Dict[str, Any]:
    """
    Make an API call to an agent with retry logic and failure tracking.

    Args:
        method (str): The HTTP method to use.
        base_url (str): The base URL of the agent's API.
        endpoint (str): The API endpoint to call.
        payload (Dict[str, Any]): The payload to send with the request.
        logger (logging.Logger): Logger instance.
        player_id (int): ID of the player (0 or 1).

    Returns:
        Dict[str, Any]: The JSON response from the API.

    Raises:
        TimeoutError: If the player exceeds their time limit.
        AgentFailure: If both players are failing or one player consistently fails.
    """
    max_retries = 5
    base_delay = 1

    try:
        for attempt in range(max_retries):
            try:
                response = requests.request(method, base_url + endpoint, json=payload, timeout=5.0)
                response.raise_for_status()
                failure_tracker.record_success(player_id)
                return response.json()
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            ) as e:
                if attempt == max_retries - 1:
                    raise

                delay = base_delay * (2**attempt)
                logger.info(f"Backing off for {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)

    except Exception as e:
        failure_tracker.record_failure(player_id)
        logger.error(f"Bot {player_id} failed API call: {str(e)}")
        raise


bankrolls = [0] * NUM_PLAYERS  # Track total bankrolls across all hands

# TODO: fix csv writer once, 6 player support is finished
@deprecated
def run_api_match(
    base_url_0: str,
    base_url_1: str,
    logger: logging.Logger,
    num_hands: int = 1000,
    csv_path: str = "./match.csv",
    team_0_name: str = "Team 0",
    team_1_name: str = "Team 1",
) -> Dict[str, Any]:
    """
    Run a match of multiple hands between two API-based agents.
    Each iteration creates a new PokerEnv instance representing one hand.
    """
    global bankrolls
    csv_headers = [
        "hand_number",
        "street",
        "active_team",
        "team_0_bankroll",
        "team_1_bankroll",
        "action_type",
        "action_amount",
        "team_0_cards",
        "team_1_cards",
        "board_cards",
        "team_0_discarded",
        "team_1_discarded",
        "team_0_bet",
        "team_1_bet",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        # Comment header
        writer = None
        # csv_file.write(f"Team 0: {team_0_name}, Team 1: {team_1_name}\n")

        # writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        # writer.writeheader()

        def format_error(e):
            return f'ERROR Raised: "{str(e)}". Stacktrace:\n{traceback.format_exc()}'

        for hand_number in range(num_hands):
            env = PokerEnv(logger=logger)  # env for a single hand
            try:
                res = play_hand(env, base_url_0, base_url_1, logger, writer, hand_number)
                bankrolls[0] += res["bot0_reward"]
                bankrolls[1] += res["bot1_reward"]
                if hand_number % 50 == 0:
                    logger.info(
                        f"Hand number: {hand_number}, {team_0_name} bankroll: {bankrolls[0]}, {team_1_name} bankroll: {bankrolls[1]}"
                    )
            except TimeoutError as te:
                # Determine winner based on which player exceeded time
                winner = 1 if "Player 0" in str(te) else 0
                return get_match_result("timeout", winner=winner)
            except AgentFailure as af:
                if "Player 0 has failed" in str(af):
                    # player 0 timeout
                    return get_match_result("timeout", winner=1)
                elif "Player 1 has failed" in str(af):
                    # player 1 timeout
                    return get_match_result("timeout", winner=0)
                return get_match_result("error", error=format_error(af))
            except Exception as e:
                logger.error(f"Unexpected error during hand {hand_number}: {e}")
                return get_match_result("error", error=format_error(e))

        logger.info("All hands completed")
        logger.info(f"Final results - {team_0_name} bankroll: {bankrolls[0]}, {team_1_name} bankroll: {bankrolls[1]}")
        logger.info(f"Time used - {team_0_name}: {time_used_0:.2f} seconds, {team_1_name}: {time_used_1:.2f} seconds")
        logger.info(f"Time limit: {TIME_LIMIT_SECONDS} seconds")

        return get_match_result("completed", rewards=(bankrolls[0], bankrolls[1]))


time_used_0 = 0.0
time_used_1 = 0.0
time_used = [0.0] * NUM_PLAYERS

# TODO handle if there exists less than 6 players at the table
@deprecated
def play_hand(
    env: PokerEnv, 
    base_urls: List[str], # for 6 players
    logger: logging.Logger, 
    writer: csv.DictWriter, 
    hand_number: int
):
    
    global bankrolls, time_used
    
    small_blind_player = hand_number % NUM_PLAYERS

    # Initialize per-hand variables
    observations, info = env.reset(options={"small_blind_player": small_blind_player})
    info["hand_number"] = hand_number
    
    assert len(observations) == NUM_PLAYERS
    
    rewards = [0] * NUM_PLAYERS
    terminated = truncated = False
    
    for i, obs in enumerate(observations):
        obs["time_used"] = time_used[i]
        obs["time_left"] = TIME_LIMIT_SECONDS - time_used[i]


    # Loop until hand terminates
    while not terminated:
        # TODO keep logs for bot history inside state
        action = None
        for player, (obs, url) in enumerate(zip(observations, base_urls)):
            bot_payload = prepare_payload(obs, rewards[i], terminated, truncated, info)
            if player == obs['acting_agent']:
                action_start = time.time()
                action = call_agent_api("GET", url, GET_ACTION_ENDPOINT, bot_payload, logger, player)
                action_duration = time.time() - action_start
                time_used[player] += action_duration
                if time_used[player] > TIME_LIMIT_SECONDS:
                    raise TimeoutError(f"Player {player} exceeded time limit")
                              
            # broadcast to other players
            else:
                # maybe do something with the return value? -dylan
                call_agent_api("POST", url, SEND_OBS_ENDPOINT, bot_payload, logger, player)
                
        # TODO: Update log action to be consistent with 6-player game
        # # Log action
        # current_state = {
        #     "hand_number": hand_number,
        #     "street": get_street_name(obs0["street"]),
        #     "active_team": obs0["acting_agent"],
        #     "team_0_bankroll": bankrolls[0],
        #     "team_1_bankroll": bankrolls[1],
        #     "team_0_cards": [env.int_card_to_str(c) for c in env.player_cards[0] if c != -1],
        #     "team_1_cards": [env.int_card_to_str(c) for c in env.player_cards[1] if c != -1],
        #     "board_cards": [env.int_card_to_str(c) for c in env.community_cards[: obs0["street"] + 2] if c != -1],
        #     "team_0_discarded": env.int_card_to_str(env.discarded_cards[0]) if env.discarded_cards[0] != -1 else "",
        #     "team_1_discarded": env.int_card_to_str(env.discarded_cards[1]) if env.discarded_cards[1] != -1 else "",
        #     "team_0_bet": obs0["my_bet"] if obs0["acting_agent"] == 0 else obs0["opp_bet"],
        #     "team_1_bet": obs1["my_bet"] if obs1["acting_agent"] == 1 else obs1["opp_bet"],
        #     "action_type": action_type.name,
        #     "action_amount": action["action"][1],
        # }
        # writer.writerow(current_state)

        # Step environment
        (observations), (rewards), terminated, truncated, info = env.step(action=action["action"])
        info["hand_number"] = hand_number  # Maintain hand number after each step
        
        for i, obs in enumerate(observations):
            obs["time_used"] = time_used[i]
            obs["time_left"] = TIME_LIMIT_SECONDS - time_used[i]

    # game has terminated; prepare and send final observation
    for player in range(NUM_PLAYERS):
        obs, reward, url = observations[i], rewards[i], base_urls[i]
        bot_payload = prepare_payload(obs, reward, terminated, truncated, info)
        call_agent_api("POST", url, SEND_OBS_ENDPOINT, bot_payload, logger, player)

    return rewards

    
    
def get_match_result(
    status: str,
    winner: Optional[int] = None,
    rewards: Optional[Tuple[float, float]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a standardized match result dictionary.

    Args:
        status (str): Match status ('completed', 'timeout', or 'error')
        winner (Optional[int]): Winner's player ID (0 or 1) for timeout cases
        rewards (Optional[Tuple[float, float]]): Final bankrolls (bot0_reward, bot1_reward)
        error (Optional[str]): Error message for error status

    Returns:
        Dict[str, Any]: Standardized match result containing:
            - status: 'completed', 'timeout', or 'error'
            - result: 'win' (player 0 won), 'loss' (player 1 won), 'tie', or 'invalid'
            - bot0_reward/bot1_reward: Final bankroll amounts (if available)
            - error: Error message (if status is 'error')
    """
    global time_used_0, time_used_1

    result = {"status": status}

    # Add result field for completed matches
    if status == "completed" and rewards:
        # Determine winner based on rewards
        if rewards[0] > rewards[1]:
            result["result"] = "win"  # Player 0 won
        elif rewards[1] > rewards[0]:
            result["result"] = "loss"  # Player 1 won
        else:
            result["result"] = "tie"  # Equal rewards
    elif status == "timeout" and winner is not None:
        # Convert timeout winner to win/loss
        result["result"] = "win" if winner == 0 else "loss"
    else:
        result["result"] = "invalid"

    # Add rewards if available
    if rewards:
        result["bot0_reward"] = rewards[0]
        result["bot1_reward"] = rewards[1]

    # Add time used data
    result["bot0_time_used"] = time_used_0
    result["bot1_time_used"] = time_used_1

    if error:
        result["error"] = error

    return result


def log_game_state(logger: logging.Logger, obs0: Dict[str, Any], obs1: Dict[str, Any]) -> None:
    """
    Log the current game state.

    Args:
        logger (logging.Logger): The logger object to use for logging.
        obs0 (Dict[str, Any]): Observation for the first agent.
        obs1 (Dict[str, Any]): Observation for the second agent.
    """
    logger.debug("#####################")
    logger.debug(f"Turn: {obs0['acting_agent']}")
    logger.debug(f"Bot0 cards: {obs0['my_cards']}, Bot1 cards: {obs1['my_cards']}")
    logger.debug(f"Community cards: {obs0['community_cards']}")
    logger.debug(f"Bot0 bet: {obs0['my_bet']}, Bot1 bet: {obs1['my_bet']}")
    logger.debug("#####################")


def format_bankroll_log(game_number: int, bankrolls: list) -> str:
    """Format bankroll data as a JSON string for logging"""
    bankroll_data = {
        "type": "bankroll_update",
        "game_number": game_number,
        "bot0_bankroll": int(bankrolls[0]),
        "bot1_bankroll": int(bankrolls[1]),
    }
    return json.dumps(bankroll_data)

