import random

from agents.agent import Agent
from gym_env import PokerEnv

action_types = PokerEnv.ActionType
int_to_card = PokerEnv.int_to_card

class FoldAgent(Agent):
    def __name__(self):
        return "FoldAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]

        # Mandatory discard phase: keep first two cards
        if valid_actions[action_types.DISCARD.value]:
            return (action_types.DISCARD.value, 0, 0, 1)

        # Otherwise always fold
        return (action_types.FOLD.value, 0, 0, 0)


class CallingStationAgent(Agent):
    def __name__(self):
        return "CallingStationAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]

        # Mandatory discard phase: keep first two cards
        if valid_actions[action_types.DISCARD.value]:
            return (action_types.DISCARD.value, 0, 0, 1)

        # Otherwise, classic calling station: call if possible, else check
        if valid_actions[action_types.CALL.value]:
            action_type = action_types.CALL.value
        else:
            action_type = action_types.CHECK.value

        return (action_type, 0, 0, 0)


class AllInAgent(Agent):
    def __name__(self):
        return "AllInAgent"

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]

        # Mandatory discard phase: keep first two cards
        if valid_actions[action_types.DISCARD.value]:
            return (action_types.DISCARD.value, 0, 0, 1)

        if observation["street"] == 0:
            self.logger.debug(f"Hole cards: {[int_to_card(c) for c in observation['my_cards']]}")

        if valid_actions[action_types.RAISE.value]:
            action_type = action_types.RAISE.value
            raise_amount = observation["max_raise"]
            if raise_amount > 20:
                self.logger.info(f"Going all-in for {raise_amount}")
        elif valid_actions[action_types.CALL.value]:
            action_type = action_types.CALL.value
            raise_amount = 0
        else:
            action_type = action_types.CHECK.value
            raise_amount = 0

        return (action_type, raise_amount, 0, 0)


class RandomAgent(Agent):
    def __init__(self, stream: bool = False):
        super().__init__(stream)  # This sets up the logger and API logic
        self.action_types = PokerEnv.ActionType

    def act(self, observation, reward, terminated, truncated, info):
        valid_actions = observation["valid_actions"]
        
        # 1. Mandatory Discard Phase Handling
        if valid_actions[self.action_types.DISCARD.value]:
            keep_indices = random.sample(range(5), 2)
            return (self.action_types.DISCARD.value, 0, keep_indices[0], keep_indices[1])

        # 2. Random Betting Phase Handling
        possible_indices = [i for i, is_valid in enumerate(valid_actions) if is_valid]
        action_type = random.choice(possible_indices)
        
        raise_amount = 0
        if action_type == self.action_types.RAISE.value:
            raise_amount = random.randint(observation["min_raise"], observation["max_raise"])

        return (action_type, raise_amount, 0, 0)


all_agent_classes = (RandomAgent)

