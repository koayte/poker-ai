import argparse
import ast
import csv
import importlib
import os
import sys
from collections import defaultdict, deque


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gym_env import PokerEnv  # noqa: E402


ACTION_MAP = {
    "FOLD": PokerEnv.ActionType.FOLD.value,
    "RAISE": PokerEnv.ActionType.RAISE.value,
    "CHECK": PokerEnv.ActionType.CHECK.value,
    "CALL": PokerEnv.ActionType.CALL.value,
    "DISCARD": PokerEnv.ActionType.DISCARD.value,
}


def card_str_to_int(card_str: str) -> int:
    ranks = PokerEnv.RANKS
    suits = PokerEnv.SUITS
    return suits.index(card_str[1]) * len(ranks) + ranks.index(card_str[0])


def parse_cards(cell: str):
    if not cell or cell == "[]":
        return []
    return [card_str_to_int(c) for c in ast.literal_eval(cell)]


def load_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("hand_number,"))
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def build_hand_data(rows):
    rows_by_hand = defaultdict(list)
    for row in rows:
        rows_by_hand[int(row["hand_number"])].append(row)

    actions_by_hand = {}
    decks_by_hand = {}

    for hand_no, hand_rows in rows_by_hand.items():
        # Logged nico actions (team_1)
        q = deque()
        for row in hand_rows:
            if int(row["active_team"]) == 1:
                action_type = ACTION_MAP[row["action_type"]]
                action_amount = int(row["action_amount"])
                keep_1 = int(row["action_keep_1"])
                keep_2 = int(row["action_keep_2"])
                q.append((action_type, action_amount, keep_1, keep_2))
        actions_by_hand[hand_no] = q

        # Reconstruct deck order consumed by env.reset:
        # [team_0 five] + [team_1 five] + [community five] + [remaining cards]
        first = hand_rows[0]
        p0 = parse_cards(first["team_0_cards"])
        p1 = parse_cards(first["team_1_cards"])

        board5 = []
        for row in reversed(hand_rows):
            board = parse_cards(row["board_cards"])
            if len(board) == 5:
                board5 = board
                break

        if len(p0) != 5 or len(p1) != 5 or len(board5) != 5:
            deck = list(range(27))
        else:
            used = set(p0 + p1 + board5)
            remaining = [c for c in range(27) if c not in used]
            deck = p0 + p1 + board5 + remaining

        decks_by_hand[hand_no] = deck

    return rows_by_hand, actions_by_hand, decks_by_hand


def safe_fallback_action(obs):
    valid = obs["valid_actions"]
    if valid[PokerEnv.ActionType.DISCARD.value]:
        return (PokerEnv.ActionType.DISCARD.value, 0, 0, 1)
    if valid[PokerEnv.ActionType.CHECK.value]:
        return (PokerEnv.ActionType.CHECK.value, 0, 0, 0)
    if valid[PokerEnv.ActionType.CALL.value]:
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0)
    return (PokerEnv.ActionType.FOLD.value, 0, 0, 0)


def legalize_logged_action(obs, logged_action):
    """
    Project a logged action onto the current state's legal action set.
    This keeps replay running even when trajectories diverge.
    Returns (action_tuple, was_modified).
    """
    valid = obs["valid_actions"]
    a_type, amount, keep_1, keep_2 = logged_action

    # DISCARD phase: must provide two distinct indices [0..4].
    if valid[PokerEnv.ActionType.DISCARD.value]:
        i = max(0, min(4, int(keep_1)))
        j = max(0, min(4, int(keep_2)))
        if i == j:
            j = 1 if i == 0 else 0
        action = (PokerEnv.ActionType.DISCARD.value, 0, i, j)
        modified = action != logged_action
        return action, modified

    # If logged action type itself is legal, keep it with raise clamping if needed.
    if 0 <= a_type < len(valid) and valid[a_type]:
        if a_type == PokerEnv.ActionType.RAISE.value:
            min_r = obs["min_raise"]
            max_r = obs["max_raise"]
            if max_r < min_r:
                max_r = min_r
            clamped = max(min_r, min(max_r, int(amount)))
            action = (a_type, clamped, 0, 0)
            modified = clamped != amount
            return action, modified
        return (a_type, int(amount), int(keep_1), int(keep_2)), False

    # Logged action is illegal in this diverged state. Choose closest legal intent.
    if valid[PokerEnv.ActionType.CHECK.value]:
        return (PokerEnv.ActionType.CHECK.value, 0, 0, 0), True
    if valid[PokerEnv.ActionType.CALL.value]:
        return (PokerEnv.ActionType.CALL.value, 0, 0, 0), True
    if valid[PokerEnv.ActionType.RAISE.value]:
        min_r = obs["min_raise"]
        return (PokerEnv.ActionType.RAISE.value, min_r, 0, 0), True
    return (PokerEnv.ActionType.FOLD.value, 0, 0, 0), True


def main():
    parser = argparse.ArgumentParser(description="Replay match_nico cards with live team_0 agent vs logged team_1 actions.")
    parser.add_argument("--csv", default="matches/match_nico.csv", help="Path to match_nico.csv")
    parser.add_argument("--agent", default="submission.player.PlayerAgent", help="Team_0 agent class path")
    parser.add_argument("--max-hands", type=int, default=None, help="Optional cap on hands to replay")
    parser.add_argument("--strict", action="store_true", help="If set, stop when logged team_1 action queue is exhausted")
    parser.add_argument(
        "--legalize-team1",
        action="store_true",
        help="Project logged team_1 actions onto legal actions when trajectory diverges.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    rows_by_hand, actions_by_hand, decks_by_hand = build_hand_data(rows)

    module_path, class_name = args.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(module_path), class_name)
    agent0 = agent_cls(stream=False)

    hand_ids = sorted(rows_by_hand.keys())
    if args.max_hands is not None:
        hand_ids = hand_ids[: args.max_hands]

    bankroll0 = 0
    bankroll1 = 0
    invalid0 = 0
    invalid1 = 0
    fallback_actions = 0
    legalized_actions = 0

    for hand_no in hand_ids:
        env = PokerEnv(num_hands=1)
        (obs0, obs1), info = env.reset(
            options={"small_blind_player": hand_no % 2, "cards": decks_by_hand[hand_no]}
        )
        info["hand_number"] = hand_no

        reward0 = 0
        reward1 = 0
        terminated = False
        truncated = False

        while not terminated:
            acting = obs0["acting_agent"]
            if acting == 0:
                action = agent0.act(obs0, reward0, terminated, truncated, info)
            else:
                if actions_by_hand[hand_no]:
                    logged = actions_by_hand[hand_no].popleft()
                    if args.legalize_team1:
                        action, changed = legalize_logged_action(obs1, logged)
                        if changed:
                            legalized_actions += 1
                    else:
                        action = logged
                else:
                    if args.strict:
                        raise RuntimeError(
                            f"Logged team_1 actions exhausted in hand {hand_no}."
                        )
                    fallback_actions += 1
                    action = safe_fallback_action(obs1)

            (obs0, obs1), (reward0, reward1), terminated, truncated, step_info = env.step(action)
            info["hand_number"] = hand_no

            if step_info.get("invalid_action"):
                if acting == 0:
                    invalid0 += 1
                else:
                    invalid1 += 1

        bankroll0 += reward0
        bankroll1 += reward1
        agent0.observe(obs0, reward0, terminated, truncated, info)

    print("Replay summary (fixed cards from CSV; team_1 uses logged actions)")
    print(f"hands={len(hand_ids)}")
    print(f"bankroll_team0={bankroll0}")
    print(f"bankroll_team1={bankroll1}")
    print(f"invalid_actions_team0={invalid0}")
    print(f"invalid_actions_team1={invalid1}")
    print(f"team1_fallback_actions={fallback_actions}")
    print(f"team1_legalized_actions={legalized_actions}")


if __name__ == "__main__":
    main()
