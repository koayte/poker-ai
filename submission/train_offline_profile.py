import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def parse_args():
    parser = argparse.ArgumentParser(description="Train offline poker profile from match CSV files.")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more match CSV paths.",
    )
    parser.add_argument(
        "--team-id",
        type=int,
        choices=[0, 1],
        default=0,
        help="Your team id in the CSV logs (0 or 1). Default: 0",
    )
    parser.add_argument(
        "--output",
        default="submission/offline_profile.json",
        help="Output JSON path written for player.py to load.",
    )
    return parser.parse_args()


def load_rows(csv_paths):
    rows = []
    for p in csv_paths:
        with open(p, newline="", encoding="utf-8") as f:
            all_lines = f.readlines()

        # Some match logs start with metadata/comment lines before CSV header.
        header_idx = None
        for i, line in enumerate(all_lines):
            normalized = line.strip().lower()
            if normalized.startswith("hand_number,"):
                header_idx = i
                break

        if header_idx is None:
            raise ValueError(f"Could not find CSV header with 'hand_number' in file: {p}")

        reader = csv.DictReader(all_lines[header_idx:])
        for row in reader:
            if not row:
                continue
            normalized_row = {
                (k.strip().lower() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }
            # Skip malformed blank rows.
            if not normalized_row.get("hand_number"):
                continue
            rows.append(normalized_row)
    return rows


def group_by_hand(rows):
    by_hand = defaultdict(list)
    for row in rows:
        hand = int(row["hand_number"])
        by_hand[hand].append(row)
    return by_hand


def compute_profile(rows, team_id):
    opp_id = 1 - team_id
    by_hand = group_by_hand(rows)

    raise_hands = 0
    opp_fold_after_our_raise = 0
    opp_actions_seen = 0
    opp_aggressive_actions = 0
    showdowns = 0
    opp_bluff_showdowns = 0

    for hand in sorted(by_hand):
        seq = by_hand[hand]
        our_raised = False
        opp_aggressive = False

        for row in seq:
            actor = int(row["active_team"])
            action = row["action_type"]

            if actor == team_id and action == "RAISE":
                our_raised = True
            if actor == opp_id:
                opp_actions_seen += 1
                if action == "RAISE":
                    opp_aggressive_actions += 1
                    opp_aggressive = True

        last = seq[-1]
        t0_cards = last.get("team_0_cards", "")
        t1_cards = last.get("team_1_cards", "")
        board = last.get("board_cards", "")

        # Heuristic showdown detector from final row visibility.
        is_showdown = (
            t0_cards.startswith("[") and t1_cards.startswith("[") and board.startswith("[") and board.count(",") >= 4
        )
        if is_showdown:
            showdowns += 1

        bankroll = int(last["team_0_bankroll"] if team_id == 0 else last["team_1_bankroll"])

        if our_raised:
            raise_hands += 1
            if bankroll > 0 and not is_showdown:
                opp_fold_after_our_raise += 1

        # Bluff proxy on showdown is intentionally conservative in this trainer.
        if is_showdown and opp_aggressive:
            pass

    fold_to_raise = opp_fold_after_our_raise / raise_hands if raise_hands else 0.35
    aggression = opp_aggressive_actions / opp_actions_seen if opp_actions_seen else 0.30
    bluff_freq = opp_bluff_showdowns / showdowns if showdowns else 0.12

    profile = {
        "version": 1,
        "source": "offline_csv_training",
        "base_thresholds": {
            "0": {"raise": 0.69, "call": 0.46, "bluff": [0.31, 0.40]},
            "1": {"raise": 0.68, "call": 0.45, "bluff": [0.30, 0.39]},
            "2": {"raise": 0.65, "call": 0.43, "bluff": [0.28, 0.37]},
            "3": {"raise": 0.61, "call": 0.41, "bluff": [0.24, 0.33]},
        },
        "simulation_budgets": {
            "discard": 300,
            "bet": 300,
            "preflop": 140,
        },
        "opponent_priors": {
            "fold_to_raise": clamp(fold_to_raise, 0.05, 0.95),
            "aggression": clamp(aggression, 0.05, 0.95),
            "bluff_freq": clamp(bluff_freq, 0.0, 0.50),
            "raise_samples": max(20, min(200, raise_hands)),
            "action_samples": max(80, min(800, opp_actions_seen)),
            "showdown_samples": max(20, min(300, showdowns)),
        },
        "training_summary": {
            "matches_rows": len(rows),
            "raise_hands": raise_hands,
            "opp_actions_seen": opp_actions_seen,
            "showdowns": showdowns,
        },
    }

    return profile


def main():
    args = parse_args()
    rows = load_rows(args.csv_files)
    profile = compute_profile(rows, args.team_id)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Wrote offline profile to {out}")


if __name__ == "__main__":
    main()
