import itertools
import random
import time

from agents.agent import Agent
from gym_env import PokerEnv


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.rng = random.Random()

        self.ranks = "23456789A"
        self.suits = "dhs"
        self.rank_to_value = {r: (14 if r == "A" else int(r)) for r in self.ranks}
        self.full_deck = tuple(range(27))

        self.simulations_discard = 300
        self.simulations_bet = 300
        self.simulations_preflop = 140
        self.max_decision_seconds = 0.45

        # Balanced baseline thresholds; opponent model applies bounded shifts.
        self.base_thresholds = {
            0: {"raise": 0.68, "call": 0.45, "bluff": (0.32, 0.42)},
            1: {"raise": 0.67, "call": 0.44, "bluff": (0.30, 0.40)},
            2: {"raise": 0.64, "call": 0.42, "bluff": (0.28, 0.38)},
            3: {"raise": 0.60, "call": 0.40, "bluff": (0.24, 0.34)},
        }

        self.equity_cache = {}
        self.current_hand_number = None
        self.prev_obs = None

        self.hand_raised = False
        self.opp_aggressive_this_hand = False

        self.stats = {
            "hands": 0,
            "raise_hands": 0,
            "opp_fold_after_our_raise": 0,
            "opp_actions_seen": 0,
            "opp_aggressive_actions": 0,
            "showdowns": 0,
            "opp_bluff_showdowns": 0,
        }

    def __name__(self):
        return "PlayerAgent"

    def _card_rank_value(self, card_int: int) -> int:
        return self.rank_to_value[self.ranks[card_int % 9]]

    def _card_suit_index(self, card_int: int) -> int:
        return card_int // 9

    def _card_str(self, card_int: int) -> str:
        return PokerEnv.int_card_to_str(card_int)

    def _visible_cards(self, my_cards, community_cards, opp_discarded_cards):
        known = set()
        for c in my_cards:
            if c != -1:
                known.add(c)
        for c in community_cards:
            if c != -1:
                known.add(c)
        for c in opp_discarded_cards:
            if c != -1:
                known.add(c)
        return known

    def _remaining_deck(self, known_cards):
        return [c for c in self.full_deck if c not in known_cards]

    def _straight_high(self, rank_values):
        unique = set(rank_values)
        straights = [
            ({14, 2, 3, 4, 5}, 5),
            ({2, 3, 4, 5, 6}, 6),
            ({3, 4, 5, 6, 7}, 7),
            ({4, 5, 6, 7, 8}, 8),
            ({5, 6, 7, 8, 9}, 9),
            ({6, 7, 8, 9, 14}, 14),
        ]
        best = None
        for needed, high in straights:
            if needed.issubset(unique):
                best = high if best is None else max(best, high)
        return best

    def _score_five_card(self, cards5):
        ranks = [self._card_rank_value(c) for c in cards5]
        suits = [self._card_suit_index(c) for c in cards5]

        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        groups = sorted(((cnt, r) for r, cnt in rank_counts.items()), reverse=True)
        is_flush = len(set(suits)) == 1
        straight_high = self._straight_high(ranks)
        is_straight = straight_high is not None

        # Category order: 7 SF, 6 FH, 5 F, 4 S, 3 Trips, 2 TwoPair, 1 Pair, 0 High.
        if is_flush and is_straight:
            return (7, straight_high)

        if groups[0][0] == 3 and groups[1][0] == 2:
            trip_rank = groups[0][1]
            pair_rank = groups[1][1]
            return (6, trip_rank, pair_rank)

        if is_flush:
            return (5, *sorted(ranks, reverse=True))

        if is_straight:
            return (4, straight_high)

        if groups[0][0] == 3:
            trip_rank = groups[0][1]
            kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
            return (3, trip_rank, *kickers)

        if groups[0][0] == 2 and groups[1][0] == 2:
            pair_ranks = sorted([groups[0][1], groups[1][1]], reverse=True)
            kicker = [r for r in ranks if r not in pair_ranks][0]
            return (2, pair_ranks[0], pair_ranks[1], kicker)

        if groups[0][0] == 2:
            pair_rank = groups[0][1]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return (1, pair_rank, *kickers)

        return (0, *sorted(ranks, reverse=True))

    def _score_best_hand(self, hole2, board5):
        seven = list(hole2) + list(board5)
        best = None
        for combo in itertools.combinations(seven, 5):
            score = self._score_five_card(combo)
            if best is None or score > best:
                best = score
        return best

    def _opp_discard_strength_shift(self, opp_discarded_cards):
        shown = [c for c in opp_discarded_cards if c != -1]
        if len(shown) != 3:
            return 0.0

        avg_rank = sum(self._card_rank_value(c) for c in shown) / 3.0
        # High discarded ranks suggest opponent likely kept weaker structure.
        if avg_rank >= 9.0:
            return 0.015
        if avg_rank <= 5.0:
            return -0.01
        return 0.0

    def _pick_best_two_from_five_given_flop(self, hole5, flop3):
        best_pair = (hole5[0], hole5[1])
        best_score = None
        for i, j in itertools.combinations(range(5), 2):
            pair = (hole5[i], hole5[j])
            score = self._score_five_card(list(pair) + list(flop3))
            if best_score is None or score > best_score:
                best_score = score
                best_pair = pair
        return best_pair

    def _make_cache_key(self, kind, my_cards, community, opp_discards, sims):
        return (
            kind,
            tuple(sorted(my_cards)),
            tuple(community),
            tuple(sorted([c for c in opp_discards if c != -1])),
            sims,
        )

    def _estimate_equity_two_card(
        self,
        my_two,
        community,
        opp_discarded_cards,
        sims,
        start_time,
    ):
        cache_key = self._make_cache_key("two", my_two, community, opp_discarded_cards, sims)
        cached = self.equity_cache.get(cache_key)
        if cached is not None:
            return cached

        known = self._visible_cards(my_two, community, opp_discarded_cards)
        deck = self._remaining_deck(known)
        board_needed = 5 - len(community)
        if board_needed < 0:
            board_needed = 0

        wins = 0.0
        trials = 0
        need = 2 + board_needed
        if need > len(deck):
            return 0.5

        for _ in range(sims):
            if time.perf_counter() - start_time > self.max_decision_seconds:
                break

            sample = self.rng.sample(deck, need)
            opp_two = sample[:2]
            board = list(community) + sample[2:]

            my_score = self._score_best_hand(my_two, board)
            opp_score = self._score_best_hand(opp_two, board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
            trials += 1

        if trials == 0:
            return 0.5

        equity = wins / trials
        equity += self._opp_discard_strength_shift(opp_discarded_cards)
        equity = min(0.99, max(0.01, equity))
        self.equity_cache[cache_key] = equity
        return equity

    def _estimate_equity_preflop(
        self,
        my_five,
        community,
        opp_discarded_cards,
        sims,
        start_time,
    ):
        cache_key = self._make_cache_key("pre", my_five, community, opp_discarded_cards, sims)
        cached = self.equity_cache.get(cache_key)
        if cached is not None:
            return cached

        known = self._visible_cards(my_five, community, opp_discarded_cards)
        deck = self._remaining_deck(known)
        if len(deck) < 10:
            return 0.5

        wins = 0.0
        trials = 0
        for _ in range(sims):
            if time.perf_counter() - start_time > self.max_decision_seconds:
                break

            sample = self.rng.sample(deck, 10)
            opp_five = sample[:5]
            board = sample[5:10]
            flop = board[:3]

            my_two = self._pick_best_two_from_five_given_flop(my_five, flop)
            opp_two = self._pick_best_two_from_five_given_flop(opp_five, flop)

            my_score = self._score_best_hand(my_two, board)
            opp_score = self._score_best_hand(opp_two, board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
            trials += 1

        if trials == 0:
            return 0.5

        equity = min(0.99, max(0.01, wins / trials))
        self.equity_cache[cache_key] = equity
        return equity

    def _choose_discard_indices(self, my_five, flop3, opp_discarded_cards, start_time):
        best_pair = (0, 1)
        best_equity = -1.0

        for i, j in itertools.combinations(range(5), 2):
            keep = [my_five[i], my_five[j]]
            eq = self._estimate_equity_two_card(
                keep,
                flop3,
                opp_discarded_cards,
                sims=self.simulations_discard,
                start_time=start_time,
            )
            current_score = self._score_five_card(keep + flop3)
            best_score = self._score_five_card([my_five[best_pair[0]], my_five[best_pair[1]]] + flop3)
            if eq > best_equity or (abs(eq - best_equity) < 1e-9 and current_score > best_score):
                best_equity = eq
                best_pair = (i, j)

        return best_pair, best_equity

    def _safe_raise_amount(self, obs, equity):
        min_raise = obs["min_raise"]
        max_raise = obs["max_raise"]
        if max_raise <= 0:
            return 0
        if min_raise > max_raise:
            min_raise = max_raise

        # Size larger with stronger equity while keeping bounded in legal range.
        frac = min(1.0, max(0.0, (equity - 0.55) / 0.40))
        amount = int(min_raise + frac * (max_raise - min_raise))
        return max(min_raise, min(max_raise, amount))

    def _opponent_adjustments(self):
        raise_hands = max(1, self.stats["raise_hands"])
        opp_actions = max(1, self.stats["opp_actions_seen"])
        showdowns = max(1, self.stats["showdowns"])

        fold_to_raise = self.stats["opp_fold_after_our_raise"] / raise_hands
        aggression = self.stats["opp_aggressive_actions"] / opp_actions
        bluff_freq = self.stats["opp_bluff_showdowns"] / showdowns

        # Medium-strength adaptation.
        raise_shift = 0.0
        call_shift = 0.0
        bluff_shift = 0.0

        if self.stats["hands"] >= 12:
            if fold_to_raise > 0.55:
                raise_shift -= 0.02
                bluff_shift += 0.03
            elif fold_to_raise < 0.30:
                raise_shift += 0.015

            if aggression > 0.42:
                call_shift -= 0.025
                bluff_shift -= 0.02
            elif aggression < 0.22:
                call_shift += 0.01

            if bluff_freq > 0.25:
                call_shift -= 0.015

        # Keep adjustments bounded and stable.
        raise_shift = max(-0.04, min(0.04, raise_shift))
        call_shift = max(-0.04, min(0.04, call_shift))
        bluff_shift = max(-0.05, min(0.05, bluff_shift))

        return {
            "raise_shift": raise_shift,
            "call_shift": call_shift,
            "bluff_shift": bluff_shift,
            "fold_to_raise": fold_to_raise,
            "aggression": aggression,
            "bluff_freq": bluff_freq,
        }

    def _update_opponent_action_signal(self, observation):
        # On our turn, compare to the previous observation from our prior turn.
        if self.prev_obs is None:
            return

        if observation.get("street") != self.prev_obs.get("street"):
            return

        prev_opp_bet = self.prev_obs.get("opp_bet", 0)
        curr_opp_bet = observation.get("opp_bet", 0)

        self.stats["opp_actions_seen"] += 1
        if curr_opp_bet > prev_opp_bet:
            self.stats["opp_aggressive_actions"] += 1
            self.opp_aggressive_this_hand = True

    def _update_showdown_bluff_proxy(self, observation, info):
        if "player_0_cards" not in info or "player_1_cards" not in info or "community_cards" not in info:
            return

        my_cards = sorted(self._card_str(c) for c in observation.get("my_cards", []) if c != -1)
        p0 = sorted(info["player_0_cards"])
        p1 = sorted(info["player_1_cards"])
        board = info["community_cards"]

        if my_cards == p0:
            opp = p1
        elif my_cards == p1:
            opp = p0
        else:
            return

        to_int = {PokerEnv.int_card_to_str(i): i for i in self.full_deck}
        try:
            opp_int = [to_int[s] for s in opp]
            board_int = [to_int[s] for s in board]
        except KeyError:
            return

        opp_score = self._score_best_hand(opp_int, board_int)
        category = opp_score[0]
        self.stats["showdowns"] += 1
        # Approx bluff proxy: opponent took aggressive action this hand but ended weak.
        if self.opp_aggressive_this_hand and category <= 1:
            self.stats["opp_bluff_showdowns"] += 1

    def _start_new_hand_if_needed(self, info):
        hand_number = info.get("hand_number")
        if hand_number is None:
            return
        if self.current_hand_number == hand_number:
            return

        self.current_hand_number = hand_number
        self.equity_cache.clear()
        self.prev_obs = None
        self.hand_raised = False
        self.opp_aggressive_this_hand = False

    def act(self, observation, reward, terminated, truncated, info):
        self._start_new_hand_if_needed(info)
        self._update_opponent_action_signal(observation)

        valid_actions = observation["valid_actions"]
        street = observation["street"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = list(observation.get("opp_discarded_cards", [-1, -1, -1]))

        start_time = time.perf_counter()

        # Mandatory discard round.
        if valid_actions[self.action_types.DISCARD.value]:
            if len(my_cards) < 5 or len(community) < 3:
                action = (self.action_types.DISCARD.value, 0, 0, 1)
            else:
                best_pair, eq = self._choose_discard_indices(my_cards, community[:3], opp_discards, start_time)
                self.logger.debug(f"Discard decision keep={best_pair} eq={eq:.3f}")
                action = (self.action_types.DISCARD.value, 0, best_pair[0], best_pair[1])

            self.prev_obs = observation
            return action

        # If betting with 2 cards, use post-discard equity. If 5 cards, use pre-flop rollouts.
        if len(my_cards) == 2:
            equity = self._estimate_equity_two_card(
                my_cards,
                community,
                opp_discards,
                sims=self.simulations_bet,
                start_time=start_time,
            )
        else:
            equity = self._estimate_equity_preflop(
                my_cards[:5],
                community,
                opp_discards,
                sims=self.simulations_preflop,
                start_time=start_time,
            )

        continue_cost = max(0, observation["opp_bet"] - observation["my_bet"])
        pot_size = observation.get("pot_size", observation["my_bet"] + observation["opp_bet"])
        pot_odds = continue_cost / (pot_size + continue_cost) if continue_cost > 0 else 0.0

        th = self.base_thresholds.get(street, self.base_thresholds[3]).copy()
        adj = self._opponent_adjustments()
        th["raise"] += adj["raise_shift"]
        th["call"] += adj["call_shift"]
        bluff_low, bluff_high = th["bluff"]
        bluff_low += adj["bluff_shift"]
        bluff_high += adj["bluff_shift"]

        # Strong value raise.
        if valid_actions[self.action_types.RAISE.value] and equity >= th["raise"]:
            raise_amount = self._safe_raise_amount(observation, equity)
            if raise_amount > 0:
                self.hand_raised = True
                self.prev_obs = observation
                return (self.action_types.RAISE.value, raise_amount, 0, 0)

        # Exploitative bluff raise against high fold-to-raise opponents.
        if (
            valid_actions[self.action_types.RAISE.value]
            and bluff_low <= equity <= bluff_high
            and adj["fold_to_raise"] > 0.52
            and continue_cost == 0
        ):
            raise_amount = max(observation["min_raise"], min(observation["max_raise"], observation["min_raise"]))
            if raise_amount > 0:
                self.hand_raised = True
                self.prev_obs = observation
                return (self.action_types.RAISE.value, raise_amount, 0, 0)

        # Continue when equity clears either threshold or pot-odds benchmark.
        call_threshold = min(th["call"], pot_odds + 0.06)
        if valid_actions[self.action_types.CALL.value] and equity >= call_threshold:
            self.prev_obs = observation
            return (self.action_types.CALL.value, 0, 0, 0)

        if valid_actions[self.action_types.CHECK.value]:
            self.prev_obs = observation
            return (self.action_types.CHECK.value, 0, 0, 0)

        self.prev_obs = observation
        return (self.action_types.FOLD.value, 0, 0, 0)

    def observe(self, observation, reward, terminated, truncated, info):
        if not terminated:
            return

        self.stats["hands"] += 1
        if self.hand_raised:
            self.stats["raise_hands"] += 1
            # Approximation: positive non-showdown result after we raised indicates fold response.
            if reward > 0 and "player_0_cards" not in info:
                self.stats["opp_fold_after_our_raise"] += 1

        self._update_showdown_bluff_proxy(observation, info)

