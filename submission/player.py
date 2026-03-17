import itertools
import json
import os
import random
import time

from agents.agent import Agent
from gym_env import PokerEnv
from gym_env import WrappedEval


class PlayerAgent(Agent):
    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self.evaluator = WrappedEval()
        self.int_to_card = PokerEnv.int_to_card
        self.rng = random.Random()

        # Match-level configuration for conservative lock-in strategy.
        self.total_hands = 1000
        # If we fold at first legal action each hand, worst-case loss is the posted big blind.
        self.max_fold_loss_per_hand = 1.5

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

        self.offline_profile = {}
        self.offline_prior = {
            "fold_to_raise": 0.35,
            "aggression": 0.30,
            "bluff_freq": 0.12,
            "raise_samples": 20,
            "action_samples": 80,
            "showdown_samples": 20,
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

        self.cumulative_profit = 0
        self.lock_in_mode = False

        # Dynamic aggression controls to prevent opponents from coasting to lock-in.
        self.early_aggression_hands = 180
        self.anti_lock_margin = 0.80
        self.anti_lock_min_remaining_hands = 80

        self._load_offline_profile()
        self._apply_offline_profile()

    def __name__(self):
        return "PlayerAgent"

    @staticmethod
    def _bounded(x, lo, hi):
        return max(lo, min(hi, x))

    def _load_offline_profile(self):
        profile_path = os.path.join(os.path.dirname(__file__), "offline_profile.json")
        if not os.path.exists(profile_path):
            self.logger.info("No offline profile found at %s", profile_path)
            return

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        except Exception as exc:
            self.logger.warning("Failed to load offline profile (%s)", exc)
            return

        if not isinstance(profile, dict):
            self.logger.warning("Offline profile format invalid; expected object")
            return

        self.offline_profile = profile

        # Optional learned baseline thresholds by street.
        learned_thresholds = profile.get("base_thresholds")
        if isinstance(learned_thresholds, dict):
            for k, val in learned_thresholds.items():
                try:
                    street = int(k)
                except (TypeError, ValueError):
                    continue
                if not isinstance(val, dict):
                    continue

                base = self.base_thresholds.get(street)
                if base is None:
                    continue

                if "raise" in val:
                    base["raise"] = float(self._bounded(float(val["raise"]), 0.45, 0.90))
                if "call" in val:
                    base["call"] = float(self._bounded(float(val["call"]), 0.20, 0.80))
                if "bluff" in val and isinstance(val["bluff"], list) and len(val["bluff"]) == 2:
                    low = float(self._bounded(float(val["bluff"][0]), 0.10, 0.80))
                    high = float(self._bounded(float(val["bluff"][1]), low, 0.90))
                    base["bluff"] = (low, high)

        # Optional learned simulation budgets.
        sims = profile.get("simulation_budgets", {})
        if isinstance(sims, dict):
            if "discard" in sims:
                self.simulations_discard = int(self._bounded(int(sims["discard"]), 80, 700))
            if "bet" in sims:
                self.simulations_bet = int(self._bounded(int(sims["bet"]), 80, 700))
            if "preflop" in sims:
                self.simulations_preflop = int(self._bounded(int(sims["preflop"]), 40, 400))

        # Learned opponent priors (used as pseudocounts at match start).
        priors = profile.get("opponent_priors", {})
        if isinstance(priors, dict):
            for key in self.offline_prior:
                if key in priors:
                    self.offline_prior[key] = priors[key]

    def _apply_offline_profile(self):
        # Convert learned priors into pseudocounts so adaptation starts informed.
        try:
            raise_samples = max(1, int(self.offline_prior.get("raise_samples", 20)))
            action_samples = max(1, int(self.offline_prior.get("action_samples", 80)))
            showdown_samples = max(1, int(self.offline_prior.get("showdown_samples", 20)))

            fold_to_raise = float(self._bounded(float(self.offline_prior.get("fold_to_raise", 0.35)), 0.01, 0.99))
            aggression = float(self._bounded(float(self.offline_prior.get("aggression", 0.30)), 0.01, 0.99))
            bluff_freq = float(self._bounded(float(self.offline_prior.get("bluff_freq", 0.12)), 0.0, 0.90))

            self.stats["raise_hands"] = raise_samples
            self.stats["opp_fold_after_our_raise"] = int(round(raise_samples * fold_to_raise))
            self.stats["opp_actions_seen"] = action_samples
            self.stats["opp_aggressive_actions"] = int(round(action_samples * aggression))
            self.stats["showdowns"] = showdown_samples
            self.stats["opp_bluff_showdowns"] = int(round(showdown_samples * bluff_freq))
        except Exception as exc:
            self.logger.warning("Failed to apply offline profile priors (%s)", exc)

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

    def _score_five_card(self, cards5):
        treys_cards = list(map(self.int_to_card, cards5))
        return self.evaluator.evaluate(treys_cards[:2], treys_cards[2:])

    def _score_best_hand(self, hole2, board5):
        treys_hand = list(map(self.int_to_card, hole2))
        treys_board = list(map(self.int_to_card, board5))
        return self.evaluator.evaluate(treys_hand, treys_board)

    # 5. Opponent discard signal 
    def _opp_discard_strength_shift(self, opp_discarded_cards):
        shown = [c for c in opp_discarded_cards if c != -1]
        if len(shown) != 3:
            return 0.0

        avg_rank = sum(self._card_rank_value(c) for c in shown) / 3.0
        # High discarded ranks suggest opponent likely kept weaker structure.
        if avg_rank >= 9.0:
            # If opponent discarded mostly high cards, may have retained weak set
            # Increase equity slightly 
            return 0.015
        if avg_rank <= 5.0:
            # If opponent discarded mostly low cards, may have retained stronger set
            # Decrease equity slightly
            return -0.01
        return 0.0
    
    # Helper, used in preflop simulation to approximate both players' discard behavior later 
    def _pick_best_two_from_five_given_flop(self, hole5, flop3):
        best_pair = (hole5[0], hole5[1])
        best_score = None
        for i, j in itertools.combinations(range(5), 2):
            pair = (hole5[i], hole5[j])
            score = self._score_five_card(list(pair) + list(flop3))
            if best_score is None or score < best_score:
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
            if my_score < opp_score:
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
            if my_score < opp_score:
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
            if eq > best_equity or (abs(eq - best_equity) < 1e-9 and current_score < best_score):
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

    def _phase_adjustments(self, info, observation):
        hand_number = info.get("hand_number", 0)
        remaining_hands = max(0, self.total_hands - hand_number)

        # Estimate opponent's lock-in threshold from our perspective.
        opponent_profit = -self.cumulative_profit
        opponent_safe_margin = self.max_fold_loss_per_hand * remaining_hands
        opponent_near_lock = (
            remaining_hands >= self.anti_lock_min_remaining_hands
            and opponent_profit > self.anti_lock_margin * opponent_safe_margin
        )

        early_pressure = hand_number < self.early_aggression_hands

        phase = {
            "early_pressure": early_pressure,
            "anti_lock_attack": opponent_near_lock,
            "raise_shift": 0.0,
            "call_shift": 0.0,
            "bluff_low_shift": 0.0,
            "bluff_high_shift": 0.0,
            "size_boost": 0.0,
        }

        if early_pressure:
            phase["raise_shift"] -= 0.03
            phase["bluff_low_shift"] -= 0.02
            phase["bluff_high_shift"] += 0.03
            phase["size_boost"] += 0.10

        if opponent_near_lock:
            # Apply stronger pressure before opponent can safely fold out match.
            phase["raise_shift"] -= 0.05
            phase["call_shift"] += 0.01
            phase["bluff_low_shift"] -= 0.02
            phase["bluff_high_shift"] += 0.04
            phase["size_boost"] += 0.15

        # Clamp for stability.
        phase["raise_shift"] = self._bounded(phase["raise_shift"], -0.10, 0.02)
        phase["call_shift"] = self._bounded(phase["call_shift"], -0.05, 0.03)
        phase["bluff_low_shift"] = self._bounded(phase["bluff_low_shift"], -0.05, 0.02)
        phase["bluff_high_shift"] = self._bounded(phase["bluff_high_shift"], -0.02, 0.08)
        phase["size_boost"] = self._bounded(phase["size_boost"], 0.0, 0.30)

        return phase

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
        rank_class = self.evaluator.get_rank_class(opp_score)
        self.stats["showdowns"] += 1
        # Approx bluff proxy: opponent took aggressive action this hand but ended weak.
        if self.opp_aggressive_this_hand and rank_class >= 8:
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

        # Enter lock-in only when lead is larger than worst-case remaining losses,
        # assuming we fold immediately on every future hand.
        remaining_hands = max(0, self.total_hands - hand_number)
        safe_margin = self.max_fold_loss_per_hand * remaining_hands
        # Recompute lock-in every hand (reversible), so we don't get stuck in fold mode.
        self.lock_in_mode = self.cumulative_profit > safe_margin

    def act(self, observation, reward, terminated, truncated, info):
        self._start_new_hand_if_needed(info)
        self._update_opponent_action_signal(observation)

        valid_actions = observation["valid_actions"]
        street = observation["street"]
        my_cards = [c for c in observation["my_cards"] if c != -1]
        community = [c for c in observation["community_cards"] if c != -1]
        opp_discards = list(observation.get("opp_discarded_cards", [-1, -1, -1]))

        start_time = time.perf_counter()

        # Lock in match win: once our lead is mathematically safe under conservative
        # per-hand fold loss, prefer folding immediately whenever legal.
        if self.lock_in_mode and valid_actions[self.action_types.FOLD.value]:
            self.prev_obs = observation
            print(f"Lock-in mode active: folding hand {self.current_hand_number} to preserve lead.")
            return (self.action_types.FOLD.value, 0, 0, 0)

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
        phase = self._phase_adjustments(info, observation)
        th["raise"] += adj["raise_shift"]
        th["call"] += adj["call_shift"]
        th["raise"] += phase["raise_shift"]
        th["call"] += phase["call_shift"]
        bluff_low, bluff_high = th["bluff"]
        bluff_low += adj["bluff_shift"]
        bluff_high += adj["bluff_shift"]
        bluff_low += phase["bluff_low_shift"]
        bluff_high += phase["bluff_high_shift"]
        bluff_low = self._bounded(bluff_low, 0.15, 0.70)
        bluff_high = self._bounded(bluff_high, bluff_low, 0.90)
        th["raise"] = self._bounded(th["raise"], 0.45, 0.90)
        th["call"] = self._bounded(th["call"], 0.20, 0.80)

        # Strong value raise.
        if valid_actions[self.action_types.RAISE.value] and equity >= th["raise"]:
            raise_amount = self._safe_raise_amount(observation, equity)
            if phase["size_boost"] > 0 and raise_amount > 0:
                min_raise = observation["min_raise"]
                max_raise = observation["max_raise"]
                if min_raise > max_raise:
                    min_raise = max_raise
                boosted = int(raise_amount + phase["size_boost"] * max(0, max_raise - min_raise))
                raise_amount = max(min_raise, min(max_raise, boosted))
            if raise_amount > 0:
                self.hand_raised = True
                self.prev_obs = observation
                return (self.action_types.RAISE.value, raise_amount, 0, 0)

        # Exploitative bluff raise against high fold-to-raise opponents.
        if (
            valid_actions[self.action_types.RAISE.value]
            and bluff_low <= equity <= bluff_high
            and adj["fold_to_raise"] > 0.52
            and (self.stats["hands"] >= 10 or phase["early_pressure"])
            and (street <= 2 or phase["anti_lock_attack"])
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

        self.cumulative_profit += reward

        self.stats["hands"] += 1
        if self.hand_raised:
            self.stats["raise_hands"] += 1
            # Approximation: positive non-showdown result after we raised indicates fold response.
            if reward > 0 and "player_0_cards" not in info:
                self.stats["opp_fold_after_our_raise"] += 1

        self._update_showdown_bluff_proxy(observation, info)

