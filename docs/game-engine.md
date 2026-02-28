# Game Engine Deep Dive

## Overview

The poker game engine implements the [tournament variant](/docs/rules) of Texas Hold'em using OpenAI's Gym framework (now Gymnasium). This document focuses on the technical implementation details.

## Environment Structure

### Core Components

```python
class PokerEnv(gym.Env):
    SMALL_BLIND_PLAYER = 0
    BIG_BLIND_PLAYER = 1
    MAX_PLAYER_BET = 100
```

The environment maintains game state including:

- Player cards
- Community cards
- Current bets
- Street number
- Acting player
- Bankrolls
- Discarded/drawn cards

## Action Space

### Available Actions

```python
class ActionType(Enum):
    FOLD = 0
    RAISE = 1
    CHECK = 2
    CALL = 3
    DISCARD = 4
    INVALID = 5
```

Each action is a tuple of four values:

```python
(action_type: int, raise_amount: int, keep_card_1: int, keep_card_2: int)
```

- `action_type`: One of the ActionType enum values
- `raise_amount`: Amount to raise (1 to MAX_PLAYER_BET)
- `keep_card_1`, `keep_card_2`: During the discard round (flop), the two indices (0–4) of the cards to **keep** from your 5 hole cards; the other 3 are discarded. Not used for non-discard actions.

## Observation Space

Each player receives an observation dictionary containing:

```python
{
    "street": int,              # Current street (0-3)
    "acting_agent": int,        # Which player acts next (0 or 1)
    "my_cards": Tuple[int, ...], # Your hole cards (5 slots; -1 if not yet dealt or after discard only 2 are used)
    "community_cards": Tuple[int, ...], # Visible community cards (5 slots; -1 for not yet dealt)
    "my_bet": int,             # Your current bet
    "opp_bet": int,            # Opponent's current bet
    "my_discarded_cards": Tuple[int, ...],  # Your 3 discarded cards (-1 until discarded)
    "opp_discarded_cards": Tuple[int, ...], # Opponent's 3 discarded cards (-1 until revealed)
    "min_raise": int,          # Minimum raise amount
    "max_raise": int,           # Maximum raise amount
    "valid_actions": List[bool] # Which actions are currently valid
}
```

## Card Representation

The tournament uses a **27-card deck**: ranks 2–9 and A, suits ♦♥♠ only (no face cards, no clubs). Cards are represented as integers from 0–26:

```python
RANKS = "23456789A"
SUITS = "dhs"  # diamonds, hearts, spades
```

Card index is `rank_index * 3 + suit_index`. Example mappings:

- 0 = 2♦
- 9 = 5♦
- 10 = 5♥
- 19 = 8♥
- 20 = 8♠
- 26 = A♠

## Game Flow

### 1. Initialization

```python
env = PokerEnv(num_hands=1000)
(obs0, obs1), info = env.reset()
```

### 2. Deal, Discard, and Betting Streets

- **Initial deal**: Each player is dealt 5 cards.
- **Street 0 — Pre-flop**: Blinds are posted; first round of betting.
- **Street 1 — Flop**: The first three community cards are dealt. There is a **discard round**: each player chooses 2 cards to keep (and discards the other 3); both players act in order. Discarded cards are revealed. Then flop betting.
- **Street 2 — Turn**: Fourth community card; betting.
- **Street 3 — River**: Fifth community card; betting; showdown if no fold.

### 3. Taking Actions

```python
obs, reward, terminated, truncated, info = env.step((action_type, raise_amount, keep_card_1, keep_card_2))
```

### 4. Hand Evaluation

Uses the `treys` library to evaluate final hands (adapted for the 27-card deck; four-of-a-kind is impossible):

```python
evaluator = Evaluator()
hand_rank = evaluator.evaluate(player_cards, board_cards)
# Lower rank = better hand
```

## Example Game Sequences

### Example 1: Basic Call/Check Sequence

```python
rigged_deck = [25, 14, 1, 4, 8, 16, 9, 11, 23]
actions = [
    (ActionType.CALL.value, 0, -1, -1),   # P0 calls big blind
    (ActionType.CHECK.value, 0, -1, -1),   # P1 checks
    (ActionType.RAISE.value, 2, -1, -1),   # P0 raises 2
    (ActionType.CALL.value, 0, -1, -1),    # P1 calls
    # ... continues to showdown
]
```

### Example 2: Max Bet Scenario

```python
rigged_deck = [24, 10, 14, 5]
actions = [
    (ActionType.CALL.value, 0, -1, -1),     # P0 calls big blind
    (ActionType.RAISE.value, 98, -1, -1),   # P1 raises to max
    (ActionType.FOLD.value, 0, -1, -1),     # P0 folds
]
```

### Example 3: Discard Action

During the flop (street 1), each player must discard 3 cards and keep 2. You specify which two card indices (0–4) to **keep**:

```python
# Player keeps cards at indices 0 and 1, discards the other 3
action = (ActionType.DISCARD.value, 0, 0, 1)

# Observation includes my_discarded_cards and opp_discarded_cards (3 cards each)
```

## Important Rules and Constraints

1. **Betting Limits**
   - Minimum raise: Previous raise amount
   - Maximum bet: 100 (MAX_PLAYER_BET)
   - Small blind: 1
   - Big blind: 2

2. **Deal and Discard**
   - Each player is dealt 5 cards at the start of the hand.
   - On the flop (street 1), there is one mandatory discard round: each player chooses 2 cards to keep and discards the other 3, in betting order. Both players must discard before flop betting.
   - Discarded cards are revealed to the opponent and removed from the deck for that hand.

3. **Valid Actions**
   - Can't check if opponent has bet
   - Can't call if bets are equal
   - Can't raise beyond MAX_PLAYER_BET
   - Can't discard after street 1

## Common Pitfalls

1. **Invalid Actions**
   - Always check `valid_actions` before choosing an action
   - Invalid actions are treated as folds

2. **Raise Amounts**
   - Must be between `min_raise` and `max_raise`
   - Raises are cumulative (total bet, not increment)

3. **Discarding**
   - Mandatory on the flop (street 1); each player keeps 2 cards and discards 3.
   - Action: use `DISCARD` with `keep_card_1` and `keep_card_2` as two distinct indices from 0 to 4 (your five hole cards).

4. **Observation Processing**
   - Convert numpy values to native Python types
   - Handle -1 values for hidden/unused cards

> For terminology and strategic concepts, see the [Terminology Guide](/docs/terminology).
