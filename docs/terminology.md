# Poker Terminology

> For complete game rules and tournament structure, see the [Rules Documentation](/docs/rules).
> For technical implementation details, see the [Game Engine Documentation](/docs/game-engine).

## Core Game Structure

### Match

A match consists of 1000 hands between two bots. The bot with the most chips at the end of all hands wins the match. Each bot starts each match with 1000 chips.

### Hand

A single poker hand is one complete round of play, from dealing cards to awarding the pot. Each hand:

- Starts with each player being dealt **5 cards** (hole cards)
- On the flop, there is a **discard round** where each player **discards 3 cards** and keeps 2
- Progresses through 4 betting streets (pre-flop, flop, turn, river)
- Ends when either one player folds or showdown

### Streets

A street is one complete round of betting. This variant uses 4 streets, with a mandatory discard round on the flop:

- **Pre-flop** (Street 0):
  - Each player is dealt **5 cards**
  - Small blind (1 chip) and big blind (2 chips) are posted
  - First round of betting occurs

- **Flop** (Street 1):
  - The first three community cards are dealt
  - **Discard round**: each player **discards 3 cards** and keeps 2, in betting order. Both players must discard. Discarded cards are revealed to the opponent.
  - Then flop betting occurs

- **Turn** (Street 2):
  - Fourth community card is dealt
  - Betting occurs

- **River** (Street 3):
  - Fifth community card is dealt
  - Final betting; if no one folds, showdown occurs

## Basic Terms

### Positions

- **Small Blind (SB)**: Player 0, posts 1 chip before cards are dealt
- **Big Blind (BB)**: Player 1, posts 2 chips before cards are dealt

### Actions

- **Fold**: Give up the hand and any chips bet
- **Check**: Pass the action when no additional bet is required
- **Call**: Match the current bet amount
- **Raise**: Increase the current bet amount
- **Discard**: On the flop, choose 2 cards to keep from your 5 hole cards; the other 3 are discarded and revealed to the opponent (tournament-specific rule)

> See the [Game Engine Documentation](/docs/game-engine#action-space) for technical implementation details.

## Hand Rankings

> See [Rules Documentation](/docs/rules#hand-rankings) for detailed hand rankings and examples.

## Strategic Concepts

### Pot Odds

The ratio of the current pot size to the cost of a call. Used to make mathematically sound calling decisions.

Example:

- Pot contains 10 chips
- Opponent bets 5 chips
- Pot odds are 15:5 or 3:1

### Position

Your place in the betting order, which affects strategic decisions:

- **Out of Position (OOP)**: Acting first (Small Blind)
- **In Position (IP)**: Acting last (Big Blind)

### Ranges

The collection of possible hands an opponent might have based on their actions.

## Engine-Specific Terms

For technical details about how the game engine represents and handles:

- [Card Representation](/docs/game-engine#card-representation)
- [Action Space](/docs/game-engine#action-space)
- [Observation Space](/docs/game-engine#observation-space)

The following terms are commonly used in the API:

### Observation Dictionary Keys

- `street`: Current betting round (0-3: pre-flop, flop, turn, river)
- `acting_agent`: Which player acts next (0 or 1)
- `my_cards`: Your hole cards
- `community_cards`: Visible shared cards

## Common Abbreviations

- **BB**: Big Blind
- **SB**: Small Blind
- **OOP**: Out of Position
- **IP**: In Position
- **EV**: Expected Value
- **SPR**: Stack-to-Pot Ratio
