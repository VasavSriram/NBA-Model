# NBA Player Prop & Arbitrage Model

A quantitative sports analytics system that combines statistical modeling, probability theory, and real-time odds data to identify +EV (positive expected value) betting opportunities on NBA player props.

Built as a student-led research project at the University at Albany, advised by Prof. Chun-Yu Ho (Department of Economics).

---

## What It Does

The system pulls live NBA player data and betting odds, runs them through a custom probability model, and flags props where our estimated true probability exceeds the bookmaker's implied probability — i.e., where there is an edge.

The full pipeline:

```
NBA API (game logs, play-by-play)
        ↓
Player Stats Database (season avg, L5, L10)
        ↓
Probability Model (Normal / Poisson distribution)
        ↓
Edge Calculator (EV, Kelly Criterion, vig removal)
        ↓
Odds API (live bookmaker lines)
        ↓
+EV Scanner (ranked opportunities output)
```

---

## Modules

### `probability_model.py`
The statistical core of the system. For any player prop (e.g. "LeBron over 24.5 points"), it calculates P(over) and P(under) using the appropriate distribution:

- **Normal distribution** — used for high-volume stats (PTS, PRA, MIN) with continuity correction to account for the discrete nature of game outcomes
- **Poisson distribution** — used for low-count stats (3PM, STL, BLK) and for rebounds/assists when the player's average is below 5

When a game log is available (≥5 games), it calculates the actual sample standard deviation. Otherwise it estimates std_dev from empirically-derived coefficients of variation (CV) for each stat type.

The effective mean is a weighted blend across time windows:
```
effective_avg = 0.40 × season_avg + 0.35 × last10_avg + 0.25 × last5_avg
```
Head-to-head history against the opponent is incorporated when available (≥2 games found):
```
h2h_avg (3+ games): 0.50 × H2H + 0.30 × last5 + 0.20 × season
h2h_avg (2 games):  0.30 × H2H + 0.40 × last5 + 0.30 × season
```

### `edge_cal.py`
Pure-math betting utilities with no external dependencies:

- **Odds conversion** — American ↔ decimal ↔ implied probability
- **Edge** = model probability − book implied probability
- **Expected Value** = (p × profit) − ((1 − p) × stake)
- **Kelly Criterion** — optimal fraction of bankroll to wager: `f* = (b×p − q) / b`
- **Half/Quarter Kelly** — conservative variants for uncertain edges
- **Vig removal** — strips the bookmaker's overround to find the fair (no-vig) price
- **Parlay analysis** — combined probability, decimal payout, and EV for multi-leg bets
- **Sharp line comparison** — benchmarks retail odds against a sharp book's de-vigged price

### `odds_api.py`
Connects to [The Odds API](https://the-odds-api.com) to fetch live NBA player prop lines across all major US bookmakers. Features:

- Fetches all supported markets: PTS, REB, AST, 3PM, STL, BLK, PRA, PR, PA
- SQLite caching (2-hour TTL) to conserve monthly API quota
- API usage tracking with warnings when quota runs low
- Manual prop entry for offline verification
- Daily scan workflow: fetches → caches → analyzes → ranks +EV opportunities
- Generates a markdown EV report (`ev_report_YYYY-MM-DD.md`)

### `main.py`
Interactive player analysis tool with three modes:

**Mode 1 — Single Player Analysis**
- Pulls full season game log via NBA API
- Displays season stats card (PPG, RPG, APG, FG%, 3P%, FT%, etc.)
- Fetches play-by-play for last N games: shows every scoring play, rebound, assist, and substitution with minute timestamps
- Tracks **fatigue**: calculates rest days between games, flags back-to-backs (B2B) and long gaps (potential absence/rest)
- Optional: filter by opponent team (vs. LAL) or compare head-to-head against another player

**Mode 5 — Batch Game Day Analysis**
- Takes multiple matchups as input (e.g. "Lakers vs Celtics, Warriors vs Suns")
- Automatically pulls top 8 players per team by minutes played
- Outputs PPG/RPG/APG for each player in their last N games + their historical averages vs. today's opponent
- Saves full game day report to a text file

### `team_history.py`
Fetches and displays full box scores for a team's last N games. Outputs per-player stats (MIN, PTS, REB, AST, STL, BLK, FG%, 3P%, ±) for every player who appeared.

### `player_stats_db.py`
SQLite-backed player stats cache. Stores and retrieves season averages, recent game logs, and per-game splits to avoid redundant API calls.

### `parlay_builder.py`
Builds and evaluates multi-leg parlays using the probability model and edge calculator.

### `prop_analyzer.py`
Interactive prop analysis: enter a player, stat type, line, and odds — get back P(over), P(under), edge, EV, and Kelly fraction.

### `probability_grid.py`
Generates probability grids across a range of lines and stat types for a given player, useful for visualizing where value exists relative to bookmaker pricing.

### `charts.py`
Visualization layer using matplotlib/plotly: plots player stat trends, probability distributions, and EV charts.

---

## Setup

**Requirements**
```
Python 3.10+
nba_api
pandas
matplotlib
plotly
scipy
numpy
requests
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Odds API Key**

This project uses [The Odds API](https://the-odds-api.com) (free tier: 500 requests/month). Set your key as an environment variable:
```bash
export ODDS_API_KEY=your_key_here
```

---

## Usage

**Player analysis (interactive):**
```bash
python main.py
```
Follow the prompts — enter a player name, number of games, and optionally an opponent team or rival player.

**Team box score history:**
```bash
python team_history.py
```

**Daily +EV prop scan:**
```bash
python odds_api.py
```
Fetches today's games, analyzes all available props, and prints the top opportunities ranked by EV.

---

## Example Output

**Single player — season card:**
```
====================================================
  Jalen Brunson
  Season: 2024-25   GP: 58   Record (team W-L): 38-20
====================================================
  PPG: 26.4   RPG: 3.6   APG: 7.5   MPG: 34.2
  SPG: 0.9    BPG: 0.2   TOV: 2.8
  FG%: 47.2   3P%: 39.1  FT%: 85.4
====================================================
```

**+EV prop scanner:**
```
  Player                 Stat   Line  Side  Odds    Prob   Edge      EV  Fair
  -------------------------------------------------------------------------
  Jalen Brunson          PTS    25.5  OVER  -108   58.3%  +7.0%   +$1.22  -138
  Anthony Edwards        3PM     3.5  OVER  -115   55.1%  +3.8%   +$0.61  -123
```

---

## Methodology Notes

- **Continuity correction**: For integer-outcome stats modeled with a normal distribution, we apply a ±0.5 correction at the threshold to map the discrete distribution onto the continuous normal curve.
- **CV-based estimation**: When game log data is insufficient, standard deviation is estimated using empirically-derived coefficients of variation (e.g. CV ≈ 0.35 for PTS, 0.60 for 3PM).
- **Probability clamping**: All probabilities are bounded to [0.001, 0.999] to prevent degenerate Kelly fractions.
- **Fatigue modeling**: Rest days are computed from game dates; back-to-backs and long gaps are flagged as context for performance expectations.

---

## Limitations

- Model assumes statistical independence between game outcomes (no lineup injury adjustments)
- Poisson model assumes stationarity of the rate parameter across games
- Arbitrage detection is bounded by the Odds API's bookmaker coverage
- Results are for research and educational purposes

---

## Tech Stack

`Python` · `SQLite` · `NBA API` · `SciPy` · `NumPy` · `pandas` · `matplotlib` · `plotly` · `The Odds API`
