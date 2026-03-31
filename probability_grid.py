"""
probability_grid.py - NBA Player Probability Grid Generator

Generates probability tables showing P(X+) for multiple stat thresholds
for every player in tonight's NBA games.
"""

import argparse
import csv
import itertools
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import odds_api
import player_stats_db
import probability_model
from player_stats_db import PlayerNotFoundError, PlayerStats, StatsNotAvailableError


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StatProbabilities:
    """Probabilities for a single stat at multiple thresholds."""
    stat_type: str
    thresholds: List[float]
    probabilities: List[float]
    season_avg: float
    last5_avg: float
    effective_avg: float
    distribution: str  # 'normal' or 'poisson'


@dataclass
class PlayerGrid:
    """Complete probability grid for one player."""
    player_name: str
    player_id: int
    team: str
    opponent: str
    game_time: str
    matchup: str  # "LAL @ MIL"

    stats: Dict[str, StatProbabilities]

    games_played: int
    minutes_avg: float


@dataclass
class GameGrid:
    """Probability grids for all players in a single game."""
    game_id: str
    home_team: str
    away_team: str
    matchup: str
    game_time: str

    players: List[PlayerGrid]


@dataclass
class DailyGrid:
    """All probability grids for today's games."""
    date: str
    games: List[GameGrid]
    total_players: int
    generation_time: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Configuration for probability grid generation."""
    stats: List[str] = field(default_factory=lambda: ['PTS', 'REB', 'AST', '3PM'])
    thresholds: Dict[str, List[float]] = field(default_factory=lambda: {
        'PTS': [10, 15, 20, 25, 30, 35, 40],
        'REB': [3, 5, 7, 10, 12, 15],
        'AST': [3, 5, 7, 10, 12],
        '3PM': [1, 2, 3, 4, 5],
    })
    min_games_played: int = 10
    min_minutes_avg: float = 15.0
    min_probability_display: float = 0.05


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_LINE_CHAR = '═'
_WIDTH = 72


def _parse_minutes(min_val) -> float:
    """Parse minutes value which may be 'MM:SS' string or a number."""
    if isinstance(min_val, (int, float)):
        return float(min_val)
    if isinstance(min_val, str):
        if ':' in min_val:
            parts = min_val.split(':')
            try:
                return float(parts[0]) + float(parts[1]) / 60
            except (ValueError, IndexError):
                pass
        try:
            return float(min_val)
        except ValueError:
            pass
    return 0.0


def _format_game_time(raw_time: str) -> str:
    """Convert ISO datetime string to readable ET time."""
    if not raw_time:
        return ""
    try:
        dt = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
        et = dt - timedelta(hours=4)  # approximate UTC → ET
        hour = et.hour % 12 or 12
        ampm = 'AM' if et.hour < 12 else 'PM'
        return f"{hour}:{et.strftime('%M')} {ampm} ET"
    except Exception:
        return raw_time


def _stat_key(stat_type: str) -> str:
    """Map canonical stat type to the key used in PlayerStats avg dicts."""
    mapping = {
        'PTS': 'points',
        'REB': 'rebounds',
        'AST': 'assists',
        '3PM': 'three_pm',
        'STL': 'steals',
        'BLK': 'blocks',
        'MIN': 'minutes',
        'TOV': 'turnovers',
    }
    return mapping.get(stat_type.upper(), stat_type.lower())


def _print_progress(current: int, total: int, width: int = 40) -> None:
    """Overwrite current line with a progress bar."""
    if total <= 0:
        return
    pct = min(current / total, 1.0)
    filled = int(width * pct)
    bar = '\u2588' * filled + '\u2591' * (width - filled)
    print(f"\r  {bar} {current}/{total}", end='', flush=True)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def generate_player_grid(
    player_name: str,
    opponent: str = "",
    matchup: str = "",
    game_time: str = "",
    config: GridConfig = None,
) -> Optional[PlayerGrid]:
    """
    Generate a full probability grid for a single player.
    Returns None if the player doesn't meet minimum criteria or stats are unavailable.
    """
    if config is None:
        config = GridConfig()

    try:
        stats: PlayerStats = player_stats_db.get_player_stats(player_name)
    except (PlayerNotFoundError, StatsNotAvailableError):
        return None
    except Exception:
        return None

    if stats.games_played < config.min_games_played:
        return None

    # Average minutes from game log
    minutes_avg = 0.0
    if stats.game_log:
        min_vals = [_parse_minutes(g.get('min', 0)) for g in stats.game_log]
        min_vals = [m for m in min_vals if m > 0]
        if min_vals:
            minutes_avg = sum(min_vals) / len(min_vals)

    if minutes_avg > 0 and minutes_avg < config.min_minutes_avg:
        return None

    grid_stats: Dict[str, StatProbabilities] = {}
    for stat_type in config.stats:
        thresholds = config.thresholds.get(stat_type, [])
        if not thresholds:
            continue

        key = _stat_key(stat_type)
        season_avg = (stats.season_avg or {}).get(key, 0.0)
        last5_avg = (stats.last5_avg or {}).get(key, 0.0)
        effective_avg = (stats.effective_avg or {}).get(key, 0.0)

        if effective_avg <= 0:
            continue

        distribution = probability_model.recommend_distribution(stat_type, effective_avg)
        std_dev = probability_model.estimate_std_dev(effective_avg, stat_type)

        probs = []
        for line in thresholds:
            if distribution == 'normal':
                prob = probability_model.calc_over_prob_normal(effective_avg, std_dev, line)
            else:
                prob = probability_model.calc_over_prob_poisson(effective_avg, line)
            probs.append(round(prob, 3))

        grid_stats[stat_type] = StatProbabilities(
            stat_type=stat_type,
            thresholds=thresholds,
            probabilities=probs,
            season_avg=season_avg,
            last5_avg=last5_avg,
            effective_avg=effective_avg,
            distribution=distribution,
        )

    if not grid_stats:
        return None

    # Infer opponent from matchup string when not provided directly
    if not opponent and matchup and stats.team:
        parts = matchup.split(' @ ')
        if len(parts) == 2:
            if stats.team.lower() in parts[0].lower() or parts[0].lower() in stats.team.lower():
                opponent = parts[1]
            else:
                opponent = parts[0]

    return PlayerGrid(
        player_name=stats.player_name,
        player_id=stats.player_id,
        team=stats.team or "",
        opponent=opponent,
        game_time=game_time,
        matchup=matchup,
        stats=grid_stats,
        games_played=stats.games_played,
        minutes_avg=minutes_avg,
    )


def generate_game_grid(
    home_team: str,
    away_team: str,
    game_id: str = "",
    game_time: str = "",
    config: GridConfig = None,
) -> GameGrid:
    """Generate probability grids for all available players in a single game."""
    if config is None:
        config = GridConfig()

    matchup = f"{away_team} @ {home_team}"
    player_grids: List[PlayerGrid] = []

    players_info: List[Dict] = []
    if game_id:
        try:
            props = odds_api.fetch_player_props(game_id)
            seen: set = set()
            for prop in props:
                if prop.player_name not in seen:
                    seen.add(prop.player_name)
                    players_info.append({
                        'player_name': prop.player_name,
                        'game_time': prop.game_time or game_time,
                    })
        except Exception:
            pass

    for info in players_info:
        pg = generate_player_grid(
            player_name=info['player_name'],
            matchup=matchup,
            game_time=info.get('game_time', game_time),
            config=config,
        )
        if pg is not None:
            player_grids.append(pg)

    player_grids.sort(key=lambda p: p.minutes_avg, reverse=True)

    return GameGrid(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        matchup=matchup,
        game_time=game_time,
        players=player_grids,
    )


def generate_daily_grid(config: GridConfig = None) -> DailyGrid:
    """
    Generate probability grids for all players across every game today.
    Prints progress as it runs.
    """
    if config is None:
        config = GridConfig()

    t_start = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    print("  Fetching today's games...")
    try:
        games_data = odds_api.fetch_nba_games()
    except Exception as e:
        print(f"  Warning: could not fetch games: {e}")
        games_data = []

    if not games_data:
        print("  No games found for today.")
        return DailyGrid(
            date=today, games=[], total_players=0,
            generation_time=time.time() - t_start,
        )

    # First pass: collect all player names per game (one props fetch per game)
    all_game_players: List[List[Dict]] = []
    for game in games_data:
        game_id = game.get('id', '')
        game_time = game.get('commence_time', '')
        players: List[Dict] = []
        try:
            props = odds_api.fetch_player_props(game_id)
            seen: set = set()
            for prop in props:
                if prop.player_name not in seen:
                    seen.add(prop.player_name)
                    players.append({
                        'player_name': prop.player_name,
                        'game_time': prop.game_time or game_time,
                    })
        except Exception:
            pass
        all_game_players.append(players)

    total_expected = sum(len(p) for p in all_game_players)
    print(f"  Found {len(games_data)} games, {total_expected} players")
    print(f"\n  Generating probability grids...")

    processed = 0
    game_grids: List[GameGrid] = []

    for i, game in enumerate(games_data):
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        game_time = game.get('commence_time', '')
        matchup = f"{away_team} @ {home_team}"

        player_grids: List[PlayerGrid] = []
        for info in all_game_players[i]:
            pg = generate_player_grid(
                player_name=info['player_name'],
                matchup=matchup,
                game_time=info.get('game_time', game_time),
                config=config,
            )
            if pg is not None:
                player_grids.append(pg)
            processed += 1
            _print_progress(processed, total_expected)

        player_grids.sort(key=lambda p: p.minutes_avg, reverse=True)

        game_grids.append(GameGrid(
            game_id=game.get('id', ''),
            home_team=home_team,
            away_team=away_team,
            matchup=matchup,
            game_time=game_time,
            players=player_grids,
        ))

    print()  # newline after progress bar

    total_players = sum(len(g.players) for g in game_grids)
    return DailyGrid(
        date=today,
        games=game_grids,
        total_players=total_players,
        generation_time=time.time() - t_start,
    )


def get_todays_players() -> List[Dict]:
    """
    Return list of players with props available today.
    Each entry: {player_name, team, opponent, game_time, matchup}
    """
    result = []
    try:
        games = odds_api.fetch_nba_games()
    except Exception:
        return result

    for game in games:
        game_id = game.get('id', '')
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        game_time = game.get('commence_time', '')
        matchup = f"{away_team} @ {home_team}"

        try:
            props = odds_api.fetch_player_props(game_id)
            seen: set = set()
            for prop in props:
                if prop.player_name in seen:
                    continue
                seen.add(prop.player_name)
                result.append({
                    'player_name': prop.player_name,
                    'team': '',
                    'opponent': '',
                    'game_time': prop.game_time or game_time,
                    'matchup': matchup,
                })
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Filtering Functions
# ---------------------------------------------------------------------------

def filter_by_probability(
    grid: DailyGrid,
    min_prob: float,
    stat_type: str = None,
) -> List[Tuple[PlayerGrid, str, float, float]]:
    """
    Find all (player, stat, threshold, probability) where prob >= min_prob.
    Optionally filter to a specific stat type.
    Returns sorted by probability descending.
    """
    results: List[Tuple[PlayerGrid, str, float, float]] = []
    for game in grid.games:
        for player in game.players:
            for stype, sp in player.stats.items():
                if stat_type and stype != stat_type:
                    continue
                for threshold, prob in zip(sp.thresholds, sp.probabilities):
                    if prob >= min_prob:
                        results.append((player, stype, threshold, prob))
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def filter_high_confidence_legs(
    grid: DailyGrid,
    min_prob: float = 0.75,
) -> List[Dict]:
    """
    Return all legs with probability >= min_prob as dicts ready for parlay building.
    Format: {player, stat, line, side, prob, matchup, game_time}
    """
    raw = filter_by_probability(grid, min_prob)
    return [
        {
            'player': player.player_name,
            'stat': stat,
            'line': line,
            'side': 'OVER',
            'prob': prob,
            'matchup': player.matchup,
            'game_time': player.game_time,
        }
        for player, stat, line, prob in raw
    ]


def find_best_threshold(
    player_grid: PlayerGrid,
    stat_type: str,
    target_prob: float,
) -> Optional[Tuple[float, float]]:
    """
    Find the highest threshold that still meets target_prob.
    Returns (threshold, actual_probability), or None if none qualify.
    """
    if stat_type not in player_grid.stats:
        return None
    sp = player_grid.stats[stat_type]
    best: Optional[Tuple[float, float]] = None
    for threshold, prob in zip(sp.thresholds, sp.probabilities):
        if prob >= target_prob:
            if best is None or threshold > best[0]:
                best = (threshold, prob)
    return best


# ---------------------------------------------------------------------------
# Display Functions
# ---------------------------------------------------------------------------

def print_player_grid(player: PlayerGrid, config: GridConfig = None) -> None:
    """Print a formatted probability grid for a single player."""
    if config is None:
        config = GridConfig()

    header = f"{player.player_name.upper()} ({player.team})"
    if player.opponent:
        header += f" vs {player.opponent}"
    if player.game_time:
        header += f" | {_format_game_time(player.game_time)}"

    print(_LINE_CHAR * _WIDTH)
    print(header)

    summary_parts: List[str] = []
    for stat_type in ('PTS', 'REB', 'AST'):
        if stat_type in player.stats:
            avg = player.stats[stat_type].effective_avg
            labels = {'PTS': 'PPG', 'REB': 'RPG', 'AST': 'APG'}
            summary_parts.append(f"{avg:.1f} {labels[stat_type]}")
    summary_parts.append(f"{player.games_played} games")
    print("  " + " | ".join(summary_parts))
    print(_LINE_CHAR * _WIDTH)

    for stat_type, sp in player.stats.items():
        label = f"  {stat_type}:"
        parts = []
        for threshold, prob in zip(sp.thresholds, sp.probabilities):
            if prob >= config.min_probability_display:
                parts.append(f"{threshold:>4g}+ ({prob * 100:.0f}%)")
        if parts:
            print(f"{label:<7} {'   '.join(parts)}")

    print()


def print_game_grid(game: GameGrid, top_n_players: int = 10) -> None:
    """Print grids for the top N players in a game (sorted by minutes)."""
    print(f"\n{'═' * _WIDTH}")
    time_str = _format_game_time(game.game_time)
    suffix = f" | {time_str}" if time_str else ""
    print(f"GAME: {game.matchup}{suffix}")
    print(f"{'═' * _WIDTH}")
    for player in game.players[:top_n_players]:
        print_player_grid(player)


def print_daily_summary(grid: DailyGrid) -> None:
    """Print a summary of all games and player counts."""
    print(f"\n{'═' * _WIDTH}")
    print(f"DAILY SUMMARY — {grid.date}")
    print(f"{'═' * _WIDTH}")
    print(
        f"  {len(grid.games)} games | {grid.total_players} players | "
        f"Generated in {grid.generation_time:.1f}s"
    )
    for game in grid.games:
        time_str = _format_game_time(game.game_time)
        print(f"  • {game.matchup:<32} {len(game.players):>3} players   {time_str}")
    print()


def print_high_prob_legs(legs: List[Dict], min_prob: float = 0.75) -> None:
    """Print a formatted table of high-probability legs."""
    shown = [l for l in legs if l['prob'] >= min_prob]
    if not shown:
        print(f"\n  No legs found with probability >= {min_prob:.0%}")
        return

    inner = _WIDTH - 2
    title = f"  HIGH PROBABILITY LEGS (>{min_prob:.0%})  "

    print(f"\n╔{'═' * inner}╗")
    print(f"║{title:<{inner}}║")
    print(f"╠{'═' * inner}╣")
    header = f"  {'PLAYER':<22} {'STAT':<5} {'LINE':>5}  {'PROB':>6}   {'MATCHUP'}"
    print(f"║{header:<{inner}}║")
    print(f"║  {'─' * (inner - 2)}║")
    for leg in shown:
        row = (
            f"  {leg['player']:<22} {leg['stat']:<5} {leg['line']:>5.1f}  "
            f"{leg['prob'] * 100:>5.1f}%   {leg['matchup']}"
        )
        print(f"║{row:<{inner}}║")
    print(f"╚{'═' * inner}╝")
    print(f"\n  Found {len(shown)} legs with >{min_prob:.0%} probability")


# ---------------------------------------------------------------------------
# Parlay Integration
# ---------------------------------------------------------------------------

def estimate_parlay_probability(legs: List[Dict]) -> float:
    """
    Calculate combined probability of hitting all legs.
    Assumes independence — same-game parlays may be correlated.
    """
    if not legs:
        return 0.0
    prob = 1.0
    for leg in legs:
        prob *= leg['prob']
    return round(prob, 4)


def build_high_prob_parlay(
    grid: DailyGrid,
    num_legs: int = 4,
    min_leg_prob: float = 0.75,
    diversify_games: bool = True,
) -> List[Dict]:
    """
    Build a parlay from the highest-probability legs.
    With diversify_games=True, spreads selections across different matchups first.
    """
    candidates = filter_high_confidence_legs(grid, min_leg_prob)
    if not candidates:
        return []

    if not diversify_games:
        return candidates[:num_legs]

    selected: List[Dict] = []
    seen_matchups: set = set()

    # First pass: one leg per matchup
    for leg in candidates:
        if leg['matchup'] not in seen_matchups:
            selected.append(leg)
            seen_matchups.add(leg['matchup'])
        if len(selected) >= num_legs:
            break

    # Second pass: fill remaining slots
    if len(selected) < num_legs:
        for leg in candidates:
            if leg not in selected:
                selected.append(leg)
            if len(selected) >= num_legs:
                break

    return selected[:num_legs]


def suggest_parlays(
    grid: DailyGrid,
    target_prob: float = 0.50,
    num_legs: int = 3,
    top_candidates: int = 30,
) -> List[List[Dict]]:
    """
    Find parlay combinations whose combined probability is close to target_prob.
    Returns up to 5 options sorted by closeness to the target.
    """
    candidates = filter_high_confidence_legs(grid, 0.60)[:top_candidates]
    if len(candidates) < num_legs:
        return []

    tolerance = 0.15
    matching: List[List[Dict]] = []

    for combo in itertools.combinations(candidates, num_legs):
        combined = estimate_parlay_probability(list(combo))
        if abs(combined - target_prob) <= tolerance:
            matching.append(list(combo))
        if len(matching) >= 50:
            break

    matching.sort(key=lambda c: abs(estimate_parlay_probability(c) - target_prob))
    return matching[:5]


# ---------------------------------------------------------------------------
# Probability-Based Parlay Builder
# ---------------------------------------------------------------------------

def build_probability_parlay(
    grid: DailyGrid,
    num_legs: int = 4,
    min_leg_prob: float = 0.70,
    target_combined_prob: float = 0.40,
    diversify_games: bool = True,
) -> List[Dict]:
    """
    Build a parlay optimized for WIN PROBABILITY, not EV.

    This is different from EV-based parlays:
    - EV parlays: Find mispriced lines (value)
    - Probability parlays: Find likely outcomes (reliability)

    Args:
        grid: DailyGrid with all player probabilities
        num_legs: Number of legs in parlay
        min_leg_prob: Minimum probability for each leg (default 70%)
        target_combined_prob: Target combined hit rate (default 40%)
        diversify_games: Spread across different games

    Returns:
        List of leg dicts ready for betting
    """
    candidates = filter_high_confidence_legs(grid, min_prob=min_leg_prob)

    if len(candidates) < num_legs:
        return []

    candidates.sort(key=lambda x: x['prob'], reverse=True)

    if diversify_games:
        selected: List[Dict] = []
        seen_matchups: set = set()

        for leg in candidates:
            if leg['matchup'] not in seen_matchups:
                selected.append(leg)
                seen_matchups.add(leg['matchup'])
            if len(selected) >= num_legs:
                break

        if len(selected) < num_legs:
            for leg in candidates:
                if leg not in selected:
                    selected.append(leg)
                if len(selected) >= num_legs:
                    break
    else:
        selected = candidates[:num_legs]

    return selected[:num_legs]


def compare_parlay_strategies(grid: DailyGrid, ev_props: List = None) -> None:
    """
    Print comparison of probability-based vs EV-based parlays.

    Shows user both approaches side by side.
    """
    print("\n" + "=" * 70)
    print("  PARLAY STRATEGY COMPARISON")
    print("=" * 70)

    prob_parlay = build_probability_parlay(grid, num_legs=4, min_leg_prob=0.75)

    print("\n  PROBABILITY-BASED PARLAY (optimized for WIN RATE)")
    print("  " + "-" * 60)
    if prob_parlay:
        combined = 1.0
        for i, leg in enumerate(prob_parlay, 1):
            print(f"  {i}. {leg['player']:<20} {leg['stat']} O{leg['line']:<5} ({leg['prob']:.0%})")
            combined *= leg['prob']
        print(f"\n  Combined probability: {combined:.1%}")
        print(f"  Strategy: HIGH WIN RATE, lower payouts")
    else:
        print("  No qualifying legs found")

    print("\n  EV-BASED PARLAY (run parlay_builder.py)")
    print("  " + "-" * 60)
    print("  Strategy: Find MISPRICED lines for value")
    print("  Run: python parlay_builder.py")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def _ensure_reports_dir() -> str:
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def generate_grid_report(grid: DailyGrid) -> str:
    """Generate a markdown report and save to reports/probability_grid_YYYY-MM-DD.md."""
    lines = [
        f"# NBA Probability Grid — {grid.date}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"{len(grid.games)} games | {grid.total_players} players",
        "",
    ]

    for game in grid.games:
        time_str = _format_game_time(game.game_time)
        lines.append(f"## {game.matchup}  |  {time_str}")
        lines.append("")
        for player in game.players:
            lines.append(f"### {player.player_name} ({player.team})")
            lines.append(f"*{player.games_played} games | {player.minutes_avg:.1f} min avg*")
            lines.append("")
            for stat_type, sp in player.stats.items():
                cells = " | ".join(
                    f"{t:g}+ ({p*100:.0f}%)"
                    for t, p in zip(sp.thresholds, sp.probabilities)
                )
                lines.append(f"**{stat_type}:** {cells}  ")
            lines.append("")

    content = "\n".join(lines)
    reports_dir = _ensure_reports_dir()
    filepath = os.path.join(reports_dir, f"probability_grid_{grid.date}.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Saved markdown report: {filepath}")
    return filepath


def generate_html_report(grid: DailyGrid) -> str:
    """
    Generate an HTML report with color-coded probability tables.
    Green >80%, yellow 60-80%, red <60%.
    Saved to reports/probability_grid_YYYY-MM-DD.html.
    """
    html: List[str] = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>NBA Probability Grid — {grid.date}</title>",
        "<style>",
        "body{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}",
        "h1{color:#4cc9f0}",
        "h2{color:#7209b7;border-bottom:1px solid #444;padding-bottom:4px;margin-top:30px}",
        "h3{color:#f72585;margin-bottom:4px}",
        "table{border-collapse:collapse;margin-bottom:12px}",
        "td,th{padding:5px 11px;border:1px solid #333;text-align:center}",
        "th{background:#0f3460;color:#4cc9f0;text-align:left;padding-right:16px}",
        ".hi{background:#d8f3dc;color:#1b4332}",
        ".md{background:#fff3b0;color:#7b5e00}",
        ".lo{background:#ffd6d6;color:#7b0000}",
        "small{color:#aaa}",
        "</style></head><body>",
        f"<h1>NBA Probability Grid — {grid.date}</h1>",
        f"<p>{len(grid.games)} games | {grid.total_players} players | "
        f"Generated {datetime.now().strftime('%H:%M')}</p>",
    ]

    for game in grid.games:
        time_str = _format_game_time(game.game_time)
        html.append(f"<h2>{game.matchup} &mdash; {time_str}</h2>")
        for player in game.players:
            html.append(f"<h3>{player.player_name} ({player.team})</h3>")
            html.append(
                f"<small>{player.games_played} games | "
                f"{player.minutes_avg:.1f} min avg</small>"
            )
            html.append("<table>")
            for stat_type, sp in player.stats.items():
                html.append("<tr>")
                html.append(f"<th>{stat_type}</th>")
                for t, p in zip(sp.thresholds, sp.probabilities):
                    cls = "hi" if p >= 0.80 else ("md" if p >= 0.60 else "lo")
                    html.append(
                        f"<td class='{cls}'>{t:g}+<br><strong>{p*100:.0f}%</strong></td>"
                    )
                html.append("</tr>")
            html.append("</table>")

    html.append("</body></html>")
    content = "\n".join(html)

    reports_dir = _ensure_reports_dir()
    filepath = os.path.join(reports_dir, f"probability_grid_{grid.date}.html")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Saved HTML report: {filepath}")
    return filepath


def export_to_csv(grid: DailyGrid, filepath: str = None) -> str:
    """Export all player/stat/threshold/probability rows to CSV."""
    if filepath is None:
        reports_dir = _ensure_reports_dir()
        filepath = os.path.join(reports_dir, f"probability_grid_{grid.date}.csv")

    fieldnames = [
        'player', 'team', 'opponent', 'matchup', 'game_time',
        'stat', 'threshold', 'probability',
        'effective_avg', 'season_avg', 'last5_avg', 'distribution',
    ]
    rows = []
    for game in grid.games:
        for player in game.players:
            for stat_type, sp in player.stats.items():
                for threshold, prob in zip(sp.thresholds, sp.probabilities):
                    rows.append({
                        'player': player.player_name,
                        'team': player.team,
                        'opponent': player.opponent,
                        'matchup': player.matchup,
                        'game_time': player.game_time,
                        'stat': stat_type,
                        'threshold': threshold,
                        'probability': prob,
                        'effective_avg': sp.effective_avg,
                        'season_avg': sp.season_avg,
                        'last5_avg': sp.last5_avg,
                        'distribution': sp.distribution,
                    })

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved CSV: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# CLI Helpers
# ---------------------------------------------------------------------------

def _print_header(date_str: str) -> None:
    title = f"  NBA PROBABILITY GRID — {date_str}  "
    w = max(_WIDTH, len(title) + 4)
    print(f"\n╔{'═' * (w - 2)}╗")
    print(f"║{title:^{w - 2}}║")
    print(f"╚{'═' * (w - 2)}╝\n")


def _print_parlay_box(parlay: List[Dict], num_legs: int, target_prob: float) -> None:
    """Print a formatted parlay suggestion box."""
    if not parlay:
        print("\n  No parlay suggestions available (try lowering --min-prob).")
        return

    combined = estimate_parlay_probability(parlay)
    inner = _WIDTH - 2

    print(f"\n╔{'═' * inner}╗")
    title = f"  SUGGESTED {len(parlay)}-LEG PARLAY (Target: ~{target_prob:.0%} hit rate)  "
    print(f"║{title:<{inner}}║")
    print(f"╠{'═' * inner}╣")
    for i, leg in enumerate(parlay, 1):
        row = (
            f"  {i}. {leg['player']:<24} OVER {leg['line']:>5.1f} {leg['stat']:<4}  "
            f"({leg['prob']*100:.0f}%)   {leg['matchup']}"
        )
        print(f"║{row:<{inner}}║")
    print(f"║  {'─' * (inner - 2)}║")
    print(f"║  Combined Probability: {combined*100:.1f}%{'':<{inner - 26}}║")
    note = "  (Independent assumption — same-game legs may be correlated)"
    print(f"║{note:<{inner}}║")
    print(f"╚{'═' * inner}╝")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog='probability_grid.py',
        description='NBA Player Probability Grid Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  today             Generate grid for today's games (default)
  player NAME       Show grid for specific player
  game TEAM         Show grid for specific game
  legs              Show all high-probability legs
  parlay            Build a high-probability parlay

Examples:
  python probability_grid.py
  python probability_grid.py player "LeBron James"
  python probability_grid.py game Lakers
  python probability_grid.py legs --min-prob 0.80
  python probability_grid.py parlay --legs 4
        """,
    )

    parser.add_argument(
        'command', nargs='?', default='today',
        choices=['today', 'player', 'game', 'legs', 'parlay', 'prob-parlay'],
    )
    parser.add_argument(
        'target', nargs='?', default=None,
        help="Player name or team for the 'player'/'game' commands",
    )
    parser.add_argument(
        '--min-prob', type=float, default=0.05,
        help='Minimum probability to display (default: 0.05)',
    )
    parser.add_argument(
        '--stats', type=str, default='PTS,REB,AST,3PM',
        help='Comma-separated stats to include (default: PTS,REB,AST,3PM)',
    )
    parser.add_argument(
        '--legs', type=int, default=4,
        help='Number of parlay legs (default: 4)',
    )
    parser.add_argument(
        '--target-prob', type=float, default=0.50,
        help='Target combined parlay probability (default: 0.50)',
    )
    parser.add_argument('--save', action='store_true', help='Save markdown report to file')
    parser.add_argument('--html', action='store_true', help='Generate HTML report')
    parser.add_argument('--csv', action='store_true', help='Export all data to CSV')

    args = parser.parse_args()

    stat_list = [s.strip().upper() for s in args.stats.split(',')]
    config = GridConfig(stats=stat_list, min_probability_display=args.min_prob)

    today_label = datetime.now().strftime("%B %d, %Y")
    _print_header(today_label)

    # ---- player ----
    if args.command == 'player':
        name = args.target or ''
        if not name:
            parser.error("'player' command requires a player name, e.g.: player \"LeBron James\"")
        pg = generate_player_grid(name, config=config)
        if pg:
            print_player_grid(pg, config)
        else:
            print(f"  Could not generate grid for '{name}' — check name or stats availability.")
        return

    # ---- game ----
    if args.command == 'game':
        team_filter = (args.target or '').lower()
        try:
            games_data = odds_api.fetch_nba_games()
        except Exception as e:
            print(f"  Error fetching games: {e}")
            return

        matched = False
        for game in games_data:
            home = game.get('home_team', '')
            away = game.get('away_team', '')
            if team_filter and team_filter not in home.lower() and team_filter not in away.lower():
                continue
            matched = True
            gg = generate_game_grid(
                home_team=home,
                away_team=away,
                game_id=game.get('id', ''),
                game_time=game.get('commence_time', ''),
                config=config,
            )
            print_game_grid(gg)

        if not matched:
            print(f"  No game found matching '{args.target}'.")
        return

    # ---- today / legs / parlay — all need the full daily grid ----
    grid = generate_daily_grid(config)

    if args.command == 'today':
        print_daily_summary(grid)
        for game in grid.games:
            print_game_grid(game)

    # High-prob legs
    high_threshold = 0.80 if args.command == 'today' else max(args.min_prob, 0.75)
    legs = filter_high_confidence_legs(grid, min_prob=high_threshold)

    if args.command == 'legs':
        print_high_prob_legs(legs, min_prob=high_threshold)
    else:
        print_high_prob_legs(legs, min_prob=0.80)

    # Parlay
    if args.command == 'parlay':
        parlay = build_high_prob_parlay(
            grid, num_legs=args.legs, min_leg_prob=0.75, diversify_games=True,
        )
        _print_parlay_box(parlay, args.legs, args.target_prob)
        print(f"\n  Run 'python probability_grid.py parlay --legs 3' for different sizes")
    elif args.command == 'prob-parlay':
        min_prob = args.min_prob if args.min_prob > 0.5 else 0.70
        prob_parlay = build_probability_parlay(
            grid,
            num_legs=args.legs,
            min_leg_prob=min_prob,
        )
        if prob_parlay:
            combined = estimate_parlay_probability(prob_parlay)
            print(f"\n  PROBABILITY-OPTIMIZED {len(prob_parlay)}-LEG PARLAY")
            print("  " + "=" * 60)
            for i, leg in enumerate(prob_parlay, 1):
                print(f"  {i}. {leg['player']:<22} {leg['stat']} OVER {leg['line']:<5.1f}  ({leg['prob']:.0%})")
            print("  " + "-" * 60)
            print(f"  Combined Win Probability: {combined:.1%}")
            print(f"\n  Note: This optimizes for WIN RATE, not value.")
            print(f"  For +EV parlays, run: python parlay_builder.py")
        else:
            print("\n  No probability parlay found with current filters.")
            print("  Try: --min-prob 0.65 or --legs 3")
    elif args.command == 'today':
        parlay = build_high_prob_parlay(grid, num_legs=4, min_leg_prob=0.75)
        _print_parlay_box(parlay, 4, 0.50)
        print(f"\n  Run 'python probability_grid.py parlay --legs 3' for different sizes")

    # ---- Optional file output ----
    if args.save:
        generate_grid_report(grid)
    if args.html:
        generate_html_report(grid)
    if args.csv:
        export_to_csv(grid)


if __name__ == '__main__':
    main()
