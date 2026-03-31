"""
prop_analyzer.py -- Main user interface for finding +EV NBA player props.

Orchestrates the full workflow: fetch odds -> get player stats ->
calculate probabilities -> find edges -> rank opportunities.

This is the primary script users run daily.

Dependencies: odds_api, player_stats_db, probability_model, edge_cal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import odds_api
import player_stats_db
import probability_model
import edge_cal


# ---------------------------------------------------------------------------
# Project directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "prop_analyzer_config.json")
_REPORTS_DIR = os.path.join(_SCRIPT_DIR, "reports")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerConfig:
    """Configuration for the prop analyzer."""

    # Filtering
    min_edge: float = 0.02
    max_edge: float = 0.30
    min_prob: float = 0.55
    max_prob: float = 0.95
    min_games_played: int = 10

    # Stat types to analyze
    stat_types: List[str] = field(default_factory=lambda: ["PTS", "REB", "AST", "3PM"])

    # Books to include (None = all)
    bookmakers: Optional[List[str]] = None

    # Sizing
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25
    max_bet_size: float = 50.0

    # Output
    top_n: int = 20
    save_report: bool = True

    # Cache
    max_cache_age_minutes: int = 120
    force_refresh: bool = False


def load_config(path: str = None) -> AnalyzerConfig:
    """Load config from JSON file if it exists, otherwise return defaults."""
    path = path or _CONFIG_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AnalyzerConfig(**{
                k: v for k, v in data.items()
                if k in AnalyzerConfig.__dataclass_fields__
            })
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"  Warning: Could not load config ({e}), using defaults.")
    return AnalyzerConfig()


def save_config(config: AnalyzerConfig, path: str = None) -> None:
    """Save config to JSON for persistence."""
    path = path or _CONFIG_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"  Config saved to: {path}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RankedProp:
    """A prop with full analysis, ready for display/betting."""

    # Core info
    rank: int
    player_name: str
    stat_type: str
    line: float

    # Recommendation
    side: str                       # 'OVER' or 'UNDER'
    book_odds: int
    bookmaker: str

    # Model outputs
    model_prob: float
    book_implied_prob: float
    edge: float
    ev_per_10: float
    fair_odds: int

    # Context
    season_avg: float
    last5_avg: float
    effective_avg: float
    games_played: int
    distribution: str

    # Sizing
    kelly_fraction: float
    half_kelly_fraction: float
    suggested_stake: float

    # Game info
    game_id: str
    matchup: str
    game_time: str

    # Metadata
    confidence: str
    data_age_minutes: int


@dataclass
class DailySummary:
    """Summary of daily analysis."""

    date: str
    games_analyzed: int
    total_props_fetched: int
    total_props_analyzed: int
    props_skipped: int

    plus_ev_count: int
    plus_ev_over_count: int
    plus_ev_under_count: int

    avg_edge: float
    max_edge: float
    total_ev: float

    top_props: List[RankedProp]

    api_requests_used: int
    api_requests_remaining: int

    analysis_time_seconds: float


# ---------------------------------------------------------------------------
# Stat type labels (for display)
# ---------------------------------------------------------------------------

_STAT_LABELS: Dict[str, str] = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "3PM": "Threes",
    "STL": "Steals",
    "BLK": "Blocks",
}


# ---------------------------------------------------------------------------
# Helper: format game time
# ---------------------------------------------------------------------------

def _format_game_time(dt: datetime) -> str:
    """Format a datetime to readable ET time string."""
    try:
        return dt.strftime("%-I:%M %p ET")
    except ValueError:
        # Windows doesn't support '-' in strftime
        return dt.strftime("%I:%M %p ET").lstrip("0")


def _format_matchup(away_team: str, home_team: str) -> str:
    """Format team names into a matchup string."""
    away_short = away_team.split()[-1] if away_team else "???"
    home_short = home_team.split()[-1] if home_team else "???"
    return f"{away_short} @ {home_short}"


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def calculate_confidence(edge: float, games_played: int, prob: float) -> str:
    """Determine confidence level for a prop.

    HIGH:   edge > 5% AND games_played > 30 AND prob between 0.60-0.85
    MEDIUM: edge > 3% AND games_played > 15
    LOW:    everything else
    """
    if edge > 0.05 and games_played > 30 and 0.60 <= prob <= 0.85:
        return "HIGH"
    if edge > 0.03 and games_played > 15:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Stake calculation
# ---------------------------------------------------------------------------

def calculate_suggested_stake(kelly_frac: float, config: AnalyzerConfig) -> float:
    """Calculate suggested bet size using Kelly fraction and config.

    Applies the configured kelly_fraction multiplier to bankroll,
    caps at max_bet_size, and rounds to nearest $0.50.
    """
    raw = kelly_frac * config.kelly_fraction * config.bankroll
    capped = min(raw, config.max_bet_size)
    # Round to nearest $0.50
    rounded = round(capped * 2) / 2
    return max(0.0, rounded)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_props(config: AnalyzerConfig = None) -> Tuple[List[RankedProp], DailySummary]:
    """Main analysis function.

    Fetches props (or uses cache), analyzes each, filters, ranks.
    Returns (ranked_props, summary).
    """
    start_time = time.time()
    if config is None:
        config = load_config()

    # Check API usage
    api_used, api_remaining = 0, 0
    try:
        usage = odds_api.check_usage()
        api_used = usage.get("requests_used", 0)
        api_remaining = usage.get("requests_remaining", 0)
        print(f"\n  API Usage: {api_remaining}/500 remaining")
    except odds_api.OddsAPIError:
        print("  Could not check API usage.")
        # Try to get from local log
        local_usage = odds_api.get_usage_stats()
        api_used = local_usage.get("requests_used", 0)
        api_remaining = local_usage.get("requests_remaining", 0)

    # Determine whether to fetch fresh or use cache
    cache_fresh = odds_api.is_cache_fresh(config.max_cache_age_minutes)

    if config.force_refresh or not cache_fresh:
        if not cache_fresh:
            print("\n  Fetching today's games...")
        else:
            print("\n  Force refreshing props...")
        try:
            props = odds_api.fetch_all_todays_props(stat_types=config.stat_types)
            if props:
                odds_api.cache_props(props)
        except odds_api.RateLimitError:
            print("  WARNING: API rate limited. Using cached data.")
            props = odds_api.get_cached_props(config.max_cache_age_minutes * 2)
        except odds_api.OddsAPIError as e:
            print(f"  WARNING: API error ({e}). Using cached data.")
            props = odds_api.get_cached_props(config.max_cache_age_minutes * 2)
    else:
        props = odds_api.get_cached_props(config.max_cache_age_minutes)
        print(f"\n  Using cached props ({len(props)} props)")

    if not props:
        print("  No games or props available today.")
        elapsed = time.time() - start_time
        empty_summary = DailySummary(
            date=datetime.now().strftime("%Y-%m-%d"),
            games_analyzed=0,
            total_props_fetched=0,
            total_props_analyzed=0,
            props_skipped=0,
            plus_ev_count=0,
            plus_ev_over_count=0,
            plus_ev_under_count=0,
            avg_edge=0.0,
            max_edge=0.0,
            total_ev=0.0,
            top_props=[],
            api_requests_used=api_used,
            api_requests_remaining=api_remaining,
            analysis_time_seconds=round(elapsed, 1),
        )
        return [], empty_summary

    total_fetched = len(props)

    # Filter by bookmakers if configured
    if config.bookmakers:
        books_lower = [b.lower() for b in config.bookmakers]
        props = [p for p in props if p.bookmaker.lower() in books_lower]

    # Filter by stat types
    allowed_stats = {s.upper() for s in config.stat_types}
    props = [p for p in props if p.stat_type in allowed_stats]

    # Count unique games
    game_ids = {p.game_id for p in props}
    games_analyzed = len(game_ids)

    # Analyze each prop
    print(f"\n  Analyzing props...")
    analyses: List[odds_api.PropAnalysis] = []
    skipped = 0
    total_to_analyze = len(props)

    for i, prop in enumerate(props, 1):
        # Progress bar
        if total_to_analyze > 0:
            pct = i / total_to_analyze
            bar_len = 40
            filled = int(bar_len * pct)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            print(f"\r  {bar} {i}/{total_to_analyze}", end="", flush=True)

        try:
            analysis = odds_api.analyze_prop(prop)
        except Exception:
            skipped += 1
            continue

        if analysis is None:
            skipped += 1
            continue

        analyses.append(analysis)

    print()  # newline after progress bar

    if skipped:
        print(f"  Skipped {skipped} props (player not found or unsupported stat)")

    # Rank and filter
    ranked = rank_props(analyses, config)

    # Update API usage after analysis
    try:
        local_usage = odds_api.get_usage_stats()
        api_used = local_usage.get("requests_used", api_used)
        api_remaining = local_usage.get("requests_remaining", api_remaining)
    except Exception:
        pass

    # Build summary
    plus_ev = [r for r in ranked]
    over_count = sum(1 for r in plus_ev if r.side == "OVER")
    under_count = sum(1 for r in plus_ev if r.side == "UNDER")
    edges = [r.edge for r in plus_ev] if plus_ev else [0.0]
    evs = [r.ev_per_10 for r in plus_ev] if plus_ev else [0.0]

    elapsed = time.time() - start_time

    summary = DailySummary(
        date=datetime.now().strftime("%Y-%m-%d"),
        games_analyzed=games_analyzed,
        total_props_fetched=total_fetched,
        total_props_analyzed=len(analyses),
        props_skipped=skipped,
        plus_ev_count=len(plus_ev),
        plus_ev_over_count=over_count,
        plus_ev_under_count=under_count,
        avg_edge=sum(edges) / len(edges) if edges else 0.0,
        max_edge=max(edges) if edges else 0.0,
        total_ev=sum(evs),
        top_props=ranked[:config.top_n],
        api_requests_used=api_used,
        api_requests_remaining=api_remaining,
        analysis_time_seconds=round(elapsed, 1),
    )

    return ranked, summary


def rank_props(
    analyses: List[odds_api.PropAnalysis],
    config: AnalyzerConfig,
) -> List[RankedProp]:
    """Convert PropAnalysis objects to RankedProp, filter, sort, assign ranks."""
    candidates: List[RankedProp] = []

    for a in analyses:
        # Only consider OVER bets (user cannot bet under)
        side = "OVER"
        odds = a.prop.over_odds
        prob = a.prob_over
        edge = a.over_edge
        ev = a.over_ev

        # Skip if over side is not +EV
        if ev <= 0:
            continue

        # Apply config filters
        if edge < config.min_edge or edge > config.max_edge:
            continue
        if prob < config.min_prob or prob > config.max_prob:
            continue
        if a.games_played < config.min_games_played:
            continue

        # Book implied probability
        book_implied = edge_cal.american_to_implied_prob(odds)

        # Kelly fractions
        full_kelly = a.kelly_fraction
        half_kelly = full_kelly / 2

        # Data age
        now = datetime.now()
        try:
            age_delta = now - a.prop.fetched_at
            data_age = int(age_delta.total_seconds() / 60)
        except (TypeError, AttributeError):
            data_age = 0

        # Confidence
        confidence = calculate_confidence(edge, a.games_played, prob)

        # Matchup and game time
        matchup = _format_matchup(a.prop.away_team, a.prop.home_team)
        game_time = _format_game_time(a.prop.game_time)

        # Suggested stake
        stake = calculate_suggested_stake(full_kelly, config)

        candidates.append(RankedProp(
            rank=0,  # assigned below
            player_name=a.prop.player_name,
            stat_type=a.prop.stat_type,
            line=a.prop.line,
            side=side,
            book_odds=odds,
            bookmaker=a.prop.bookmaker,
            model_prob=round(prob, 4),
            book_implied_prob=round(book_implied, 4),
            edge=round(edge, 4),
            ev_per_10=round(ev, 2),
            fair_odds=a.fair_odds,
            season_avg=a.season_avg,
            last5_avg=a.last5_avg,
            effective_avg=a.effective_avg,
            games_played=a.games_played,
            distribution=a.distribution_used,
            kelly_fraction=round(full_kelly, 4),
            half_kelly_fraction=round(half_kelly, 4),
            suggested_stake=stake,
            game_id=a.prop.game_id,
            matchup=matchup,
            game_time=game_time,
            confidence=confidence,
            data_age_minutes=data_age,
        ))

    # Sort by EV descending
    candidates.sort(key=lambda r: r.ev_per_10, reverse=True)

    # Assign ranks
    for i, prop in enumerate(candidates, 1):
        prop.rank = i

    return candidates


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_summary(summary: DailySummary) -> None:
    """Print formatted summary box."""
    d = summary.date
    try:
        dt = datetime.strptime(d, "%Y-%m-%d")
        date_display = dt.strftime("%B %d, %Y")
    except ValueError:
        date_display = d

    w = 72
    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + f"  SUMMARY".ljust(w) + "\u2551")
    print("\u2560" + "\u2550" * w + "\u2563")

    line1 = (
        f"  Props Analyzed: {summary.total_props_analyzed:<6}"
        f"  +EV Found: {summary.plus_ev_count:<6}"
        f"  Skipped: {summary.props_skipped} (no stats/combo)"
    )
    print("\u2551" + line1.ljust(w) + "\u2551")

    avg_edge_str = f"{summary.avg_edge:.1%}" if summary.plus_ev_count else "N/A"
    max_edge_str = f"{summary.max_edge:.1%}" if summary.plus_ev_count else "N/A"
    total_ev_str = f"${summary.total_ev:.2f}" if summary.plus_ev_count else "$0.00"
    line2 = (
        f"  Avg Edge: {avg_edge_str:<12}"
        f"  Max Edge: {max_edge_str:<12}"
        f"  Total EV: {total_ev_str}"
    )
    print("\u2551" + line2.ljust(w) + "\u2551")

    line3 = f"  API Usage: {summary.api_requests_remaining}/500 remaining"
    print("\u2551" + line3.ljust(w) + "\u2551")

    line4 = f"  Analysis time: {summary.analysis_time_seconds:.1f}s"
    print("\u2551" + line4.ljust(w) + "\u2551")

    print("\u255a" + "\u2550" * w + "\u255d")


def print_top_props(props: List[RankedProp], n: int = 20) -> None:
    """Print formatted table of top N props."""
    if not props:
        print("\n  No +EV opportunities found.")
        return

    top = props[:n]
    w = 72

    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + "  TOP +EV OPPORTUNITIES".ljust(w) + "\u2551")
    print("\u2560" + "\u2550" * w + "\u2563")

    header = (
        f"  {'#':<4}"
        f"{'PLAYER':<20}"
        f"{'STAT':<5}"
        f"{'LINE':>5}  "
        f"{'SIDE':<5}"
        f"{'ODDS':>5}  "
        f"{'PROB':>5}  "
        f"{'EDGE':>5}  "
        f"{'EV':>6}  "
        f"{'CONF':<4}"
    )
    print("\u2551" + header[:w].ljust(w) + "\u2551")
    print("\u2551" + "  " + "\u2500" * (w - 2) + "\u2551")

    for p in top:
        name = p.player_name[:19]
        odds_str = f"{p.book_odds:+d}"
        prob_str = f"{p.model_prob:.0%}"
        edge_str = f"+{p.edge:.1%}"
        ev_str = f"${p.ev_per_10:.2f}"
        conf = p.confidence[:3]

        row = (
            f"  {p.rank:<4}"
            f"{name:<20}"
            f"{p.stat_type:<5}"
            f"{p.line:>5.1f}  "
            f"{p.side:<5}"
            f"{odds_str:>5}  "
            f"{prob_str:>5}  "
            f"{edge_str:>5}  "
            f"{ev_str:>6}  "
            f"{conf:<4}"
        )
        print("\u2551" + row[:w].ljust(w) + "\u2551")

    print("\u255a" + "\u2550" * w + "\u255d")

    # Stale data warning
    if top and top[0].data_age_minutes > 120:
        print(f"\n  WARNING: Data is {top[0].data_age_minutes} minutes old. "
              "Run with --refresh for fresh odds.")


def print_prop_detail(prop: RankedProp) -> None:
    """Print detailed view of a single prop."""
    w = 60
    sep = "\u2550" * w

    stat_label = _STAT_LABELS.get(prop.stat_type, prop.stat_type)

    print()
    print(sep)
    print(f"{prop.player_name.upper()} - {stat_label} {prop.side} {prop.line}")
    print(sep)
    print()
    print(f"  Game: {prop.matchup} | {prop.game_time}")
    print(f"  Book: {prop.bookmaker.title()} | Odds: {prop.book_odds:+d}")
    print()
    print("  AVERAGES")

    stat_suffix = _STAT_LABELS.get(prop.stat_type, prop.stat_type)
    print(f"    Season:     {prop.season_avg:.1f} ({prop.games_played} games)")
    print(f"    Last 5:     {prop.last5_avg:.1f}")
    print(f"    Effective:  {prop.effective_avg:.1f}")
    print()
    print("  ANALYSIS")
    print(f"    Model Prob:     {prop.model_prob:.1%}")
    print(f"    Book Implied:   {prop.book_implied_prob:.1%}")
    print(f"    Edge:           +{prop.edge:.1%}")
    print(f"    Fair Odds:      {prop.fair_odds:+d}")
    print(f"    Distribution:   {prop.distribution}")
    print()
    print("  RECOMMENDATION")
    print(f"    Side:           {prop.side}")
    print(f"    EV per $10:     ${prop.ev_per_10:.2f}")
    print(f"    Confidence:     {prop.confidence}")

    kelly_label = "quarter" if abs(prop.suggested_stake) > 0 else "N/A"
    print(f"    Kelly (full):   {prop.kelly_fraction:.2%}")
    print(f"    Suggested Bet:  ${prop.suggested_stake:.2f}")
    print()
    print(f"  Data fetched {prop.data_age_minutes} minutes ago")
    print(sep)


# ---------------------------------------------------------------------------
# Filtering functions
# ---------------------------------------------------------------------------

def filter_by_player(props: List[RankedProp], player_name: str) -> List[RankedProp]:
    """Filter props by player name (case-insensitive partial match)."""
    query = player_name.lower()
    return [p for p in props if query in p.player_name.lower()]


def filter_by_stat(props: List[RankedProp], stat_type: str) -> List[RankedProp]:
    """Filter to specific stat type."""
    st = stat_type.upper()
    return [p for p in props if p.stat_type == st]


def filter_by_game(props: List[RankedProp], team: str) -> List[RankedProp]:
    """Filter to props from games involving a team."""
    query = team.lower()
    return [p for p in props if query in p.matchup.lower()]


def filter_by_book(props: List[RankedProp], bookmaker: str) -> List[RankedProp]:
    """Filter to specific bookmaker."""
    query = bookmaker.lower()
    return [p for p in props if query in p.bookmaker.lower()]


def filter_high_confidence(props: List[RankedProp]) -> List[RankedProp]:
    """Return only HIGH confidence props."""
    return [p for p in props if p.confidence == "HIGH"]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_daily_report(
    props: List[RankedProp],
    summary: DailySummary,
) -> str:
    """Generate comprehensive markdown report and save to file."""
    os.makedirs(_REPORTS_DIR, exist_ok=True)

    date_str = summary.date
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        date_display = dt.strftime("%B %d, %Y")
    except ValueError:
        date_display = date_str

    lines = [
        f"# NBA +EV Props Report - {date_display}",
        "",
        f"Generated at {datetime.now().strftime('%I:%M %p')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Games Analyzed | {summary.games_analyzed} |",
        f"| Total Props Fetched | {summary.total_props_fetched} |",
        f"| Props Analyzed | {summary.total_props_analyzed} |",
        f"| Props Skipped | {summary.props_skipped} |",
        f"| +EV Opportunities | {summary.plus_ev_count} |",
        f"| +EV Over | {summary.plus_ev_over_count} |",
        f"| +EV Under | {summary.plus_ev_under_count} |",
        f"| Average Edge | {summary.avg_edge:.1%} |",
        f"| Max Edge | {summary.max_edge:.1%} |",
        f"| Total EV (per $10) | ${summary.total_ev:.2f} |",
        f"| Analysis Time | {summary.analysis_time_seconds:.1f}s |",
        "",
        "## Methodology",
        "",
        "- Probabilities calculated using normal distribution (PTS, high-count stats) "
        "or Poisson distribution (3PM, STL, BLK, low-count stats)",
        "- Effective average: 40% season + 60% last-5 weighted blend",
        "- Edge = model probability - book implied probability",
        "- Kelly criterion used for position sizing (quarter Kelly default)",
        "- Confidence based on edge magnitude, sample size, and probability range",
        "",
    ]

    if props:
        lines.append("## Top +EV Opportunities")
        lines.append("")
        lines.append("| Rank | Player | Stat | Line | Side | Odds | Prob | Edge | EV | Conf |")
        lines.append("|------|--------|------|------|------|------|------|------|-----|------|")

        for p in props:
            lines.append(
                f"| {p.rank} | {p.player_name} | {p.stat_type} | {p.line} | "
                f"{p.side} | {p.book_odds:+d} | {p.model_prob:.1%} | "
                f"+{p.edge:.1%} | ${p.ev_per_10:.2f} | {p.confidence} |"
            )

        lines.append("")
        lines.append("## Detailed Breakdown")
        lines.append("")

        for p in props[:10]:  # Top 10 detailed
            stat_label = _STAT_LABELS.get(p.stat_type, p.stat_type)
            lines.append(f"### {p.rank}. {p.player_name} - {stat_label} {p.side} {p.line}")
            lines.append("")
            lines.append(f"- **Game:** {p.matchup} | {p.game_time}")
            lines.append(f"- **Book:** {p.bookmaker.title()} | Odds: {p.book_odds:+d}")
            lines.append(f"- **Season Avg:** {p.season_avg:.1f} | **Last 5:** {p.last5_avg:.1f} | "
                         f"**Effective:** {p.effective_avg:.1f}")
            lines.append(f"- **Model Prob:** {p.model_prob:.1%} | "
                         f"**Book Implied:** {p.book_implied_prob:.1%}")
            lines.append(f"- **Edge:** +{p.edge:.1%} | **EV:** ${p.ev_per_10:.2f} | "
                         f"**Fair Odds:** {p.fair_odds:+d}")
            lines.append(f"- **Confidence:** {p.confidence} | "
                         f"**Suggested Stake:** ${p.suggested_stake:.2f}")
            lines.append("")

    else:
        lines.append("## Results")
        lines.append("")
        lines.append("No +EV opportunities found matching filter criteria.")
        lines.append("")

    lines.extend([
        "## Warnings",
        "",
        "- Lines move quickly. Verify current odds before placing any bet.",
        "- This model is for educational/research purposes.",
        "- Past performance does not guarantee future results.",
        "- Always practice responsible bankroll management.",
        "",
        "---",
        f"*Report generated by prop_analyzer.py*",
    ])

    content = "\n".join(lines) + "\n"

    filepath = os.path.join(_REPORTS_DIR, f"daily_report_{date_str}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Report saved to: {filepath}")

    return content


def generate_quick_picks(props: List[RankedProp], n: int = 5) -> str:
    """Generate short text summary of top N picks."""
    date_str = datetime.now().strftime("%b %d")
    picks = props[:n]

    lines = [f"NBA +EV Picks - {date_str}", ""]

    total_ev = 0.0
    total_stake = 0.0
    for i, p in enumerate(picks, 1):
        side_char = "O" if p.side == "OVER" else "U"
        lines.append(
            f"{i}. {p.player_name} {side_char}{p.line:.0f} {p.stat_type} "
            f"({p.book_odds:+d}) | {p.model_prob:.0%} | +{p.edge:.1%} edge"
        )
        total_ev += p.ev_per_10
        total_stake += 10.0

    if picks:
        lines.append("")
        lines.append(f"Total EV: +${total_ev:.2f} on ${total_stake:.0f}")

    result = "\n".join(lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def interactive_session() -> None:
    """Start an interactive CLI session."""
    config = load_config()
    ranked_props: List[RankedProp] = []
    summary: Optional[DailySummary] = None
    current_view: List[RankedProp] = []

    print("\n  NBA Prop Analyzer - Interactive Mode")
    print("  Type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "q", "exit"):
            print("  Goodbye!")
            break

        elif cmd in ("scan", "s"):
            ranked_props, summary = analyze_props(config)
            current_view = ranked_props
            if summary:
                print_summary(summary)
            print_top_props(current_view, config.top_n)

        elif cmd == "top":
            n = int(parts[1]) if len(parts) > 1 else config.top_n
            view = current_view if current_view else ranked_props
            print_top_props(view, n)

        elif cmd == "detail":
            if len(parts) < 2:
                print("  Usage: detail <rank>")
                continue
            try:
                rank = int(parts[1])
            except ValueError:
                print("  Rank must be a number.")
                continue
            source = current_view if current_view else ranked_props
            match = [p for p in source if p.rank == rank]
            if match:
                print_prop_detail(match[0])
            else:
                print(f"  No prop at rank {rank}.")

        elif cmd == "filter":
            if len(parts) < 3:
                print("  Usage: filter player|stat|game|book <value>")
                continue
            filter_type = parts[1].lower()
            value = " ".join(parts[2:])
            source = ranked_props

            if filter_type == "player":
                current_view = filter_by_player(source, value)
            elif filter_type == "stat":
                current_view = filter_by_stat(source, value)
            elif filter_type == "game":
                current_view = filter_by_game(source, value)
            elif filter_type == "book":
                current_view = filter_by_book(source, value)
            elif filter_type == "high":
                current_view = filter_high_confidence(source)
            else:
                print(f"  Unknown filter: {filter_type}")
                continue

            # Re-rank the filtered view
            for i, p in enumerate(current_view, 1):
                p.rank = i

            print(f"  Filtered: {len(current_view)} props")
            print_top_props(current_view, config.top_n)

        elif cmd == "refresh":
            config.force_refresh = True
            ranked_props, summary = analyze_props(config)
            current_view = ranked_props
            config.force_refresh = False
            if summary:
                print_summary(summary)
            print_top_props(current_view, config.top_n)

        elif cmd == "config":
            print("\n  Current Configuration:")
            for k, v in asdict(config).items():
                print(f"    {k}: {v}")
            print()

        elif cmd == "set":
            if len(parts) < 3:
                print("  Usage: set <key> <value>")
                continue
            key = parts[1]
            value_str = " ".join(parts[2:])
            if key not in AnalyzerConfig.__dataclass_fields__:
                print(f"  Unknown config key: {key}")
                continue
            try:
                field_type = type(getattr(config, key))
                if field_type == bool:
                    val = value_str.lower() in ("true", "1", "yes")
                elif field_type == float:
                    val = float(value_str)
                elif field_type == int:
                    val = int(value_str)
                else:
                    val = value_str
                setattr(config, key, val)
                print(f"  Set {key} = {val}")
            except (ValueError, TypeError) as e:
                print(f"  Error setting {key}: {e}")

        elif cmd == "report":
            if not ranked_props:
                print("  Run 'scan' first.")
                continue
            generate_daily_report(ranked_props, summary)

        elif cmd == "picks":
            if not ranked_props:
                print("  Run 'scan' first.")
                continue
            generate_quick_picks(
                current_view if current_view else ranked_props, 5
            )

        elif cmd == "reset":
            current_view = ranked_props
            print("  Filters cleared.")

        elif cmd == "help":
            print("""
  Commands:
    scan / s          Run full analysis
    top [n]           Show top N props (default: 20)
    detail <rank>     Show details for prop at rank
    filter player <name>   Filter by player name
    filter stat <type>     Filter by stat type (PTS, REB, etc.)
    filter game <team>     Filter by team
    filter book <name>     Filter by bookmaker
    filter high            Show only HIGH confidence
    reset             Clear filters
    refresh           Force refresh from API
    config            Show current config
    set <key> <value> Update config value
    report            Generate and save daily report
    picks             Generate quick picks summary
    help              Show this help
    quit / q          Exit
""")
        else:
            print(f"  Unknown command: {cmd}. Type 'help' for commands.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="NBA +EV Player Prop Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prop_analyzer.py                    # Run scan with defaults
  python prop_analyzer.py --min-edge 0.03    # Higher edge threshold
  python prop_analyzer.py top --top 10       # Show top 10
  python prop_analyzer.py detail 1           # Detail on rank 1
  python prop_analyzer.py --stat PTS         # Only points props
  python prop_analyzer.py interactive        # Start interactive mode
""",
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="scan",
        choices=["scan", "top", "detail", "report", "picks", "interactive", "config"],
        help="Command to run (default: scan)",
    )
    parser.add_argument(
        "rank",
        nargs="?",
        type=int,
        default=None,
        help="Rank number for 'detail' command",
    )

    parser.add_argument("--min-edge", type=float, default=None, help="Minimum edge (default: 0.02)")
    parser.add_argument("--max-edge", type=float, default=None, help="Maximum edge (default: 0.30)")
    parser.add_argument("--min-prob", type=float, default=None, help="Minimum probability (default: 0.55)")
    parser.add_argument("--top", type=int, default=None, dest="top_n", help="Number of top props to show")
    parser.add_argument("--refresh", action="store_true", help="Force refresh from API")
    parser.add_argument("--bankroll", type=float, default=None, help="Bankroll for Kelly sizing")
    parser.add_argument("--stat", type=str, default=None, help="Filter to specific stat type")
    parser.add_argument("--player", type=str, default=None, help="Filter to specific player")
    parser.add_argument("--book", type=str, default=None, help="Filter to specific bookmaker")
    parser.add_argument("--save", action="store_true", help="Save report to file")

    return parser


def main() -> None:
    """Main entry point: parse CLI arguments, run requested command."""
    parser = build_parser()
    args = parser.parse_args()

    # Load and override config
    config = load_config()

    if args.min_edge is not None:
        config.min_edge = args.min_edge
    if args.max_edge is not None:
        config.max_edge = args.max_edge
    if args.min_prob is not None:
        config.min_prob = args.min_prob
    if args.top_n is not None:
        config.top_n = args.top_n
    if args.refresh:
        config.force_refresh = True
    if args.bankroll is not None:
        config.bankroll = args.bankroll
    if args.save:
        config.save_report = True
    if args.stat:
        config.stat_types = [args.stat.upper()]

    # Print header
    date_display = datetime.now().strftime("%B %d, %Y")
    w = 72
    print()
    print("\u2554" + "\u2550" * w + "\u2557")
    print("\u2551" + f"  NBA PROPS ANALYZER - {date_display}".ljust(w) + "\u2551")
    print("\u255a" + "\u2550" * w + "\u255d")

    # Handle commands
    if args.command == "interactive":
        interactive_session()
        return

    if args.command == "config":
        print("\n  Current Configuration:")
        for k, v in asdict(config).items():
            print(f"    {k}: {v}")
        return

    # For all other commands, run analysis first
    ranked_props, summary = analyze_props(config)

    # Apply post-analysis filters
    filtered = ranked_props
    if args.player:
        filtered = filter_by_player(filtered, args.player)
    if args.stat:
        filtered = filter_by_stat(filtered, args.stat.upper())
    if args.book:
        filtered = filter_by_book(filtered, args.book)

    if args.command == "scan":
        print_summary(summary)
        print_top_props(filtered, config.top_n)

        if config.save_report and ranked_props:
            generate_daily_report(ranked_props, summary)

        if ranked_props:
            print(f"\n  Run 'python prop_analyzer.py detail <N>' for full analysis on any prop.")
            print(f"  Run 'python prop_analyzer.py picks' for quick picks summary.")

    elif args.command == "top":
        print_top_props(filtered, config.top_n)

    elif args.command == "detail":
        rank = args.rank
        if rank is None:
            print("  Usage: python prop_analyzer.py detail <rank>")
            return
        match = [p for p in filtered if p.rank == rank]
        if match:
            print_prop_detail(match[0])
        else:
            print(f"  No prop at rank {rank}.")

    elif args.command == "report":
        if ranked_props:
            generate_daily_report(ranked_props, summary)
        else:
            print("  No props to report.")

    elif args.command == "picks":
        generate_quick_picks(filtered, config.top_n if config.top_n < 10 else 5)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
