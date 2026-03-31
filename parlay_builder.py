"""
parlay_builder.py -- Build optimal +EV parlays from single +EV props.

Takes ranked props from prop_analyzer.py and constructs parlays that
maximize expected value while managing risk through correlation adjustments,
diversification, and conservative Kelly sizing.

Dependencies: prop_analyzer, edge_cal, itertools, collections
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import edge_cal
import player_stats_db
import probability_model
from prop_analyzer import RankedProp, AnalyzerConfig, analyze_props


# ---------------------------------------------------------------------------
# Project directory
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPORTS_DIR = os.path.join(_SCRIPT_DIR, "reports")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParlayLeg:
    """A single leg in a parlay."""

    player_name: str
    stat_type: str
    line: float
    side: str               # 'OVER' or 'UNDER'
    book_odds: int
    decimal_odds: float
    model_prob: float
    edge: float
    game_id: str
    matchup: str
    bookmaker: str


@dataclass
class Parlay:
    """A complete parlay with analysis."""

    legs: List[ParlayLeg]
    num_legs: int

    # Probabilities
    combined_prob: float            # Product of leg probabilities
    adjusted_prob: float            # After correlation adjustment

    # Odds
    parlay_decimal: float           # Product of decimal odds
    parlay_american: int            # Converted to American
    fair_american: int              # What odds SHOULD be

    # Value
    edge: float                     # adjusted_prob - implied_prob
    ev_per_10: float               # EV on $10 stake
    is_plus_ev: bool

    # Risk metrics
    variance_score: float           # Higher = more volatile
    confidence: str                 # 'HIGH', 'MEDIUM', 'LOW'

    # Sizing
    kelly_fraction: float
    suggested_stake: float

    # Metadata
    same_game_count: int            # Number of legs from same game
    games_involved: int             # Number of unique games
    correlation_penalty: float      # Applied to probability

    # Display
    description: str                # "Siakam O15 + Nembhard O5 + ..."


@dataclass
class ParlayBuilderConfig:
    """Configuration for parlay building."""

    # Stat types (OVER only)
    stat_types: List[str] = field(default_factory=lambda: ['PTS', 'REB', 'AST', '3PM'])

    # Leg limits
    min_legs: int = 2
    max_legs: int = 6

    # Quality filters
    min_leg_prob: float = 0.60          # Each leg must have >= 60% prob
    min_leg_edge: float = 0.02          # Each leg must have >= 2% edge
    min_parlay_prob: float = 0.10       # Combined prob >= 10%
    min_parlay_ev: float = 0.50         # Minimum EV per $10

    # Correlation
    same_game_correlation_penalty: float = 0.02   # Reduce prob by 2% per same-game pair
    max_same_game_legs: int = 3                   # Max legs from single game

    # Diversification
    prefer_cross_game: bool = True      # Prefer legs from different games
    max_legs_per_game: int = 2          # Soft cap on legs from one game

    # Generation
    max_parlays_to_generate: int = 100  # Cap on combinations to evaluate
    top_n_to_show: int = 10             # Number of parlays to display

    # Sizing
    bankroll: float = 1000.0
    kelly_fraction: float = 0.10        # More conservative for parlays
    max_parlay_stake: float = 25.0


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def props_to_legs(props: List[RankedProp]) -> List[ParlayLeg]:
    """Convert RankedProp objects to ParlayLeg objects."""
    legs = []
    for p in props:
        legs.append(ParlayLeg(
            player_name=p.player_name,
            stat_type=p.stat_type,
            line=p.line,
            side=p.side,
            book_odds=p.book_odds,
            decimal_odds=edge_cal.american_to_decimal(p.book_odds),
            model_prob=p.model_prob,
            edge=p.edge,
            game_id=p.matchup,
            matchup=p.matchup,
            bookmaker=p.bookmaker,
        ))
    return legs


def _leg_short_desc(leg: ParlayLeg) -> str:
    """Short description for a leg, e.g. 'Siakam O15.0 PTS'."""
    side_char = "O" if leg.side == "OVER" else "U"
    last_name = leg.player_name.split()[-1] if leg.player_name else "???"
    return f"{last_name} {side_char}{leg.line}"


# ---------------------------------------------------------------------------
# Correlation adjustment
# ---------------------------------------------------------------------------

def apply_correlation_penalty(
    combined_prob: float,
    legs: List[ParlayLeg],
    config: ParlayBuilderConfig,
) -> Tuple[float, float]:
    """Apply correlation penalty for same-game leg pairs.

    Returns (adjusted_prob, total_penalty_applied).
    """
    game_counts = Counter(leg.game_id for leg in legs)
    same_game_pairs = sum(n * (n - 1) // 2 for n in game_counts.values())
    penalty = same_game_pairs * config.same_game_correlation_penalty
    adjusted = combined_prob * (1 - penalty)
    return max(0.01, adjusted), penalty


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def calculate_parlay_confidence(parlay: Parlay) -> str:
    """Determine confidence level for a parlay.

    HIGH:   edge > 5%, all legs prob > 70%, <= 3 legs
    MEDIUM: edge > 3%, all legs prob > 60%
    LOW:    everything else
    """
    min_leg_prob = min(leg.model_prob for leg in parlay.legs)

    if parlay.edge > 0.05 and min_leg_prob > 0.70 and parlay.num_legs <= 3:
        return "HIGH"
    if parlay.edge > 0.03 and min_leg_prob > 0.60:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Core parlay calculation
# ---------------------------------------------------------------------------

def calculate_parlay(
    legs: List[ParlayLeg],
    config: ParlayBuilderConfig,
) -> Parlay:
    """Calculate all parlay metrics for a set of legs."""
    # Combined probability (assumes independence)
    combined_prob = 1.0
    for leg in legs:
        combined_prob *= leg.model_prob

    # Correlation adjustment
    adjusted_prob, penalty = apply_correlation_penalty(combined_prob, legs, config)

    # Parlay odds (product of decimal odds)
    parlay_decimal = 1.0
    for leg in legs:
        parlay_decimal *= leg.decimal_odds

    parlay_american = edge_cal.decimal_to_american(parlay_decimal)

    # Fair odds from adjusted probability
    clamped_prob = max(0.001, min(0.999, adjusted_prob))
    fair_american = edge_cal.probability_to_fair_odds(clamped_prob)

    # Implied probability from book odds
    implied_prob = edge_cal.american_to_implied_prob(parlay_american)
    parlay_edge = adjusted_prob - implied_prob

    # EV on $10 stake
    profit_if_win = 10.0 * (parlay_decimal - 1)
    ev_per_10 = (adjusted_prob * profit_if_win) - ((1 - adjusted_prob) * 10.0)

    is_plus_ev = ev_per_10 > 0

    # Variance score: higher with more legs and lower probabilities
    variance_score = parlay_decimal * (1 - adjusted_prob)

    # Game diversification info
    game_ids = {leg.game_id for leg in legs}
    games_involved = len(game_ids)
    game_counts = Counter(leg.game_id for leg in legs)
    same_game_count = sum(1 for c in game_counts.values() if c > 1)

    # Kelly fraction (conservative for parlays)
    b = parlay_decimal - 1  # profit-to-stake ratio
    if b > 0 and adjusted_prob > 0:
        kelly = (b * adjusted_prob - (1 - adjusted_prob)) / b
        kelly = max(0.0, kelly)
    else:
        kelly = 0.0

    # Suggested stake
    raw_stake = kelly * config.kelly_fraction * config.bankroll
    capped = min(raw_stake, config.max_parlay_stake)
    suggested_stake = max(0.0, round(capped * 2) / 2)  # round to $0.50

    # Description
    description = " + ".join(_leg_short_desc(leg) for leg in legs)

    parlay = Parlay(
        legs=legs,
        num_legs=len(legs),
        combined_prob=round(combined_prob, 4),
        adjusted_prob=round(adjusted_prob, 4),
        parlay_decimal=round(parlay_decimal, 3),
        parlay_american=parlay_american,
        fair_american=fair_american,
        edge=round(parlay_edge, 4),
        ev_per_10=round(ev_per_10, 2),
        is_plus_ev=is_plus_ev,
        variance_score=round(variance_score, 2),
        confidence="LOW",  # placeholder, set below
        kelly_fraction=round(kelly, 4),
        suggested_stake=suggested_stake,
        same_game_count=same_game_count,
        games_involved=games_involved,
        correlation_penalty=round(penalty, 4),
        description=description,
    )
    parlay.confidence = calculate_parlay_confidence(parlay)
    return parlay


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def filter_compatible_legs(legs: List[ParlayLeg]) -> List[ParlayLeg]:
    """Remove conflicting legs (same player + stat, keep best edge)."""
    best: Dict[str, ParlayLeg] = {}
    for leg in legs:
        key = f"{leg.player_name}|{leg.stat_type}"
        if key not in best or leg.edge > best[key].edge:
            best[key] = leg
    return list(best.values())


def _is_valid_combination(
    combo: Tuple[ParlayLeg, ...],
    config: ParlayBuilderConfig,
) -> bool:
    """Check if a combination of legs respects constraints."""
    game_counts = Counter(leg.game_id for leg in combo)

    # Hard cap: max same-game legs
    if any(c > config.max_same_game_legs for c in game_counts.values()):
        return False

    # No duplicate player+stat
    seen = set()
    for leg in combo:
        key = f"{leg.player_name}|{leg.stat_type}"
        if key in seen:
            return False
        seen.add(key)

    return True


def generate_combinations(
    legs: List[ParlayLeg],
    min_legs: int,
    max_legs: int,
    config: ParlayBuilderConfig,
) -> List[List[ParlayLeg]]:
    """Generate all valid leg combinations, respecting constraints.

    Caps total combinations at config.max_parlays_to_generate.
    """
    valid = []
    budget = config.max_parlays_to_generate

    for size in range(min_legs, max_legs + 1):
        if size > len(legs):
            break
        for combo in itertools.combinations(legs, size):
            if budget <= 0:
                return valid
            if _is_valid_combination(combo, config):
                valid.append(list(combo))
                budget -= 1

    return valid


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_parlays(
    props: List[RankedProp],
    config: ParlayBuilderConfig = None,
) -> List[Parlay]:
    """Main function: takes ranked props, returns ranked parlays.

    1. Filter props to those meeting min_leg_prob and min_leg_edge
    2. Convert to ParlayLeg objects
    3. Remove conflicting legs
    4. Generate combinations (2-leg through max_legs)
    5. Calculate each parlay with correlation adjustments
    6. Filter to +EV only meeting thresholds
    7. Sort by EV descending
    8. Return top N
    """
    if config is None:
        config = ParlayBuilderConfig()

    # Filter qualifying props
    qualifying = [
        p for p in props
        if p.model_prob >= config.min_leg_prob
        and p.edge >= config.min_leg_edge
    ]

    if len(qualifying) < config.min_legs:
        return []

    # Convert and deduplicate
    legs = props_to_legs(qualifying)
    legs = filter_compatible_legs(legs)

    if len(legs) < config.min_legs:
        return []

    # Sort legs by edge descending for better pruning
    legs.sort(key=lambda l: l.edge, reverse=True)

    # Generate combinations
    combos = generate_combinations(legs, config.min_legs, config.max_legs, config)

    # Calculate and filter parlays
    parlays = []
    for combo in combos:
        parlay = calculate_parlay(combo, config)

        if not parlay.is_plus_ev:
            continue
        if parlay.adjusted_prob < config.min_parlay_prob:
            continue
        if parlay.ev_per_10 < config.min_parlay_ev:
            continue

        parlays.append(parlay)

    # Sort by EV descending
    parlays.sort(key=lambda p: p.ev_per_10, reverse=True)

    return parlays[:config.top_n_to_show]


# ---------------------------------------------------------------------------
# Optimization functions
# ---------------------------------------------------------------------------

def find_optimal_parlay(
    legs: List[ParlayLeg],
    target_legs: int,
    config: ParlayBuilderConfig = None,
) -> Optional[Parlay]:
    """Find the single best N-leg parlay using greedy selection."""
    if config is None:
        config = ParlayBuilderConfig()

    if len(legs) < target_legs:
        return None

    # Sort by a composite score: edge * prob (balances value and likelihood)
    sorted_legs = sorted(legs, key=lambda l: l.edge * l.model_prob, reverse=True)

    best_parlay = None
    best_ev = -float("inf")

    # Try combinations of the top candidates (greedy + limited search)
    pool = sorted_legs[:min(len(sorted_legs), target_legs + 6)]

    for combo in itertools.combinations(pool, target_legs):
        if not _is_valid_combination(combo, config):
            continue
        parlay = calculate_parlay(list(combo), config)
        if parlay.is_plus_ev and parlay.ev_per_10 > best_ev:
            best_ev = parlay.ev_per_10
            best_parlay = parlay

    return best_parlay


def build_diverse_parlays(
    props: List[RankedProp],
    n: int = 5,
    config: ParlayBuilderConfig = None,
) -> List[Parlay]:
    """Build N parlays that minimize leg overlap.

    Each subsequent parlay avoids legs from previous parlays where possible.
    """
    if config is None:
        config = ParlayBuilderConfig()

    all_parlays = build_parlays(props, config)
    if not all_parlays:
        return []

    diverse: List[Parlay] = []
    used_keys: set = set()

    for parlay in all_parlays:
        leg_keys = {f"{l.player_name}|{l.stat_type}" for l in parlay.legs}
        overlap = len(leg_keys & used_keys)

        # Allow at most 1 shared leg
        if overlap <= 1:
            diverse.append(parlay)
            used_keys.update(leg_keys)

        if len(diverse) >= n:
            break

    return diverse


def build_same_game_parlay(
    props: List[RankedProp],
    game_id: str = None,
    team: str = None,
    config: ParlayBuilderConfig = None,
) -> List[Parlay]:
    """Build same-game parlays for a specific game.

    Identify the game by game_id or team name substring.
    """
    if config is None:
        config = ParlayBuilderConfig()

    # Filter props to the target game
    if game_id:
        game_props = [p for p in props if p.matchup == game_id]
    elif team:
        team_lower = team.lower()
        game_props = [
            p for p in props
            if team_lower in p.matchup.lower()
        ]
    else:
        return []

    if len(game_props) < config.min_legs:
        return []

    # Build with SGP-appropriate config
    sgp_config = ParlayBuilderConfig(
        min_legs=config.min_legs,
        max_legs=min(config.max_legs, config.max_same_game_legs),
        min_leg_prob=config.min_leg_prob,
        min_leg_edge=config.min_leg_edge,
        min_parlay_prob=config.min_parlay_prob,
        min_parlay_ev=config.min_parlay_ev,
        same_game_correlation_penalty=config.same_game_correlation_penalty,
        max_same_game_legs=config.max_same_game_legs,
        prefer_cross_game=False,
        max_legs_per_game=config.max_same_game_legs,
        max_parlays_to_generate=config.max_parlays_to_generate,
        top_n_to_show=config.top_n_to_show,
        bankroll=config.bankroll,
        kelly_fraction=config.kelly_fraction,
        max_parlay_stake=config.max_parlay_stake,
    )

    return build_parlays(game_props, sgp_config)


# ---------------------------------------------------------------------------
# Special parlay types
# ---------------------------------------------------------------------------

def build_round_robin(
    legs: List[ParlayLeg],
    subset_size: int,
    config: ParlayBuilderConfig = None,
) -> List[Parlay]:
    """Build all parlays of a given size from the provided legs."""
    if config is None:
        config = ParlayBuilderConfig()

    if subset_size > len(legs) or subset_size < 2:
        return []

    parlays = []
    for combo in itertools.combinations(legs, subset_size):
        if _is_valid_combination(combo, config):
            parlay = calculate_parlay(list(combo), config)
            if parlay.is_plus_ev:
                parlays.append(parlay)

    parlays.sort(key=lambda p: p.ev_per_10, reverse=True)
    return parlays


def build_progressive_parlay(
    legs: List[ParlayLeg],
    sizes: List[int] = None,
    config: ParlayBuilderConfig = None,
) -> Dict[int, Optional[Parlay]]:
    """Build best parlay at each requested size.

    Returns dict mapping size -> best parlay (or None if impossible).
    """
    if sizes is None:
        sizes = [2, 3, 4]
    if config is None:
        config = ParlayBuilderConfig()

    result: Dict[int, Optional[Parlay]] = {}
    for size in sizes:
        result[size] = find_optimal_parlay(legs, size, config)
    return result


# ---------------------------------------------------------------------------
# Filtering functions
# ---------------------------------------------------------------------------

def filter_by_game(parlays: List[Parlay], team: str) -> List[Parlay]:
    """Filter to parlays involving a specific team."""
    team_lower = team.lower()
    return [
        p for p in parlays
        if any(team_lower in leg.matchup.lower() for leg in p.legs)
    ]


def filter_by_legs(parlays: List[Parlay], num_legs: int) -> List[Parlay]:
    """Filter to parlays with exactly N legs."""
    return [p for p in parlays if p.num_legs == num_legs]


def filter_high_confidence(parlays: List[Parlay]) -> List[Parlay]:
    """Return only HIGH confidence parlays."""
    return [p for p in parlays if p.confidence == "HIGH"]


def filter_parlays_by_player(parlays: List[Parlay], player_name: str) -> List[Parlay]:
    """Filter to parlays containing a specific player (case-insensitive partial match)."""
    player_lower = player_name.lower()
    return [
        p for p in parlays
        if any(player_lower in leg.player_name.lower() for leg in p.legs)
    ]


def filter_parlays_by_team(parlays: List[Parlay], team: str) -> List[Parlay]:
    """Filter to parlays involving a specific team/game (case-insensitive partial match)."""
    team_lower = team.lower()
    return [
        p for p in parlays
        if any(team_lower in leg.matchup.lower() for leg in p.legs)
    ]


def filter_parlays_by_legs(parlays: List[Parlay], num_legs: int) -> List[Parlay]:
    """Filter to parlays with exactly N legs."""
    return [p for p in parlays if p.num_legs == num_legs]


def filter_parlays_by_min_prob(parlays: List[Parlay], min_prob: float) -> List[Parlay]:
    """Filter to parlays with combined probability >= min_prob."""
    return [p for p in parlays if p.adjusted_prob >= min_prob]


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def _fmt_odds(odds: int) -> str:
    """Format American odds with sign."""
    return f"+{odds}" if odds > 0 else str(odds)


def _fmt_pct(val: float) -> str:
    """Format a decimal as percentage string."""
    return f"{val * 100:.1f}%"


def print_parlay(parlay: Parlay) -> None:
    """Print detailed view of a single parlay."""
    border = "\u2550" * 66

    print(f"\n{border}")
    header = (
        f"{parlay.num_legs}-LEG PARLAY | "
        f"{_fmt_odds(parlay.parlay_american)} (Fair: {_fmt_odds(parlay.fair_american)}) | "
        f"EV: ${parlay.ev_per_10:.2f}"
    )
    print(header)
    print(border)

    print("\n  LEGS:")
    for i, leg in enumerate(parlay.legs, 1):
        odds_str = _fmt_odds(leg.book_odds)
        print(
            f"    {i}. {leg.player_name} {leg.side} {leg.line} {leg.stat_type} "
            f"({odds_str}) | {_fmt_pct(leg.model_prob)} | {leg.matchup}"
        )

    # Correlation adjustment display
    if parlay.correlation_penalty > 0:
        adj_str = f"-{_fmt_pct(parlay.correlation_penalty)} ({parlay.same_game_count} same-game groups)"
    else:
        adj_str = "none (cross-game)"

    print(f"\n  ANALYSIS:")
    print(f"    Combined Prob:     {_fmt_pct(parlay.combined_prob)}")
    print(f"    Correlation Adj:   {adj_str}")
    print(f"    Adjusted Prob:     {_fmt_pct(parlay.adjusted_prob)}")
    print(f"    Book Implied:      {_fmt_pct(edge_cal.american_to_implied_prob(parlay.parlay_american))}")
    print(f"    Edge:              +{_fmt_pct(parlay.edge)}")

    payout = 10.0 * parlay.parlay_decimal
    print(f"\n  RECOMMENDATION:")
    print(f"    Confidence:        {parlay.confidence}")
    print(f"    Suggested Stake:   ${parlay.suggested_stake:.2f}")
    print(f"    Potential Payout:  ${parlay.suggested_stake * parlay.parlay_decimal:.2f}")

    print(f"\n{border}")


def print_parlays_table(parlays: List[Parlay], n: int = 10) -> None:
    """Print summary table of top parlays."""
    shown = parlays[:n]

    if not shown:
        print("\n  No +EV parlays found.")
        return

    top = "\u2554" + "\u2550" * 70 + "\u2557"
    mid = "\u2560" + "\u2550" * 70 + "\u2563"
    bot = "\u255a" + "\u2550" * 70 + "\u255d"
    sep = "\u2551"

    print(f"\n{top}")
    print(f"{sep}  TOP {len(shown)} PARLAYS{' ' * (58 - len(str(len(shown))))}{sep}")
    print(mid)
    print(f"{sep}  {'#':<4}{'LEGS':<6}{'ODDS':<9}{'PROB':<8}{'EDGE':<9}{'EV':<8}{'CONF':<6}{'DESCRIPTION':<22}{sep}")
    print(f"{sep}  {'─' * 66}  {sep}")

    for i, p in enumerate(shown, 1):
        # Truncate description to fit
        desc = p.description
        if len(desc) > 20:
            desc = desc[:17] + "..."

        print(
            f"{sep}  {i:<4}{p.num_legs:<6}"
            f"{_fmt_odds(p.parlay_american):<9}"
            f"{_fmt_pct(p.adjusted_prob):<8}"
            f"+{_fmt_pct(p.edge):<8}"
            f"${p.ev_per_10:<7.2f}"
            f"{p.confidence[:3]:<6}"
            f"{desc:<22}{sep}"
        )

    print(bot)


def print_parlay_ticket(parlay: Parlay) -> str:
    """Generate a clean copy-paste format for betting apps.

    Returns the ticket string and also prints it.
    """
    lines = []
    lines.append(
        f"PARLAY: {parlay.num_legs} Legs | "
        f"Target Odds: {_fmt_odds(parlay.parlay_american)} or better"
    )
    lines.append("")

    for leg in parlay.legs:
        lines.append(
            f"  \u25a1 {leg.player_name} {leg.side} {leg.line} {leg.stat_type}"
        )

    lines.append("")
    lines.append(
        f"Combined Prob: {_fmt_pct(parlay.adjusted_prob)} | "
        f"Fair Odds: {_fmt_odds(parlay.fair_american)}"
    )
    lines.append(
        f"Stake: ${parlay.suggested_stake:.2f} | "
        f"Potential: ${parlay.suggested_stake * parlay.parlay_decimal:.2f}"
    )

    ticket = "\n".join(lines)
    print(f"\n{ticket}")
    return ticket


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_parlay_report(
    parlays: List[Parlay],
    singles: List[RankedProp],
) -> str:
    """Generate markdown report with both singles and parlays.

    Saves to reports/parlay_report_YYYY-MM-DD.md and returns the path.
    """
    os.makedirs(_REPORTS_DIR, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(_REPORTS_DIR, f"parlay_report_{today}.md")

    lines = []
    lines.append(f"# NBA Parlay Report - {today}")
    lines.append("")

    # Singles section
    lines.append("## Top Singles")
    lines.append("")
    if singles:
        lines.append("| # | Player | Stat | Side | Line | Odds | Edge | EV |")
        lines.append("|---|--------|------|------|------|------|------|-----|")
        for s in singles[:10]:
            lines.append(
                f"| {s.rank} | {s.player_name} | {s.stat_type} | {s.side} | "
                f"{s.line} | {_fmt_odds(s.book_odds)} | "
                f"+{_fmt_pct(s.edge)} | ${s.ev_per_10:.2f} |"
            )
    else:
        lines.append("No +EV singles found.")
    lines.append("")

    # Parlays section
    lines.append("## Top Parlays")
    lines.append("")
    if parlays:
        for i, p in enumerate(parlays[:10], 1):
            lines.append(f"### Parlay #{i} ({p.num_legs} legs)")
            lines.append("")
            for j, leg in enumerate(p.legs, 1):
                lines.append(
                    f"{j}. {leg.player_name} {leg.side} {leg.line} {leg.stat_type} "
                    f"({_fmt_odds(leg.book_odds)}) - {_fmt_pct(leg.model_prob)}"
                )
            lines.append("")
            lines.append(f"- **Odds:** {_fmt_odds(p.parlay_american)} (Fair: {_fmt_odds(p.fair_american)})")
            lines.append(f"- **Adjusted Prob:** {_fmt_pct(p.adjusted_prob)}")
            lines.append(f"- **Edge:** +{_fmt_pct(p.edge)}")
            lines.append(f"- **EV per $10:** ${p.ev_per_10:.2f}")
            lines.append(f"- **Confidence:** {p.confidence}")
            lines.append(f"- **Suggested Stake:** ${p.suggested_stake:.2f}")
            lines.append("")
    else:
        lines.append("No +EV parlays found.")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Parlays built from individually +EV single props")
    lines.append("- Same-game legs receive a correlation penalty (2% per pair)")
    lines.append("- Kelly criterion at 10% fraction (conservative for parlays)")
    lines.append("- Probabilities from weighted player stat model (season + recent)")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by NBA Parlay Builder*")

    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n  Report saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Custom parlay builder
# ---------------------------------------------------------------------------

_CUSTOM_THRESHOLDS = {
    'PTS': [10, 15, 20, 25, 30, 35, 40],
    'REB': [3, 5, 7, 10, 12, 15],
    'AST': [3, 5, 7, 10, 12],
    '3PM': [1, 2, 3, 4, 5],
}

_STAT_KEY_MAP = {
    'PTS': 'points',
    'REB': 'rebounds',
    'AST': 'assists',
    '3PM': 'three_pm',
}


def get_player_probability_grid(player_name: str, opponent: str = None) -> Optional[Dict]:
    """Get probability grid for a single player across all stat thresholds.

    When `opponent` is provided, each threshold also carries an h2h_prob
    computed from the player's historical stats vs that team.
    """
    try:
        stats = player_stats_db.get_player_stats_with_h2h(player_name, opponent)
    except Exception:
        print(f"  Could not find player: {player_name}")
        return None

    h2h_games = int(stats.h2h_avg.get('games_vs', 0)) if stats.h2h_avg else 0

    grid: Dict = {
        'player_name': stats.player_name,
        'team': stats.team,
        'games_played': stats.games_played,
        'opponent': opponent,
        'h2h_games': h2h_games,
        'stats': {},
    }

    for stat_type, lines in _CUSTOM_THRESHOLDS.items():
        stat_key = _STAT_KEY_MAP[stat_type]
        effective_avg = (stats.effective_avg or {}).get(stat_key, 0)
        if effective_avg <= 0:
            continue

        season_avg = (stats.season_avg or {}).get(stat_key, 0)
        last5_avg = (stats.last5_avg or {}).get(stat_key, 0)
        h2h_avg_val = stats.h2h_avg.get(stat_key) if stats.h2h_avg else None

        line_probs = []
        for line in lines:
            std_prob, h2h_prob = probability_model.calc_prob_with_h2h(
                stat_type, line, season_avg, last5_avg, h2h_avg_val, h2h_games
            )
            line_probs.append({
                'line': line,
                'prob': std_prob,
                'h2h_prob': h2h_prob,
                'has_h2h': h2h_avg_val is not None and h2h_games >= 2,
            })

        grid['stats'][stat_type] = {
            'season_avg': round(season_avg, 1),
            'last5_avg': round(last5_avg, 1),
            'h2h_avg': round(h2h_avg_val, 1) if h2h_avg_val is not None else None,
            'effective_avg': round(effective_avg, 1),
            'lines': line_probs,
        }

    return grid


def _print_player_selection_grid(grid: Dict) -> None:
    """Print the probability grid for one player during selection."""
    h2h_suffix = ""
    if grid.get('opponent') and grid.get('h2h_games', 0) > 0:
        h2h_suffix = f" | vs {grid['opponent']}: {grid['h2h_games']} games"
    elif grid.get('opponent'):
        h2h_suffix = f" | vs {grid['opponent']}: no H2H data"

    print(f"\n  {'═' * 74}")
    print(f"  {grid['player_name'].upper()} ({grid['team']}) | {grid['games_played']} games{h2h_suffix}")
    print(f"  {'═' * 74}")
    for stat_type, data in grid['stats'].items():
        avg_str = f"Avg: {data['effective_avg']}"
        if data['h2h_avg'] is not None:
            avg_str += f" (H2H: {data['h2h_avg']})"

        line_strs = []
        for l in data['lines']:
            if l['prob'] < 0.10:
                continue
            if l['has_h2h'] and abs(l['prob'] - l['h2h_prob']) > 0.01:
                line_strs.append(
                    f"{l['line']}+ {l['prob']*100:.0f}% [H2H:{l['h2h_prob']*100:.0f}%]"
                )
            else:
                line_strs.append(f"{l['line']}+ ({l['prob']*100:.0f}%)")

        print(f"  {stat_type:<4} [{avg_str}]")
        print(f"       {' | '.join(line_strs)}")
    print()


def _print_custom_parlay_analysis(legs: List[Dict]) -> None:
    """Print combined parlay analysis table for custom-selected legs.

    Shows both standard and H2H-weighted probabilities when H2H data is present.
    """
    combined_std = 1.0
    combined_h2h = 1.0
    has_any_h2h = any(leg.get('has_h2h') for leg in legs)

    for leg in legs:
        combined_std *= leg['prob']
        combined_h2h *= leg.get('h2h_prob', leg['prob'])

    fair_american_std = edge_cal.decimal_to_american(1 / max(0.001, min(0.999, combined_std)))
    fair_american_h2h = edge_cal.decimal_to_american(1 / max(0.001, min(0.999, combined_h2h)))

    # Use H2H combined prob for EV analysis when available
    display_prob = combined_h2h if has_any_h2h else combined_std
    display_fair = fair_american_h2h if has_any_h2h else fair_american_std

    print("\n" + "═" * 78)
    print(f"  CUSTOM {len(legs)}-LEG PARLAY ANALYSIS")
    print("═" * 78)

    if has_any_h2h:
        print(f"\n  {'#':<3} {'PLAYER':<22} {'STAT':<5} {'LINE':<6} {'PROB':<8} {'H2H PROB':<10} {'FAIR ODDS'}")
    else:
        print(f"\n  {'#':<3} {'PLAYER':<22} {'STAT':<5} {'LINE':<6} {'PROB':<8} {'FAIR ODDS':<12}")
    print("  " + "─" * 74)

    for i, leg in enumerate(legs, 1):
        std_prob = leg['prob']
        h2h_prob = leg.get('h2h_prob', std_prob)
        display_leg_prob = h2h_prob if leg.get('has_h2h') else std_prob
        fair_str = f"{edge_cal.decimal_to_american(1 / max(0.001, min(0.999, display_leg_prob))):+d}"

        if has_any_h2h:
            h2h_str = f"{h2h_prob * 100:>5.1f}%" if leg.get('has_h2h') else "   -  "
            print(
                f"  {i:<3} {leg['player']:<22} {leg['stat']:<5} {leg['line']:<6.1f}"
                f" {std_prob * 100:>5.1f}%   {h2h_str:<10} {fair_str}"
            )
        else:
            print(
                f"  {i:<3} {leg['player']:<22} {leg['stat']:<5} {leg['line']:<6.1f}"
                f" {std_prob * 100:>5.1f}%   {fair_str:<12}"
            )

    print("  " + "─" * 74)

    print(f"\n  COMBINED ANALYSIS:")
    print(f"  {'─' * 50}")
    print(f"  Standard Combined Probability:  {combined_std * 100:.1f}%")
    if has_any_h2h:
        print(f"  H2H-Weighted Combined Prob:     {combined_h2h * 100:.1f}%  \u2190 USE THIS")
        print(f"  Fair Parlay Odds (H2H):         {fair_american_h2h:+d}")
    else:
        print(f"  Fair Parlay Odds:               {fair_american_std:+d}")

    print(f"\n  EV AT DIFFERENT BOOK ODDS:")
    print(f"  {'─' * 50}")

    for book_odds in [200, 300, 400, 500, 600, 800, 1000, 1500]:
        if book_odds < display_fair - 100:
            continue
        book_decimal = edge_cal.american_to_decimal(book_odds)
        book_implied = edge_cal.american_to_implied_prob(book_odds)
        edge = display_prob - book_implied
        ev_per_10 = (display_prob * (book_decimal - 1) * 10) - ((1 - display_prob) * 10)
        ev_str = f"${ev_per_10:+.2f}"
        edge_str = f"{edge * 100:+.1f}%"
        status = "\u2713 +EV" if ev_per_10 > 0 else "\u2717 -EV"
        print(f"  Book +{book_odds}:  Edge {edge_str:>7}  |  EV/10: {ev_str:>7}  |  {status}")

    print(f"\n  {'─' * 50}")
    print(f"  Target: {display_fair:+d} or better for +EV")

    print(f"\n  BETTING TICKET:")
    print(f"  {'─' * 50}")
    for leg in legs:
        h2h_note = (
            f" [H2H:{leg.get('h2h_prob', 0) * 100:.0f}%]"
            if leg.get('has_h2h') else ""
        )
        print(f"  \u25a1 {leg['player']} OVER {leg['line']} {leg['stat']} ({leg['prob'] * 100:.0f}%{h2h_note})")

    prob_label = "H2H Combined" if has_any_h2h else "Combined"
    print(f"\n  {prob_label} Probability: {display_prob * 100:.1f}%")
    print(f"  Target Odds: {display_fair:+d} or better")
    print("═" * 78)


def build_custom_parlay_interactive(player_names: List[str], opponent: str = None) -> None:
    """Show probability grids for each player, prompt for leg selections, then analyze."""
    print("\n  Loading player stats...")

    grids = []
    for name in player_names:
        grid = get_player_probability_grid(name, opponent)
        if grid:
            grids.append(grid)
        else:
            print(f"  \u26a0 Skipping '{name}' - not found")

    if len(grids) < 2:
        print("  Need at least 2 valid players to build a parlay.")
        return

    print("\n" + "=" * 74)
    print("  PLAYER PROBABILITY GRIDS")
    print("=" * 74)
    for grid in grids:
        _print_player_selection_grid(grid)

    print("=" * 74)
    print("  SELECT YOUR LEGS")
    print("=" * 74)
    print("\n  For each player, enter: STAT LINE  (e.g. 'PTS 20' or 'REB 7')")
    print("  Press Enter to skip a player.\n")

    selected_legs = []

    for grid in grids:
        player = grid['player_name']
        available_stats = list(grid['stats'].keys())

        while True:
            try:
                choice = input(f"  {player} [{'/'.join(available_stats)}]: ").strip().upper()
            except EOFError:
                break

            if not choice:
                print(f"    Skipped {player}")
                break

            parts = choice.split()
            if len(parts) != 2:
                print("    Invalid format. Use: STAT LINE  (e.g. 'PTS 20')")
                continue

            stat_type, line_str = parts
            try:
                line = float(line_str)
            except ValueError:
                print("    Invalid line number.")
                continue

            if stat_type not in grid['stats']:
                print(f"    {stat_type} not available. Choose from: {available_stats}")
                continue

            stat_data = grid['stats'][stat_type]
            matched = next((l for l in stat_data['lines'] if l['line'] == line), None)
            if matched is None:
                valid_lines = [l['line'] for l in stat_data['lines']]
                print(f"    Line {line} not available. Choose from: {valid_lines}")
                continue

            prob = matched['prob']
            h2h_prob = matched.get('h2h_prob', prob)
            has_h2h = matched.get('has_h2h', False)

            selected_legs.append({
                'player': player,
                'stat': stat_type,
                'line': line,
                'prob': prob,
                'h2h_prob': h2h_prob,
                'has_h2h': has_h2h,
                'team': grid['team'],
            })
            h2h_note = f" [H2H:{h2h_prob * 100:.0f}%]" if has_h2h else ""
            print(f"    \u2713 Added: {player} OVER {line} {stat_type} ({prob * 100:.0f}%{h2h_note})")
            break

    if len(selected_legs) < 2:
        print("\n  Need at least 2 legs for a parlay.")
        return

    _print_custom_parlay_analysis(selected_legs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_date_header() -> str:
    """Format today's date for the header."""
    return datetime.now().strftime("%B %d, %Y")


def _print_header() -> None:
    """Print the main header."""
    top = "\u2554" + "\u2550" * 74 + "\u2557"
    bot = "\u255a" + "\u2550" * 74 + "\u255d"
    sep = "\u2551"
    title = f"NBA PARLAY BUILDER - {_format_date_header()}"
    padding = 74 - len(title) - 2
    print(f"\n{top}")
    print(f"{sep}  {title}{' ' * padding}{sep}")
    print(bot)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="parlay_builder",
        description="Build optimal +EV parlays from NBA player props.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build (default)
    build_cmd = subparsers.add_parser("build", help="Build parlays (default)")
    build_cmd.add_argument("--min-legs", type=int, default=2, help="Minimum legs (default: 2)")
    build_cmd.add_argument("--max-legs", type=int, default=6, help="Maximum legs (default: 6)")
    build_cmd.add_argument("--min-prob", type=float, default=0.60, help="Min leg probability (default: 0.60)")
    build_cmd.add_argument("--top", type=int, default=10, help="Number of parlays to show (default: 10)")
    build_cmd.add_argument("--refresh", action="store_true", help="Force refresh props from API")
    build_cmd.add_argument("--save", action="store_true", help="Save report to file")
    build_cmd.add_argument("--player", type=str, default=None,
        help="Filter to parlays containing this player (partial match)")
    build_cmd.add_argument("--team", type=str, default=None,
        help="Filter to parlays involving this team/game (partial match)")
    build_cmd.add_argument("--legs", type=int, default=None,
        help="Show only parlays with exactly N legs")
    build_cmd.add_argument("--min-combined-prob", type=float, default=None,
        help="Minimum combined probability (e.g., 0.40)")

    # show
    show_cmd = subparsers.add_parser("show", help="Show details for parlay #N")
    show_cmd.add_argument("n", type=int, help="Parlay number to show")
    show_cmd.add_argument("--refresh", action="store_true")

    # ticket
    ticket_cmd = subparsers.add_parser("ticket", help="Generate betting ticket for parlay #N")
    ticket_cmd.add_argument("n", type=int, help="Parlay number")
    ticket_cmd.add_argument("--refresh", action="store_true")

    # sgp
    sgp_cmd = subparsers.add_parser("sgp", help="Build same-game parlays for team")
    sgp_cmd.add_argument("team", type=str, help="Team name (e.g., Pacers)")
    sgp_cmd.add_argument("--refresh", action="store_true")

    # progressive
    prog_cmd = subparsers.add_parser("progressive", help="Build best 2, 3, 4-leg parlays")
    prog_cmd.add_argument("--refresh", action="store_true")

    # custom parlay
    custom_cmd = subparsers.add_parser("custom", help="Build parlay from specific players you choose")
    custom_cmd.add_argument("players", nargs="+", help="Player names (2-6 players)")
    custom_cmd.add_argument("--vs", type=str, default=None,
        help="Opponent team for H2H analysis (e.g., --vs Bucks)")
    custom_cmd.add_argument("--refresh", action="store_true")

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    _print_header()

    # Default to 'build' if no command given
    command = args.command or "build"

    # Custom command goes directly to player stats — no prop loading needed
    if command == "custom":
        player_names = args.players
        opponent = getattr(args, 'vs', None)
        if opponent:
            print(f"\n  Building custom parlay vs {opponent}")
        print(f"  Players: {', '.join(player_names)}")
        build_custom_parlay_interactive(player_names, opponent)
        return

    # Analyzer config for fetching props
    analyzer_config = AnalyzerConfig()
    if hasattr(args, "refresh") and args.refresh:
        analyzer_config.force_refresh = True

    print("\n  Loading +EV props from prop_analyzer...")
    try:
        ranked_props, summary = analyze_props(analyzer_config)
    except Exception as e:
        print(f"\n  ERROR: Could not load props: {e}")
        sys.exit(1)

    if not ranked_props:
        print("  No +EV props found. Nothing to build parlays from.")
        sys.exit(0)

    print(f"  Found {len(ranked_props)} +EV props to work with")

    # Parlay builder config
    parlay_config = ParlayBuilderConfig()
    if hasattr(args, "min_legs"):
        parlay_config.min_legs = args.min_legs
    if hasattr(args, "max_legs"):
        parlay_config.max_legs = args.max_legs
    if hasattr(args, "min_prob"):
        parlay_config.min_leg_prob = args.min_prob
    if hasattr(args, "top"):
        parlay_config.top_n_to_show = args.top
    # If exact leg count requested, constrain builder to only generate that size
    if hasattr(args, "legs") and args.legs is not None:
        parlay_config.min_legs = args.legs
        parlay_config.max_legs = args.legs

    if command == "build":
        print("\n  Building parlays...")
        parlays = build_parlays(ranked_props, parlay_config)

        if not parlays:
            print("  No +EV parlays found with current settings.")
            return

        print(f"  Found {len(parlays)} +EV parlays")

        # Apply filters
        if hasattr(args, 'player') and args.player:
            parlays = filter_parlays_by_player(parlays, args.player)
            print(f"  Filtered to parlays with '{args.player}': {len(parlays)} found")

        if hasattr(args, 'team') and args.team:
            parlays = filter_parlays_by_team(parlays, args.team)
            print(f"  Filtered to parlays with '{args.team}': {len(parlays)} found")

        if hasattr(args, 'legs') and args.legs is not None:
            parlays = filter_parlays_by_legs(parlays, args.legs)
            print(f"  Filtered to {args.legs}-leg parlays: {len(parlays)} found")

        if hasattr(args, 'min_combined_prob') and args.min_combined_prob is not None:
            parlays = filter_parlays_by_min_prob(parlays, args.min_combined_prob)
            print(f"  Filtered to >{args.min_combined_prob:.0%} probability: {len(parlays)} found")

        if not parlays:
            print("  No parlays match your filters.")
            return

        print_parlays_table(parlays, parlay_config.top_n_to_show)

        # Show best parlay detail
        if parlays:
            print(f"\n  BEST PARLAY DETAIL: #1")
            print_parlay(parlays[0])

        if hasattr(args, "save") and args.save:
            generate_parlay_report(parlays, ranked_props)

        print(f"\n  Run 'python parlay_builder.py show N' for details on parlay #N")
        print(f"  Run 'python parlay_builder.py ticket 1' to get betting ticket")

    elif command == "show":
        parlays = build_parlays(ranked_props, parlay_config)
        idx = args.n - 1
        if 0 <= idx < len(parlays):
            print_parlay(parlays[idx])
        else:
            print(f"\n  Parlay #{args.n} not found. Only {len(parlays)} parlays available.")

    elif command == "ticket":
        parlays = build_parlays(ranked_props, parlay_config)
        idx = args.n - 1
        if 0 <= idx < len(parlays):
            print_parlay_ticket(parlays[idx])
        else:
            print(f"\n  Parlay #{args.n} not found. Only {len(parlays)} parlays available.")

    elif command == "sgp":
        print(f"\n  Building same-game parlays for {args.team}...")
        parlays = build_same_game_parlay(ranked_props, team=args.team, config=parlay_config)
        if parlays:
            print_parlays_table(parlays)
            print_parlay(parlays[0])
        else:
            print(f"  No +EV same-game parlays found for {args.team}.")

    elif command == "progressive":
        print("\n  Building progressive parlays (best at each size)...")
        legs = props_to_legs(ranked_props)
        legs = filter_compatible_legs(legs)
        # Filter legs to meet quality thresholds
        legs = [l for l in legs if l.model_prob >= parlay_config.min_leg_prob
                and l.edge >= parlay_config.min_leg_edge]

        prog = build_progressive_parlay(legs, [2, 3, 4], parlay_config)
        for size, parlay in sorted(prog.items()):
            if parlay:
                print(f"\n  BEST {size}-LEG PARLAY:")
                print_parlay(parlay)
            else:
                print(f"\n  No valid {size}-leg parlay found.")


if __name__ == "__main__":
    main()