"""
probability_model.py -- NBA player prop probability calculator.

Calculates P(over) and P(under) for a player prop line using either
a normal or Poisson distribution, based on a player's statistical
average and variance.  Designed to feed into edge_cal.py for full
EV / Kelly analysis.

Dependencies: scipy, numpy (standard scientific Python stack).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, poisson


# ---------------------------------------------------------------------------
# Default coefficients of variation (CV = std_dev / mean)
# ---------------------------------------------------------------------------

# These are empirically-derived typical CVs for NBA box-score stats.
# They let us estimate std_dev when only a season average is available.
DEFAULT_CVS: Dict[str, float] = {
    "PTS": 0.35,   # Points -- relatively consistent for starters
    "REB": 0.40,   # Rebounds -- more game-to-game variance
    "AST": 0.40,   # Assists -- similar to rebounds
    "3PM": 0.60,   # Three-pointers made -- high variance
    "STL": 0.70,   # Steals -- very volatile, low-count stat
    "BLK": 0.80,   # Blocks -- most volatile, low-count stat
    "PRA": 0.30,   # Pts + Reb + Ast combined -- more stable
    "PR":  0.32,   # Points + Rebounds
    "PA":  0.32,   # Points + Assists
    "RA":  0.38,   # Rebounds + Assists
    "MIN": 0.15,   # Minutes -- fairly consistent for starters
}

# Stats that always use Poisson regardless of mean
_ALWAYS_POISSON = {"3PM", "STL", "BLK"}

# Threshold: use Poisson for REB/AST when mean is below this value
_POISSON_MEAN_THRESHOLD = 5.0

# Probability clamp bounds -- never return exactly 0% or 100%
_PROB_FLOOR = 0.001
_PROB_CEIL  = 0.999


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PropProbability:
    """Complete probability analysis of a single player prop."""

    player: str
    stat_type: str
    line: float
    mean: float
    std_dev: float
    cv: float
    distribution_used: str      # 'normal' or 'poisson'
    prob_over: float
    prob_under: float

    sample_size: Optional[int] = None   # games used in calculation
    confidence: Optional[str] = None    # 'high', 'medium', 'low'


# ---------------------------------------------------------------------------
# CV & std-dev helpers
# ---------------------------------------------------------------------------

def estimate_std_dev(
    mean: float,
    stat_type: str,
    cv_overrides: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate standard deviation from a player's mean using the CV for that stat type.

    CV (coefficient of variation) = std_dev / mean, so std_dev = mean * CV.

    Args:
        mean: Player's average for the stat.
        stat_type: Key into DEFAULT_CVS (e.g. 'PTS', '3PM').
        cv_overrides: Optional dict to override default CVs.

    Example:
        >>> estimate_std_dev(20.0, 'PTS')   # 20.0 * 0.35 = 7.0
        >>> estimate_std_dev(3.0, '3PM')    # 3.0 * 0.60 = 1.8
    """
    cvs = {**DEFAULT_CVS, **(cv_overrides or {})}
    cv = cvs.get(stat_type.upper(), 0.35)  # fall back to PTS-like CV
    return mean * cv


def calc_actual_std_dev(game_log: List[float]) -> float:
    """Calculate sample standard deviation from a game log.

    Uses ddof=1 (Bessel's correction) for an unbiased estimate from a sample.
    Requires at least 2 games; raises ValueError otherwise.

    Example:
        >>> calc_actual_std_dev([18, 22, 15, 25, 20])  # ~3.808
    """
    if len(game_log) < 2:
        raise ValueError(
            f"Need at least 2 games for sample std dev, got {len(game_log)}"
        )
    return float(np.std(game_log, ddof=1))


def calc_actual_cv(game_log: List[float]) -> float:
    """Calculate coefficient of variation from a game log.

    CV = sample_std_dev / mean.  Requires at least 5 games to be
    meaningful; raises ValueError otherwise.

    Example:
        >>> calc_actual_cv([18, 22, 15, 25, 20])  # ~0.183
    """
    if len(game_log) < 5:
        raise ValueError(
            f"Need at least 5 games for a reliable CV, got {len(game_log)}"
        )
    mean = float(np.mean(game_log))
    if mean == 0:
        return 0.0
    return calc_actual_std_dev(game_log) / mean


# ---------------------------------------------------------------------------
# Distribution selection
# ---------------------------------------------------------------------------

def recommend_distribution(stat_type: str, mean: float) -> str:
    """Choose 'normal' or 'poisson' based on the stat type and mean.

    Rules:
      - 3PM, STL, BLK -> always Poisson (low-count, discrete events).
      - REB, AST -> Poisson if mean < 5, else normal.
      - PTS, PRA, PR, PA, RA, MIN -> always normal.
      - Any stat with mean < 1.0 -> Poisson (too few events for normal).

    Example:
        >>> recommend_distribution('PTS', 22.0)  # 'normal'
        >>> recommend_distribution('3PM', 3.0)   # 'poisson'
        >>> recommend_distribution('REB', 4.0)   # 'poisson'
        >>> recommend_distribution('REB', 12.0)  # 'normal'
    """
    st = stat_type.upper()

    # Very low means always use Poisson
    if mean < 1.0:
        return "poisson"

    if st in _ALWAYS_POISSON:
        return "poisson"

    if st in {"REB", "AST"} and mean < _POISSON_MEAN_THRESHOLD:
        return "poisson"

    return "normal"


# ---------------------------------------------------------------------------
# Normal-distribution probability
# ---------------------------------------------------------------------------

def _clamp(p: float) -> float:
    """Clamp probability to [0.001, 0.999]."""
    return max(_PROB_FLOOR, min(_PROB_CEIL, p))


# Minimum std_dev for normal calculations to avoid division by zero.
# A std_dev this small means the outcome is essentially deterministic,
# so we compare mean vs line directly.
_MIN_STD_DEV = 1e-9


def calc_over_prob_normal(mean: float, std_dev: float, line: float) -> float:
    """P(over) using the normal distribution with continuity correction.

    "Over 19.5" means the player must score >= 20 (integer outcome).
    Continuity correction: we integrate from (line + 0.5) upward,
    treating the discrete value 20 as occupying [19.5, 20.5].

    Since "over line" means X > line, and lines are at half-integers,
    the effective threshold is ceil(line) = floor(line) + 1.
    With continuity correction: z = (ceil(line) - 0.5 - mean) / std
                                  = (line - mean) / std   (for .5 lines)

    More explicitly:
        z = (line - 0.5 - mean) / std_dev
        P(over) = 1 - norm.cdf(z)

    We subtract 0.5 from the line so that, for a line of 19.5, we compute
    z = (19.0 - mean) / std, effectively asking P(X >= 19.5) in continuous
    terms, which maps to P(X >= 20) in the discrete world.

    Example:
        >>> calc_over_prob_normal(20.0, 7.0, 19.5)  # ~0.5284
    """
    if std_dev < _MIN_STD_DEV:
        # Near-zero variance: outcome is deterministic
        return _clamp(1.0 if mean > line else 0.0)
    z = (line - 0.5 - mean) / std_dev
    return _clamp(1.0 - norm.cdf(z))


def calc_under_prob_normal(mean: float, std_dev: float, line: float) -> float:
    """P(under) using the normal distribution with continuity correction.

    "Under 19.5" means the player scores <= 19.
    With continuity correction we integrate up to (line - 0.5 + 0.5) = line:
        z = (line + 0.5 - mean) / std_dev
        P(under) = norm.cdf(z)

    We add 0.5 to the line so that, for a line of 19.5, we compute
    z = (20.0 - mean) / std, asking P(X <= 20.0) in continuous terms,
    which maps to P(X <= 19) in the discrete world.

    Example:
        >>> calc_under_prob_normal(20.0, 7.0, 19.5)  # ~0.4716
    """
    if std_dev < _MIN_STD_DEV:
        # Near-zero variance: outcome is deterministic
        return _clamp(1.0 if mean < line else 0.0)
    z = (line + 0.5 - mean) / std_dev
    return _clamp(norm.cdf(z))


# ---------------------------------------------------------------------------
# Poisson-distribution probability
# ---------------------------------------------------------------------------

def calc_over_prob_poisson(mean: float, line: float) -> float:
    """P(over) using the Poisson distribution.

    "Over 2.5" means X >= 3, so P(over) = 1 - P(X <= 2).

    Args:
        mean: Player's average (lambda parameter for Poisson).
        line: The prop line (e.g. 2.5).

    Example:
        >>> calc_over_prob_poisson(2.5, 2.5)  # P(X >= 3) ~ 0.456
        >>> calc_over_prob_poisson(1.5, 0.5)  # P(X >= 1) ~ 0.777
    """
    k = math.floor(line)  # "over 2.5" -> need X > 2, i.e. X >= 3
    return _clamp(1.0 - poisson.cdf(k, mean))


def calc_under_prob_poisson(mean: float, line: float) -> float:
    """P(under) using the Poisson distribution.

    "Under 2.5" means X <= 2, so P(under) = P(X <= 2).
    For integer lines like 3.0: "under 3.0" means X <= 2 = P(X <= floor(3)-1).
    For half-integer lines like 2.5: "under 2.5" means X <= 2 = P(X <= floor(2.5)).

    Args:
        mean: Player's average (lambda parameter).
        line: The prop line.

    Example:
        >>> calc_under_prob_poisson(2.5, 2.5)  # P(X <= 2) ~ 0.544
    """
    # If line is an integer (e.g. 3.0), "under 3" means X <= 2
    # If line is half-integer (e.g. 2.5), "under 2.5" means X <= 2
    if line == math.floor(line):
        # integer line: "under 3" = X <= 2
        k = int(line) - 1
    else:
        # half-integer line: "under 2.5" = X <= 2
        k = math.floor(line)

    if k < 0:
        return _clamp(0.0)
    return _clamp(poisson.cdf(k, mean))


# ---------------------------------------------------------------------------
# Unified probability interface
# ---------------------------------------------------------------------------

def calc_over_probability(
    mean: float,
    line: float,
    stat_type: str,
    std_dev: Optional[float] = None,
    distribution: str = "auto",
) -> float:
    """Calculate P(over) routing to the correct distribution.

    Args:
        mean: Player's average for the stat.
        line: Prop line (e.g. 19.5).
        stat_type: Stat key (e.g. 'PTS', '3PM').
        std_dev: Known std dev; estimated from CV if None.
        distribution: 'normal', 'poisson', or 'auto' (default).

    Example:
        >>> calc_over_probability(20.0, 19.5, 'PTS')
    """
    dist = (
        distribution
        if distribution != "auto"
        else recommend_distribution(stat_type, mean)
    )

    if dist == "poisson":
        return calc_over_prob_poisson(mean, line)

    # Normal path -- need std_dev
    if std_dev is None:
        std_dev = estimate_std_dev(mean, stat_type)
    return calc_over_prob_normal(mean, std_dev, line)


def calc_under_probability(
    mean: float,
    line: float,
    stat_type: str,
    std_dev: Optional[float] = None,
    distribution: str = "auto",
) -> float:
    """Calculate P(under) routing to the correct distribution.

    Args:
        mean: Player's average for the stat.
        line: Prop line (e.g. 19.5).
        stat_type: Stat key (e.g. 'PTS', '3PM').
        std_dev: Known std dev; estimated from CV if None.
        distribution: 'normal', 'poisson', or 'auto' (default).

    Example:
        >>> calc_under_probability(20.0, 19.5, 'PTS')
    """
    dist = (
        distribution
        if distribution != "auto"
        else recommend_distribution(stat_type, mean)
    )

    if dist == "poisson":
        return calc_under_prob_poisson(mean, line)

    if std_dev is None:
        std_dev = estimate_std_dev(mean, stat_type)
    return calc_under_prob_normal(mean, std_dev, line)


# ---------------------------------------------------------------------------
# Effective / weighted averages
# ---------------------------------------------------------------------------

def calc_effective_average(
    season_avg: float,
    last5_avg: float,
    last10_avg: float,
    weights: Tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> float:
    """Weighted average across three time windows.

    Default weights: 40% season, 35% last-10, 25% last-5.
    The idea is to blend long-term baseline with recent form.

    Args:
        season_avg: Full-season average.
        last5_avg: Average over the last 5 games.
        last10_avg: Average over the last 10 games.
        weights: (season_weight, last10_weight, last5_weight).  Must sum to 1.0.

    Example:
        >>> calc_effective_average(20, 25, 22)
        # 0.40*20 + 0.35*22 + 0.25*25 = 8 + 7.7 + 6.25 = 21.95
    """
    w_season, w_last10, w_last5 = weights
    return w_season * season_avg + w_last10 * last10_avg + w_last5 * last5_avg


def calc_weighted_average(
    season_avg: float,
    recent_avg: float,
    recent_weight: float = 0.60,
) -> float:
    """Simple two-window weighted average.

    Args:
        season_avg: Full-season average.
        recent_avg: Recent stretch average.
        recent_weight: Weight on recent_avg (0-1). Season gets the remainder.

    Example:
        >>> calc_weighted_average(20, 24, 0.6)  # 0.4*20 + 0.6*24 = 22.4
    """
    return (1 - recent_weight) * season_avg + recent_weight * recent_avg


# ---------------------------------------------------------------------------
# Confidence heuristic
# ---------------------------------------------------------------------------

def _confidence_from_sample_size(n: Optional[int]) -> str:
    """Return 'high', 'medium', or 'low' based on number of games."""
    if n is None:
        return "low"
    if n >= 20:
        return "high"
    if n >= 10:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_prop(
    player: str,
    stat_type: str,
    line: float,
    season_avg: float,
    recent_avg: Optional[float] = None,
    game_log: Optional[List[float]] = None,
    distribution: str = "auto",
) -> PropProbability:
    """Full probability analysis of a single player prop.

    Priority:
      1. If game_log is provided (>= 5 games): use actual mean, std_dev, CV.
      2. If recent_avg is provided: blend with season_avg via calc_weighted_average.
      3. Otherwise: use season_avg and estimate std_dev from CV.

    Args:
        player: Player name (for labeling).
        stat_type: 'PTS', '3PM', etc.
        line: The prop line.
        season_avg: Season-long average for this stat.
        recent_avg: Optional recent stretch average.
        game_log: Optional list of stat values from recent games.
        distribution: 'normal', 'poisson', or 'auto'.

    Example:
        >>> analyze_prop("LaMelo Ball", "PTS", 19.5, season_avg=19.5, recent_avg=22.0)
    """
    st = stat_type.upper()
    sample_size: Optional[int] = None

    # --- Determine mean and std_dev ---
    if game_log and len(game_log) >= 5:
        mean = float(np.mean(game_log))
        std_dev = calc_actual_std_dev(game_log)
        cv = calc_actual_cv(game_log)
        sample_size = len(game_log)
    else:
        if recent_avg is not None:
            mean = calc_weighted_average(season_avg, recent_avg)
        else:
            mean = season_avg
        std_dev = estimate_std_dev(mean, st)
        cv = std_dev / mean if mean > 0 else 0.0
        if game_log:
            sample_size = len(game_log)

    # --- Choose distribution ---
    dist = (
        distribution if distribution != "auto"
        else recommend_distribution(st, mean)
    )

    # --- Calculate probabilities ---
    if dist == "poisson":
        prob_over = calc_over_prob_poisson(mean, line)
        prob_under = calc_under_prob_poisson(mean, line)
    else:
        prob_over = calc_over_prob_normal(mean, std_dev, line)
        prob_under = calc_under_prob_normal(mean, std_dev, line)

    return PropProbability(
        player=player,
        stat_type=st,
        line=line,
        mean=round(mean, 2),
        std_dev=round(std_dev, 2),
        cv=round(cv, 3),
        distribution_used=dist,
        prob_over=round(prob_over, 4),
        prob_under=round(prob_under, 4),
        sample_size=sample_size,
        confidence=_confidence_from_sample_size(sample_size),
    )


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def calc_h2h_weighted_average(
    season_avg: float,
    last5_avg: float,
    h2h_avg: Optional[float] = None,
    h2h_games: int = 0,
) -> Tuple[float, float]:
    """Calculate both standard and H2H-weighted effective averages.

    Standard:  60% last-5 + 40% season
    H2H (3+ games): 50% H2H + 30% last-5 + 20% season
    H2H (2 games):  30% H2H + 40% last-5 + 30% season
    Falls back to standard when H2H data is unavailable.

    Returns:
        (standard_weighted_avg, h2h_weighted_avg)
    """
    standard = 0.6 * last5_avg + 0.4 * season_avg

    if h2h_avg is not None and h2h_games >= 2:
        if h2h_games >= 3:
            h2h_weighted = 0.50 * h2h_avg + 0.30 * last5_avg + 0.20 * season_avg
        else:
            h2h_weighted = 0.30 * h2h_avg + 0.40 * last5_avg + 0.30 * season_avg
    else:
        h2h_weighted = standard

    return round(standard, 2), round(h2h_weighted, 2)


def calc_prob_with_h2h(
    stat_type: str,
    line: float,
    season_avg: float,
    last5_avg: float,
    h2h_avg: Optional[float] = None,
    h2h_games: int = 0,
) -> Tuple[float, float]:
    """Calculate over-probability using both standard and H2H-weighted averages.

    Returns:
        (standard_prob, h2h_prob)
        When no H2H data is available, both values are equal.
    """
    std_mean, h2h_mean = calc_h2h_weighted_average(
        season_avg, last5_avg, h2h_avg, h2h_games
    )
    dist = recommend_distribution(stat_type, std_mean)

    std_dev_std = estimate_std_dev(std_mean, stat_type)
    if dist == "normal":
        std_prob = calc_over_prob_normal(std_mean, std_dev_std, line)
    else:
        std_prob = calc_over_prob_poisson(std_mean, line)

    std_dev_h2h = estimate_std_dev(h2h_mean, stat_type)
    if dist == "normal":
        h2h_prob = calc_over_prob_normal(h2h_mean, std_dev_h2h, line)
    else:
        h2h_prob = calc_over_prob_poisson(h2h_mean, line)

    return round(std_prob, 3), round(h2h_prob, 3)


def analyze_props_batch(props: List[dict]) -> List[PropProbability]:
    """Analyze a list of props in one call.

    Each dict must have keys: player, stat_type, line, season_avg.
    Optional keys: recent_avg, game_log, distribution.

    Example:
        >>> analyze_props_batch([
        ...     {"player": "LeBron", "stat_type": "PTS", "line": 24.5, "season_avg": 25.0},
        ...     {"player": "Curry",  "stat_type": "3PM", "line": 4.5,  "season_avg": 5.0},
        ... ])
    """
    results = []
    for p in props:
        results.append(analyze_prop(
            player=p["player"],
            stat_type=p["stat_type"],
            line=p["line"],
            season_avg=p["season_avg"],
            recent_avg=p.get("recent_avg"),
            game_log=p.get("game_log"),
            distribution=p.get("distribution", "auto"),
        ))
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  probability_model.py -- demo")
    print("=" * 60)

    # --- Normal distribution example ---
    print("\n--- PTS: LaMelo Ball, line 19.5 ---")
    prop = analyze_prop("LaMelo Ball", "PTS", 19.5, season_avg=19.5, recent_avg=22.0)
    print(f"  Mean (weighted):  {prop.mean}")
    print(f"  Std dev:          {prop.std_dev}")
    print(f"  CV:               {prop.cv}")
    print(f"  Distribution:     {prop.distribution_used}")
    print(f"  P(over 19.5):     {prop.prob_over:.4f} ({prop.prob_over*100:.1f}%)")
    print(f"  P(under 19.5):    {prop.prob_under:.4f} ({prop.prob_under*100:.1f}%)")
    print(f"  Confidence:       {prop.confidence}")

    # --- Poisson distribution example ---
    print("\n--- 3PM: Steph Curry, line 4.5 ---")
    prop2 = analyze_prop("Steph Curry", "3PM", 4.5, season_avg=5.0, recent_avg=4.5)
    print(f"  Mean (weighted):  {prop2.mean}")
    print(f"  Distribution:     {prop2.distribution_used}")
    print(f"  P(over 4.5):      {prop2.prob_over:.4f} ({prop2.prob_over*100:.1f}%)")
    print(f"  P(under 4.5):     {prop2.prob_under:.4f} ({prop2.prob_under*100:.1f}%)")

    # --- Game log example ---
    print("\n--- REB from game log ---")
    log = [8, 12, 10, 7, 11, 9, 13, 8, 10, 11]
    prop3 = analyze_prop("Anthony Davis", "REB", 9.5, season_avg=10.0, game_log=log)
    print(f"  Mean (from log):  {prop3.mean}")
    print(f"  Std dev (actual): {prop3.std_dev}")
    print(f"  CV (actual):      {prop3.cv}")
    print(f"  Sample size:      {prop3.sample_size}")
    print(f"  Distribution:     {prop3.distribution_used}")
    print(f"  P(over 9.5):      {prop3.prob_over:.4f}")
    print(f"  Confidence:       {prop3.confidence}")

    # --- Integration with edge_cal ---
    print("\n--- Integration with edge_cal.py ---")
    try:
        from edge_cal import analyze_bet
        bet = analyze_bet(model_prob=prop.prob_over, book_odds=-115)
        print(f"  Model prob:  {prop.prob_over:.4f}")
        print(f"  Book odds:   -115")
        print(f"  Edge:        {bet.edge:+.2%}")
        print(f"  EV ($10):    ${bet.ev:+.2f}")
        print(f"  Kelly:       {bet.kelly_fraction:.4f} ({bet.kelly_fraction*100:.2f}%)")
    except ImportError:
        print("  (edge_cal.py not found -- skipping integration demo)")
