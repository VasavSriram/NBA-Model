"""
edge_cal.py — Sports betting math module.

Pure-math utilities for odds conversion, expected value, Kelly criterion,
vig removal, and parlay analysis.  No external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BetAnalysis:
    """Complete analysis of a single bet."""

    book_odds: int
    book_implied_prob: float
    model_prob: float
    edge: float                 # model_prob - implied_prob
    ev: float                   # expected value in dollars
    ev_percent: float           # ev as percentage of stake
    is_plus_ev: bool
    kelly_fraction: float
    half_kelly_fraction: float
    quarter_kelly_fraction: float
    fair_odds: int              # what odds SHOULD be based on model_prob


@dataclass
class ParlayAnalysis:
    """Complete analysis of a parlay."""

    legs: List[int]             # American odds of each leg
    leg_probs: List[float]      # Model probability of each leg
    combined_prob: float
    parlay_decimal: float
    parlay_american: int
    fair_american: int          # Fair odds based on combined prob
    ev: float
    is_plus_ev: bool


# ---------------------------------------------------------------------------
# Core odds conversion
# ---------------------------------------------------------------------------

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability.

    Negative odds (favorites):  abs(odds) / (abs(odds) + 100)
    Positive odds (underdogs):  100 / (odds + 100)

    Examples:
        >>> american_to_implied_prob(-110)  # ≈ 0.5238
        >>> american_to_implied_prob(+150)  # 0.40
        >>> american_to_implied_prob(-200)  # ≈ 0.6667
        >>> american_to_implied_prob(+200)  # ≈ 0.3333
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    # positive odds (or zero, treated as +100 equivalent math)
    return 100 / (odds + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) odds.

    Negative odds: 1 + 100 / abs(odds)
    Positive odds: 1 + odds / 100

    Examples:
        >>> american_to_decimal(-110)  # ≈ 1.909
        >>> american_to_decimal(+150)  # 2.50
        >>> american_to_decimal(-200)  # 1.50
        >>> american_to_decimal(+200)  # 3.00
    """
    if odds < 0:
        return 1 + (100 / abs(odds))
    return 1 + (odds / 100)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds.

    >= 2.0  →  positive American odds: round((decimal - 1) * 100)
    <  2.0  →  negative American odds: round(-100 / (decimal - 1))

    Examples:
        >>> decimal_to_american(1.909)  # -110
        >>> decimal_to_american(2.50)   # +150
    """
    if decimal_odds <= 1.0:
        raise ValueError(
            f"Decimal odds must be > 1.0 (got {decimal_odds}). "
            "Odds of 1.0 mean zero profit, which has no American equivalent."
        )
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def probability_to_fair_odds(prob: float) -> int:
    """Convert a true probability to fair American odds (no vig).

    > 0.5  →  negative (favorite): round(-100 * prob / (1 - prob))
    < 0.5  →  positive (underdog): round(100 * (1 - prob) / prob)
    = 0.5  →  +100 (even money)

    Examples:
        >>> probability_to_fair_odds(0.60)    # -150
        >>> probability_to_fair_odds(0.40)    # +150
        >>> probability_to_fair_odds(0.5238)  # -110
    """
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError(
            f"Probability must be between 0 and 1 exclusive (got {prob}). "
            "A 0% or 100% event has no meaningful American odds."
        )
    if prob == 0.5:
        return 100
    if prob > 0.5:
        return round(-100 * prob / (1 - prob))
    return round(100 * (1 - prob) / prob)


# ---------------------------------------------------------------------------
# Edge and EV
# ---------------------------------------------------------------------------

def calculate_edge(model_prob: float, book_odds: int) -> float:
    """Return edge as model_prob minus the book's implied probability.

    Positive → you have edge.  Negative → the book has edge.

    Example:
        >>> calculate_edge(0.55, -110)  # ≈ +0.0262 (2.62%)
    """
    return model_prob - american_to_implied_prob(book_odds)


def calculate_ev(model_prob: float, book_odds: int, stake: float = 10.0) -> float:
    """Return expected value in dollars for a single bet.

    EV = (model_prob * profit_if_win) - ((1 - model_prob) * stake)
    where profit_if_win = stake * (decimal_odds - 1).

    Example:
        >>> calculate_ev(0.55, -110, 10.0)  # ≈ +$0.50
    """
    decimal_odds = american_to_decimal(book_odds)
    profit = stake * (decimal_odds - 1)  # net profit on a win
    return (model_prob * profit) - ((1 - model_prob) * stake)


def is_plus_ev(model_prob: float, book_odds: int) -> bool:
    """Return True when the bet has positive expected value.

    Example:
        >>> is_plus_ev(0.55, -110)  # True
    """
    return calculate_ev(model_prob, book_odds) > 0


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_criterion(model_prob: float, book_odds: int) -> float:
    """Full Kelly fraction of bankroll to wager.

    f* = (b * p - q) / b
    where b = decimal_odds - 1 (profit per $1 risked),
          p = model_prob,
          q = 1 - p.

    Returns max(0, f*) — never recommends a negative bet.

    Example:
        >>> kelly_criterion(0.55, -110)  # ≈ 0.055 (5.5%)
    """
    b = american_to_decimal(book_odds) - 1  # profit-to-stake ratio
    if b <= 0:
        # Decimal odds of 1.0 or below means zero or negative profit —
        # no rational bet exists, so Kelly says don't bet.
        return 0.0
    p = model_prob
    q = 1 - p
    f_star = (b * p - q) / b
    return max(0.0, f_star)


def half_kelly(model_prob: float, book_odds: int) -> float:
    """Half-Kelly: more conservative, reduces variance while capturing edge.

    Example:
        >>> half_kelly(0.55, -110)  # ≈ 0.0275 (2.75%)
    """
    return kelly_criterion(model_prob, book_odds) / 2


def quarter_kelly(model_prob: float, book_odds: int) -> float:
    """Quarter-Kelly: very conservative, good for uncertain edges.

    Example:
        >>> quarter_kelly(0.55, -110)  # ≈ 0.01375 (1.375%)
    """
    return kelly_criterion(model_prob, book_odds) / 4


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

def calculate_vig(odds_side1: int, odds_side2: int) -> float:
    """Return the book's vig (overround) as a decimal.

    Vig = (implied_prob_1 + implied_prob_2) - 1.0

    Example:
        >>> calculate_vig(-110, -110)  # ≈ 0.0476 (4.76%)
        >>> calculate_vig(-150, 130)   # ≈ 0.0348 (3.48%)
    """
    total_implied = (
        american_to_implied_prob(odds_side1)
        + american_to_implied_prob(odds_side2)
    )
    return total_implied - 1.0


def remove_vig(odds_side1: int, odds_side2: int) -> Tuple[float, float]:
    """Return fair (no-vig) probabilities for each side.

    Each implied probability is divided by the sum of both implied
    probabilities, so the result sums to exactly 1.0.

    Example:
        >>> remove_vig(-110, -110)  # (0.50, 0.50)
        >>> remove_vig(-150, 130)   # ≈ (0.5797, 0.4203)
    """
    imp1 = american_to_implied_prob(odds_side1)
    imp2 = american_to_implied_prob(odds_side2)
    total = imp1 + imp2
    return (imp1 / total, imp2 / total)


def no_vig_odds(odds_side1: int, odds_side2: int) -> Tuple[int, int]:
    """Return fair American odds for each side after removing the vig.

    Example:
        >>> no_vig_odds(-110, -110)  # (100, 100)
        >>> no_vig_odds(-150, 130)   # ≈ (-138, 138)
    """
    fair1, fair2 = remove_vig(odds_side1, odds_side2)
    return (probability_to_fair_odds(fair1), probability_to_fair_odds(fair2))


# ---------------------------------------------------------------------------
# Parlays
# ---------------------------------------------------------------------------

def calculate_parlay_decimal(legs: List[int]) -> float:
    """Multiply the decimal odds of every leg to get the parlay decimal payout.

    Example:
        >>> calculate_parlay_decimal([-110, -110])  # ≈ 3.645
    """
    result = 1.0
    for american_odds in legs:
        result *= american_to_decimal(american_odds)
    return result


def calculate_parlay_american(legs: List[int]) -> int:
    """Return the overall American odds of a parlay.

    Example:
        >>> calculate_parlay_american([-110, -110])  # +264 (approx)
    """
    return decimal_to_american(calculate_parlay_decimal(legs))


def calculate_parlay_probability(leg_probs: List[float]) -> float:
    """Combined probability of all legs hitting (assumes independence).

    Example:
        >>> calculate_parlay_probability([0.55, 0.60, 0.70])  # 0.231
    """
    result = 1.0
    for p in leg_probs:
        result *= p
    return result


def calculate_parlay_ev(
    leg_probs: List[float],
    parlay_american_odds: int,
    stake: float = 10.0,
) -> float:
    """Expected value of a parlay in dollars.

    EV = (combined_prob * profit) - ((1 - combined_prob) * stake)

    Example:
        >>> calculate_parlay_ev([0.55, 0.60, 0.70], 500, 10.0)
    """
    combined = calculate_parlay_probability(leg_probs)
    decimal_odds = american_to_decimal(parlay_american_odds)
    profit = stake * (decimal_odds - 1)
    return (combined * profit) - ((1 - combined) * stake)


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------

def analyze_bet(
    model_prob: float,
    book_odds: int,
    stake: float = 10.0,
) -> BetAnalysis:
    """Return a complete analysis of a single bet.

    Example:
        >>> analyze_bet(0.55, -110)
    """
    imp = american_to_implied_prob(book_odds)
    ev = calculate_ev(model_prob, book_odds, stake)

    # Clamp model_prob away from exact 0/1 for fair_odds calculation —
    # a true 0% or 100% probability has no American odds representation.
    clamped_prob = max(0.001, min(0.999, model_prob))

    return BetAnalysis(
        book_odds=book_odds,
        book_implied_prob=imp,
        model_prob=model_prob,
        edge=model_prob - imp,
        ev=ev,
        ev_percent=(ev / stake) * 100 if stake > 0 else 0.0,
        is_plus_ev=ev > 0,
        kelly_fraction=kelly_criterion(model_prob, book_odds),
        half_kelly_fraction=half_kelly(model_prob, book_odds),
        quarter_kelly_fraction=quarter_kelly(model_prob, book_odds),
        fair_odds=probability_to_fair_odds(clamped_prob),
    )


def analyze_parlay(
    leg_probs: List[float],
    leg_odds: List[int],
    stake: float = 10.0,
) -> ParlayAnalysis:
    """Return a complete analysis of a parlay.

    Example:
        >>> analyze_parlay([0.55, 0.60], [-110, -150])
    """
    combined = calculate_parlay_probability(leg_probs)
    parlay_dec = calculate_parlay_decimal(leg_odds)
    parlay_am = decimal_to_american(parlay_dec)
    clamped_combined = max(0.001, min(0.999, combined))
    fair_am = probability_to_fair_odds(clamped_combined)
    ev = calculate_parlay_ev(leg_probs, parlay_am, stake)

    return ParlayAnalysis(
        legs=leg_odds,
        leg_probs=leg_probs,
        combined_prob=combined,
        parlay_decimal=parlay_dec,
        parlay_american=parlay_am,
        fair_american=fair_am,
        ev=ev,
        is_plus_ev=ev > 0,
    )


def compare_to_sharp(
    your_odds: int,
    sharp_odds_side1: int,
    sharp_odds_side2: int,
) -> Dict[str, object]:
    """Compare your book's line to a sharp book's no-vig line.

    Pros find +EV by checking whether the retail book's odds are softer
    than the sharp book's fair (de-vigged) price.

    Args:
        your_odds:        The American odds your book is offering you.
        sharp_odds_side1: Sharp book's odds on *your* side.
        sharp_odds_side2: Sharp book's odds on the *other* side.

    Returns:
        Dict with sharp_fair_prob, your_implied_prob, edge, ev,
        and a plain-English recommendation.

    Example:
        >>> compare_to_sharp(105, -110, -110)
    """
    sharp_fair_prob, _ = remove_vig(sharp_odds_side1, sharp_odds_side2)
    your_implied = american_to_implied_prob(your_odds)
    edge = sharp_fair_prob - your_implied
    ev = calculate_ev(sharp_fair_prob, your_odds)

    if edge > 0.02:
        recommendation = "Strong +EV — bet it."
    elif edge > 0:
        recommendation = "Slight +EV — consider betting."
    else:
        recommendation = "No edge — pass."

    return {
        "sharp_fair_prob": round(sharp_fair_prob, 4),
        "your_implied_prob": round(your_implied, 4),
        "edge": round(edge, 4),
        "ev": round(ev, 2),
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  edge_cal.py — demo")
    print("=" * 60)

    # --- Odds conversions ---
    print("\n--- Odds conversions ---")
    for odds in [-110, 150, -200, 200]:
        imp = american_to_implied_prob(odds)
        dec = american_to_decimal(odds)
        print(f"  {odds:+d}  ->  implied {imp:.4f}  decimal {dec:.3f}")

    # --- Single-bet analysis ---
    print("\n--- Single-bet analysis (55% model, -110 line) ---")
    analysis = analyze_bet(model_prob=0.55, book_odds=-110, stake=10.0)
    print(f"  Edge:          {analysis.edge:+.4f} ({analysis.edge * 100:+.2f}%)")
    print(f"  EV ($10):      ${analysis.ev:+.2f} ({analysis.ev_percent:+.2f}%)")
    print(f"  +EV?           {analysis.is_plus_ev}")
    print(f"  Full Kelly:    {analysis.kelly_fraction:.4f} ({analysis.kelly_fraction * 100:.2f}%)")
    print(f"  Half Kelly:    {analysis.half_kelly_fraction:.4f}")
    print(f"  Quarter Kelly: {analysis.quarter_kelly_fraction:.4f}")
    print(f"  Fair odds:     {analysis.fair_odds:+d}")

    # --- Vig removal ---
    print("\n--- Vig removal (-110 / -110) ---")
    vig = calculate_vig(-110, -110)
    fair1, fair2 = remove_vig(-110, -110)
    nv1, nv2 = no_vig_odds(-110, -110)
    print(f"  Vig:        {vig:.4f} ({vig * 100:.2f}%)")
    print(f"  Fair probs: {fair1:.4f} / {fair2:.4f}")
    print(f"  Fair odds:  {nv1:+d} / {nv2:+d}")

    # --- Parlay ---
    print("\n--- 2-leg parlay (both -110) ---")
    parlay = analyze_parlay(
        leg_probs=[0.55, 0.55],
        leg_odds=[-110, -110],
        stake=10.0,
    )
    print(f"  Combined prob: {parlay.combined_prob:.4f}")
    print(f"  Decimal odds:  {parlay.parlay_decimal:.3f}")
    print(f"  American odds: {parlay.parlay_american:+d}")
    print(f"  EV ($10):      ${parlay.ev:+.2f}")

    # --- Sharp comparison ---
    print("\n--- Sharp comparison (your +105 vs sharp -110/-110) ---")
    comp = compare_to_sharp(105, -110, -110)
    for k, v in comp.items():
        print(f"  {k}: {v}")
