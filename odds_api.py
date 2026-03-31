"""
odds_api.py -- Fetch NBA player prop odds from The Odds API.

Integrates with edge_cal.py, probability_model.py, and player_stats_db.py
to find +EV betting opportunities on NBA player props.

Dependencies: requests, sqlite3, player_stats_db, probability_model, edge_cal
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

import edge_cal
import probability_model
import player_stats_db


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT = "basketball_nba"
_DEFAULT_API_KEY = ""

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds_cache.db")

_CACHE_MINUTES_PROPS = 120     # 2 hours default for props
_CACHE_MINUTES_GAMES = 60      # 1 hour for game list

# API market name <-> our canonical stat type
API_TO_STAT: Dict[str, str] = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "3PM",
    "player_blocks": "BLK",
    "player_steals": "STL",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds": "PR",
    "player_points_assists": "PA",
}

STAT_TO_API: Dict[str, str] = {v: k for k, v in API_TO_STAT.items()}

# All supported prop markets for fetching
_ALL_MARKETS = ",".join(API_TO_STAT.keys())

# Maps our canonical stat -> player_stats_db column name
_STAT_TO_DB_COL: Dict[str, str] = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "3PM": "three_pm",
    "STL": "steals",
    "BLK": "blocks",
    "PRA": None,   # combo stat -- handled specially
    "PR": None,
    "PA": None,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OddsAPIError(Exception):
    """General API error."""
    pass


class RateLimitError(OddsAPIError):
    """Raised when API rate limit (429) is hit."""
    pass


class NoPropsAvailableError(Exception):
    """Raised when no props are found."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerProp:
    """A single player prop line from a bookmaker."""

    player_name: str
    stat_type: str          # 'PTS', 'REB', 'AST', '3PM', etc.
    line: float             # e.g. 19.5
    over_odds: int          # American odds, e.g. -115
    under_odds: int         # American odds, e.g. -105
    bookmaker: str          # e.g. 'draftkings', 'fanduel'
    game_id: str
    home_team: str
    away_team: str
    game_time: datetime
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class PropAnalysis:
    """Complete analysis of a prop combining odds with model probability."""

    prop: PlayerProp

    # From player_stats_db
    season_avg: float
    last5_avg: float
    effective_avg: float
    games_played: int

    # From probability_model
    prob_over: float
    prob_under: float
    distribution_used: str

    # From edge_cal
    over_edge: float
    under_edge: float
    over_ev: float
    under_ev: float
    best_side: str          # 'over', 'under', or 'none'
    best_edge: float
    best_ev: float
    is_plus_ev: bool
    kelly_fraction: float
    fair_odds: int


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """Return API key from environment variable or default."""
    return os.environ.get("ODDS_API_KEY", _DEFAULT_API_KEY)


# ---------------------------------------------------------------------------
# Database (odds cache)
# ---------------------------------------------------------------------------

def _get_cache_conn() -> sqlite3.Connection:
    """Return connection to the odds cache database."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_cache_db() -> None:
    """Create cache tables if they don't exist."""
    conn = _get_cache_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS props_cache (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                stat_type   TEXT NOT NULL,
                line        REAL NOT NULL,
                over_odds   INTEGER,
                under_odds  INTEGER,
                bookmaker   TEXT NOT NULL,
                game_id     TEXT NOT NULL,
                home_team   TEXT,
                away_team   TEXT,
                game_time   TEXT,
                fetched_at  TEXT NOT NULL,
                UNIQUE(player_name, stat_type, line, bookmaker, game_id)
            );

            CREATE TABLE IF NOT EXISTS api_requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint    TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                requests_used     INTEGER,
                requests_remaining INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_props_cache_fetched
                ON props_cache(fetched_at DESC);

            CREATE INDEX IF NOT EXISTS idx_props_cache_player
                ON props_cache(player_name, stat_type);
        """)
        conn.commit()
    finally:
        conn.close()


# Auto-init on import
_init_cache_db()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _api_get(endpoint: str, params: Dict = None,
             retries: int = 3) -> requests.Response:
    """Make a GET request to the Odds API with retry logic.

    Tracks usage via response headers and raises on rate limit.
    """
    if params is None:
        params = {}
    params.setdefault("apiKey", get_api_key())

    last_err = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)  # 2s, 4s backoff

            resp = requests.get(f"{_BASE_URL}{endpoint}", params=params,
                                timeout=30)

            # Track usage
            used = resp.headers.get("x-requests-used")
            remaining = resp.headers.get("x-requests-remaining")
            if used is not None:
                log_api_request(endpoint, int(used), int(remaining or 0))

            if resp.status_code == 429:
                raise RateLimitError(
                    f"Rate limit reached. Used: {used}, Remaining: {remaining}. "
                    "Wait before making more requests."
                )

            if resp.status_code == 401:
                raise OddsAPIError("Invalid API key.")

            if resp.status_code != 200:
                raise OddsAPIError(
                    f"API returned {resp.status_code}: {resp.text[:200]}"
                )

            return resp

        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                continue
            raise OddsAPIError(
                f"Network error after {retries} attempts: {last_err}"
            )

    raise OddsAPIError(f"API call failed after {retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

def log_api_request(endpoint: str, used: int, remaining: int) -> None:
    """Store API usage in the cache database."""
    conn = _get_cache_conn()
    try:
        conn.execute(
            "INSERT INTO api_requests (endpoint, timestamp, requests_used, requests_remaining) "
            "VALUES (?, ?, ?, ?)",
            (endpoint, datetime.now().isoformat(), used, remaining),
        )
        conn.commit()
    finally:
        conn.close()

    if remaining < 50:
        print(f"  WARNING: Only {remaining} API requests remaining this month!")


def check_usage() -> Dict:
    """Check API usage without consuming a counted request.

    The /sports endpoint is free and returns usage headers.
    """
    resp = _api_get("/sports")
    return {
        "requests_used": int(resp.headers.get("x-requests-used", 0)),
        "requests_remaining": int(resp.headers.get("x-requests-remaining", 0)),
    }


def get_usage_stats() -> Dict:
    """Return usage stats from the local log."""
    conn = _get_cache_conn()
    try:
        # Most recent request
        latest = conn.execute(
            "SELECT requests_used, requests_remaining, timestamp "
            "FROM api_requests ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # Requests today
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_count = conn.execute(
            "SELECT COUNT(*) FROM api_requests WHERE timestamp LIKE ?",
            (f"{today_str}%",),
        ).fetchone()[0]

        if latest:
            return {
                "requests_used": latest["requests_used"],
                "requests_remaining": latest["requests_remaining"],
                "requests_today": today_count,
                "last_request": latest["timestamp"],
            }
        return {
            "requests_used": 0,
            "requests_remaining": 0,
            "requests_today": 0,
            "last_request": None,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fetch games
# ---------------------------------------------------------------------------

def fetch_nba_games() -> List[Dict]:
    """Fetch today's NBA games from the API.

    Returns list of dicts with: id, home_team, away_team, commence_time.
    """
    resp = _api_get(f"/sports/{_SPORT}/events")
    events = resp.json()

    games = []
    for ev in events:
        games.append({
            "id": ev["id"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "commence_time": ev["commence_time"],
        })
    return games


# ---------------------------------------------------------------------------
# Fetch player props
# ---------------------------------------------------------------------------

def _parse_props_response(data: Dict, game_id: str, home_team: str,
                          away_team: str, game_time: str) -> List[PlayerProp]:
    """Parse the API response for a single game into PlayerProp objects."""
    props = []

    try:
        gt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        gt = datetime.now()

    for bookmaker in data.get("bookmakers", []):
        book_key = bookmaker["key"]

        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            stat_type = API_TO_STAT.get(market_key)
            if stat_type is None:
                continue  # unknown market, skip

            # Group outcomes by (player_name, point) to pair Over/Under
            grouped: Dict[tuple, Dict[str, int]] = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                point = outcome.get("point")
                side = outcome.get("name", "").lower()
                price = outcome.get("price")

                if not player or point is None or price is None:
                    continue

                key = (player, float(point))
                if key not in grouped:
                    grouped[key] = {}
                grouped[key][side] = int(price)

            # Create PlayerProp for each complete pair
            for (player, line), sides in grouped.items():
                over_odds = sides.get("over")
                under_odds = sides.get("under")

                if over_odds is None or under_odds is None:
                    continue  # incomplete pair

                props.append(PlayerProp(
                    player_name=player,
                    stat_type=stat_type,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    bookmaker=book_key,
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    game_time=gt,
                    fetched_at=datetime.now(),
                ))

    return props


def fetch_player_props(game_id: str,
                       markets: List[str] = None) -> List[PlayerProp]:
    """Fetch player props for a specific game.

    Args:
        game_id: The Odds API game ID.
        markets: List of API market names (e.g. ['player_points']).
                 Defaults to all supported markets.

    Returns:
        List of PlayerProp objects.
    """
    if markets:
        markets_str = ",".join(markets)
    else:
        markets_str = _ALL_MARKETS

    resp = _api_get(
        f"/sports/{_SPORT}/events/{game_id}/odds",
        params={
            "regions": "us",
            "markets": markets_str,
            "oddsFormat": "american",
        },
    )
    data = resp.json()

    return _parse_props_response(
        data,
        game_id=data.get("id", game_id),
        home_team=data.get("home_team", ""),
        away_team=data.get("away_team", ""),
        game_time=data.get("commence_time", ""),
    )


def fetch_all_todays_props(
    stat_types: List[str] = None,
) -> List[PlayerProp]:
    """Fetch props for all today's games.

    Args:
        stat_types: Optional list of canonical stat types to filter
                    (e.g. ['PTS', 'REB']). Fetches all if None.

    Returns:
        Combined list of all PlayerProp objects.
    """
    games = fetch_nba_games()
    if not games:
        print("  No NBA games found today.")
        return []

    print(f"  Found {len(games)} games")

    # Convert stat_types to API market names
    api_markets = None
    if stat_types:
        api_markets = []
        for st in stat_types:
            api_name = STAT_TO_API.get(st.upper())
            if api_name:
                api_markets.append(api_name)
        if not api_markets:
            raise ValueError(
                f"No valid stat types in {stat_types}. "
                f"Valid: {list(STAT_TO_API.keys())}"
            )

    all_props: List[PlayerProp] = []

    for i, game in enumerate(games, 1):
        away = game["away_team"].split()[-1]  # "Los Angeles Clippers" -> "Clippers"
        home = game["home_team"].split()[-1]
        print(f"  Fetching props for game {i}/{len(games)}: {away} @ {home}...")

        try:
            props = fetch_player_props(game["id"], markets=api_markets)
            all_props.extend(props)
        except OddsAPIError as e:
            print(f"    Warning: {e}")
            continue

        # Rate-limit pause between games
        if i < len(games):
            time.sleep(0.5)

    print(f"  Found {len(all_props)} total props")
    return all_props


# ---------------------------------------------------------------------------
# Props caching
# ---------------------------------------------------------------------------

def cache_props(props: List[PlayerProp]) -> None:
    """Store props in the cache database, upserting on conflict."""
    if not props:
        return

    conn = _get_cache_conn()
    try:
        for p in props:
            game_time_str = (
                p.game_time.isoformat()
                if isinstance(p.game_time, datetime)
                else str(p.game_time)
            )
            conn.execute("""
                INSERT INTO props_cache
                    (player_name, stat_type, line, over_odds, under_odds,
                     bookmaker, game_id, home_team, away_team,
                     game_time, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_name, stat_type, line, bookmaker, game_id)
                DO UPDATE SET
                    over_odds = excluded.over_odds,
                    under_odds = excluded.under_odds,
                    game_time = excluded.game_time,
                    fetched_at = excluded.fetched_at
            """, (
                p.player_name, p.stat_type, p.line,
                p.over_odds, p.under_odds,
                p.bookmaker, p.game_id,
                p.home_team, p.away_team,
                game_time_str, p.fetched_at.isoformat(),
            ))
        conn.commit()
    finally:
        conn.close()


def get_cached_props(max_age_minutes: int = _CACHE_MINUTES_PROPS) -> List[PlayerProp]:
    """Return cached props if within max_age."""
    cutoff = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()

    conn = _get_cache_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM props_cache WHERE fetched_at > ? ORDER BY player_name",
            (cutoff,),
        ).fetchall()

        props = []
        for r in rows:
            try:
                gt = datetime.fromisoformat(r["game_time"])
            except (ValueError, TypeError):
                gt = datetime.now()
            try:
                fa = datetime.fromisoformat(r["fetched_at"])
            except (ValueError, TypeError):
                fa = datetime.now()

            props.append(PlayerProp(
                player_name=r["player_name"],
                stat_type=r["stat_type"],
                line=r["line"],
                over_odds=r["over_odds"],
                under_odds=r["under_odds"],
                bookmaker=r["bookmaker"],
                game_id=r["game_id"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                game_time=gt,
                fetched_at=fa,
            ))
        return props
    finally:
        conn.close()


def is_cache_fresh(max_age_minutes: int = _CACHE_MINUTES_PROPS) -> bool:
    """Check if cache has fresh data."""
    cutoff = (datetime.now() - timedelta(minutes=max_age_minutes)).isoformat()
    conn = _get_cache_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM props_cache WHERE fetched_at > ?",
            (cutoff,),
        ).fetchone()
        return row[0] > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Manual entry
# ---------------------------------------------------------------------------

def add_manual_prop(
    player_name: str,
    stat_type: str,
    line: float,
    over_odds: int,
    under_odds: int,
    bookmaker: str = "manual",
    game_id: str = "manual",
    home_team: str = "",
    away_team: str = "",
) -> PlayerProp:
    """Create and cache a manually entered prop.

    Useful when the API is rate limited or for verification.
    """
    prop = PlayerProp(
        player_name=player_name,
        stat_type=stat_type.upper(),
        line=float(line),
        over_odds=int(over_odds),
        under_odds=int(under_odds),
        bookmaker=bookmaker,
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        game_time=datetime.now(),
        fetched_at=datetime.now(),
    )
    cache_props([prop])
    return prop


def update_prop_odds(
    player_name: str,
    stat_type: str,
    line: float,
    over_odds: int = None,
    under_odds: int = None,
) -> bool:
    """Update odds for an existing cached prop.

    Returns True if a row was updated, False if no matching prop found.
    """
    conn = _get_cache_conn()
    try:
        updates = []
        params: list = []

        if over_odds is not None:
            updates.append("over_odds = ?")
            params.append(over_odds)
        if under_odds is not None:
            updates.append("under_odds = ?")
            params.append(under_odds)

        if not updates:
            return False

        updates.append("fetched_at = ?")
        params.append(datetime.now().isoformat())

        params.extend([player_name, stat_type.upper(), line])

        cursor = conn.execute(
            f"UPDATE props_cache SET {', '.join(updates)} "
            "WHERE player_name = ? AND stat_type = ? AND line = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Integration with model
# ---------------------------------------------------------------------------

def analyze_prop(prop: PlayerProp,
                 refresh_stats: bool = False) -> Optional[PropAnalysis]:
    """Analyze a single prop: stats -> probability -> edge.

    Returns PropAnalysis or None if the player can't be found in stats DB.
    """
    # Fetch player stats
    try:
        stats = player_stats_db.get_player_stats(
            prop.player_name, refresh=refresh_stats
        )
    except player_stats_db.PlayerNotFoundError:
        return None  # skip unknown players

    # Map stat type to the column name in player_stats_db
    col_name = _STAT_TO_DB_COL.get(prop.stat_type)
    if col_name is None:
        # Combo stat (PRA, PR, PA) -- not directly in DB
        return None

    # Get averages
    season_avg = stats.season_avg.get(col_name, 0.0)
    last5_avg = stats.last5_avg.get(col_name, 0.0)
    effective_avg = stats.effective_avg.get(col_name, 0.0)

    # Build game log values
    game_log_values = [
        float(g[col_name]) for g in stats.game_log
        if g.get(col_name) is not None
    ]

    # Calculate probability
    prob_result = probability_model.analyze_prop(
        player=prop.player_name,
        stat_type=prop.stat_type,
        line=prop.line,
        season_avg=season_avg,
        recent_avg=last5_avg,
        game_log=game_log_values if len(game_log_values) >= 5 else None,
    )

    # Calculate edge for OVER
    over_bet = edge_cal.analyze_bet(prob_result.prob_over, prop.over_odds)

    # Calculate edge for UNDER
    under_bet = edge_cal.analyze_bet(prob_result.prob_under, prop.under_odds)

    # Calculate edge for OVER only (user can only bet over)
    best_side = "over" if over_bet.ev > 0 else "none"
    best_edge = over_bet.edge
    best_ev = over_bet.ev
    best_kelly = over_bet.kelly_fraction if over_bet.ev > 0 else 0.0
    best_fair = over_bet.fair_odds

    return PropAnalysis(
        prop=prop,
        season_avg=season_avg,
        last5_avg=last5_avg,
        effective_avg=effective_avg,
        games_played=stats.games_played,
        prob_over=prob_result.prob_over,
        prob_under=prob_result.prob_under,
        distribution_used=prob_result.distribution_used,
        over_edge=over_bet.edge,
        under_edge=under_bet.edge,
        over_ev=over_bet.ev,
        under_ev=under_bet.ev,
        best_side=best_side,
        best_edge=best_edge,
        best_ev=best_ev,
        is_plus_ev=best_ev > 0,
        kelly_fraction=best_kelly,
        fair_odds=best_fair,
    )


def analyze_all_props(
    props: List[PlayerProp] = None,
    min_edge: float = 0.0,
    refresh_stats: bool = False,
) -> List[PropAnalysis]:
    """Analyze a list of props. Uses cached props if none provided.

    Args:
        props: List of PlayerProp to analyze. Uses cache if None.
        min_edge: Minimum edge to include in results.
        refresh_stats: Force refresh of player stats.

    Returns:
        List of PropAnalysis sorted by best_ev descending.
    """
    if props is None:
        props = get_cached_props()
    if not props:
        raise NoPropsAvailableError("No props available to analyze.")

    results: List[PropAnalysis] = []
    skipped = 0

    for i, prop in enumerate(props, 1):
        if i % 50 == 0 or i == len(props):
            print(f"  Analyzing prop {i}/{len(props)}...")

        try:
            analysis = analyze_prop(prop, refresh_stats=refresh_stats)
        except Exception as e:
            skipped += 1
            continue

        if analysis is None:
            skipped += 1
            continue

        if analysis.best_edge >= min_edge:
            results.append(analysis)

    if skipped:
        print(f"  Skipped {skipped} props (player not found or unsupported stat)")

    results.sort(key=lambda a: a.best_ev, reverse=True)
    return results


def find_plus_ev_props(
    min_edge: float = 0.02,
    min_prob: float = 0.55,
) -> List[PropAnalysis]:
    """Fetch fresh props, analyze, and return only +EV opportunities.

    Args:
        min_edge: Minimum edge (default 2%).
        min_prob: Minimum probability on best side (avoid longshots).

    Returns:
        List of +EV PropAnalysis sorted by EV descending.
    """
    # Fetch fresh if cache is stale
    if not is_cache_fresh():
        print("\n  Fetching fresh props from API...")
        props = fetch_all_todays_props()
        cache_props(props)
    else:
        props = get_cached_props()
        print(f"\n  Using cached props ({len(props)} props)")

    if not props:
        raise NoPropsAvailableError("No props available.")

    print("\n  Analyzing props...")
    all_analyses = analyze_all_props(props, min_edge=0.0)

    # Filter to +EV with minimum probability
    plus_ev = []
    for a in all_analyses:
        if not a.is_plus_ev or a.best_edge < min_edge:
            continue
        # Check probability threshold on the best side
        prob = a.prob_over if a.best_side == "over" else a.prob_under
        if prob >= min_prob:
            plus_ev.append(a)

    plus_ev.sort(key=lambda a: a.best_ev, reverse=True)
    return plus_ev


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_top_props(analyses: List[PropAnalysis], n: int = 20) -> None:
    """Print formatted table of top N props by EV."""
    if not analyses:
        print("  No props to display.")
        return

    top = analyses[:n]

    # Header
    print()
    print(f"  {'Player':<22} {'Stat':<5} {'Line':>5} {'Side':<5} "
          f"{'Odds':>5} {'Prob':>6} {'Edge':>6} {'EV':>7} {'Fair':>5}")
    print("  " + "-" * 75)

    for a in top:
        side = a.best_side.upper()
        odds = a.prop.over_odds if a.best_side == "over" else a.prop.under_odds
        prob = a.prob_over if a.best_side == "over" else a.prob_under

        odds_str = f"{odds:+d}" if a.best_side != "none" else "---"
        prob_str = f"{prob:.1%}"
        edge_str = f"{a.best_edge:+.1%}"
        ev_str = f"${a.best_ev:+.2f}"
        fair_str = f"{a.fair_odds:+d}"
        name = a.prop.player_name[:21]

        print(f"  {name:<22} {a.prop.stat_type:<5} {a.prop.line:>5.1f} {side:<5} "
              f"{odds_str:>5} {prob_str:>6} {edge_str:>6} {ev_str:>7} {fair_str:>5}")

    print()


def generate_report(analyses: List[PropAnalysis] = None) -> str:
    """Generate a markdown report of +EV opportunities.

    Saves to ev_report_YYYY-MM-DD.md and returns the content.
    """
    if analyses is None:
        analyses = []

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%I:%M %p")

    plus_ev = [a for a in analyses if a.is_plus_ev]
    plus_ev.sort(key=lambda a: a.best_ev, reverse=True)

    lines = [
        f"# NBA Props EV Report - {date_str}",
        f"Generated at {time_str}",
        "",
        "## Summary",
        f"- Total props analyzed: {len(analyses)}",
        f"- +EV opportunities: {len(plus_ev)}",
        "",
    ]

    if plus_ev:
        # Data freshness
        fetch_times = [a.prop.fetched_at for a in plus_ev]
        oldest = min(fetch_times)
        age = now - oldest
        lines.append(f"- Data freshness: {int(age.total_seconds() / 60)} minutes old")
        lines.append("")

        lines.append("## Top Plays")
        lines.append("")
        lines.append("| Player | Stat | Line | Side | Odds | Prob | Edge | EV | Kelly |")
        lines.append("|--------|------|------|------|------|------|------|-----|-------|")

        for a in plus_ev:
            side = a.best_side.upper()
            odds = a.prop.over_odds if a.best_side == "over" else a.prop.under_odds
            prob = a.prob_over if a.best_side == "over" else a.prob_under

            lines.append(
                f"| {a.prop.player_name} | {a.prop.stat_type} | {a.prop.line} | "
                f"{side} | {odds:+d} | {prob:.1%} | {a.best_edge:+.1%} | "
                f"${a.best_ev:+.2f} | {a.kelly_fraction:.1%} |"
            )

        lines.append("")
        lines.append("## Notes")
        lines.append("- EV is per $10 stake")
        lines.append("- Edge = model probability - book implied probability")
        lines.append("- Kelly = recommended fraction of bankroll (full Kelly)")
        lines.append("- Always verify lines haven't moved before placing bets")
    else:
        lines.append("No +EV opportunities found.")

    content = "\n".join(lines) + "\n"

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, f"ev_report_{date_str}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Report saved to: {filepath}")

    return content


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def daily_scan(min_edge: float = 0.02) -> List[PropAnalysis]:
    """Main function for daily use.

    Checks cache freshness, fetches if needed, analyzes all props,
    returns +EV opportunities.
    """
    print("\n=== NBA Props EV Scanner ===")

    # Check usage
    try:
        usage = check_usage()
        print(f"  API Usage: {usage['requests_remaining']}/500 remaining this month")
    except OddsAPIError:
        print("  Could not check API usage")

    # Get props
    if is_cache_fresh():
        props = get_cached_props()
        print(f"\n  Using cached props ({len(props)} props)")
    else:
        print("\n  Fetching today's games...")
        props = fetch_all_todays_props()
        if props:
            cache_props(props)

    if not props:
        print("  No props available.")
        return []

    # Analyze
    print(f"\n  Analyzing {len(props)} props...")
    all_analyses = analyze_all_props(props, min_edge=0.0)

    # Filter to +EV
    plus_ev = [a for a in all_analyses if a.is_plus_ev and a.best_edge >= min_edge]
    plus_ev.sort(key=lambda a: a.best_ev, reverse=True)

    # Print results
    if plus_ev:
        print(f"\n=== TOP +EV OPPORTUNITIES ===")
        print_top_props(plus_ev, n=20)
        print(f"  Found {len(plus_ev)} +EV props (edge > {min_edge:.0%})")

        # Save report
        generate_report(all_analyses)
    else:
        print(f"\n  No +EV props found with edge > {min_edge:.0%}")

    return plus_ev


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plus_ev = daily_scan(min_edge=0.02)
