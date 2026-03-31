"""
player_stats_db.py -- NBA player stats fetcher with SQLite caching.

Bridges raw NBA data to probability_model.py and edge_cal.py.
Fetches player stats from nba_api, caches in SQLite to avoid repeated
API calls, and provides clean interfaces for prop analysis.

Dependencies: nba_api, sqlite3, probability_model, edge_cal
"""

from __future__ import annotations

import os
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

import probability_model
import edge_cal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nba_stats.db")

_CACHE_HOURS_SEASON = 12   # season stats cache TTL
_CACHE_HOURS_GAMELOG = 12   # game log cache TTL
_CACHE_HOURS_PLAYER = 6    # full PlayerStats staleness threshold

# Stat name mapping -- normalise all the ways a stat can be referenced
STAT_MAP: Dict[str, str] = {
    "points": "PTS",
    "pts": "PTS",
    "rebounds": "REB",
    "reb": "REB",
    "assists": "AST",
    "ast": "AST",
    "three_pm": "3PM",
    "3pm": "3PM",
    "threes": "3PM",
    "steals": "STL",
    "stl": "STL",
    "blocks": "BLK",
    "blk": "BLK",
    "turnovers": "TOV",
    "tov": "TOV",
    "minutes": "MIN",
    "min": "MIN",
}

# Maps canonical stat key -> game_logs column name
_STAT_TO_COL: Dict[str, str] = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "3PM": "three_pm",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "MIN": "minutes",
}

# Maps nba_api header -> game_logs column
_API_HEADER_MAP: Dict[str, str] = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "FG3M": "three_pm",
    "STL": "steals",
    "BLK": "blocks",
    "TOV": "turnovers",
    "MIN": "minutes",
    "PLUS_MINUS": "plus_minus",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PlayerNotFoundError(Exception):
    """Raised when a player name cannot be resolved."""
    pass


class StatsNotAvailableError(Exception):
    """Raised when stats are not available for a player/season."""
    pass


class APIError(Exception):
    """Raised when the nba_api call fails after retries."""
    pass


# ---------------------------------------------------------------------------
# Helpers (reused patterns from main.py / team_history.py)
# ---------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Strip accents and lowercase for accent-insensitive comparison."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def safe_print(s: str):
    """Print with non-ASCII characters replaced so cp1252 never crashes."""
    print(s.encode("ascii", "replace").decode())


def current_nba_season() -> str:
    """Return the current NBA season string, e.g. '2025-26'."""
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def _normalize_stat(stat: str) -> str:
    """Resolve a stat name to its canonical form (e.g. 'points' -> 'PTS')."""
    key = stat.strip().lower()
    if key in STAT_MAP:
        return STAT_MAP[key]
    # Already canonical?
    if stat.upper() in _STAT_TO_COL:
        return stat.upper()
    raise ValueError(f"Unknown stat type: '{stat}'. Valid: {list(STAT_MAP.keys())}")


def _api_call_with_retry(fn, *args, retries: int = 3, **kwargs):
    """Call an nba_api endpoint with retry + rate-limit sleep."""
    last_err = None
    for attempt in range(retries):
        try:
            time.sleep(0.6 + attempt * 0.4)  # 0.6s, 1.0s, 1.4s backoff
            return fn(*args, **kwargs).get_dict()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s exponential backoff
    raise APIError(f"API call failed after {retries} attempts: {last_err}")


def _parse_game_date(raw: str) -> str:
    """Parse NBA API date string (e.g. 'MAR 15, 2025') to 'YYYY-MM-DD'."""
    try:
        dt = datetime.strptime(raw.strip().title(), "%b %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return raw


def _parse_matchup(matchup: str) -> Tuple[str, str]:
    """Parse matchup string -> (opponent_abbrev, home_or_away).

    'CHA vs. LAL' -> ('LAL', 'home')
    'CHA @ LAL'   -> ('LAL', 'away')
    """
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        return parts[1].strip(), "home"
    elif " @ " in matchup:
        parts = matchup.split(" @ ")
        return parts[1].strip(), "away"
    return "UNK", "unknown"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create database tables if they don't exist."""
    conn = get_db_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                player_id   INTEGER PRIMARY KEY,
                full_name   TEXT NOT NULL,
                team_abbreviation TEXT,
                position    TEXT,
                is_active   BOOLEAN DEFAULT 1,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS season_stats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id   INTEGER NOT NULL,
                season      TEXT NOT NULL,
                games_played INTEGER,
                minutes     REAL,
                points      REAL,
                rebounds    REAL,
                assists     REAL,
                steals      REAL,
                blocks      REAL,
                three_pm    REAL,
                turnovers   REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_id, season),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );

            CREATE TABLE IF NOT EXISTS game_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id   INTEGER NOT NULL,
                game_id     TEXT NOT NULL,
                game_date   DATE,
                opponent    TEXT,
                home_away   TEXT,
                minutes     REAL,
                points      INTEGER,
                rebounds    INTEGER,
                assists     INTEGER,
                steals      INTEGER,
                blocks      INTEGER,
                three_pm    INTEGER,
                turnovers   INTEGER,
                plus_minus  INTEGER,
                UNIQUE(player_id, game_id),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );

            CREATE INDEX IF NOT EXISTS idx_game_logs_player_date
                ON game_logs(player_id, game_date DESC);

            CREATE INDEX IF NOT EXISTS idx_season_stats_player_season
                ON season_stats(player_id, season);
        """)
        conn.commit()
    finally:
        conn.close()


# Auto-init on import
init_db()


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def find_player(name: str) -> Optional[Dict]:
    """Fuzzy search for a player by name (case/accent insensitive).

    Returns dict with player_id, full_name, team, position or None.
    """
    all_players = players.get_players()
    norm_input = normalize(name.strip())

    # Exact match
    for p in all_players:
        if normalize(p["full_name"]) == norm_input:
            return {
                "player_id": p["id"],
                "full_name": p["full_name"],
                "team": "",  # nba_api static doesn't include team
                "position": "",
                "is_active": p.get("is_active", True),
            }

    # Partial match -- all input words must appear in player name
    input_words = norm_input.split()
    matches = []
    for p in all_players:
        pname = normalize(p["full_name"])
        if all(w in pname for w in input_words):
            matches.append(p)

    # Single partial match -> return it
    if len(matches) == 1:
        p = matches[0]
        return {
            "player_id": p["id"],
            "full_name": p["full_name"],
            "team": "",
            "position": "",
            "is_active": p.get("is_active", True),
        }

    # Multiple matches -> return None (ambiguous)
    return None


def find_player_id(name: str) -> int:
    """Return player_id for a name, raising PlayerNotFoundError if not found."""
    result = find_player(name)
    if result is None:
        # Build suggestion list for error message
        all_players = players.get_players()
        norm_input = normalize(name.strip())
        input_words = norm_input.split()
        suggestions = [
            p["full_name"] for p in all_players
            if any(w in normalize(p["full_name"]) for w in input_words)
        ]
        msg = f"Player '{name}' not found."
        if suggestions:
            top = sorted(suggestions)[:6]
            msg += " Did you mean: " + ", ".join(top) + "?"
        raise PlayerNotFoundError(msg)
    return result["player_id"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _is_cache_fresh(last_updated_str: Optional[str], max_hours: int) -> bool:
    """Check if a cached timestamp is within max_hours of now."""
    if last_updated_str is None:
        return False
    try:
        last_updated = datetime.fromisoformat(last_updated_str)
    except (ValueError, TypeError):
        return False
    return datetime.now() - last_updated < timedelta(hours=max_hours)


def _cache_player(conn: sqlite3.Connection, player_info: Dict) -> None:
    """Insert or update a player record in the cache."""
    conn.execute("""
        INSERT INTO players (player_id, full_name, team_abbreviation, position, is_active, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            full_name = excluded.full_name,
            team_abbreviation = excluded.team_abbreviation,
            position = excluded.position,
            is_active = excluded.is_active,
            last_updated = excluded.last_updated
    """, (
        player_info["player_id"],
        player_info["full_name"],
        player_info.get("team", ""),
        player_info.get("position", ""),
        player_info.get("is_active", True),
        datetime.now().isoformat(),
    ))
    conn.commit()


# ---------------------------------------------------------------------------
# Stats fetching (with caching)
# ---------------------------------------------------------------------------

def _fetch_game_log_from_api(player_id: int, season: str) -> Tuple[List[str], List[list]]:
    """Fetch game log from nba_api with retry logic. Returns (headers, rows)."""
    data = _api_call_with_retry(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
    )
    rs = data["resultSets"][0]
    return rs["headers"], rs["rowSet"]


def _store_game_logs(conn: sqlite3.Connection, player_id: int,
                     headers: List[str], rows: List[list]) -> None:
    """Store game log rows into the database."""
    col = {h: i for i, h in enumerate(headers)}

    for row in rows:
        matchup = row[col["MATCHUP"]]
        opponent, home_away = _parse_matchup(matchup)
        game_date = _parse_game_date(row[col["GAME_DATE"]])

        # Parse minutes -- can be float or "MM:SS" string
        raw_min = row[col["MIN"]]
        try:
            minutes = float(raw_min) if raw_min is not None else 0.0
        except (ValueError, TypeError):
            minutes = 0.0

        conn.execute("""
            INSERT INTO game_logs
                (player_id, game_id, game_date, opponent, home_away,
                 minutes, points, rebounds, assists, steals, blocks,
                 three_pm, turnovers, plus_minus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(player_id, game_id) DO UPDATE SET
                game_date = excluded.game_date,
                opponent = excluded.opponent,
                home_away = excluded.home_away,
                minutes = excluded.minutes,
                points = excluded.points,
                rebounds = excluded.rebounds,
                assists = excluded.assists,
                steals = excluded.steals,
                blocks = excluded.blocks,
                three_pm = excluded.three_pm,
                turnovers = excluded.turnovers,
                plus_minus = excluded.plus_minus
        """, (
            player_id,
            row[col["Game_ID"]],
            game_date,
            opponent,
            home_away,
            minutes,
            row[col["PTS"]] or 0,
            row[col["REB"]] or 0,
            row[col["AST"]] or 0,
            row[col["STL"]] or 0,
            row[col["BLK"]] or 0,
            row[col["FG3M"]] or 0,
            row[col["TOV"]] or 0,
            row[col["PLUS_MINUS"]] or 0,
        ))
    conn.commit()


def _store_season_stats(conn: sqlite3.Connection, player_id: int,
                        season: str, stats: Dict) -> None:
    """Store computed season averages into the cache."""
    conn.execute("""
        INSERT INTO season_stats
            (player_id, season, games_played, minutes, points, rebounds,
             assists, steals, blocks, three_pm, turnovers, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, season) DO UPDATE SET
            games_played = excluded.games_played,
            minutes = excluded.minutes,
            points = excluded.points,
            rebounds = excluded.rebounds,
            assists = excluded.assists,
            steals = excluded.steals,
            blocks = excluded.blocks,
            three_pm = excluded.three_pm,
            turnovers = excluded.turnovers,
            last_updated = excluded.last_updated
    """, (
        player_id, season,
        stats["games_played"],
        stats["minutes"],
        stats["points"],
        stats["rebounds"],
        stats["assists"],
        stats["steals"],
        stats["blocks"],
        stats["three_pm"],
        stats["turnovers"],
        datetime.now().isoformat(),
    ))
    conn.commit()


def fetch_season_stats(player_id: int, season: str = None) -> Dict:
    """Fetch season averages for a player, using cache if fresh.

    Returns dict: {points, rebounds, assists, steals, blocks,
                   three_pm, minutes, turnovers, games_played}
    """
    if season is None:
        season = current_nba_season()

    conn = get_db_connection()
    try:
        # Check cache
        row = conn.execute(
            "SELECT * FROM season_stats WHERE player_id = ? AND season = ?",
            (player_id, season)
        ).fetchone()

        if row and _is_cache_fresh(row["last_updated"], _CACHE_HOURS_SEASON):
            return {
                "points": row["points"],
                "rebounds": row["rebounds"],
                "assists": row["assists"],
                "steals": row["steals"],
                "blocks": row["blocks"],
                "three_pm": row["three_pm"],
                "minutes": row["minutes"],
                "turnovers": row["turnovers"],
                "games_played": row["games_played"],
            }

        # Fetch from API
        headers, rows = _fetch_game_log_from_api(player_id, season)
        if not rows:
            raise StatsNotAvailableError(
                f"No game log found for player {player_id} in {season}"
            )

        # Store game logs
        _store_game_logs(conn, player_id, headers, rows)

        # Compute averages
        col = {h: i for i, h in enumerate(headers)}
        gp = len(rows)

        def avg(stat_key):
            vals = [r[col[stat_key]] for r in rows if r[col[stat_key]] is not None]
            return round(sum(vals) / len(vals), 1) if vals else 0.0

        stats = {
            "points": avg("PTS"),
            "rebounds": avg("REB"),
            "assists": avg("AST"),
            "steals": avg("STL"),
            "blocks": avg("BLK"),
            "three_pm": avg("FG3M"),
            "minutes": avg("MIN"),
            "turnovers": avg("TOV"),
            "games_played": gp,
        }

        _store_season_stats(conn, player_id, season, stats)
        return stats

    finally:
        conn.close()


def fetch_game_log(player_id: int, season: str = None,
                   last_n: int = None) -> List[Dict]:
    """Fetch game log for a player, using cache if fresh.

    Returns list of game dicts (most recent first), each with:
    game_date, opponent, home_away, points, rebounds, assists,
    steals, blocks, three_pm, minutes, turnovers, plus_minus
    """
    if season is None:
        season = current_nba_season()

    conn = get_db_connection()
    try:
        # Check if we have cached game logs
        cached = conn.execute("""
            SELECT game_date FROM game_logs
            WHERE player_id = ?
            ORDER BY game_date DESC LIMIT 1
        """, (player_id,)).fetchone()

        need_fetch = True
        if cached:
            # Check if most recent cached game is within cache window
            try:
                last_game = datetime.strptime(cached["game_date"], "%Y-%m-%d")
                # If we fetched recently (within cache hours), use cache
                # We check by looking at whether we have season_stats cached recently
                ss_row = conn.execute(
                    "SELECT last_updated FROM season_stats WHERE player_id = ? AND season = ?",
                    (player_id, season)
                ).fetchone()
                if ss_row and _is_cache_fresh(ss_row["last_updated"], _CACHE_HOURS_GAMELOG):
                    need_fetch = False
            except (ValueError, TypeError):
                pass

        if need_fetch:
            headers, rows = _fetch_game_log_from_api(player_id, season)
            if rows:
                _store_game_logs(conn, player_id, headers, rows)

        # Read from cache
        if last_n is not None:
            db_rows = conn.execute("""
                SELECT game_date, opponent, home_away, minutes, points, rebounds,
                       assists, steals, blocks, three_pm, turnovers, plus_minus
                FROM game_logs
                WHERE player_id = ?
                ORDER BY game_date DESC
                LIMIT ?
            """, (player_id, int(last_n))).fetchall()
        else:
            db_rows = conn.execute("""
                SELECT game_date, opponent, home_away, minutes, points, rebounds,
                       assists, steals, blocks, three_pm, turnovers, plus_minus
                FROM game_logs
                WHERE player_id = ?
                ORDER BY game_date DESC
            """, (player_id,)).fetchall()

        return [dict(row) for row in db_rows]

    finally:
        conn.close()


def get_stat_values(player_id: int, stat: str, last_n: int = 10) -> List[float]:
    """Return list of stat values from last N games.

    This is what you feed into probability_model.calc_actual_cv().

    Args:
        player_id: NBA player ID.
        stat: Stat name (e.g. 'points', 'PTS', 'three_pm', '3PM').
        last_n: Number of recent games to include.

    Example:
        >>> get_stat_values(12345, 'points', 10)
        [22, 18, 25, 20, ...]
    """
    canonical = _normalize_stat(stat)
    col_name = _STAT_TO_COL[canonical]
    games = fetch_game_log(player_id, last_n=last_n)
    return [float(g[col_name]) for g in games if g[col_name] is not None]


# ---------------------------------------------------------------------------
# Computed stats
# ---------------------------------------------------------------------------

def get_last_n_average(player_id: int, stat: str, n: int = 5) -> float:
    """Calculate average of a stat over the last N games.

    Example:
        >>> get_last_n_average(12345, 'points', 5)
        22.4
    """
    values = get_stat_values(player_id, stat, last_n=n)
    if not values:
        raise StatsNotAvailableError(
            f"No game data available for player {player_id}"
        )
    return round(sum(values) / len(values), 2)


def get_effective_average(player_id: int, stat: str,
                          recent_weight: float = 0.6) -> float:
    """Blend season avg with last-5 avg.

    Formula: (1 - recent_weight) * season_avg + recent_weight * last5_avg

    Example:
        >>> get_effective_average(12345, 'points')
        20.4
    """
    canonical = _normalize_stat(stat)
    col_name = _STAT_TO_COL[canonical]

    season = fetch_season_stats(player_id)
    season_avg = season[col_name]
    last5_avg = get_last_n_average(player_id, stat, n=5)

    return round(
        probability_model.calc_weighted_average(season_avg, last5_avg, recent_weight),
        2,
    )


def get_player_cv(player_id: int, stat: str,
                  min_games: int = 10) -> Optional[float]:
    """Calculate actual CV from game log.

    Returns None if fewer than min_games (or < 5, the calc_actual_cv minimum).

    Example:
        >>> get_player_cv(12345, 'points')
        0.32
    """
    effective_min = max(min_games, 5)  # calc_actual_cv needs >= 5
    values = get_stat_values(player_id, stat, last_n=effective_min)

    if len(values) < effective_min:
        return None

    try:
        return round(probability_model.calc_actual_cv(values), 4)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# PlayerStats data class
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    """Complete statistical profile for a player."""

    player_id: int
    player_name: str
    team: str
    position: str
    season: str
    games_played: int

    # Season averages
    season_avg: Dict[str, float]

    # Recent performance
    last5_avg: Dict[str, float]
    last10_avg: Dict[str, float]

    # For probability model
    effective_avg: Dict[str, float]
    actual_cv: Dict[str, Optional[float]]

    # Raw game log
    game_log: List[Dict]

    last_updated: datetime = field(default_factory=datetime.now)

    # Head-to-head vs specific opponent (populated by get_player_stats_with_h2h)
    h2h_avg: Dict[str, float] = field(default_factory=dict)
    h2h_opponent: Optional[str] = None


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

_CORE_STATS = ["points", "rebounds", "assists", "steals", "blocks", "three_pm",
               "minutes", "turnovers"]


def _compute_avg_dict(games: List[Dict], stats: List[str] = None) -> Dict[str, float]:
    """Compute averages from a list of game dicts."""
    if stats is None:
        stats = _CORE_STATS
    result = {}
    for stat in stats:
        vals = [float(g[stat]) for g in games if g.get(stat) is not None]
        result[stat] = round(sum(vals) / len(vals), 2) if vals else 0.0
    return result


def get_player_stats(player_name: str, refresh: bool = False) -> PlayerStats:
    """The main function -- fetch everything for a player.

    Handles player lookup, fetches all stats, computes averages.
    If refresh=True or data is stale (> 6 hours), fetches fresh from API.

    Example:
        >>> stats = get_player_stats("LaMelo Ball")
        >>> print(f"Season: {stats.season_avg['points']} PPG")
    """
    # Resolve player
    player_info = find_player(player_name)
    if player_info is None:
        raise PlayerNotFoundError(f"Player '{player_name}' not found.")

    player_id = player_info["player_id"]
    season = current_nba_season()

    conn = get_db_connection()
    try:
        # Cache the player record
        _cache_player(conn, player_info)

        # Check staleness
        if not refresh:
            ss_row = conn.execute(
                "SELECT last_updated FROM season_stats WHERE player_id = ? AND season = ?",
                (player_id, season)
            ).fetchone()
            if ss_row and not _is_cache_fresh(ss_row["last_updated"], _CACHE_HOURS_PLAYER):
                refresh = True
    finally:
        conn.close()

    # If refresh needed, force fresh API fetch by clearing cache timestamps
    if refresh:
        conn = get_db_connection()
        try:
            # Set last_updated to epoch to force re-fetch
            conn.execute(
                "UPDATE season_stats SET last_updated = '2000-01-01T00:00:00' "
                "WHERE player_id = ? AND season = ?",
                (player_id, season)
            )
            conn.commit()
        finally:
            conn.close()

    # Fetch season stats (handles caching internally)
    season_stats = fetch_season_stats(player_id, season)

    # Fetch game log (last 15 games for the profile)
    game_log = fetch_game_log(player_id, season, last_n=15)

    # Compute averages over different windows
    last5 = game_log[:5] if len(game_log) >= 5 else game_log
    last10 = game_log[:10] if len(game_log) >= 10 else game_log

    last5_avg = _compute_avg_dict(last5)
    last10_avg = _compute_avg_dict(last10)

    # Effective average (weighted blend)
    effective_avg = {}
    for stat in _CORE_STATS:
        s_avg = season_stats.get(stat, 0.0)
        l5_avg = last5_avg.get(stat, 0.0)
        effective_avg[stat] = round(
            probability_model.calc_weighted_average(s_avg, l5_avg, 0.6), 2
        )

    # Actual CV from game log
    actual_cv: Dict[str, Optional[float]] = {}
    for stat in _CORE_STATS:
        vals = [float(g[stat]) for g in game_log if g.get(stat) is not None]
        if len(vals) >= 5:
            try:
                actual_cv[stat] = round(probability_model.calc_actual_cv(vals), 4)
            except ValueError:
                actual_cv[stat] = None
        else:
            actual_cv[stat] = None

    # Look up team from cached game log (opponent field gives us opponents, not team)
    # Try to get team from the players table cache
    team = player_info.get("team", "")
    position = player_info.get("position", "")

    return PlayerStats(
        player_id=player_id,
        player_name=player_info["full_name"],
        team=team,
        position=position,
        season=season,
        games_played=season_stats["games_played"],
        season_avg=season_stats,
        last5_avg=last5_avg,
        last10_avg=last10_avg,
        effective_avg=effective_avg,
        actual_cv=actual_cv,
        game_log=game_log,
        last_updated=datetime.now(),
    )


def analyze_player_prop(
    player_name: str,
    stat_type: str,
    line: float,
    book_odds: int = None,
    stake: float = 10.0,
) -> Dict:
    """All-in-one function: fetch stats -> probability -> edge analysis.

    Fetches player stats, calls probability_model.analyze_prop(),
    and if book_odds provided, calls edge_cal.analyze_bet().

    Args:
        player_name: Player full name.
        stat_type: Stat type (e.g. 'PTS', 'points', '3PM').
        line: The prop line (e.g. 19.5).
        book_odds: Optional American odds from the book.
        stake: Bet stake in dollars (default $10).

    Returns:
        Dict with all analysis fields.

    Example:
        >>> result = analyze_player_prop("LaMelo Ball", "PTS", 19.5, book_odds=-115)
        >>> print(f"P(over): {result['prob_over']:.1%}")
        >>> print(f"Edge: {result['edge']:.2%}")
    """
    # Normalize stat type
    canonical_stat = _normalize_stat(stat_type)

    # Get player stats
    stats = get_player_stats(player_name)

    # Get the right column name for the stat
    col_name = _STAT_TO_COL[canonical_stat]

    # Season and recent averages
    season_avg = stats.season_avg.get(col_name, 0.0)
    last5_avg = stats.last5_avg.get(col_name, 0.0)
    effective_avg = stats.effective_avg.get(col_name, 0.0)

    # Build game log values for probability model
    game_log_values = [
        float(g[col_name]) for g in stats.game_log
        if g.get(col_name) is not None
    ]

    # Run probability analysis
    prop_result = probability_model.analyze_prop(
        player=stats.player_name,
        stat_type=canonical_stat,
        line=line,
        season_avg=season_avg,
        recent_avg=last5_avg,
        game_log=game_log_values if len(game_log_values) >= 5 else None,
    )

    # Build result dict
    result = {
        "player": stats.player_name,
        "stat": canonical_stat,
        "line": line,
        "season_avg": season_avg,
        "last5_avg": last5_avg,
        "last10_avg": stats.last10_avg.get(col_name, 0.0),
        "effective_avg": effective_avg,
        "mean_used": prop_result.mean,
        "std_dev": prop_result.std_dev,
        "cv": prop_result.cv,
        "prob_over": prop_result.prob_over,
        "prob_under": prop_result.prob_under,
        "distribution": prop_result.distribution_used,
        "confidence": prop_result.confidence,
        "sample_size": prop_result.sample_size,
        "games_played": stats.games_played,
    }

    # If book odds provided, run edge analysis
    if book_odds is not None:
        bet = edge_cal.analyze_bet(
            model_prob=prop_result.prob_over,
            book_odds=book_odds,
            stake=stake,
        )
        result.update({
            "book_odds": book_odds,
            "book_implied_prob": bet.book_implied_prob,
            "edge": bet.edge,
            "ev": bet.ev,
            "ev_percent": bet.ev_percent,
            "is_plus_ev": bet.is_plus_ev,
            "kelly": bet.kelly_fraction,
            "half_kelly": bet.half_kelly_fraction,
            "quarter_kelly": bet.quarter_kelly_fraction,
            "fair_odds": bet.fair_odds,
        })

    return result


# ---------------------------------------------------------------------------
# Head-to-head helpers
# ---------------------------------------------------------------------------

# Maps common name variants to a canonical team abbreviation for matching
# against the `opponent` field stored in game_logs (which is already an abbrev).
_TEAM_ABBREV_ALIASES: Dict[str, List[str]] = {
    "ATL": ["atl", "hawks", "atlanta"],
    "BOS": ["bos", "celtics", "boston"],
    "BKN": ["bkn", "nets", "brooklyn"],
    "CHA": ["cha", "hornets", "charlotte"],
    "CHI": ["chi", "bulls", "chicago"],
    "CLE": ["cle", "cavaliers", "cavs", "cleveland"],
    "DAL": ["dal", "mavericks", "mavs", "dallas"],
    "DEN": ["den", "nuggets", "denver"],
    "DET": ["det", "pistons", "detroit"],
    "GSW": ["gsw", "warriors", "golden state"],
    "HOU": ["hou", "rockets", "houston"],
    "IND": ["ind", "pacers", "indiana"],
    "LAC": ["lac", "clippers", "los angeles clippers", "la clippers"],
    "LAL": ["lal", "lakers", "los angeles lakers", "la lakers"],
    "MEM": ["mem", "grizzlies", "memphis"],
    "MIA": ["mia", "heat", "miami"],
    "MIL": ["mil", "bucks", "milwaukee"],
    "MIN": ["min", "timberwolves", "wolves", "minnesota"],
    "NOP": ["nop", "pelicans", "new orleans"],
    "NYK": ["nyk", "knicks", "new york"],
    "OKC": ["okc", "thunder", "oklahoma city"],
    "ORL": ["orl", "magic", "orlando"],
    "PHI": ["phi", "sixers", "76ers", "philadelphia"],
    "PHX": ["phx", "suns", "phoenix"],
    "POR": ["por", "blazers", "trail blazers", "portland"],
    "SAC": ["sac", "kings", "sacramento"],
    "SAS": ["sas", "spurs", "san antonio"],
    "TOR": ["tor", "raptors", "toronto"],
    "UTA": ["uta", "jazz", "utah"],
    "WAS": ["was", "wizards", "washington"],
}

_CORE_STATS_H2H = ["points", "rebounds", "assists", "steals", "blocks", "three_pm", "minutes"]


def _resolve_opponent_abbrev(opponent_team: str) -> Optional[str]:
    """Return the canonical 3-letter abbreviation for an opponent name, or None."""
    needle = opponent_team.strip().lower()
    for abbrev, aliases in _TEAM_ABBREV_ALIASES.items():
        if needle == abbrev.lower() or needle in aliases:
            return abbrev
        # Partial substring match as fallback
        if any(needle in alias for alias in aliases):
            return abbrev
    return None


def get_vs_opponent_stats(player_name: str, opponent_team: str,
                          last_n: int = 5) -> Dict[str, float]:
    """Get a player's average stats against a specific opponent from their game log.

    Args:
        player_name: Player name (partial match supported).
        opponent_team: Opponent team name, abbreviation, or city/nickname
                       (e.g. "Bucks", "MIL", "Milwaukee").
        last_n: Maximum number of recent games vs this opponent to use.

    Returns:
        Dict of stat averages including 'games_vs' count, or empty dict if no data.
    """
    try:
        stats = get_player_stats(player_name)
    except (PlayerNotFoundError, StatsNotAvailableError):
        return {}

    if not stats.game_log:
        return {}

    # Resolve opponent to a canonical abbreviation
    abbrev = _resolve_opponent_abbrev(opponent_team)

    vs_games = []
    for game in stats.game_log:
        game_opp = game.get("opponent", "").upper()
        if abbrev:
            match = game_opp == abbrev
        else:
            # No known abbreviation — fall back to substring match on whatever was passed
            match = opponent_team.strip().upper() in game_opp
        if match:
            vs_games.append(game)
        if len(vs_games) >= last_n:
            break

    if not vs_games:
        return {}

    totals: Dict[str, float] = {s: 0.0 for s in _CORE_STATS_H2H}
    for game in vs_games:
        for stat in _CORE_STATS_H2H:
            try:
                totals[stat] += float(game.get(stat) or 0)
            except (ValueError, TypeError):
                pass

    n = len(vs_games)
    averages: Dict[str, float] = {stat: round(total / n, 1) for stat, total in totals.items()}
    averages["games_vs"] = n
    return averages


def get_player_stats_with_h2h(player_name: str, opponent: str = None) -> "PlayerStats":
    """Fetch player stats, optionally enriched with head-to-head averages vs opponent."""
    stats = get_player_stats(player_name)
    if opponent:
        stats.h2h_avg = get_vs_opponent_stats(player_name, opponent)
        stats.h2h_opponent = opponent
    return stats


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  player_stats_db.py -- demo")
    print("=" * 60)

    print("\n--- Testing player lookup ---")
    try:
        info = find_player("LaMelo Ball")
        if info:
            safe_print(f"  Found: {info['full_name']} (ID: {info['player_id']})")
        else:
            print("  Player not found")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- Fetching stats ---")
    try:
        stats = get_player_stats("LaMelo Ball")
        safe_print(f"  Player: {stats.player_name}")
        print(f"  Season: {stats.season}")
        print(f"  Games: {stats.games_played}")
        print(f"  Season PPG: {stats.season_avg.get('points', 'N/A')}")
        print(f"  Last 5 PPG: {stats.last5_avg.get('points', 'N/A')}")
        print(f"  Effective PPG: {stats.effective_avg.get('points', 'N/A')}")
        print(f"  CV (points): {stats.actual_cv.get('points', 'N/A')}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- Full prop analysis ---")
    try:
        result = analyze_player_prop("LaMelo Ball", "PTS", 19.5, book_odds=-115)
        print(f"  P(over): {result['prob_over']:.1%}")
        print(f"  P(under): {result['prob_under']:.1%}")
        print(f"  Distribution: {result['distribution']}")
        if "edge" in result:
            print(f"  Edge: {result['edge']:.2%}")
            print(f"  EV: ${result['ev']:.2f}")
            print(f"  +EV? {result['is_plus_ev']}")
            print(f"  Kelly: {result['kelly']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
