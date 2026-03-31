"""
Team History – show a team's last N games with full box-score stats for every
player who appeared in each game.

Standalone script: python team_history.py
"""

import time
import unicodedata
from datetime import date
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster,
    playergamelog,
    boxscoretraditionalv3,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def safe_print(s: str):
    print(s.encode("ascii", "replace").decode())


def current_nba_season() -> str:
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def resolve_team(name: str) -> dict:
    """Return the full team dict from any common input (city, nickname, abbrev, full name)."""
    all_teams = teams.get_teams()
    norm = normalize(name.strip())

    for t in all_teams:
        if norm in (normalize(t["full_name"]), normalize(t["nickname"]),
                    normalize(t["abbreviation"]), normalize(t["city"])):
            return t

    # Partial match fallback
    matches = [
        t for t in all_teams
        if norm in normalize(t["full_name"]) or norm in normalize(t["nickname"])
    ]
    if len(matches) == 1:
        return matches[0]

    safe_print(f"\nERROR: Team '{name}' not found.")
    if matches:
        safe_print("  Did you mean one of these?")
        for t in matches[:6]:
            safe_print(f"    {t['full_name']} ({t['abbreviation']})")
    raise SystemExit(1)


def api_call_with_retry(fn, *args, retries=3, **kwargs):
    """Call an nba_api endpoint with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(0.6)
            return fn(*args, **kwargs).get_dict()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  (Retry {attempt + 1}: {e})")
                time.sleep(2)
            else:
                raise


# ── Core logic ───────────────────────────────────────────────────────────────

def get_recent_game_ids(team_id: int, season: str, n: int) -> list[str]:
    """
    Get the team's last N game IDs by fetching a roster player's game log.
    """
    # Grab roster
    data = api_call_with_retry(commonteamroster.CommonTeamRoster,
                               team_id=team_id, season=season)
    rs = data["resultSets"][0]
    rcol = {h: i for i, h in enumerate(rs["headers"])}
    roster_rows = rs["rowSet"]

    if not roster_rows:
        print("  ERROR: Empty roster returned.")
        raise SystemExit(1)

    # Pick first player on roster and get their game log
    player_id = roster_rows[0][rcol["PLAYER_ID"]]
    gl_data = api_call_with_retry(playergamelog.PlayerGameLog,
                                  player_id=player_id, season=season)
    gl_rs = gl_data["resultSets"][0]
    gl_col = {h: i for i, h in enumerate(gl_rs["headers"])}
    gl_rows = gl_rs["rowSet"]

    # Extract unique game IDs (most recent first)
    seen = set()
    game_ids = []
    for r in gl_rows:
        gid = r[gl_col["Game_ID"]]
        if gid not in seen:
            seen.add(gid)
            game_ids.append(gid)
        if len(game_ids) >= n:
            break

    if not game_ids:
        # If the first player has no games, try more roster players
        for row in roster_rows[1:6]:
            pid = row[rcol["PLAYER_ID"]]
            gl2 = api_call_with_retry(playergamelog.PlayerGameLog,
                                      player_id=pid, season=season)
            gl2_rs = gl2["resultSets"][0]
            gl2_col = {h: i for i, h in enumerate(gl2_rs["headers"])}
            for r in gl2_rs["rowSet"]:
                gid = r[gl2_col["Game_ID"]]
                if gid not in seen:
                    seen.add(gid)
                    game_ids.append(gid)
            if game_ids:
                break

    return game_ids[:n]


def fetch_box_score(game_id: str, team_id: int) -> dict | None:
    """
    Fetch box score for a game via boxscoretraditionalv3.
    Returns a dict with game info and player rows, or None on failure.
    """
    try:
        data = api_call_with_retry(boxscoretraditionalv3.BoxScoreTraditionalV3,
                                   game_id=game_id)
    except Exception as e:
        print(f"  WARNING: Could not fetch box score for {game_id}: {e}")
        return None

    box = data.get("boxScoreTraditional", data.get("boxScore", None))
    if box is None:
        # Try alternate structure
        for key in data:
            if isinstance(data[key], dict):
                box = data[key]
                break
    if box is None:
        print(f"  WARNING: Unexpected response structure for {game_id}")
        return None

    # Game metadata
    game_id_str = box.get("gameId", game_id)
    home_team = box.get("homeTeam", {})
    away_team = box.get("awayTeam", {})

    # Determine which side is ours
    home_id = home_team.get("teamId", 0)
    away_id = away_team.get("teamId", 0)

    if home_id == team_id:
        our_team = home_team
        opp_team = away_team
        location = "vs"
    else:
        our_team = away_team
        opp_team = home_team
        location = "@"

    our_score = our_team.get("score", 0)
    opp_score = opp_team.get("score", 0)
    result = "W" if our_score > opp_score else "L"
    opp_abbrev = opp_team.get("teamTricode", opp_team.get("teamCity", "???"))

    game_date = box.get("gameTimeLocal", box.get("gameEt", ""))
    if game_date and "T" in game_date:
        game_date = game_date.split("T")[0]

    # Player stats
    players_raw = our_team.get("players", [])
    player_stats = []
    for p in players_raw:
        stats = p.get("statistics", {})
        minutes_str = stats.get("minutes", "")
        # e.g. "PT34M12.00S" or "34:12" — parse to readable
        minutes_display = _parse_minutes(minutes_str)
        if minutes_display == "0:00" or minutes_display == "":
            continue  # DNP

        player_stats.append({
            "name": f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
            "min": minutes_display,
            "pts": stats.get("points", 0),
            "reb": stats.get("reboundsTotal", 0),
            "ast": stats.get("assists", 0),
            "stl": stats.get("steals", 0),
            "blk": stats.get("blocks", 0),
            "fg_pct": _fmt_pct(stats.get("fieldGoalsMade"), stats.get("fieldGoalsAttempted")),
            "fg_made": stats.get("fieldGoalsMade", 0),
            "fg_att": stats.get("fieldGoalsAttempted", 0),
            "tp_pct": _fmt_pct(stats.get("threePointersMade"), stats.get("threePointersAttempted")),
            "tp_made": stats.get("threePointersMade", 0),
            "tp_att": stats.get("threePointersAttempted", 0),
            "plus_minus": stats.get("plusMinusPoints", 0),
        })

    return {
        "game_id": game_id_str,
        "date": game_date,
        "location": location,
        "opp": opp_abbrev,
        "our_score": our_score,
        "opp_score": opp_score,
        "result": result,
        "players": player_stats,
    }


def _parse_minutes(raw: str) -> str:
    """Convert ISO 8601 duration or MM:SS to readable MM:SS string."""
    if not raw:
        return ""
    raw = str(raw)
    # ISO format: PT34M12.00S
    if raw.startswith("PT"):
        raw = raw[2:]  # strip PT
        mins = 0
        secs = 0
        if "M" in raw:
            parts = raw.split("M")
            mins = int(parts[0])
            raw = parts[1]
        if "S" in raw:
            secs = int(float(raw.replace("S", "")))
        return f"{mins}:{secs:02d}"
    # Already MM:SS
    if ":" in raw:
        return raw
    # Just a number (minutes)
    try:
        m = int(float(raw))
        return f"{m}:00"
    except ValueError:
        return raw


def _fmt_pct(made, attempted) -> str:
    if not attempted:
        return "0.0"
    return f"{(made / attempted) * 100:.1f}"


# ── Display / Report ─────────────────────────────────────────────────────────

def format_game(game: dict, team_abbrev: str) -> list[str]:
    """Return formatted lines for a single game's box score."""
    lines = []
    header = (f"  {game['date']}  |  {team_abbrev} {game['location']} "
              f"{game['opp']}  |  {game['our_score']}-{game['opp_score']}  "
              f"({game['result']})")
    lines.append("=" * 90)
    lines.append(header)
    lines.append("=" * 90)

    # Column headers
    col_header = (f"  {'Player':<22} {'MIN':>5}  {'PTS':>3}  {'REB':>3}  "
                  f"{'AST':>3}  {'STL':>3}  {'BLK':>3}  "
                  f"{'FG%':>6}  {'3P%':>6}  {'+/-':>4}")
    lines.append(col_header)
    lines.append("  " + "-" * 86)

    for p in game["players"]:
        fg_str = f"{p['fg_made']}/{p['fg_att']}"
        tp_str = f"{p['tp_made']}/{p['tp_att']}"
        pm = p["plus_minus"]
        pm_str = f"+{pm}" if pm > 0 else str(pm)
        line = (f"  {p['name']:<22} {p['min']:>5}  {p['pts']:>3}  "
                f"{p['reb']:>3}  {p['ast']:>3}  {p['stl']:>3}  "
                f"{p['blk']:>3}  {fg_str:>6}  {tp_str:>6}  {pm_str:>4}")
        lines.append(line)

    lines.append("")
    return lines


def save_report(lines: list[str], team_abbrev: str):
    today_str = date.today().strftime("%Y-%m-%d")
    filename = f"team_history_{team_abbrev}_{today_str}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved to {filename}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 52)
    print("  NBA TEAM GAME HISTORY")
    print("=" * 52)

    team_input = input("  Team name (e.g. Blazers, Portland, POR): ").strip()
    if not team_input:
        print("  No team entered. Exiting.")
        return

    n_raw = input("  Number of past games to show [default 5]: ").strip()
    n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 5

    print()

    # Resolve team
    team = resolve_team(team_input)
    team_id = team["id"]
    abbrev = team["abbreviation"]
    season = current_nba_season()

    safe_print(f"  Team: {team['full_name']} ({abbrev})")
    print(f"  Season: {season}")
    print(f"  Fetching last {n} games...\n")

    # Get recent game IDs
    game_ids = get_recent_game_ids(team_id, season, n)
    if not game_ids:
        print("  No games found this season.")
        return

    print(f"  Found {len(game_ids)} game(s). Fetching box scores...\n")

    # Fetch box scores
    report_lines = []
    report_lines.append(f"  NBA TEAM GAME HISTORY — {team['full_name']} ({abbrev})")
    report_lines.append(f"  Season: {season}  |  Last {len(game_ids)} game(s)")
    report_lines.append("")

    for i, gid in enumerate(game_ids):
        print(f"  [{i + 1}/{len(game_ids)}] Fetching game {gid}...")
        game = fetch_box_score(gid, team_id)
        if game is None:
            continue

        game_lines = format_game(game, abbrev)
        report_lines.extend(game_lines)

        # Print to console
        for line in game_lines:
            safe_print(line)

    # Save to file
    save_report(report_lines, abbrev)


if __name__ == "__main__":
    main()
