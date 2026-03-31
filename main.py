import csv
import os
import re
import time
import unicodedata
from datetime import date, datetime
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (playergamelog, playbyplayv3,
                                      commonteamroster, leaguedashplayerstats)


def normalize(s: str) -> str:
    """Strip accents and lowercase for accent-insensitive comparison."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def safe_print(s: str):
    """Print with non-ASCII characters replaced so cp1252 never crashes."""
    print(s.encode("ascii", "replace").decode())


def prompt_config():
    """Ask the user for settings at runtime — no file editing needed."""
    print("\n" + "=" * 52)
    print("  NBA PLAYER STATS")
    print("=" * 52)

    print("  [1] Single player analysis (default)")
    print("  [5] Batch Game Day Analysis")
    mode = input("  Pick mode [default 1]: ").strip()

    if mode == "5":
        return None, None, None, None, "batch"

    name = input("  Player name (e.g. Jalen Brunson): ").strip()
    if not name:
        print("No player entered. Exiting.")

    n_raw = input("  Last N games [press Enter for 10]: ").strip()
    n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 10

    vs_team = input("  VS Team (e.g. Lakers) or press Enter to skip: ").strip() or None
    vs_player = input("  VS Player head-to-head (e.g. Stephen Curry) or press Enter to skip: ").strip() or None

    print("=" * 52 + "\n")
    return name, n, vs_team, vs_player, "single"


def current_nba_season() -> str:
    today = date.today()
    year  = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[2:]}"


# ── Player / Team lookup ──────────────────────────────────────────────────────

def get_player_id(name: str, all_players: list) -> int:
    norm_input = normalize(name.strip())

    match = next(
        (p for p in all_players if normalize(p["full_name"]) == norm_input),
        None
    )
    if match:
        return match["id"]

    input_words = norm_input.split()
    suggestions = [
        p["full_name"] for p in all_players
        if any(w in normalize(p["full_name"]) for w in input_words)
    ]
    safe_print(f"\nERROR: Player '{name}' not found. Full name required.")
    if suggestions:
        safe_print("  Did you mean one of these?")
        for s in sorted(suggestions)[:8]:
            safe_print(f"  {s}")
    print()
    raise SystemExit(1)


def resolve_team_abbrev(name: str) -> str:
    """Return NBA team abbreviation from full name, city, nickname, or abbreviation."""
    all_teams  = teams.get_teams()
    norm_input = normalize(name.strip())

    for t in all_teams:
        if (normalize(t["full_name"])   == norm_input or
                normalize(t["nickname"])    == norm_input or
                normalize(t["abbreviation"]) == norm_input or
                normalize(t["city"])        == norm_input):
            return t["abbreviation"]

    # Partial match fallback
    matches = [
        t for t in all_teams
        if norm_input in normalize(t["full_name"]) or
           norm_input in normalize(t["nickname"])
    ]
    if len(matches) == 1:
        return matches[0]["abbreviation"]

    safe_print(f"\nERROR: Team '{name}' not found.")
    if matches:
        safe_print("  Did you mean one of these?")
        for t in matches[:6]:
            safe_print(f"  {t['full_name']} ({t['abbreviation']})")
    print()
    raise SystemExit(1)


# ── Game log helpers ──────────────────────────────────────────────────────────

def fetch_game_log(player_id: int, season: str) -> tuple[list, list]:
    """Fetch player game log with retry logic for rate-limited API."""
    for attempt in range(3):
        try:
            time.sleep(0.6)
            data = playergamelog.PlayerGameLog(
                player_id=player_id, season=season
            ).get_dict()
            rs = data["resultSets"][0]
            headers, rows = rs["headers"], rs["rowSet"]
            if rows:
                return headers, rows
            # Empty rows on first attempt — might be rate-limited, retry
            if attempt < 2:
                print(f"  (Retry {attempt + 1}: empty game log, waiting...)")
                time.sleep(2)
        except Exception as e:
            if attempt < 2:
                print(f"  (Retry {attempt + 1}: {e})")
                time.sleep(2)
            else:
                raise
    return rs["headers"], rs["rowSet"]


def season_summary(headers: list, rows: list, season: str) -> dict:
    col = {h: i for i, h in enumerate(headers)}

    def avg(stat):
        vals = [r[col[stat]] for r in rows if r[col[stat]] is not None]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    def total(stat):
        return sum(r[col[stat]] for r in rows if r[col[stat]] is not None)

    gp     = len(rows)
    wins   = sum(1 for r in rows if r[col["WL"]] == "W")
    losses = gp - wins

    return {
        "season":     season,
        "gp":         gp,
        "record":     f"{wins}-{losses}",
        "ppg":        avg("PTS"),
        "rpg":        avg("REB"),
        "apg":        avg("AST"),
        "spg":        avg("STL"),
        "bpg":        avg("BLK"),
        "topg":       avg("TOV"),
        "fg_pct":     round(total("FGM") / total("FGA") * 100, 1) if total("FGA") else 0.0,
        "fg3_pct":    round(total("FG3M") / total("FG3A") * 100, 1) if total("FG3A") else 0.0,
        "ft_pct":     round(total("FTM") / total("FTA") * 100, 1) if total("FTA") else 0.0,
        "mpg":        avg("MIN"),
        "total_pts":  total("PTS"),
        "total_reb":  total("REB"),
        "total_ast":  total("AST"),
    }


def print_season_card(name: str, s: dict, label: str = ""):
    line = "=" * 52
    tag  = f"  [{label}]" if label else ""
    print(f"\n{line}")
    safe_print(f"  {name}{tag}")
    print(f"  Season: {s['season']}   GP: {s['gp']}   Record (team W-L): {s['record']}")
    print(f"{line}")
    print(f"  PPG: {s['ppg']:<7} RPG: {s['rpg']:<7} APG: {s['apg']:<7} MPG: {s['mpg']}")
    print(f"  SPG: {s['spg']:<7} BPG: {s['bpg']:<7} TOV: {s['topg']}")
    print(f"  FG%: {s['fg_pct']:<7} 3P%: {s['fg3_pct']:<7} FT%: {s['ft_pct']}")
    print(f"  Season totals  ->  PTS: {s['total_pts']}   REB: {s['total_reb']}   AST: {s['total_ast']}")
    print(f"{line}\n")


# ── Fatigue / rest tracking ───────────────────────────────────────────────────

def compute_fatigue(rows: list, headers: list) -> list:
    """
    For each game row (newest-first from API), compute:
      rest_days  — calendar days since previous game (None for first game)
      b2b        — True if rest_days == 0
      fatigue    — "HIGH" (b2b or rest 1), "MODERATE" (rest 2-3), "NORMAL" (rest 4+)
      gap_flag   — True if rest_days > 4 (possible injury/rest absence)
    Returns list of dicts aligned with rows (same order).
    """
    col = {h: i for i, h in enumerate(headers)}
    dates = []
    for r in rows:
        raw = r[col["GAME_DATE"]]  # e.g. "MAR 15, 2025"
        try:
            # NBA API returns uppercase month e.g. "MAR 15, 2025"
            # title() normalises it to "Mar 15, 2025" for strptime %b on Windows
            dates.append(datetime.strptime(raw.title(), "%b %d, %Y").date())
        except ValueError:
            dates.append(None)

    # rows are newest-first; reverse to compute forward differences
    reversed_dates = list(reversed(dates))
    reversed_fatigue = []
    for i, d in enumerate(reversed_dates):
        if i == 0 or d is None or reversed_dates[i - 1] is None:
            rest = None
        else:
            rest = (d - reversed_dates[i - 1]).days - 1  # days OFF between games

        b2b  = rest == 0 if rest is not None else False
        if rest is None:
            level = "NORMAL"
        elif rest <= 1:
            level = "HIGH"
        elif rest <= 3:
            level = "MODERATE"
        else:
            level = "NORMAL"

        gap = rest is not None and rest > 4
        reversed_fatigue.append({"rest_days": rest, "b2b": b2b, "fatigue": level, "gap_flag": gap})

    # reverse back to match original rows order (newest-first)
    return list(reversed(reversed_fatigue))


# ── Game ID selection helpers ─────────────────────────────────────────────────

def get_last_n_game_ids(headers: list, rows: list, n: int) -> list:
    """Return up to n most-recent *unique* game IDs. No backfill."""
    col = {h: i for i, h in enumerate(headers)}
    seen = set()
    ids = []
    for r in rows:
        gid = r[col["Game_ID"]]
        if gid not in seen:
            seen.add(gid)
            ids.append(gid)
    return ids[:n]


def filter_rows_vs_team(headers: list, rows: list, team_abbrev: str) -> list:
    """Keep only rows where the opponent matches team_abbrev."""
    col = {h: i for i, h in enumerate(headers)}
    filtered = []
    for r in rows:
        matchup = r[col["MATCHUP"]]  # e.g. "GSW vs. LAL" or "GSW @ LAL"
        # Opponent is always the team that is NOT the player's team
        # MATCHUP format: "TEAM vs. OPP" or "TEAM @ OPP"
        parts = re.split(r"\s+(?:vs\.|@)\s+", matchup)
        if len(parts) == 2:
            opponent = parts[1].strip()
        else:
            opponent = ""
        if normalize(opponent) == normalize(team_abbrev):
            filtered.append(r)
    return filtered


def get_shared_game_ids(headers1: list, rows1: list,
                        headers2: list, rows2: list) -> list:
    """Return game IDs that appear in both players' game logs (shared games)."""
    col1 = {h: i for i, h in enumerate(headers1)}
    col2 = {h: i for i, h in enumerate(headers2)}
    ids1 = {r[col1["Game_ID"]] for r in rows1}
    ids2 = {r[col2["Game_ID"]] for r in rows2}
    shared = ids1 & ids2
    # Preserve ordering from player 1's log (newest first)
    return [r[col1["Game_ID"]] for r in rows1 if r[col1["Game_ID"]] in shared]


# ── Play-by-play parsing ──────────────────────────────────────────────────────

def parse_iso_clock(clock_str: str, period: int) -> float:
    match = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if not match:
        return 0.0
    mins_left  = int(match.group(1))
    secs_left  = float(match.group(2))
    period_len = 5 if period > 4 else 12
    prior_mins = (period - 1) * 12 if period <= 4 else 48 + (period - 5) * 5
    return round(prior_mins + (period_len - mins_left - secs_left / 60), 2)


def get_game_events(player_id: int, game_id: str, last_name: str) -> dict:
    """
    Single API call per game.

    Assist detection: PlayByPlayV3 has NO assistPersonId field.
    Assists are detected via the description text on made shots, which
    always encodes the assister as "(LastName N AST)" or "(F. LastName N AST)".
    We match the normalised last_name (accent-stripped, lowercase).

    Returns:
      pts_events, reb_events, ast_events, sub_events
    """
    time.sleep(0.6)
    data    = playbyplayv3.PlayByPlayV3(game_id=game_id).get_dict()
    actions = data["game"]["actions"]

    # Match "(Green 5 AST)" or "(D. Green 5 AST)" — accent-stripped last name
    ast_pattern = re.compile(
        r"\((?:\w+\.\s+)?" + re.escape(last_name) + r"\s+\d+\s+AST\)",
        re.IGNORECASE
    )

    pts_events = []
    reb_events = []
    ast_events = []
    sub_events = []
    prev_off   = 0

    for action in actions:
        atype  = action["actionType"]
        desc   = action.get("description", "").strip()
        period = action["period"]
        minute = parse_iso_clock(action["clock"], period)
        base   = {"game_id": game_id, "period": period, "minute": minute, "description": desc}

        # --- Points ---
        if action["personId"] == player_id and atype in ("Made Shot", "Free Throw"):
            if atype == "Free Throw" and "MISS" in desc.upper():
                continue
            shot_value = action.get("shotValue", 0)
            pts_events.append({**base, "points": shot_value if shot_value else 1})

        # --- Rebounds ---
        if action["personId"] == player_id and atype == "Rebound":
            m = re.search(r"\(Off:(\d+)\s+Def:(\d+)\)", desc)
            if m:
                cur_off = int(m.group(1))
                is_off  = cur_off > prev_off
                prev_off = cur_off
            else:
                is_off = False
            reb_events.append({**base, "reb_type": "OFF" if is_off else "DEF"})

        # --- Assists ---
        # A made shot by someone else whose description credits our player
        if action["personId"] != player_id and atype == "Made Shot":
            if ast_pattern.search(desc):
                ast_events.append({**base})

        # --- Substitutions ---
        if atype == "Substitution":
            if action["personId"] == player_id:
                sub_events.append({**base, "sub_type": "OUT"})
            elif re.match(r"SUB:\s+" + re.escape(last_name), desc, re.IGNORECASE):
                sub_events.append({**base, "sub_type": "IN"})

    return {"pts": pts_events, "reb": reb_events, "ast": ast_events, "sub": sub_events}


# ── Display helpers ───────────────────────────────────────────────────────────

def print_game_table(i: int, total: int, game_id: str,
                     pts_total: int, reb_total: int, ast_total: int,
                     tagged: list, fatigue_info: dict | None = None,
                     opponent: str = ""):
    opp_str  = f"  vs {opponent}" if opponent else ""
    fat_str  = ""
    if fatigue_info:
        rest = fatigue_info["rest_days"]
        rest_label = f"{rest}d rest" if rest is not None else "season opener"
        b2b_label  = " [B2B]" if fatigue_info["b2b"] else ""
        gap_label  = " [GAP - possible absence]" if fatigue_info["gap_flag"] else ""
        fat_level  = fatigue_info["fatigue"]
        fat_str = f"  | fatigue: {fat_level} ({rest_label}{b2b_label}{gap_label})"

    print(f"\n[{i}/{total}] Game {game_id}{opp_str}  |  "
          f"PTS: {pts_total}  REB: {reb_total}  AST: {ast_total}{fat_str}")
    print(f"  {'-'*85}")
    print(f"  {'Per':<5} {'Min':<8} {'Type':<6} {'Val':<6} Description")
    print(f"  {'-'*85}")
    for e in tagged:
        print(f"  {e['period']:<5} {e['minute']:<8} {e['type']:<6} {str(e['value']):<6} {e['description'][:60]}")
    print(f"  {'-'*85}")


def print_window_summary(n: int, pts_totals: dict, reb_totals: dict,
                         ast_totals: dict, season_stats: dict):
    def l_avg(totals):
        v = list(totals.values())
        return round(sum(v) / len(v), 1) if v else 0.0

    print(f"\n{'='*60}")
    print(f"  LAST {n} GAMES SUMMARY  (season avg in parentheses)")
    print(f"{'='*60}")
    print(f"  PTS  avg: {l_avg(pts_totals):<6} ({season_stats['ppg']} season)  "
          f"high: {max(pts_totals.values(), default=0)}  "
          f"low: {min(pts_totals.values(), default=0)}")
    print(f"  REB  avg: {l_avg(reb_totals):<6} ({season_stats['rpg']} season)  "
          f"high: {max(reb_totals.values(), default=0)}  "
          f"low: {min(reb_totals.values(), default=0)}")
    print(f"  AST  avg: {l_avg(ast_totals):<6} ({season_stats['apg']} season)  "
          f"high: {max(ast_totals.values(), default=0)}  "
          f"low: {min(ast_totals.values(), default=0)}")
    print(f"{'='*60}\n")


def save_events_csv(all_events: list, slug: str, season: str, suffix: str = ""):
    if not all_events:
        return
    tag  = f"_{suffix}" if suffix else ""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{slug}_{season}{tag}_events.csv")
    with open(path, "w", newline="") as f:
        fields = ["game_id", "game_date", "opponent", "period", "minute",
                  "type", "value", "description"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: e.get(k, "") for k in fields} for e in all_events])
    print(f"Saved {len(all_events)} events -> {path}")


# ── Per-game processing ───────────────────────────────────────────────────────

def process_games(player_id: int, game_ids: list, last_name: str,
                  fatigue_map: dict, opponent_map: dict, date_map: dict) -> tuple:
    """
    Fetch PBP for each game_id, print table, return aggregates.
    fatigue_map: game_id -> fatigue_info dict (or empty dict)
    opponent_map: game_id -> opponent abbreviation string
    date_map:     game_id -> game date string e.g. "MAR 19, 2026"
    """
    all_events = []
    pts_totals, reb_totals, ast_totals = {}, {}, {}
    failed_games = []

    for i, game_id in enumerate(game_ids, 1):
        try:
            evts = get_game_events(player_id, game_id, last_name)
        except Exception as e:
            print(f"  WARNING: Could not fetch PBP for game {game_id}: {e}")
            failed_games.append(game_id)
            continue

        pts_total = sum(e["points"] for e in evts["pts"])
        reb_total = len(evts["reb"])
        ast_total = len(evts["ast"])
        pts_totals[game_id] = pts_total
        reb_totals[game_id] = reb_total
        ast_totals[game_id] = ast_total

        game_date = date_map.get(game_id, "")
        opponent  = opponent_map.get(game_id, "")

        tagged = []
        for e in evts["pts"]:
            tagged.append({**e, "type": "PTS", "value": e["points"],
                           "game_date": game_date, "opponent": opponent})
        for e in evts["reb"]:
            tagged.append({**e, "type": "REB", "value": e["reb_type"],
                           "game_date": game_date, "opponent": opponent})
        for e in evts["ast"]:
            tagged.append({**e, "type": "AST", "value": "-",
                           "game_date": game_date, "opponent": opponent})
        for e in evts["sub"]:
            tagged.append({**e, "type": "SUB_IN" if e["sub_type"] == "IN" else "SUB_OUT",
                           "value": e["sub_type"],
                           "game_date": game_date, "opponent": opponent})
        tagged.sort(key=lambda x: x["minute"])

        print_game_table(
            i, len(game_ids), game_id,
            pts_total, reb_total, ast_total,
            tagged,
            fatigue_info=fatigue_map.get(game_id),
            opponent=opponent_map.get(game_id, ""),
        )
        all_events.extend(tagged)

    if failed_games:
        print(f"\n  WARNING: {len(failed_games)} game(s) failed to fetch PBP data "
              f"and were excluded from the summary.")

    return all_events, pts_totals, reb_totals, ast_totals


# ── VS PLAYER helpers ─────────────────────────────────────────────────────────

def process_games_dual(pid1: int, pid2: int, game_ids: list,
                        name1: str, name2: str,
                        last_name1: str, last_name2: str) -> None:
    """
    Print merged chronological PBP table for both players per shared game.
    """
    pts1, reb1, ast1 = {}, {}, {}
    pts2, reb2, ast2 = {}, {}, {}

    for i, game_id in enumerate(game_ids, 1):
        evts1 = get_game_events(pid1, game_id, last_name1)
        evts2 = get_game_events(pid2, game_id, last_name2)

        p1_pts = sum(e["points"] for e in evts1["pts"])
        p1_reb = len(evts1["reb"])
        p1_ast = len(evts1["ast"])
        p2_pts = sum(e["points"] for e in evts2["pts"])
        p2_reb = len(evts2["reb"])
        p2_ast = len(evts2["ast"])

        pts1[game_id] = p1_pts; reb1[game_id] = p1_reb; ast1[game_id] = p1_ast
        pts2[game_id] = p2_pts; reb2[game_id] = p2_reb; ast2[game_id] = p2_ast

        def tag_player(evts, label):
            t = []
            for e in evts["pts"]:
                t.append({**e, "player": label, "type": "PTS", "value": e["points"]})
            for e in evts["reb"]:
                t.append({**e, "player": label, "type": "REB", "value": e["reb_type"]})
            for e in evts["ast"]:
                t.append({**e, "player": label, "type": "AST", "value": "-"})
            for e in evts["sub"]:
                t.append({**e, "player": label,
                          "type": "SUB_IN" if e["sub_type"] == "IN" else "SUB_OUT",
                          "value": e["sub_type"]})
            return t

        short1 = name1.split()[-1]
        short2 = name2.split()[-1]
        merged = tag_player(evts1, short1) + tag_player(evts2, short2)
        merged.sort(key=lambda x: x["minute"])

        print(f"\n[{i}/{len(game_ids)}] Game {game_id}  MATCHUP: {name1} vs {name2}")
        safe_print(f"  {name1}: PTS {p1_pts}  REB {p1_reb}  AST {p1_ast}")
        safe_print(f"  {name2}: PTS {p2_pts}  REB {p2_reb}  AST {p2_ast}")
        print(f"  {'-'*90}")
        print(f"  {'Per':<5} {'Min':<8} {'Player':<14} {'Type':<6} {'Val':<6} Description")
        print(f"  {'-'*90}")
        for e in merged:
            safe_print(f"  {e['period']:<5} {e['minute']:<8} {e['player']:<14} "
                       f"{e['type']:<6} {str(e['value']):<6} {e['description'][:50]}")
        print(f"  {'-'*90}")

    def l_avg(totals):
        v = list(totals.values())
        return round(sum(v) / len(v), 1) if v else 0.0

    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD  (last {len(game_ids)} shared games)")
    print(f"{'='*60}")
    safe_print(f"  {name1:<25} PTS {l_avg(pts1):<5} REB {l_avg(reb1):<5} AST {l_avg(ast1)}")
    safe_print(f"  {name2:<25} PTS {l_avg(pts2):<5} REB {l_avg(reb2):<5} AST {l_avg(ast2)}")
    print(f"{'='*60}\n")


# ── Utilities ─────────────────────────────────────────────────────────────────

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def _extract_last_name(full_name: str) -> str:
    """
    Return accent-stripped last name, skipping generational suffixes.
    'Tim Hardaway Jr.' -> 'hardaway'
    'Karl-Anthony Towns' -> 'towns'   (hyphen stays for re.escape)
    'Nikola Jokic'      -> 'jokic'
    """
    parts = full_name.strip().split()
    # Drop trailing suffix tokens
    while parts and normalize(parts[-1].rstrip(".")) in _NAME_SUFFIXES:
        parts.pop()
    return normalize(parts[-1]) if parts else normalize(full_name.strip().split()[-1])


# ── Batch Game Day Analysis ──────────────────────────────────────────────────

def _get_top_players_for_team(team_abbrev: str, season: str, top_n: int = 8) -> list[dict]:
    """
    Return top N players for a team by minutes played this season.
    Each entry: {"id": int, "full_name": str}
    """
    # Get team ID from abbreviation
    all_teams = teams.get_teams()
    team = next((t for t in all_teams
                 if t["abbreviation"].upper() == team_abbrev.upper()), None)
    if not team:
        print(f"  WARNING: Could not find team {team_abbrev}")
        return []

    team_id = team["id"]

    # Get roster to know who is on the team
    time.sleep(0.6)
    try:
        roster_data = commonteamroster.CommonTeamRoster(
            team_id=team_id, season=season
        ).get_dict()
        roster_rs = roster_data["resultSets"][0]
        roster_headers = roster_rs["headers"]
        roster_rows = roster_rs["rowSet"]
    except Exception as e:
        print(f"  WARNING: Could not fetch roster for {team_abbrev}: {e}")
        return []

    rcol = {h: i for i, h in enumerate(roster_headers)}
    roster_ids = {r[rcol["PLAYER_ID"]] for r in roster_rows}

    # Get league-wide player stats, filter to this team's roster
    time.sleep(0.6)
    try:
        stats_data = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="PerGame"
        ).get_dict()
        stats_rs = stats_data["resultSets"][0]
        stats_headers = stats_rs["headers"]
        stats_rows = stats_rs["rowSet"]
    except Exception as e:
        print(f"  WARNING: Could not fetch league stats: {e}")
        return []

    scol = {h: i for i, h in enumerate(stats_headers)}

    # Filter to players on this team's roster and sort by minutes
    team_players = []
    for r in stats_rows:
        pid = r[scol["PLAYER_ID"]]
        if pid in roster_ids:
            mpg = r[scol["MIN"]] or 0.0
            gp = r[scol["GP"]] or 0
            team_players.append({
                "id": pid,
                "full_name": r[scol["PLAYER_NAME"]],
                "mpg": mpg,
                "gp": gp,
            })

    # Sort by total minutes (MPG * GP) descending, take top N
    team_players.sort(key=lambda p: p["mpg"] * p["gp"], reverse=True)
    return team_players[:top_n]


def _batch_player_summary(player_id: int, player_name: str, last_name: str,
                           season: str, num_games: int,
                           vs_team_abbrev: str | None,
                           all_players_db: list) -> dict:
    """
    Compute last-N and vs-team averages for a single player.
    Returns dict with stats, or None if player has no games.
    """
    headers, rows = fetch_game_log(player_id, season)
    if not rows:
        return None

    col = {h: i for i, h in enumerate(headers)}

    # Last N games averages
    game_ids = get_last_n_game_ids(headers, rows, num_games)
    avail = len(game_ids)

    last_n_rows = []
    for r in rows:
        if r[col["Game_ID"]] in set(game_ids):
            last_n_rows.append(r)

    def avg_stat(row_list, stat):
        vals = [r[col[stat]] for r in row_list if r[col[stat]] is not None]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    result = {
        "name": player_name,
        "last_n": avail,
        "ppg": avg_stat(last_n_rows, "PTS"),
        "rpg": avg_stat(last_n_rows, "REB"),
        "apg": avg_stat(last_n_rows, "AST"),
        "vs_team": None,
    }

    # VS opponent team if requested
    if vs_team_abbrev:
        vs_rows = filter_rows_vs_team(headers, rows, vs_team_abbrev)
        if vs_rows:
            result["vs_team"] = {
                "abbrev": vs_team_abbrev,
                "gp": len(vs_rows),
                "ppg": avg_stat(vs_rows, "PTS"),
                "rpg": avg_stat(vs_rows, "REB"),
                "apg": avg_stat(vs_rows, "AST"),
            }

    return result


def batch_game_day():
    """Mode 5: Batch Game Day Analysis — analyze all key players for multiple matchups."""
    print("\n" + "=" * 52)
    print("  BATCH GAME DAY ANALYSIS")
    print("=" * 52)

    raw_matchups = input("\n  Enter matchups (e.g. Grizzlies vs Hornets, Lakers vs Magic): ").strip()
    if not raw_matchups:
        print("  No matchups entered. Exiting.")
        return

    n_raw = input("  How many past games? [default 5]: ").strip()
    num_games = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 5

    print("\n  Stats to analyze:")
    print("  [1] Points  [2] Rebounds  [3] Assists  [4] All three")
    stat_pick = input("  Pick [default 4]: ").strip()
    # Stats selection affects display — we always fetch all three from game log
    if stat_pick == "1":
        show_stats = ["PTS"]
    elif stat_pick == "2":
        show_stats = ["REB"]
    elif stat_pick == "3":
        show_stats = ["AST"]
    else:
        show_stats = ["PTS", "REB", "AST"]

    # Parse matchups
    matchups = []
    for chunk in raw_matchups.split(","):
        chunk = chunk.strip()
        parts = re.split(r"\s+vs\.?\s+", chunk, flags=re.IGNORECASE)
        if len(parts) != 2:
            print(f"  WARNING: Could not parse matchup '{chunk}', skipping.")
            continue
        matchups.append((parts[0].strip(), parts[1].strip()))

    if not matchups:
        print("  No valid matchups found. Exiting.")
        return

    season = current_nba_season()
    all_players_db = players.get_players()
    report_lines = []

    for team1_name, team2_name in matchups:
        # Resolve team abbreviations
        try:
            abbrev1 = resolve_team_abbrev(team1_name)
        except SystemExit:
            print(f"  WARNING: Could not resolve team '{team1_name}', skipping matchup.")
            continue
        try:
            abbrev2 = resolve_team_abbrev(team2_name)
        except SystemExit:
            print(f"  WARNING: Could not resolve team '{team2_name}', skipping matchup.")
            continue

        header = f"{abbrev1} vs {abbrev2} — Game Day Report"
        divider = "=" * len(header)
        print(f"\n{divider}")
        print(f"{header}")
        print(f"{divider}")
        report_lines.extend(["", divider, header, divider])

        for team_abbrev, opp_abbrev, team_label in [
            (abbrev1, abbrev2, abbrev1),
            (abbrev2, abbrev1, abbrev2),
        ]:
            team_header = f"--- {team_label} ---"
            print(f"\n{team_header}\n")
            report_lines.extend(["", team_header, ""])

            print(f"  Fetching top players for {team_abbrev}...")
            top_players = _get_top_players_for_team(team_abbrev, season)
            if not top_players:
                msg = f"  Could not fetch roster for {team_abbrev}."
                print(msg)
                report_lines.append(msg)
                continue

            safe_print(f"  Found {len(top_players)} players: "
                       + ", ".join(p["full_name"] for p in top_players))

            for p_info in top_players:
                pid = p_info["id"]
                pname = p_info["full_name"]
                last_name = _extract_last_name(pname)

                try:
                    result = _batch_player_summary(
                        pid, pname, last_name, season, num_games,
                        opp_abbrev, all_players_db
                    )
                except Exception as e:
                    msg = f"  WARNING: Failed to fetch data for {pname}: {e}"
                    safe_print(msg)
                    report_lines.append(msg)
                    continue

                if result is None:
                    msg = f"  WARNING: No games found for {pname}, skipping."
                    safe_print(msg)
                    report_lines.append(msg)
                    continue

                # Format player line
                name_line = f"{result['name']} (Last {result['last_n']} Games)"
                safe_print(f"\n  {name_line}")
                report_lines.append(f"  {name_line}")

                stat_parts = []
                if "PTS" in show_stats:
                    stat_parts.append(f"PPG: {result['ppg']}")
                if "REB" in show_stats:
                    stat_parts.append(f"RPG: {result['rpg']}")
                if "AST" in show_stats:
                    stat_parts.append(f"APG: {result['apg']}")
                stat_line = "  " + "  |  ".join(stat_parts)
                print(stat_line)
                report_lines.append(stat_line)

                # VS opponent
                if result["vs_team"]:
                    vs = result["vs_team"]
                    vs_parts = []
                    if "PTS" in show_stats:
                        vs_parts.append(f"PPG: {vs['ppg']}")
                    if "REB" in show_stats:
                        vs_parts.append(f"RPG: {vs['rpg']}")
                    if "AST" in show_stats:
                        vs_parts.append(f"APG: {vs['apg']}")
                    vs_line = f"  vs {vs['abbrev']} ({vs['gp']} games found): " + "  |  ".join(vs_parts)
                    safe_print(vs_line)
                    report_lines.append(vs_line)
                else:
                    vs_line = f"  vs {opp_abbrev}: No games found this season"
                    print(vs_line)
                    report_lines.append(vs_line)

    # Save report to file
    today_str = date.today().strftime("%Y-%m-%d")
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"gameday_report_{today_str}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n{'=' * 52}")
    print(f"  Report saved -> {report_path}")
    print(f"{'=' * 52}\n")


# ── Main ────────────────────────────────────────────────────────────────────── 


def main():
    PLAYER_NAME, NUM_GAMES, VS_TEAM, VS_PLAYER, MODE = prompt_config()

    if MODE == "batch":     
        batch_game_day()
        return

    if VS_TEAM:
        safe_print(f"  VS Team   : {VS_TEAM}")
    if VS_PLAYER:
        safe_print(f"  VS Player : {VS_PLAYER}")


    print(f"Active NBA season: {season}")

    # ── Resolve primary player ────────────────────────────────────────────────
    all_players_db = players.get_players()   # fetch once, reuse everywhere
    player_id  = get_player_id(PLAYER_NAME, all_players_db)
    true_name1 = next(p["full_name"] for p in all_players_db if p["id"] == player_id)   #simple lin search, usage of algo could speed up search
    last_name1 = _extract_last_name(true_name1)

    headers, rows = fetch_game_log(player_id, season)
    if not rows:
        safe_print(f"No games found for {PLAYER_NAME} in {season}.")
        return

    print(f"  Game log: {len(rows)} games found in {season}")
    s = season_summary(headers, rows, season)
    print_season_card(PLAYER_NAME, s)

    # Build fatigue info and opponent map for all games (indexed by game_id)
    col          = {h: i for i, h in enumerate(headers)}
    fatigue_list = compute_fatigue(rows, headers)
    fatigue_map  = {
        rows[i][col["Game_ID"]]: fatigue_list[i]
        for i in range(len(rows))
    }
    opponent_map = {
        r[col["Game_ID"]]: re.split(r"\s+(?:vs\.|@)\s+", r[col["MATCHUP"]])[-1].strip()
        for r in rows
    }
    date_map = {
        r[col["Game_ID"]]: r[col["GAME_DATE"]]
        for r in rows
    }

    slug = PLAYER_NAME.lower().replace(" ", "_")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: VS_PLAYER head-to-head
    # ══════════════════════════════════════════════════════════════════════════
    if VS_PLAYER:
        pid2 = get_player_id(VS_PLAYER, all_players_db)
        true_name2 = next(p["full_name"] for p in all_players_db if p["id"] == pid2)
        last_name2 = _extract_last_name(true_name2)


        h2, r2 = fetch_game_log(pid2, season)
        if not r2:
            safe_print(f"No games found for {VS_PLAYER} in {season}.")
            return
        s2 = season_summary(h2, r2, season)
        print_season_card(VS_PLAYER, s2)

        shared_ids = get_shared_game_ids(headers, rows, h2, r2)
        if not shared_ids:
            safe_print(f"No shared games found between {PLAYER_NAME} and {VS_PLAYER} in {season}.")
            return

        game_ids = shared_ids[:NUM_GAMES]
        if len(shared_ids) < NUM_GAMES:
            safe_print(f"Found {len(shared_ids)} shared games (requested {NUM_GAMES}) — showing all {len(game_ids)}.\n")
        else:
            safe_print(f"Found {len(shared_ids)} shared games — showing last {len(game_ids)}.\n")
        process_games_dual(player_id, pid2, game_ids,
                           PLAYER_NAME, VS_PLAYER, last_name1, last_name2)

        # Also show the primary player's last-N games (standard breakdown)
        solo_ids = get_last_n_game_ids(headers, rows, NUM_GAMES)
        avail = len(solo_ids)
        if avail < NUM_GAMES:
            safe_print(f"\nNote: Only {avail} games played this season (requested {NUM_GAMES}). "
                       f"Showing all {avail}.\n")
        else:
            safe_print(f"\nFetching PBP for {PLAYER_NAME}'s last {len(solo_ids)} games...\n")
        all_events, pts_totals, reb_totals, ast_totals = process_games(
            player_id, solo_ids, last_name1, fatigue_map, opponent_map, date_map
        )
        actual_count = len(pts_totals)
        print_window_summary(actual_count, pts_totals, reb_totals, ast_totals, s)
        save_events_csv(all_events, slug, season, suffix="vs_player")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2: VS_TEAM filter
    # ══════════════════════════════════════════════════════════════════════════
    if VS_TEAM:
        team_abbrev   = resolve_team_abbrev(VS_TEAM)
        filtered_rows = filter_rows_vs_team(headers, rows, team_abbrev)
        if not filtered_rows:
            safe_print(f"No games vs {VS_TEAM} ({team_abbrev}) found in {season}.")
            return

        vs_s = season_summary(headers, filtered_rows, season)
        print(f"  -- Stats vs {VS_TEAM} ({team_abbrev}) this season: "
              f"GP {vs_s['gp']}  PPG {vs_s['ppg']}  RPG {vs_s['rpg']}  APG {vs_s['apg']} --\n")

        game_ids = get_last_n_game_ids(headers, filtered_rows, NUM_GAMES)
        safe_print(f"Fetching PBP for last {len(game_ids)} games vs {VS_TEAM}...\n")

        all_events, pts_totals, reb_totals, ast_totals = process_games(
            player_id, game_ids, last_name1, fatigue_map, opponent_map, date_map
        )
        actual_count = len(pts_totals)
        print_window_summary(actual_count, pts_totals, reb_totals, ast_totals, vs_s)
        save_events_csv(all_events, slug, season, suffix=f"vs_{team_abbrev}")

        # Also show the primary player's last-N games (standard breakdown)
        solo_ids = get_last_n_game_ids(headers, rows, NUM_GAMES)
        avail = len(solo_ids)
        if avail < NUM_GAMES:
            safe_print(f"\nNote: Only {avail} games played this season (requested {NUM_GAMES}). "
                       f"Showing all {avail}.\n")
        else:
            safe_print(f"\nFetching PBP for {PLAYER_NAME}'s last {len(solo_ids)} games...\n")
        all_events, pts_totals, reb_totals, ast_totals = process_games(
            player_id, solo_ids, last_name1, fatigue_map, opponent_map, date_map
        )
        actual_count = len(pts_totals)
        print_window_summary(actual_count, pts_totals, reb_totals, ast_totals, s)
        save_events_csv(all_events, slug, season)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 3: Standard last-N games
    # ══════════════════════════════════════════════════════════════════════════
    game_ids = get_last_n_game_ids(headers, rows, NUM_GAMES)
    avail    = len(game_ids)
    if avail < NUM_GAMES:
        print(f"Note: Only {avail} games played this season (requested {NUM_GAMES}). "
              f"Showing all {avail}.\n")

    safe_print(f"Fetching PBP for last {len(game_ids)} games...\n")
    all_events, pts_totals, reb_totals, ast_totals = process_games(
        player_id, game_ids, last_name1, fatigue_map, opponent_map, date_map
    )
    actual_count = len(pts_totals)  # games that were actually processed
    print_window_summary(actual_count, pts_totals, reb_totals, ast_totals, s)
    save_events_csv(all_events, slug, season)


if __name__ == "__main__":
    main()
