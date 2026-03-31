"""
charts.py  —  Visualisation tool for NBA player stats
Can run standalone (fetches data from NBA API directly)
or read CSV files saved by main.py.

Run:
    python charts.py
"""

import csv
import os
import re
import time
import unicodedata
from pathlib import Path
from datetime import date, datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import matplotlib.lines as mlines
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from nba_api.stats.endpoints import playbyplayv3, playergamelog, commonplayerinfo
    from nba_api.stats.static import players, teams
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

_BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = _BASE_DIR / "data"
CHART_DIR = _BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)


def _current_nba_season() -> str:
    today = date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[2:]}"


def _normalize(s: str) -> str:
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _safe_print(s: str):
    print(s.encode("ascii", "replace").decode())


def _get_player_id(name: str) -> tuple:
    """Look up NBA player by name, return (player_id, full_name)."""
    if not HAS_NBA_API:
        raise RuntimeError("nba_api is not installed.")
    from nba_api.stats.static import players as nba_players
    norm = _normalize(name)
    all_p = nba_players.get_players()
    # Exact match first
    for p in all_p:
        if _normalize(p["full_name"]) == norm:
            return p["id"], p["full_name"]
    # Last name match
    matches = [p for p in all_p if _normalize(p["last_name"]) == norm]
    if len(matches) == 1:
        return matches[0]["id"], matches[0]["full_name"]
    # Partial match
    matches = [p for p in all_p if norm in _normalize(p["full_name"])]
    if len(matches) == 1:
        return matches[0]["id"], matches[0]["full_name"]
    if len(matches) > 1:
        print("  Multiple matches found:")
        for i, p in enumerate(matches):
            print(f"    [{i+1}] {p['full_name']}")
        pick = input("  Pick a number: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(matches):
            m = matches[int(pick) - 1]
            return m["id"], m["full_name"]
    raise SystemExit(f"Could not find player: {name}")


def _resolve_team_abbrev(name: str) -> str | None:
    if not HAS_NBA_API:
        return None
    norm = _normalize(name)
    for t in teams.get_teams():
        if (_normalize(t["full_name"]) == norm or
                _normalize(t["nickname"]) == norm or
                _normalize(t["abbreviation"]) == norm or
                _normalize(t["city"]) == norm):
            return t["abbreviation"]
    matches = [t for t in teams.get_teams()
               if norm in _normalize(t["full_name"]) or norm in _normalize(t["nickname"])]
    if len(matches) == 1:
        return matches[0]["abbreviation"]
    return None


def _fetch_game_log(player_id: int, season: str) -> tuple[list, list]:
    """Fetch player game log with retry logic."""
    for attempt in range(3):
        try:
            time.sleep(0.6)
            data = playergamelog.PlayerGameLog(
                player_id=player_id, season=season
            ).get_dict()
            rs = data["resultSets"][0]
            if rs["rowSet"] or attempt == 2:
                return rs["headers"], rs["rowSet"]
            print(f"    (Retry {attempt + 1}: empty game log, waiting...)")
            time.sleep(2)
        except Exception as e:
            if attempt < 2:
                print(f"    (Retry {attempt + 1}: {e})")
                time.sleep(2)
            else:
                raise
    return [], []


def _filter_rows_vs_team(headers: list, rows: list, team_abbrev: str) -> list:
    """Keep only rows where the opponent matches team_abbrev."""
    col = {h: i for i, h in enumerate(headers)}
    filtered = []
    for r in rows:
        matchup = r[col["MATCHUP"]]
        parts = re.split(r"\s+(?:vs\.|@)\s+", matchup)
        if len(parts) == 2:
            opponent = parts[1].strip()
        else:
            opponent = ""
        if _normalize(opponent) == _normalize(team_abbrev):
            filtered.append(r)
    return filtered


def _parse_iso_clock(clock_str: str, period: int) -> float:
    match = re.match(r"PT(\d+)M([\d.]+)S", clock_str)
    if not match:
        return 0.0
    mins_left  = int(match.group(1))
    secs_left  = float(match.group(2))
    period_len = 5 if period > 4 else 12
    prior_mins = (period - 1) * 12 if period <= 4 else 48 + (period - 5) * 5
    return round(prior_mins + (period_len - mins_left - secs_left / 60), 2)


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def _extract_last_name(full_name: str) -> str:
    parts = full_name.strip().split()
    while parts and _normalize(parts[-1].rstrip(".")) in _NAME_SUFFIXES:
        parts.pop()
    return _normalize(parts[-1]) if parts else _normalize(full_name.strip().split()[-1])


def _get_game_events(player_id: int, game_id: str, last_name: str) -> dict:
    """Fetch PBP for a single game and extract pts/reb/ast events."""
    time.sleep(0.6)
    data    = playbyplayv3.PlayByPlayV3(game_id=game_id).get_dict()
    actions = data["game"]["actions"]

    ast_pattern = re.compile(
        r"\((?:\w+\.\s+)?" + re.escape(last_name) + r"\s+\d+\s+AST\)",
        re.IGNORECASE
    )

    pts_events, reb_events, ast_events, sub_events = [], [], [], []
    prev_off = 0

    for action in actions:
        atype  = action["actionType"]
        desc   = action.get("description", "").strip()
        period = action["period"]
        minute = _parse_iso_clock(action["clock"], period)
        base   = {"game_id": game_id, "period": period, "minute": minute, "description": desc}

        if action["personId"] == player_id and atype in ("Made Shot", "Free Throw"):
            if atype == "Free Throw" and "MISS" in desc.upper():
                continue
            shot_value = action.get("shotValue", 0)
            pts_events.append({**base, "points": shot_value if shot_value else 1})

        if action["personId"] == player_id and atype == "Rebound":
            m = re.search(r"\(Off:(\d+)\s+Def:(\d+)\)", desc)
            if m:
                cur_off = int(m.group(1))
                is_off  = cur_off > prev_off
                prev_off = cur_off
            else:
                is_off = False
            reb_events.append({**base, "reb_type": "OFF" if is_off else "DEF"})

        if action["personId"] != player_id and atype == "Made Shot":
            if ast_pattern.search(desc):
                ast_events.append({**base})

        if atype == "Substitution":
            if action["personId"] == player_id:
                sub_events.append({"minute": minute, "sub_type": "OUT"})
            elif re.match(r"SUB:\s+" + re.escape(last_name),
                          desc, re.IGNORECASE):
                sub_events.append({"minute": minute, "sub_type": "IN"})

    return {"pts": pts_events, "reb": reb_events, "ast": ast_events, "sub": sub_events}


def fetch_vs_team_data_from_api(player_name: str, team_input: str) -> tuple[list[dict], str, str, float]:
    """
    Fetch game data for player vs team directly from NBA API.
    Returns (rows_as_dicts, player_name, season, season_ppg).
    rows_as_dicts mimics the CSV row format used elsewhere in charts.py.
    """
    if not HAS_NBA_API:
        print("  nba_api not installed. Cannot fetch live data.")
        print("  Run: pip install nba_api")
        return [], player_name, "", 0.0

    season = _current_nba_season()
    player_id, true_name = _get_player_id(player_name)
    last_name = _extract_last_name(true_name)

    _safe_print(f"  Fetching game log for {true_name} ({season})...")
    headers, rows = _fetch_game_log(player_id, season)
    if not rows:
        _safe_print(f"  No games found for {true_name} in {season}.")
        return [], true_name, season, 0.0

    col = {h: i for i, h in enumerate(headers)}

    # Compute season PPG from the full game log
    all_pts = [r[col["PTS"]] for r in rows if r[col["PTS"]] is not None]
    season_ppg = round(sum(all_pts) / len(all_pts), 1) if all_pts else 0.0

    # Resolve team and filter
    abbrev = _resolve_team_abbrev(team_input)
    if abbrev is None:
        print(f"  Team '{team_input}' not recognised.")
        return [], true_name, season, season_ppg

    vs_rows = _filter_rows_vs_team(headers, rows, abbrev)
    if not vs_rows:
        print(f"  No games vs {abbrev} found in {season}.")
        return [], true_name, season, season_ppg

    print(f"  Found {len(vs_rows)} games vs {abbrev}. Fetching play-by-play...")

    # Build opponent/date maps for matched games
    csv_rows = []
    for r in vs_rows:
        game_id  = r[col["Game_ID"]]
        raw_date = r[col["GAME_DATE"]]
        matchup  = r[col["MATCHUP"]]
        opponent = re.split(r"\s+(?:vs\.|@)\s+", matchup)[-1].strip()

        try:
            evts = _get_game_events(player_id, game_id, last_name)
        except Exception as e:
            print(f"    WARNING: Could not fetch PBP for game {game_id}: {e}")
            continue

        for e in evts["pts"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "PTS", "value": str(e["points"]), "description": e["description"],
            })
        for e in evts["reb"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "REB", "value": e["reb_type"], "description": e["description"],
            })
        for e in evts["ast"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "AST", "value": "-", "description": e["description"],
            })
        for e in evts["sub"]:
            sub_type = "SUB_IN" if e["sub_type"] == "IN" else "SUB_OUT"
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": 0, "minute": e["minute"],
                "type": sub_type, "value": e["sub_type"], "description": "",
            })

    print(f"  Fetched {len(csv_rows)} events across {len(vs_rows)} games vs {abbrev}.")
    return csv_rows, true_name, season, season_ppg


def fetch_last_n_data_from_api(player_name: str, n: int) -> tuple[list[dict], str, str]:
    """
    Fetch last N games for a player directly from NBA API.
    Returns (rows_as_dicts, player_name, season).
    """
    if not HAS_NBA_API:
        print("  nba_api not installed. Cannot fetch live data.")
        print("  Run: pip install nba_api")
        return [], player_name, ""

    season = _current_nba_season()
    player_id, true_name = _get_player_id(player_name)
    last_name = _extract_last_name(true_name)

    _safe_print(f"  Fetching game log for {true_name} ({season})...")
    headers, rows = _fetch_game_log(player_id, season)
    if not rows:
        _safe_print(f"  No games found for {true_name} in {season}.")
        return [], true_name, season

    col = {h: i for i, h in enumerate(headers)}

    # Deduplicate game IDs and take first N
    seen = set()
    unique_rows = []
    for r in rows:
        gid = r[col["Game_ID"]]
        if gid not in seen:
            seen.add(gid)
            unique_rows.append(r)
    target_rows = unique_rows[:n]

    avail = len(target_rows)
    if avail < n:
        print(f"  Note: Only {avail} games played this season (requested {n}). Showing all {avail}.")

    print(f"  Fetching play-by-play for {avail} games...")

    csv_rows = []
    for r in target_rows:
        game_id  = r[col["Game_ID"]]
        raw_date = r[col["GAME_DATE"]]
        matchup  = r[col["MATCHUP"]]
        opponent = re.split(r"\s+(?:vs\.|@)\s+", matchup)[-1].strip()

        try:
            evts = _get_game_events(player_id, game_id, last_name)
        except Exception as e:
            print(f"    WARNING: Could not fetch PBP for game {game_id}: {e}")
            continue

        for e in evts["pts"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "PTS", "value": str(e["points"]), "description": e["description"],
            })
        for e in evts["reb"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "REB", "value": e["reb_type"], "description": e["description"],
            })
        for e in evts["ast"]:
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": e["period"], "minute": e["minute"],
                "type": "AST", "value": "-", "description": e["description"],
            })
        for e in evts["sub"]:
            sub_type = "SUB_IN" if e["sub_type"] == "IN" else "SUB_OUT"
            csv_rows.append({
                "game_id": game_id, "game_date": raw_date, "opponent": opponent,
                "period": 0, "minute": e["minute"],
                "type": sub_type, "value": e["sub_type"], "description": "",
            })

    print(f"  Fetched {len(csv_rows)} events across {avail} games.")
    return csv_rows, true_name, season


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Data processing ───────────────────────────────────────────────────────────

# ── Stat-type configuration ──────────────────────────────────────────────────

STAT_CONFIG = {
    "PTS": {"label": "Points",   "per_game": "PPG", "unit": "pts", "color": "#c6ff00",
            "suffix": "_pts", "flat_note": "Flat = not scoring"},
    "AST": {"label": "Assists",  "per_game": "APG", "unit": "ast", "color": "#00e5ff",
            "suffix": "_ast", "flat_note": "Flat = no assists"},
    "REB": {"label": "Rebounds", "per_game": "RPG", "unit": "reb", "color": "#ff9a00",
            "suffix": "_reb", "flat_note": "Flat = no rebounds"},
}


def build_cumulative_curve(events: list[dict], stat_type: str = "PTS") -> list[float]:
    """
    Returns 481 values (index 0–480) = cumulative stat at each 0.1-min step.
    For PTS uses the numeric value; for REB/AST each event counts as 1.
    """
    curve = [0.0] * 481
    for evt in events:
        try:
            minute = float(evt["minute"])
            if stat_type == "PTS":
                val = int(evt["value"])
            else:
                val = 1
        except (ValueError, KeyError):
            continue
        idx = min(int(round(minute * 10)), 480)
        curve[idx] += val
    for i in range(1, 481):
        curve[i] += curve[i - 1]
    return curve


def avg_curve(curves_dict: dict, gids: list) -> list[float]:
    if not gids:
        return [0.0] * 481
    return [
        sum(curves_dict[g][i] for g in gids) / len(gids)
        for i in range(481)
    ]


def step_up_markers(curve: list[float], x: list[float]):
    """Return (x_vals, y_vals) at every integer minute where the curve increases."""
    mx, my = [], []
    for minute in range(1, 49):          # integer minutes 1..48
        idx = minute * 10
        if curve[idx] > curve[idx - 10]:
            mx.append(x[idx])
            my.append(curve[idx])
    return mx, my


# ── Substitution interval helper ──────────────────────────────────────────────

def build_court_intervals(sub_rows: list[dict], game_duration: float = 48.0) -> list[tuple[float, float, bool]]:
    """
    Return a list of (start_minute, end_minute, on_court) covering 0..game_duration.
    sub_rows must be sorted by minute and each have r["type"] in ("SUB_IN","SUB_OUT").
    """
    if not sub_rows:
        return [(0.0, game_duration, True)]

    # If first event is SUB_OUT the player started on court; SUB_IN means bench
    on_court = sub_rows[0]["type"] == "SUB_OUT"
    intervals: list[tuple[float, float, bool]] = []
    cursor = 0.0

    for r in sub_rows:
        minute = float(r["minute"])
        if minute > cursor:
            intervals.append((cursor, minute, on_court))
        if r["type"] == "SUB_OUT":
            on_court = False
        else:
            on_court = True
        cursor = minute

    if cursor < game_duration:
        intervals.append((cursor, game_duration, on_court))

    return intervals


def _plot_game_segments(ax, x, curve, intervals, colour, linewidth, on_alpha, off_alpha, zorder):
    """Plot a curve with different alpha for on-court vs bench intervals."""
    for start, end, on_court in intervals:
        i0 = int(round(start * 10))
        i1 = min(int(round(end * 10)), 480)
        if i1 <= i0:
            continue
        seg_x = x[i0:i1 + 1]
        seg_y = curve[i0:i1 + 1]
        alpha = on_alpha if on_court else off_alpha
        ax.plot(seg_x, seg_y, color=colour, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _plot_sub_markers(ax, sub_rows, curve, colour):
    """Plot SUB_IN (filled) and SUB_OUT (hollow) markers on the curve."""
    # Pre-compute which minutes have overlapping markers (within 0.1 tolerance)
    minutes = [float(r["minute"]) for r in sub_rows]
    overlapping = set()
    for i, m1 in enumerate(minutes):
        for j, m2 in enumerate(minutes):
            if i < j and abs(m1 - m2) <= 0.1:
                overlapping.add(i)
                overlapping.add(j)

    for i, r in enumerate(sub_rows):
        minute = minutes[i]
        plot_minute = min(minute, 48.0)
        idx = min(int(round(minute * 10)), 480)
        y_val = curve[idx]
        if i in overlapping:
            if r["type"] == "SUB_OUT":
                plot_minute = max(0.0, min(plot_minute - 0.3, 48.0))
            else:
                plot_minute = max(0.0, min(plot_minute + 0.3, 48.0))
        if r["type"] == "SUB_OUT":
            ax.scatter(plot_minute, y_val, marker="o", facecolors="none",
                       edgecolors=colour, s=80, linewidths=1.5, zorder=6)
        else:
            ax.scatter(plot_minute, y_val, marker="o", color=colour,
                       s=80, zorder=6)


def _get_sub_rows(rows, gid):
    """Extract and sort SUB_IN/SUB_OUT rows for a given game."""
    subs = [r for r in rows if r["game_id"] == gid and r["type"] in ("SUB_IN", "SUB_OUT")]
    subs.sort(key=lambda r: float(r["minute"]))
    # Deduplicate consecutive same-type events
    deduped = []
    for s in subs:
        if not deduped or s["type"] != deduped[-1]["type"]:
            deduped.append(s)
    return deduped


# ── Chart ─────────────────────────────────────────────────────────────────────

# Quarter background colours — alternating very dark blues
QUARTER_FILLS = ["#0d1b2a", "#112233"]

def plot_cumulative_scoring(rows: list[dict], player_name: str,
                             season: str, last_n: int | None, out_path: Path,
                             stat_type: str = "PTS"):

    cfg  = STAT_CONFIG[stat_type]
    unit = cfg["unit"]

    # ── Group events by game ─────────────────────────────────────────────────
    games: dict[str, list] = {}
    for r in rows:
        if r["type"] != stat_type:
            continue
        gid = r["game_id"]
        if gid not in games:
            games[gid] = []
        games[gid].append(r)

    if not games:
        print(f"No {stat_type} events found in this CSV.")
        return

    # Oldest-first so chart reads left=oldest, right=most recent
    all_game_order = list(reversed(list(games.keys())))
    curves         = {gid: build_cumulative_curve(games[gid], stat_type) for gid in all_game_order}

    # Which games to show as individual lines (respect last_n filter)
    n_total    = len(all_game_order)
    if last_n and last_n > n_total:
        print(f"  Note: Only {n_total} games available (requested {last_n}). Showing all {n_total}.")
    game_order = all_game_order[-last_n:] if (last_n and last_n < n_total) else all_game_order
    n_games    = len(game_order)
    x          = [i / 10 for i in range(481)]   # 0.0 → 48.0

    all_avg_c    = avg_curve(curves, all_game_order)   # white line: ALL csv games
    recent_avg_c = avg_curve(curves, game_order)        # coloured line: shown games

    # Season per-game avg = average final total across all games
    season_avg = round(all_avg_c[480], 1)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    # ── Quarter shading (drawn first, behind everything) ──────────────────────
    for q in range(4):
        ax.axvspan(q * 12, (q + 1) * 12,
                   facecolor=QUARTER_FILLS[q % 2], alpha=1.0, zorder=0)

    # ── Build per-game metadata (date + opponent) from first row of each game ──
    game_meta: dict[str, dict] = {}
    for r in rows:
        gid = r["game_id"]
        if gid not in game_meta:
            raw_date = r.get("game_date", "")
            opponent = r.get("opponent", "")
            try:
                from datetime import datetime as _dt
                formatted = _dt.strptime(raw_date.title(), "%b %d, %Y").strftime("%b %d")
            except (ValueError, AttributeError):
                formatted = raw_date[:6] if raw_date else ""
            game_meta[gid] = {"date": formatted, "opponent": opponent}

    # ── Individual game lines — distinct tab10 colours ────────────────────────
    cmap         = plt.cm.tab10
    game_handles = []
    for i, gid in enumerate(game_order):
        colour    = cmap(i % 10)
        total     = int(round(curves[gid][480]))
        meta      = game_meta.get(gid, {})
        date_str  = meta.get("date", f"Game {i + 1}")
        opp_str   = meta.get("opponent", "")
        label     = f"{date_str} vs {opp_str}:  {total} {unit}" if opp_str else f"{date_str}:  {total} {unit}"
        sub_rows  = _get_sub_rows(rows, gid)
        intervals = build_court_intervals(sub_rows)
        _plot_game_segments(ax, x, curves[gid], intervals, colour, 1.6, 0.6, 0.15, 2)
        _plot_sub_markers(ax, sub_rows, curves[gid], colour)
        game_handles.append(mlines.Line2D([], [], color=colour, linewidth=2,
                                          alpha=0.8, label=label))

    # ── All-games average — WHITE, thick ─────────────────────────────────────
    ax.plot(x, all_avg_c, color="white", linewidth=3, zorder=4,
            label=f"All games avg ({n_total} games)  —  {season_avg} {unit} avg")
    mx, my = step_up_markers(all_avg_c, x)
    ax.scatter(mx, my, color="white", s=55, zorder=5, marker="o", edgecolors="#0d1b2a", linewidths=0.8)

    # ── Shown-games average — stat colour, thick dashed ──────────────────────
    recent_label = f"Last {n_games} games avg" if n_games < n_total else f"All {n_games} games avg"
    ax.plot(x, recent_avg_c, color=cfg["color"], linewidth=3,
            linestyle="--", zorder=4, label=recent_label)
    mx2, my2 = step_up_markers(recent_avg_c, x)
    ax.scatter(mx2, my2, color=cfg["color"], s=55, zorder=5, marker="o",
               edgecolors="#0d1b2a", linewidths=0.8)

    # ── Season avg horizontal dashed line ────────────────────────────────────
    ax.axhline(y=season_avg, color="#ff6b6b", linewidth=1.5,
               linestyle="--", zorder=3, alpha=0.85,
               label=f"Season {cfg['per_game']} avg  ({season_avg} {unit})")

    # ── Quarter labels inside each region ────────────────────────────────────
    y_max = max(max(curves[g]) for g in game_order) * 1.05
    ax.set_ylim(0, y_max)
    for q, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        ax.text(q * 12 + 6, y_max * 0.96, label,
                color="white", fontsize=22, fontweight="bold",
                ha="center", va="top", zorder=6,
                alpha=0.85)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xticklabels(["0", "6", "12", "18", "24", "30", "36", "42", "48"],
                       color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=13)
    ax.set_xlabel("Game Minute", fontsize=15, color="white", labelpad=10)
    ax.set_ylabel(f"Cumulative {cfg['label']}", fontsize=15, color="white", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color("#334466")

    # ── Legend — outside the plot on the right ────────────────────────────────
    # Split legend: average/ppg lines on top, then individual games below
    avg_handles, avg_labels = ax.get_legend_handles_labels()
    # avg_handles contains: all-avg, recent-avg, ppg-hline
    all_handles = avg_handles + game_handles
    all_labels  = avg_labels  + [h.get_label() for h in game_handles]

    legend = ax.legend(
        handles=all_handles, labels=all_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        facecolor="#112233",
        edgecolor="#334466",
        labelcolor="white",
        fontsize=12,
        title=f"{player_name}  ·  {season}",
        title_fontsize=13,
        framealpha=0.95,
        handlelength=2.5,
    )
    legend.get_title().set_color("white")

    # ── Title ─────────────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%b %d, %Y")
    shown_note = f"Showing last {n_games} of {n_total}" if n_games < n_total else f"{n_games} games"
    ax.set_title(
        f"{player_name}  ·  {season}  —  Cumulative {cfg['label']} by Minute\n"
        f"{shown_note}  |  Generated {date_str}  |  {cfg['flat_note']}",
        color="white", fontsize=17, pad=18, fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.savefig(str(out_path).replace(".png", ".svg"),
                format="svg", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    _safe_print(f"  Saved -> {out_path}")
    os.startfile(str(out_path).replace(".png", ".svg"))


# ── Team matching helper ──────────────────────────────────────────────────────

# Common full-name / city / nickname → abbreviation mappings
_TEAM_ALIASES: dict[str, str] = {
    "atlanta hawks": "ATL", "hawks": "ATL", "atlanta": "ATL",
    "boston celtics": "BOS", "celtics": "BOS", "boston": "BOS",
    "brooklyn nets": "BKN", "nets": "BKN", "brooklyn": "BKN",
    "charlotte hornets": "CHA", "hornets": "CHA", "charlotte": "CHA",
    "chicago bulls": "CHI", "bulls": "CHI", "chicago": "CHI",
    "cleveland cavaliers": "CLE", "cavaliers": "CLE", "cavs": "CLE", "cleveland": "CLE",
    "dallas mavericks": "DAL", "mavericks": "DAL", "mavs": "DAL", "dallas": "DAL",
    "denver nuggets": "DEN", "nuggets": "DEN", "denver": "DEN",
    "detroit pistons": "DET", "pistons": "DET", "detroit": "DET",
    "golden state warriors": "GSW", "warriors": "GSW", "golden state": "GSW",
    "houston rockets": "HOU", "rockets": "HOU", "houston": "HOU",
    "indiana pacers": "IND", "pacers": "IND", "indiana": "IND",
    "los angeles clippers": "LAC", "clippers": "LAC",
    "los angeles lakers": "LAL", "lakers": "LAL",
    "memphis grizzlies": "MEM", "grizzlies": "MEM", "memphis": "MEM",
    "miami heat": "MIA", "heat": "MIA", "miami": "MIA",
    "milwaukee bucks": "MIL", "bucks": "MIL", "milwaukee": "MIL",
    "minnesota timberwolves": "MIN", "timberwolves": "MIN", "wolves": "MIN", "minnesota": "MIN",
    "new orleans pelicans": "NOP", "pelicans": "NOP", "new orleans": "NOP",
    "new york knicks": "NYK", "knicks": "NYK", "new york": "NYK",
    "oklahoma city thunder": "OKC", "thunder": "OKC", "oklahoma city": "OKC", "okc": "OKC",
    "orlando magic": "ORL", "magic": "ORL", "orlando": "ORL",
    "philadelphia 76ers": "PHI", "76ers": "PHI", "sixers": "PHI", "philadelphia": "PHI",
    "phoenix suns": "PHX", "suns": "PHX", "phoenix": "PHX",
    "portland trail blazers": "POR", "trail blazers": "POR", "blazers": "POR", "portland": "POR",
    "sacramento kings": "SAC", "kings": "SAC", "sacramento": "SAC",
    "san antonio spurs": "SAS", "spurs": "SAS", "san antonio": "SAS",
    "toronto raptors": "TOR", "raptors": "TOR", "toronto": "TOR",
    "utah jazz": "UTA", "jazz": "UTA", "utah": "UTA",
    "washington wizards": "WAS", "wizards": "WAS", "washington": "WAS",
}


def resolve_abbrev(user_input: str) -> str | None:
    """Return the 2-3 letter team abbreviation from any common name/alias."""
    key = user_input.strip().lower()
    # Direct abbreviation (e.g. "LAL")
    if key.upper() in _TEAM_ALIASES.values():
        return key.upper()
    return _TEAM_ALIASES.get(key)


def format_date(raw: str) -> str:
    """'MAR 19, 2026' -> 'Mar 19'"""
    try:
        return datetime.strptime(raw.title(), "%b %d, %Y").strftime("%b %d")
    except (ValueError, AttributeError):
        return raw[:6] if raw else ""


# ── VS-TEAM chart ─────────────────────────────────────────────────────────────

def plot_vs_team_scoring(rows: list[dict], player_name: str, season: str,
                          team_input: str, season_ppg: float,
                          csv_stem: str, out_dir: Path,
                          stat_type: str = "PTS"):
    cfg  = STAT_CONFIG[stat_type]
    unit = cfg["unit"]

    abbrev = resolve_abbrev(team_input)
    if abbrev is None:
        # Try the NBA API resolver as well
        if HAS_NBA_API:
            abbrev = _resolve_team_abbrev(team_input)
        if abbrev is None:
            print(f"  Team '{team_input}' not recognised. Skipping VS TEAM chart.")
            return

    # Filter rows to only games vs this team
    vs_rows = [r for r in rows if r.get("opponent", "").upper() == abbrev]

    # Also look for a dedicated vs-team CSV (e.g. _vs_BKN_events.csv)
    vs_csv_candidates = list(DATA_DIR.glob(f"*_vs_{abbrev}_events.csv"))
    for vp in vs_csv_candidates:
        extra_rows = load_csv(vp)
        existing_gids = {r["game_id"] for r in vs_rows}
        for r in extra_rows:
            if r.get("opponent", "").upper() == abbrev and r["game_id"] not in existing_gids:
                vs_rows.append(r)

    # If no CSV data found, fetch directly from NBA API
    if not any(r["type"] == stat_type for r in vs_rows):
        if HAS_NBA_API:
            print(f"\n  No vs-{abbrev} data in CSV. Fetching from NBA API...")
            api_rows, api_name, api_season, api_ppg = fetch_vs_team_data_from_api(
                player_name, team_input
            )
            if api_rows:
                vs_rows = api_rows
                if api_name:
                    player_name = api_name
                if api_season:
                    season = api_season
                if api_ppg > 0 and stat_type == "PTS":
                    season_ppg = api_ppg
            else:
                print(f"  No games found vs {abbrev}.")
                return
        else:
            print(f"  No games found vs {abbrev} in CSV data.")
            print(f"  Install nba_api to fetch data live: pip install nba_api")
            return

    # Find unique game_ids that had stat events vs this team
    seen_games: dict[str, list] = {}
    for r in vs_rows:
        if r["type"] != stat_type:
            continue
        gid = r["game_id"]
        if gid not in seen_games:
            seen_games[gid] = []
        seen_games[gid].append(r)

    n_vs = len(seen_games)
    if n_vs == 0:
        print(f"  No {stat_type} events found vs {abbrev}.")
        return

    # Oldest-first
    game_order = list(reversed(list(seen_games.keys())))
    curves     = {gid: build_cumulative_curve(seen_games[gid], stat_type) for gid in game_order}
    x          = [i / 10 for i in range(481)]

    vs_avg_c  = avg_curve(curves, game_order)
    vs_avg    = round(vs_avg_c[480], 1)
    diff      = round(vs_avg - season_ppg, 1)
    diff_str  = f"+{diff}" if diff >= 0 else str(diff)

    # Per-game metadata
    game_meta: dict[str, dict] = {}
    for r in vs_rows:
        gid = r["game_id"]
        if gid not in game_meta:
            game_meta[gid] = {
                "date": format_date(r.get("game_date", "")),
                "opponent": r.get("opponent", ""),
            }

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    for q in range(4):
        ax.axvspan(q * 12, (q + 1) * 12,
                   facecolor=QUARTER_FILLS[q % 2], alpha=1.0, zorder=0)

    cmap         = plt.cm.tab10
    game_handles = []
    for i, gid in enumerate(game_order):
        colour    = cmap(i % 10)
        total     = int(round(curves[gid][480]))
        meta      = game_meta.get(gid, {})
        date_str  = meta.get("date", f"Game {i + 1}")
        label     = f"{date_str} vs {abbrev}:  {total} {unit}"
        sub_rows  = _get_sub_rows(vs_rows, gid)
        intervals = build_court_intervals(sub_rows)
        _plot_game_segments(ax, x, curves[gid], intervals, colour, 1.8, 0.6, 0.15, 2)
        _plot_sub_markers(ax, sub_rows, curves[gid], colour)
        game_handles.append(mlines.Line2D([], [], color=colour, linewidth=2,
                                          alpha=0.85, label=label))

    # VS-team average — WHITE
    ax.plot(x, vs_avg_c, color="white", linewidth=3, zorder=4,
            label=f"Avg vs {abbrev} ({n_vs} games)  —  {vs_avg} {unit}")
    mx, my = step_up_markers(vs_avg_c, x)
    ax.scatter(mx, my, color="white", s=55, zorder=5, marker="o",
               edgecolors="#0d1b2a", linewidths=0.8)

    # Season avg reference line
    ax.axhline(y=season_ppg, color="#ff6b6b", linewidth=1.5, linestyle="--",
               zorder=3, alpha=0.85, label=f"Season {cfg['per_game']} avg  ({season_ppg} {unit})")

    # ── Stats text box ────────────────────────────────────────────────────────
    box_lines = [
        f"vs {abbrev}  —  {season}",
        f"Games played:  {n_vs}",
        f"{cfg['per_game']} vs {abbrev}:  {vs_avg}",
        f"Season {cfg['per_game']}:      {season_ppg}",
        f"Difference:       {diff_str} {unit}",
    ]
    box_text = "\n".join(box_lines)
    ax.text(0.01, 0.99, box_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            color="white", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#112233",
                      edgecolor="#334466", alpha=0.92),
            zorder=7)

    # ── Quarter labels ────────────────────────────────────────────────────────
    y_max = max(max(curves[g]) for g in game_order) * 1.05
    ax.set_ylim(0, y_max)
    for q, ql in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        ax.text(q * 12 + 6, y_max * 0.96, ql,
                color="white", fontsize=22, fontweight="bold",
                ha="center", va="top", zorder=6, alpha=0.85)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xticklabels(["0", "6", "12", "18", "24", "30", "36", "42", "48"],
                       color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=13)
    ax.set_xlabel("Game Minute", fontsize=15, color="white", labelpad=10)
    ax.set_ylabel(f"Cumulative {cfg['label']}", fontsize=15, color="white", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color("#334466")

    # ── Legend ────────────────────────────────────────────────────────────────
    avg_handles, avg_labels = ax.get_legend_handles_labels()
    all_handles = avg_handles + game_handles
    all_labels  = avg_labels  + [h.get_label() for h in game_handles]
    legend = ax.legend(
        handles=all_handles, labels=all_labels,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0, facecolor="#112233", edgecolor="#334466",
        labelcolor="white", fontsize=12,
        title=f"{player_name}  vs  {abbrev}",
        title_fontsize=13, framealpha=0.95, handlelength=2.5,
    )
    legend.get_title().set_color("white")

    # ── Title ─────────────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%b %d, %Y")
    ax.set_title(
        f"{player_name}  ·  {season}  —  Cumulative {cfg['label']} vs {abbrev}\n"
        f"{n_vs} games  |  {vs_avg} {cfg['per_game']} vs {abbrev}  ({diff_str} vs season avg)  |  Generated {date_str}",
        color="white", fontsize=17, pad=18, fontweight="bold",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = out_dir / f"{csv_stem}_vs_{abbrev}{cfg['suffix']}_cumulative.png"
    plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.savefig(str(out_path).replace(".png", ".svg"),
                format="svg", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    _safe_print(f"  Saved -> {out_path}")
    os.startfile(str(out_path).replace(".png", ".svg"))


# ── H2H API fetch ─────────────────────────────────────────────────────────────

def fetch_h2h_data_from_api(player1_name: str, player2_name: str,
                             n: int) -> tuple[list[dict], list[dict], str, str, str, str]:
    """
    Fetch head-to-head data for two players from NBA API.
    Returns (p1_rows, p2_rows, p1_true_name, p2_true_name, season, shared_msg).
    Each rows list mimics the CSV row format used elsewhere in charts.py.
    """
    if not HAS_NBA_API:
        print("  nba_api not installed. Cannot fetch live data.")
        print("  Run: pip install nba_api")
        return [], [], player1_name, player2_name, "", ""

    season = _current_nba_season()

    pid1, true1 = _get_player_id(player1_name)
    pid2, true2 = _get_player_id(player2_name)
    ln1 = _extract_last_name(true1)
    ln2 = _extract_last_name(true2)

    _safe_print(f"  Fetching game logs for {true1} and {true2} ({season})...")
    h1, r1 = _fetch_game_log(pid1, season)
    h2, r2 = _fetch_game_log(pid2, season)

    if not r1:
        _safe_print(f"  No games found for {true1} in {season}.")
        return [], [], true1, true2, season, ""
    if not r2:
        _safe_print(f"  No games found for {true2} in {season}.")
        return [], [], true1, true2, season, ""

    # Find shared game IDs
    col1 = {h: i for i, h in enumerate(h1)}
    col2 = {h: i for i, h in enumerate(h2)}
    ids1 = {r[col1["Game_ID"]] for r in r1}
    ids2 = {r[col2["Game_ID"]] for r in r2}
    shared = ids1 & ids2
    shared_ids = [r[col1["Game_ID"]] for r in r1 if r[col1["Game_ID"]] in shared]

    if not shared_ids:
        _safe_print(f"  No shared games found between {true1} and {true2} in {season}.")
        return [], [], true1, true2, season, ""

    game_ids = shared_ids[:n]
    if len(shared_ids) < n:
        shared_msg = f"Found {len(shared_ids)} shared games (requested {n}) — showing all {len(game_ids)}."
    else:
        shared_msg = f"Found {len(shared_ids)} shared games — showing last {len(game_ids)}."
    _safe_print(f"  {shared_msg}")
    print(f"  Fetching play-by-play for {len(game_ids)} shared games...")

    def _build_rows(player_id, last_name, rows, col):
        csv_rows = []
        for gid in game_ids:
            row = next((r for r in rows if r[col["Game_ID"]] == gid), None)
            if row is None:
                continue
            raw_date = row[col["GAME_DATE"]]
            matchup  = row[col["MATCHUP"]]
            opponent = re.split(r"\s+(?:vs\.|@)\s+", matchup)[-1].strip()
            try:
                evts = _get_game_events(player_id, gid, last_name)
            except Exception as e:
                print(f"    WARNING: Could not fetch PBP for game {gid}: {e}")
                continue
            for ev in evts["pts"]:
                csv_rows.append({
                    "game_id": gid, "game_date": raw_date, "opponent": opponent,
                    "period": ev["period"], "minute": ev["minute"],
                    "type": "PTS", "value": str(ev["points"]), "description": ev["description"],
                })
            for ev in evts["reb"]:
                csv_rows.append({
                    "game_id": gid, "game_date": raw_date, "opponent": opponent,
                    "period": ev["period"], "minute": ev["minute"],
                    "type": "REB", "value": ev["reb_type"], "description": ev["description"],
                })
            for ev in evts["ast"]:
                csv_rows.append({
                    "game_id": gid, "game_date": raw_date, "opponent": opponent,
                    "period": ev["period"], "minute": ev["minute"],
                    "type": "AST", "value": "-", "description": ev["description"],
                })
            for ev in evts["sub"]:
                sub_type = "SUB_IN" if ev["sub_type"] == "IN" else "SUB_OUT"
                csv_rows.append({
                    "game_id": gid, "game_date": raw_date, "opponent": opponent,
                    "period": 0, "minute": ev["minute"],
                    "type": sub_type, "value": ev["sub_type"], "description": "",
                })
        return csv_rows

    p1_rows = _build_rows(pid1, ln1, r1, col1)
    p2_rows = _build_rows(pid2, ln2, r2, col2)

    total = len(p1_rows) + len(p2_rows)
    print(f"  Fetched {total} events across {len(game_ids)} shared games.")
    return p1_rows, p2_rows, true1, true2, season, shared_msg


# ── VS-PLAYER overlay chart ──────────────────────────────────────────────────

def plot_vs_player_scoring(rows1: list[dict], rows2: list[dict],
                            name1: str, name2: str, season: str,
                            stat_type: str, last_n: int | None,
                            out_path: Path):
    """
    Overlay chart: both players' cumulative curves on the same axes.
    rows1/rows2 must already be filtered to shared game IDs only.
    """
    cfg  = STAT_CONFIG[stat_type]
    unit = cfg["unit"]

    # ── Group events by game for each player ─────────────────────────────────
    def _group(rows):
        games: dict[str, list] = {}
        for r in rows:
            if r["type"] != stat_type:
                continue
            gid = r["game_id"]
            if gid not in games:
                games[gid] = []
            games[gid].append(r)
        return games

    games1 = _group(rows1)
    games2 = _group(rows2)

    # Use the union of game IDs (oldest-first)
    all_gids = list(dict.fromkeys(
        list(reversed(list(games1.keys()))) +
        list(reversed(list(games2.keys())))
    ))

    if not all_gids:
        print(f"  No {stat_type} events found for either player.")
        return

    # Apply last_n filter
    n_total = len(all_gids)
    if last_n and last_n > n_total:
        print(f"  Note: Only {n_total} shared games available (requested {last_n}). Showing all {n_total}.")
    game_order = all_gids[-last_n:] if (last_n and last_n < n_total) else all_gids
    n_games    = len(game_order)
    x          = [i / 10 for i in range(481)]

    # Build curves
    curves1 = {gid: build_cumulative_curve(games1.get(gid, []), stat_type) for gid in game_order}
    curves2 = {gid: build_cumulative_curve(games2.get(gid, []), stat_type) for gid in game_order}

    gids1 = [g for g in game_order if g in games1]
    gids2 = [g for g in game_order if g in games2]
    avg1 = avg_curve(curves1, gids1) if gids1 else [0.0] * 481
    avg2 = avg_curve(curves2, gids2) if gids2 else [0.0] * 481

    # Per-game metadata from rows1 (both share game IDs)
    game_meta: dict[str, dict] = {}
    for r in list(rows1) + list(rows2):
        gid = r["game_id"]
        if gid not in game_meta:
            raw_date = r.get("game_date", "")
            opponent = r.get("opponent", "")
            try:
                formatted = datetime.strptime(raw_date.title(), "%b %d, %Y").strftime("%b %d")
            except (ValueError, AttributeError):
                formatted = raw_date[:6] if raw_date else ""
            game_meta[gid] = {"date": formatted, "opponent": opponent}

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    for q in range(4):
        ax.axvspan(q * 12, (q + 1) * 12,
                   facecolor=QUARTER_FILLS[q % 2], alpha=1.0, zorder=0)

    # ── Player 1 individual lines (blue tones) ──────────────────────────────
    cmap1 = plt.cm.Blues_r
    p1_handles = []
    for i, gid in enumerate(game_order):
        if gid not in games1:
            continue
        colour = cmap1(0.25 + 0.5 * i / max(n_games - 1, 1))
        total  = int(round(curves1[gid][480]))
        meta   = game_meta.get(gid, {})
        date_s = meta.get("date", f"Game {i + 1}")
        opp    = meta.get("opponent", "")
        label  = f"{date_s} vs {opp}:  {total} {unit}" if opp else f"{date_s}:  {total} {unit}"
        sub_rows  = _get_sub_rows(rows1, gid)
        intervals = build_court_intervals(sub_rows)
        _plot_game_segments(ax, x, curves1[gid], intervals, colour, 1.4, 0.45, 0.15, 2)
        _plot_sub_markers(ax, sub_rows, curves1[gid], colour)
        p1_handles.append(mlines.Line2D([], [], color=colour, linewidth=2,
                                        alpha=0.7, label=label))

    # ── Player 2 individual lines (red/orange tones) ─────────────────────────
    cmap2 = plt.cm.Reds_r
    p2_handles = []
    for i, gid in enumerate(game_order):
        if gid not in games2:
            continue
        colour = cmap2(0.25 + 0.5 * i / max(n_games - 1, 1))
        total  = int(round(curves2[gid][480]))
        meta   = game_meta.get(gid, {})
        date_s = meta.get("date", f"Game {i + 1}")
        opp    = meta.get("opponent", "")
        label  = f"{date_s} vs {opp}:  {total} {unit}" if opp else f"{date_s}:  {total} {unit}"
        sub_rows  = _get_sub_rows(rows2, gid)
        intervals = build_court_intervals(sub_rows)
        _plot_game_segments(ax, x, curves2[gid], intervals, colour, 1.4, 0.45, 0.15, 2)
        _plot_sub_markers(ax, sub_rows, curves2[gid], colour)
        p2_handles.append(mlines.Line2D([], [], color=colour, linewidth=2,
                                        alpha=0.7, label=label))

    # ── Player 1 average — bright blue ───────────────────────────────────────
    p1_avg_total = round(avg1[480], 1)
    ax.plot(x, avg1, color="#00aaff", linewidth=3, zorder=4,
            label=f"{name1} avg  —  {p1_avg_total} {unit}")
    mx1, my1 = step_up_markers(avg1, x)
    ax.scatter(mx1, my1, color="#00aaff", s=55, zorder=5, marker="o",
               edgecolors="#0d1b2a", linewidths=0.8)

    # ── Player 2 average — red ───────────────────────────────────────────────
    p2_avg_total = round(avg2[480], 1)
    ax.plot(x, avg2, color="#ff4444", linewidth=3, zorder=4,
            label=f"{name2} avg  —  {p2_avg_total} {unit}")
    mx2, my2 = step_up_markers(avg2, x)
    ax.scatter(mx2, my2, color="#ff4444", s=55, zorder=5, marker="o",
               edgecolors="#0d1b2a", linewidths=0.8)

    # ── Quarter labels ───────────────────────────────────────────────────────
    all_curves = list(curves1.values()) + list(curves2.values())
    y_max = max(max(c) for c in all_curves) * 1.05 if all_curves else 10
    ax.set_ylim(0, y_max)
    for q, ql in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        ax.text(q * 12 + 6, y_max * 0.96, ql,
                color="white", fontsize=22, fontweight="bold",
                ha="center", va="top", zorder=6, alpha=0.85)

    # ── Axes formatting ──────────────────────────────────────────────────────
    ax.set_xlim(0, 48)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xticklabels(["0", "6", "12", "18", "24", "30", "36", "42", "48"],
                       color="white", fontsize=13)
    ax.tick_params(colors="white", labelsize=13)
    ax.set_xlabel("Game Minute", fontsize=15, color="white", labelpad=10)
    ax.set_ylabel(f"Cumulative {cfg['label']}", fontsize=15, color="white", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color("#334466")

    # ── Legend — split by player ─────────────────────────────────────────────
    avg_handles, avg_labels = ax.get_legend_handles_labels()

    # Section header handles (invisible lines used as group titles)
    p1_header = mlines.Line2D([], [], color="none", label=f"── {name1} ──")
    p2_header = mlines.Line2D([], [], color="none", label=f"── {name2} ──")

    all_handles = avg_handles + [p1_header] + p1_handles + [p2_header] + p2_handles
    all_labels  = avg_labels + [p1_header.get_label()] + [h.get_label() for h in p1_handles] \
                  + [p2_header.get_label()] + [h.get_label() for h in p2_handles]

    legend = ax.legend(
        handles=all_handles, labels=all_labels,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0, facecolor="#112233", edgecolor="#334466",
        labelcolor="white", fontsize=11,
        title=f"{name1}  vs  {name2}  ·  {season}",
        title_fontsize=13, framealpha=0.95, handlelength=2.5,
    )
    legend.get_title().set_color("white")

    # ── Title ────────────────────────────────────────────────────────────────
    date_str = datetime.now().strftime("%b %d, %Y")
    ax.set_title(
        f"{name1}  vs  {name2}  ·  {season}  —  Cumulative {cfg['label']} by Minute\n"
        f"Last {n_games} shared games  |  Generated {date_str}  |  {cfg['flat_note']}",
        color="white", fontsize=17, pad=18, fontweight="bold",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    plt.savefig(str(out_path), dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.savefig(str(out_path).replace(".png", ".svg"),
                format="svg", facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.3)
    plt.close()
    _safe_print(f"  Saved -> {out_path}")
    os.startfile(str(out_path).replace(".png", ".svg"))


# ── Season PPG helper ─────────────────────────────────────────────────────────

def _compute_season_avg(rows: list[dict], stat_type: str = "PTS") -> float:
    """Compute average per-game stat from event rows."""
    games_totals: dict[str, int] = {}
    for r in rows:
        if r["type"] != stat_type:
            continue
        gid = r["game_id"]
        if stat_type == "PTS":
            try:
                games_totals[gid] = games_totals.get(gid, 0) + int(r["value"])
            except ValueError:
                pass
        else:
            games_totals[gid] = games_totals.get(gid, 0) + 1
    return round(sum(games_totals.values()) / len(games_totals), 1) if games_totals else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def _require_api():
    if not HAS_NBA_API:
        print("  nba_api not installed. Run: pip install nba_api")
        raise SystemExit(1)


def _ask_player(prompt: str = "  Player name (e.g. Jalen Brunson): ") -> str:
    name = input(prompt).strip()
    if not name:
        print("  No player entered. Exiting.")
        raise SystemExit(1)
    return name


def _ask_stat_type() -> list[str]:
    """Prompt for stat selection. Returns list of stat_type keys."""
    print("\n  Stat to chart:")
    print("  [1] Points  [2] Assists  [3] Rebounds  [4] All three")
    pick = input("  Pick [default 1]: ").strip()
    if pick == "2":
        return ["AST"]
    if pick == "3":
        return ["REB"]
    if pick == "4":
        return ["PTS", "AST", "REB"]
    return ["PTS"]


def main():
    print("\n" + "=" * 52)
    print("  NBA CHARTS")
    print("=" * 52)

    print("\n  Chart mode:")
    print("  [1] Last N games")
    print("  [2] VS Team")
    print("  [3] VS Player (head-to-head)")
    mode = input("  Pick [default 1]: ").strip()

    stat_types = _ask_stat_type()

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: Last N games
    # ══════════════════════════════════════════════════════════════════════════
    if mode != "2" and mode != "3":
        _require_api()
        player_name = _ask_player()
        n_raw = input("  Last N games [Enter for 10]: ").strip()
        n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 10
        rows, player_name, season = fetch_last_n_data_from_api(player_name, n)
        if not rows:
            return

        n_raw  = input("  How many recent games to show in chart? [Enter for all]: ").strip()
        last_n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else None

        slug = player_name.lower().replace(" ", "_")
        for st in stat_types:
            suffix = STAT_CONFIG[st]["suffix"]
            out_path = CHART_DIR / f"{slug}_{season}{suffix}_cumulative.png"
            plot_cumulative_scoring(rows, player_name, season, last_n, out_path, stat_type=st)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2: VS Team
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "2":
        _require_api()
        player_name = _ask_player()
        team_input = input("  VS Team (e.g. Lakers or LAL): ").strip()
        if not team_input:
            print("  No team entered. Exiting.")
            return
        api_rows, true_name, season, season_ppg = fetch_vs_team_data_from_api(
            player_name, team_input
        )
        if not api_rows:
            return
        slug = true_name.lower().replace(" ", "_")
        for st in stat_types:
            season_avg = season_ppg if st == "PTS" else _compute_season_avg(api_rows, st)
            plot_vs_team_scoring(api_rows, true_name, season, team_input,
                                  season_avg, f"{slug}_{season}", CHART_DIR, stat_type=st)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 3: VS Player (head-to-head)
    # ══════════════════════════════════════════════════════════════════════════
    if mode == "3":
        _require_api()
        player1 = _ask_player("  Player 1 name (e.g. James Harden): ")
        player2 = _ask_player("  Player 2 name (e.g. Stephen Curry): ")
        n_raw = input("  Last N shared games [Enter for 10]: ").strip()
        n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else 10

        p1_rows, p2_rows, name1, name2, season, _ = fetch_h2h_data_from_api(
            player1, player2, n
        )
        if not p1_rows and not p2_rows:
            return

        n_raw  = input("  How many recent games to show in chart? [Enter for all]: ").strip()
        last_n = int(n_raw) if n_raw.isdigit() and int(n_raw) > 0 else None

        slug1 = name1.lower().replace(" ", "_")
        slug2 = name2.lower().replace(" ", "_")

        for st in stat_types:
            suffix = STAT_CONFIG[st]["suffix"]
            out = CHART_DIR / f"{slug1}_vs_{slug2}_{season}{suffix}.png"
            plot_vs_player_scoring(p1_rows, p2_rows, name1, name2, season,
                                   stat_type=st, last_n=last_n, out_path=out)
        return


if __name__ == "__main__":
    main()
