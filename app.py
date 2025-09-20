# app.py
import os
import re
import json
import html
from typing import Optional, Dict

# --- Make HTTPS robust on Windows/local: use certifi CA bundle ---
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import altair as alt
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

from google_sheets_adapter import read_sheet_to_df

# Optional Gemini import (guarded)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="HS Soccer Dashboard", layout="wide")

# Load local .env (for local dev)
load_dotenv()

# Streamlit Secrets fallback (for Cloud)
try:
    SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY") or st.secrets.get("SPREADSHEET_KEY", "YOUR_SPREADSHEET_KEY_OR_URL")
    for _k in ["GEMINI_API_KEY", "APP_PASSWORD", "GOOGLE_SERVICE_ACCOUNT_JSON"]:
        if not os.getenv(_k) and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY", "YOUR_SPREADSHEET_KEY_OR_URL")

# Tiny CSS + mobile polish
def _inject_css():
    st.markdown(
        """
        <style>
          a.tiny-open {
            display:inline-block;
            padding:2px 6px;
            font-size:12px;
            line-height:1;
            border-radius:6px;
            background:#f0f2f6;
            text-decoration:none;
          }
          a.tiny-open:hover { background:#e6e9ef; }

          /* ----- Game cards (existing) ----- */
          .game-card {
            border:1px solid #e6e9ef;
            border-radius:12px;
            padding:12px 14px;
            margin:10px 0 14px;
            background:#ffffff;
            box-shadow:0 1px 2px rgba(0,0,0,0.04);
          }
          .gc-row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
          .gc-date { font-size:0.92rem; color:#6b7280; }
          .gc-opp  { font-weight:600; font-size:1.05rem; }
          .gc-score{ font-weight:700; font-size:1.15rem; white-space:nowrap; }
          .gc-meta { margin-top:6px; display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }
          .pill { padding:2px 8px; border-radius:999px; background:#f0f2f6; font-size:12px; }
          .pill.home { background:#e8f5e9; }
          .pill.away { background:#e3f2fd; }
          .pill.div  { background:#fff7ed; }

          /* ----- KPI cards (NEW) ----- */
          .kpi-grid {
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap:12px;
            margin:6px 0 18px;
          }
          .stat-card {
            border:1px solid #e6e9ef;
            border-radius:12px;
            background:#fff;
            padding:12px 14px;
            box-shadow:0 1px 2px rgba(0,0,0,.04);
          }
          .stat-label { font-size:.85rem; color:#6b7280; margin-bottom:4px; }
          .stat-value { font-size:1.6rem; font-weight:700; line-height:1.1; }
          .stat-sub { font-size:.8rem; color:#6b7280; margin-top:2px; }

          @media (max-width: 480px) {
            .block-container { padding-top: 0.75rem; padding-left: 0.6rem; padding-right: 0.6rem; }
            a.tiny-open { padding:6px 10px; font-size:14px; }
            .stat-value { font-size:1.8rem; }
          }

          /* Print-friendly view: hide sidebar/nav when printing */
          @media print {
            section[data-testid="stSidebar"], header { display: none !important; }
            .block-container { padding: 0 !important; }
            a.tiny-open { display: none !important; }
          }

          /* ----- AI Chat Styling ----- */
          .ai-chat-message {
            background: #f8f9fa;
            border-left: 4px solid #4a90e2;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          }
          .ai-chat-user {
            background: #e3f2fd;
            border-left-color: #2196f3;
          }
          .ai-chat-assistant {
            background: #f3e5f5;
            border-left-color: #9c27b0;
          }
          .ai-quick-action {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            transition: all 0.3s ease;
          }
          .ai-quick-action:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
_inject_css()

# Optional: simple password gate (set APP_PASSWORD in Secrets/env to enable)
def require_app_password():
    pwd = os.getenv("APP_PASSWORD", "").strip()
    if not pwd:
        return  # disabled
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if not st.session_state.authed:
        st.title("Coaches Only")
        entered = st.text_input("Enter password", type="password")
        if st.button("Enter"):
            st.session_state.authed = (entered == pwd)
        st.stop()

require_app_password()

# External sources: SBLive schedule + SBLive rankings
SBLIVE_BASE = "https://www.si.com/high-school/stats/vermont"
SBLIVE_TEAM_SLUG = "397925-milton-yellowjackets"
SBLIVE_SCHEDULE_URL = f"{SBLIVE_BASE}/soccer/teams/{SBLIVE_TEAM_SLUG}/games"
SBLIVE_RANKINGS_URL = "https://www.si.com/high-school/stats/vermont/28806-division-2/soccer/rankings?formula=DIVISION_POINT_INDEX"
TEAM_NAME_CANON = "Milton"

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true","1","yes","y","t"])

def _normalize_set_piece(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    def norm(v: str) -> str:
        if not v or v in ("nan", "none"):
            return ""
        # Penalties (avoid matching "open")
        if v == "pk" or v.startswith("pk ") or v.startswith("pk-") or v.startswith("pk:"):
            return "penalty"
        if v.startswith("pen") or ("penalty" in v):
            return "penalty"
        # Explicit FK labels first
        if "fk_direct" in v:
            return "fk_direct"
        if "fk_indirect" in v:
            return "fk_indirect"
        if v == "dfk":
            return "fk_direct"
        if v == "ifk":
            return "fk_indirect"
        # Numeric shorthand (1 = direct, 2 = indirect)
        trimmed = v.strip()
        if re.match(r"^1(\D|$)", trimmed):
            return "fk_direct"
        if re.match(r"^2(\D|$)", trimmed):
            return "fk_indirect"
        # Corners
        if v in ("ck", "corners", "corner", "corner kick") or v.startswith("corner"):
            return "corner"
        # Direct FK variants
        direct_vals = {"dfk", "direct fk", "fk direct", "direct kick", "direct free kick", "direct"}
        if v in direct_vals or ("direct" in v and "fk" in v):
            return "fk_direct"
        # Indirect FK variants
        indirect_vals = {"ifk", "indirect fk", "fk indirect", "indirect kick", "indirect free kick", "indirect"}
        if v in indirect_vals or ("indirect" in v and "fk" in v):
            return "fk_indirect"
        return v
    return s.map(norm)

def _qparams_get():
    try: return st.query_params
    except Exception: return st.experimental_get_query_params()

def _qparams_set(**kwargs):
    try:
        st.query_params.clear()
        for k,v in kwargs.items(): st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

def _qparams_merge_update(**kwargs):
    """Merge update query params without dropping existing ones."""
    try:
        qp = dict(st.query_params)
    except Exception:
        qp = dict(st.experimental_get_query_params())
    # Flatten list values from experimental API
    qp2 = {k: (v[0] if isinstance(v, list) and v else v) for k,v in qp.items()}
    qp2.update({k: v for k,v in kwargs.items() if v is not None})
    _qparams_set(**qp2)

def _qp_bool(val, default=False) -> bool:
    if val is None: return default
    if isinstance(val, list): val = val[0] if val else ""
    s = str(val).strip().lower()
    return s in ("1","true","t","yes","y","on")

def _format_date(val) -> str:
    ts = pd.to_datetime(val, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%b %d, %Y")

def _result_color(res: str) -> str:
    return {"W":"green","L":"red","D":"goldenrod"}.get(res,"black")

def _result_emoji(res: str) -> str:
    return {"W":"✅","L":"❌","D":"➖"}.get(res,"")

def _status_dot(res: str) -> str:
    return f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{_result_color(res)};'></span>"

def _color_opp(name: str, res: str) -> str:
    safe = str(name) if name is not None else ""
    return f"<span style='color:{_result_color(res)};font-weight:600'>{safe}</span> {_result_emoji(res)}"

def _team_record_text(df: pd.DataFrame) -> str:
    if df.empty or "result" not in df: return "0-0"
    w = int((df["result"]=="W").sum())
    l = int((df["result"]=="L").sum())
    d = int((df["result"]=="D").sum())
    return f"{w}-{l}-{d}" if d>0 else f"{w}-{l}"

def _strip_and_alias_matches(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    if "shots_for" not in df.columns and "shots" in df.columns:
        df = df.rename(columns={"shots":"shots_for"})
    return df

def _suffix(n: int) -> str:
    return {1:"st",2:"nd",3:"rd"}.get(n if n in (1,2,3) else 0, "th")

def _row_as_clean_dict(row: pd.Series) -> Dict[str, str]:
    out = {}
    if row is None:
        return out
    for k, v in row.items():
        if pd.isna(v) or str(v).strip() == "":
            continue
        out[str(k)] = str(v)
    return out

def _minute_bucket(x) -> str:
    """Return a simple 15-min bucket label."""
    try:
        m = float(x)
    except Exception:
        return "N/A"
    if m < 0: return "N/A"
    if m <= 15: return "0-15"
    if m <= 30: return "16-30"
    if m <= 45: return "31-45"
    if m <= 60: return "46-60"
    if m <= 75: return "61-75"
    return "76-90+"

# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_matches() -> pd.DataFrame:
    df = read_sheet_to_df(SPREADSHEET_KEY, "matches")
    df = _strip_and_alias_matches(df)
    if "date" in df: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "division_game" in df: df["division_game"] = _bool_col(df["division_game"])
    if "home_away" in df:
        df["home_away"] = (df["home_away"].astype(str).str.strip().str.lower()
                           .map({"h":"H","home":"H","a":"A","away":"A"}))
    for c in ["goals_for","goals_against","shots_for","shots_against","saves"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if {"goals_for","goals_against"}.issubset(df):
        df["result"] = df.apply(lambda r: "W" if r.goals_for>r.goals_against else ("L" if r.goals_for<r.goals_against else "D"), axis=1)
    if "match_id" not in df: df["match_id"] = df.index.astype(str)
    else: df["match_id"] = df["match_id"].astype(str)
    return df

@st.cache_data(ttl=300)
def load_players() -> pd.DataFrame:
    df = read_sheet_to_df(SPREADSHEET_KEY, "players")
    if "jersey" in df: df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").fillna(0).astype(int)
    if "player_id" in df: df["player_id"] = df["player_id"].astype(str)
    return df

@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    df = read_sheet_to_df(SPREADSHEET_KEY, "events")
    df.columns = [c.strip().lower() for c in df.columns]
    if "assist" in df.columns and "assists" not in df.columns:
        df = df.rename(columns={"assist":"assists"})
    for k in ["event_id","match_id","player_id"]:
        if k in df.columns: df[k] = df[k].astype(str)
    for n in ["goals","assists","shots","fouls"]:
        if n not in df.columns: df[n] = 0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(ttl=300)
def load_plays_simple() -> pd.DataFrame:
    try:
        raw = read_sheet_to_df(SPREADSHEET_KEY, "plays")
    except Exception:
        return pd.DataFrame()
    raw.columns = [c.lower().strip() for c in raw.columns]
    if "play type" in raw and "play_type" not in raw: raw = raw.rename(columns={"play type":"play_type"})
    if "set_piece" in raw:
        sp = raw["set_piece"].astype(str).str.strip().str.lower()
        raw["set_piece"] = sp.replace({"direct":"fk_direct","indirect":"fk_indirect","fk direct":"fk_direct","fk indirect":"fk_indirect"})
    raw["taker_notes"] = raw.get("taker_id","").astype(str).fillna("")
    if "goal_created" in raw:
        raw["goal_created"] = (raw["goal_created"].astype(str).str.strip().str.lower()
                               .map({"true":True,"yes":True,"y":True,"1":True,"no":False,"false":False,"0":False})
                               .fillna(False))
    for k in ["match_id","play_call_id","play_type"]:
        if k in raw: raw[k] = raw[k].astype(str).fillna("").str.strip()
    keep = [c for c in ["match_id","set_piece","play_call_id","play_type","taker_notes","goal_created"] if c in raw]
    return raw[keep]

@st.cache_data(ttl=300)
def load_summaries() -> pd.DataFrame:
    # Support both 'summary' and 'summaries'
    for tab in ("summary", "summaries"):
        try:
            df = read_sheet_to_df(SPREADSHEET_KEY, tab)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "match_id" in df.columns:
                df["match_id"] = df["match_id"].astype(str)
            return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_goals_allowed() -> pd.DataFrame:
    """
    Read 'goals_allowed' tab.
    Expected columns (case-insensitive, flexible):
      - match_id (str)
      - goal_id (str)  [optional]
      - description or description_of_goal (str)  [optional]
      - goalie_player_id / goalkeeper_player_id / goalie (player_id as str)
      - minute (int) [optional]
      - situation (str) [optional: Open Play, Set Piece, CK, FK, PK, etc.]
    """
    try:
        df = read_sheet_to_df(SPREADSHEET_KEY, "goals_allowed")
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "description_of_goal" in df.columns and "description" not in df.columns:
        df = df.rename(columns={"description_of_goal": "description"})
    for cand in ["goalie_player_id","goalkeeper_player_id","goalie"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "goalie_player_id"})
            break
    if "goalie_player_id" not in df.columns:
        df["goalie_player_id"] = ""

    for k in ["match_id","goal_id","goalie_player_id"]:
        if k in df.columns:
            df[k] = df[k].astype(str)
    if "minute" in df.columns:
        df["minute"] = pd.to_numeric(df["minute"], errors="coerce")
    else:
        df["minute"] = pd.NA
    if "situation" not in df.columns:
        df["situation"] = ""
    if "description" not in df.columns:
        df["description"] = ""

    return df

# ---------------------------------------------------------------------
# RANKINGS HELPERS (for D2 Rank KPI)
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def _parse_ranks_from_sblive(html: str) -> Dict[str, int]:
    try:
        m = re.search(r'data-react-props="(.*?)"', html, flags=re.S)
        if not m:
            return {}
        props = json.loads(html.unescape(m.group(1)))
        ranking = props.get("query", {}).get("ranking", {})
        team_rankings = ranking.get("teamRankings") or {}
        if isinstance(team_rankings, dict):
            nodes = team_rankings.get("nodes") or []
        elif isinstance(team_rankings, list):
            nodes = team_rankings
        else:
            nodes = []
        ranks: Dict[str, int] = {}
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            team = entry.get("team") or {}
            name = str(team.get("name", "")).strip()
            if not name:
                continue
            rank_value = (
                entry.get("filteredPlace")
                or entry.get("place")
                or entry.get("rank")
                or entry.get("overallStandingPlacement")
            )
            try:
                rank_int = int(rank_value)
            except (TypeError, ValueError):
                continue
            if rank_int <= 0 or name in ranks:
                continue
            ranks[name] = rank_int
        return ranks
    except Exception:
        return {}


def _clean_text(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&mdash;|&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_all_ranks_from_si(html: str) -> Dict[str, int]:
    if not html:
        return {}
    ranks = _parse_ranks_from_sblive(html)
    if ranks:
        return ranks
    text_clean = _clean_text(html)
    pairs = re.findall(r"\b(\d{1,2})\s+([A-Z][A-Za-z0-9.\-\' ]{2,})", text_clean)
    ranks_fallback: Dict[str, int] = {}
    for num, name in pairs:
        try:
            rank = int(num)
        except Exception:
            continue
        name = name.strip()
        if rank <= 0 or name in ranks_fallback:
            continue
        ranks_fallback[name] = rank
    return ranks_fallback

def fuzzy_find_rank(ranks: Dict[str,int], target: str) -> Optional[int]:
    t = target.lower().strip()
    best = None
    for name, r in ranks.items():
        n = name.lower()
        if t == n or t in n or n in t:
            if best is None or r < best:
                best = r
    return best

# ---------------------------------------------------------------------
# AGGREGATIONS / STATS
# ---------------------------------------------------------------------
def set_piece_leaderboard_from_plays(plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize set-piece attempts by play_call_id, set_piece type, and play_type.
    Expects columns:
      - match_id (str)
      - set_piece (e.g., 'fk_direct', 'fk_indirect', 'corner', 'throw', ...)
      - play_call_id (str)  <-- used as the grouping key/name of the call
      - play_type (free text, optional)
      - goal_created (bool) <-- True if the play directly created a goal
    """
    if plays_df.empty or "play_call_id" not in plays_df.columns:
        return pd.DataFrame(columns=["set_piece", "Play Call", "play_type", "attempts", "Goals", "Goal%"])

    grp = (
        plays_df.groupby(["play_call_id", "set_piece", "play_type"], dropna=False)
        .agg(
            attempts=("play_call_id", "count"),
            goals=("goal_created", "sum"),
            goal_rate=("goal_created", "mean"),
        )
        .reset_index()
    )
    grp["goals"] = grp["goals"].fillna(0).astype(int)
    grp["Goal%"] = (grp["goal_rate"] * 100).round(1)

    out = grp.rename(columns={"play_call_id": "Play Call", "goals": "Goals"})
    # Sort primarily by attempts (desc) to surface most-used plays
    cols = ["set_piece", "Play Call", "play_type", "attempts", "Goals", "Goal%"]
    ordered = out[cols].sort_values(["Goal%", "attempts", "Play Call"], ascending=[False, False, True])
    return ordered

def build_trend_frame(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame()
    df = matches.sort_values("date").copy()
    df["GF"] = df.get("goals_for", 0)
    df["GA"] = df.get("goals_against", 0)

    sv  = df.get("saves", pd.Series([0]*len(df)))
    shf = df.get("shots_for", pd.Series([0]*len(df)))
    sha = df.get("shots_against", pd.Series([0]*len(df)))

    denom_sv = sv + df["GA"]
    df["Save%"] = (sv / denom_sv * 100).where(denom_sv > 0, 0.0)
    df["GF Conv%"] = (df["GF"] / shf * 100).where(shf > 0, 0.0)
    df["GA Conv%"] = (df["GA"] / sha * 100).where(sha > 0, 0.0)

    roll = df[["GF", "GA", "Save%", "GF Conv%", "GA Conv%"]].rolling(3, min_periods=1).mean()
    for c in roll.columns:
        df[f"R3 {c}"] = roll[c]
    df["Date"] = df["date"]
    return df

def build_comparison_trend_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a comparison frame showing all games vs last 3 games metrics."""
    if matches.empty:
        return pd.DataFrame()
    
    df = matches.sort_values("date").copy()
    df["GF"] = df.get("goals_for", 0)
    df["GA"] = df.get("goals_against", 0)

    sv  = df.get("saves", pd.Series([0]*len(df)))
    shf = df.get("shots_for", pd.Series([0]*len(df)))
    sha = df.get("shots_against", pd.Series([0]*len(df)))

    denom_sv = sv + df["GA"]
    df["Save%"] = (sv / denom_sv * 100).where(denom_sv > 0, 0.0)
    df["GF Conv%"] = (df["GF"] / shf * 100).where(shf > 0, 0.0)
    df["GA Conv%"] = (df["GA"] / sha * 100).where(sha > 0, 0.0)

    # Calculate season averages (all games)
    season_avg = {
        "GF": df["GF"].mean(),
        "GA": df["GA"].mean(),
        "Save%": df["Save%"].mean(),
        "GF Conv%": df["GF Conv%"].mean(),
        "GA Conv%": df["GA Conv%"].mean()
    }

    # Calculate last 3 games averages
    last_3_avg = {}
    if len(df) >= 3:
        last_3 = df.tail(3)
        last_3_avg = {
            "GF": last_3["GF"].mean(),
            "GA": last_3["GA"].mean(),
            "Save%": last_3["Save%"].mean(),
            "GF Conv%": last_3["GF Conv%"].mean(),
            "GA Conv%": last_3["GA Conv%"].mean()
        }
    else:
        # If less than 3 games, use all available games
        last_3_avg = season_avg.copy()

    # Create comparison data
    comparison_data = []
    for metric in ["GF", "GA", "Save%", "GF Conv%", "GA Conv%"]:
        comparison_data.append({
            "Metric": metric,
            "All Games": season_avg[metric],
            "Last 3 Games": last_3_avg[metric],
            "Difference": last_3_avg[metric] - season_avg[metric]
        })

    return pd.DataFrame(comparison_data)

def build_individual_game_trends(matches: pd.DataFrame) -> pd.DataFrame:
    """Build individual game data points for trend analysis."""
    if matches.empty:
        return pd.DataFrame()
    
    df = matches.sort_values("date").copy()
    df["GF"] = df.get("goals_for", 0)
    df["GA"] = df.get("goals_against", 0)

    sv  = df.get("saves", pd.Series([0]*len(df)))
    shf = df.get("shots_for", pd.Series([0]*len(df)))
    sha = df.get("shots_against", pd.Series([0]*len(df)))

    denom_sv = sv + df["GA"]
    df["Save%"] = (sv / denom_sv * 100).where(denom_sv > 0, 0.0)
    df["GF Conv%"] = (df["GF"] / shf * 100).where(shf > 0, 0.0)
    df["GA Conv%"] = (df["GA"] / sha * 100).where(sha > 0, 0.0)

    # Add game number and opponent info
    df["Game #"] = range(1, len(df) + 1)
    df["Opponent"] = df.get("opponent", "")
    df["Date"] = df["date"]
    
    # Mark last 3 games
    df["Last 3 Games"] = df.index >= (len(df) - 3)
    
    return df[["Game #", "Date", "Opponent", "GF", "GA", "Save%", "GF Conv%", "GA Conv%", "Last 3 Games"]]

# --- AI: match summary ---
def generate_ai_game_summary(match_row: pd.Series,
                             notes_row: Optional[pd.Series],
                             events: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        gf = int(match_row.get("goals_for", 0))
        ga = int(match_row.get("goals_against", 0))
        shots = int(match_row.get("shots_for", match_row.get("shots", 0)))
        saves = int(match_row.get("saves", 0))
        result = str(match_row.get("result", ""))
        opp = str(match_row.get("opponent", ""))
        ha = str(match_row.get("home_away", ""))
        date_txt = ""
        try:
            date_txt = pd.to_datetime(match_row.get("date")).strftime("%b %d, %Y")
        except Exception:
            pass

        coach_bits = _row_as_clean_dict(notes_row)

        sys = (
            "You are an assistant soccer analyst writing a concise match recap for coaches. "
            "Use short, plain English sentences, avoid fluff, and keep it to ~120-160 words. "
            "Be neutral and constructive. Include 1-2 actionable coaching takeaways."
        )
        user = {
            "context": {
                "match": {
                    "date": date_txt, "opponent": opp, "home_away": ha,
                    "result": result, "score": f"{gf}-{ga}",
                    "shots_for": shots, "saves": saves,
                },
                "coach_notes": coach_bits
            },
            "instructions": [
                "Open with result and score.",
                "Add one line on chance creation/shot quality if relevant.",
                "Mention formations/key dynamics if notes provided.",
                "Name our Player of the Game if provided.",
                "End with 1-2 concrete takeaways for training/prep.",
            ],
        }

        resp = model.generate_content([sys, str(user)])
        text = getattr(resp, "text", "").strip()
        return text or None
    except Exception:
        return None

# --- AI: conceded goals summary ---
def generate_ai_conceded_summary(ga_df: pd.DataFrame,
                                 matches: pd.DataFrame,
                                 players: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        pl = players.set_index("player_id") if "player_id" in players.columns else pd.DataFrame()
        mx = matches.set_index("match_id") if "match_id" in matches.columns else pd.DataFrame()

        tmp = ga_df.copy()
        if not pl.empty and "goalie_player_id" in tmp.columns:
            tmp["goalie_name"] = tmp["goalie_player_id"].map(
                lambda pid: pl.at[str(pid), "name"] if str(pid) in pl.index else ""
            )
        else:
            tmp["goalie_name"] = ""

        if not mx.empty and "match_id" in tmp.columns:
            tmp["opponent"] = tmp["match_id"].map(lambda mid: mx.at[str(mid), "opponent"] if str(mid) in mx.index else "")
            tmp["date"] = tmp["match_id"].map(lambda mid: mx.at[str(mid), "date"] if str(mid) in mx.index else "")
        else:
            tmp["opponent"] = ""
            tmp["date"] = ""

        tmp["minute_bucket"] = tmp["minute"].apply(_minute_bucket)
        by_situation = tmp["situation"].fillna("").str.title().replace({"": "Unspecified"}).value_counts().to_dict()
        by_bucket = tmp["minute_bucket"].value_counts().to_dict()
        by_goalie = tmp["goalie_name"].fillna("").replace({"": "Unspecified"}).value_counts().to_dict()

        context = {
            "total_goals_allowed": int(len(ga_df)),
            "by_situation": by_situation,
            "by_minute_bucket": by_bucket,
            "by_goalie": by_goalie,
        }

        prompt = (
            "You are a soccer defensive analyst. Review the conceded goals profile and give a brief, "
            "coach-friendly summary (120-160 words max) with 3-5 concrete actions. "
            "Focus on patterns (set pieces, late goals, specific minute windows, keeper load) and training priorities. "
            "Avoid jargon. Keep it practical.\n\n"
            f"DATA: {context}"
        )

        resp = model.generate_content(prompt)
        return getattr(resp, "text", "").strip() or None
    except Exception:
        return None

# --- AI: General team analysis and Q&A ---
def generate_ai_team_analysis(query: str,
                             matches: pd.DataFrame,
                             players: pd.DataFrame,
                             events: pd.DataFrame,
                             plays_df: pd.DataFrame,
                             goals_allowed: pd.DataFrame) -> Optional[str]:
    """Generate AI analysis based on user query about team performance."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Prepare comprehensive team data
        # Build top scorers with player names, not IDs
        top_scorers = []
        if not events.empty:
            ev_norm = events.copy()
            ev_norm.columns = [c.strip().lower() for c in ev_norm.columns]
            if "assist" in ev_norm.columns and "assists" not in ev_norm.columns:
                ev_norm = ev_norm.rename(columns={"assist": "assists"})
            agg = ev_norm.groupby("player_id", as_index=False)[[c for c in ["goals","assists"] if c in ev_norm.columns]].sum()
            # map names
            pl_map = {}
            if not players.empty and "player_id" in players.columns and "name" in players.columns:
                tmp = players[["player_id","name"]].copy()
                tmp["player_id"] = tmp["player_id"].astype(str)
                pl_map = dict(zip(tmp["player_id"], tmp["name"].astype(str)))
            agg["name"] = agg["player_id"].astype(str).map(pl_map).fillna(agg["player_id"].astype(str))
            top_scorers = agg.sort_values("goals", ascending=False).head(5)[["name","goals","assists"]].to_dict("records")

        team_data = {
            "matches": {
                "total_games": len(matches),
                "record": _team_record_text(matches),
                "goals_for": int(matches.get("goals_for", pd.Series(dtype=int)).sum()) if not matches.empty else 0,
                "goals_against": int(matches.get("goals_against", pd.Series(dtype=int)).sum()) if not matches.empty else 0,
                "shots_for": int(matches.get("shots_for", pd.Series(dtype=int)).sum()) if not matches.empty else 0,
                "shots_against": int(matches.get("shots_against", pd.Series(dtype=int)).sum()) if not matches.empty else 0,
                "saves": int(matches.get("saves", pd.Series(dtype=int)).sum()) if not matches.empty else 0,
                "recent_games": matches.tail(3)[["date", "opponent", "goals_for", "goals_against", "result"]].to_dict("records") if len(matches) >= 3 else []
            },
            "players": {
                "total_players": len(players),
                "top_scorers": top_scorers
            },
            "events": {
                "total_goals": int(events.get("goals", pd.Series(dtype=int)).sum()) if not events.empty else 0,
                "total_assists": int(events.get("assists", pd.Series(dtype=int)).sum()) if not events.empty else 0,
                "total_shots": int(events.get("shots", pd.Series(dtype=int)).sum()) if not events.empty else 0
            },
            "goals_allowed": {
                "total_conceded": len(goals_allowed),
                "by_situation": goals_allowed["situation"].value_counts().to_dict() if not goals_allowed.empty else {},
                "by_minute": goals_allowed["minute"].apply(_minute_bucket).value_counts().to_dict() if not goals_allowed.empty else {}
            },
            "set_pieces": {
                "total_attempts": len(plays_df),
                "goals_created": int(plays_df.get("goal_created", pd.Series(dtype=bool)).sum()) if not plays_df.empty else 0,
                "by_type": plays_df["set_piece"].value_counts().to_dict() if not plays_df.empty else {}
            }
        }

        system_prompt = (
            "You are an expert soccer analyst and assistant coach. Analyze the provided team data and answer the user's question "
            "with specific insights, statistics, and actionable recommendations. Be concise but thorough. "
            "Focus on patterns, trends, strengths, weaknesses, and coaching implications. "
            "Use specific numbers and examples from the data when relevant."
        )

        user_prompt = f"""
        USER QUESTION: {query}

        TEAM DATA:
        {team_data}

        Please provide a comprehensive analysis addressing the user's question with specific insights from the data.
        """

        resp = model.generate_content([system_prompt, user_prompt])
        return getattr(resp, "text", "").strip() or None
    except Exception:
        return None

def get_next_opponent_from_schedule() -> Optional[Dict[str, str]]:
    """Return next opponent using the Google Sheet schedule when possible.

    Falls back to SB Live scraping only if the sheet is unavailable.
    """
    # 1) Prefer local sheet data loaded into `matches`
    try:
        df = globals().get("matches")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty and "date" in df.columns:
            df2 = df.copy()
            df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
            today = pd.Timestamp.now().normalize()

            # Consider upcoming = today or future; optionally also where result is missing
            upcoming = df2[df2["date"] >= today].sort_values("date")
            if not upcoming.empty:
                row = upcoming.iloc[0]
                return {
                    "opponent": str(row.get("opponent", "Unknown")),
                    "date": str(row.get("date", "")),
                    "source": "Sheet",
                }
    except Exception:
        pass

    # 2) Fallback to SB Live simple parse
    try:
        html = fetch_html(SBLIVE_SCHEDULE_URL)
        text = _clean_text(html)
        lines = text.split('\n')
        for line in lines:
            if 'milton' in line.lower() and any(month in line.lower() for month in ['oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr', 'may']):
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'milton' in part.lower() and i < len(parts) - 1:
                        opponent = parts[i + 1] if i + 1 < len(parts) else "Unknown"
                        return {"opponent": opponent, "date": "Upcoming", "source": "SB Live"}
        return None
    except Exception:
        return None

def analyze_opponent_from_data(opponent_name: str, matches: pd.DataFrame) -> Dict[str, any]:
    """Analyze opponent based on historical match data."""
    if matches.empty or not opponent_name:
        return {}
    
    # Find matches against this opponent
    opponent_matches = matches[matches["opponent"].str.contains(opponent_name, case=False, na=False)]
    
    if opponent_matches.empty:
        return {"found": False, "message": f"No historical data found for {opponent_name}"}
    
    analysis = {
        "found": True,
        "total_meetings": len(opponent_matches),
        "wins": int((opponent_matches["result"] == "W").sum()),
        "losses": int((opponent_matches["result"] == "L").sum()),
        "draws": int((opponent_matches["result"] == "D").sum()),
        "avg_goals_for": float(opponent_matches["goals_for"].mean()),
        "avg_goals_against": float(opponent_matches["goals_against"].mean()),
        "recent_results": opponent_matches.tail(3)[["date", "result", "goals_for", "goals_against"]].to_dict("records")
    }
    
    return analysis

def _extract_links_with_text(html: str) -> list[tuple[str,str]]:
    """Very light HTML anchor extraction: returns list of (href, text)."""
    pairs = []
    try:
        # Match <a ... href="...">Text</a>
        for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>", html, flags=re.I|re.S):
            href = m.group(1)
            text = re.sub(r"<[^>]+>", " ", m.group(2))
            text = re.sub(r"\s+", " ", text).strip()
            pairs.append((href, text))
    except Exception:
        pass
    return pairs

def find_opponent_slug_from_our_schedule(opponent_name: str) -> Optional[str]:
    """Try to find the SI team slug for an opponent by scanning our SI schedule page for a link to that team."""
    try:
        html = fetch_html(SBLIVE_SCHEDULE_URL)
        links = _extract_links_with_text(html)
        target = opponent_name.lower().strip()
        for href, text in links:
            if target in text.lower() and "/soccer/teams/" in href and href.endswith("/games"):
                # href may be absolute or relative; take the slug portion
                m = re.search(r"/soccer/teams/([^/]+)/games", href)
                if m:
                    return m.group(1)
    except Exception:
        return None
    return None

def scrape_team_schedule_stats(team_slug: str) -> Optional[Dict[str, any]]:
    """Fetch an SI team schedule page and derive rough W-L-D, GF, GA and list of opponents.
    This is a best-effort text parse; if it fails, returns None.
    """
    try:
        url = f"{SBLIVE_BASE}/soccer/teams/{team_slug}/games"
        html = fetch_html(url)
        text = _clean_text(html)

        # Attempt to extract per-game lines containing a score like "2 - 1" and an opponent name
        games = []
        for m in re.finditer(r"([A-Za-z0-9.\-\' ]{3,})\s+(\d+)\s*[-–]\s*(\d+)", text):
            opp = m.group(1).strip()
            gf = int(m.group(2))
            ga = int(m.group(3))
            games.append({"opponent": opp, "gf": gf, "ga": ga})

        if not games:
            return None

        wins = sum(1 for g in games if g["gf"] > g["ga"]) 
        losses = sum(1 for g in games if g["gf"] < g["ga"]) 
        draws = sum(1 for g in games if g["gf"] == g["ga"]) 
        gf_total = sum(g["gf"] for g in games)
        ga_total = sum(g["ga"] for g in games)

        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "goals_for": gf_total,
            "goals_against": ga_total,
            "games": games,
        }
    except Exception:
        return None

def summarize_vs_common_opponents(opponent_stats: Dict[str, any], our_matches: pd.DataFrame) -> Dict[str, any]:
    """Compute opponent's record vs teams we have on our schedule (common opponents), using scraped opponent games.
    Returns dict with list of common opponents and opponent W-L-D and GF/GA vs those opponents.
    """
    out = {"common": [], "wins": 0, "losses": 0, "draws": 0, "gf": 0, "ga": 0}
    try:
        if not opponent_stats or our_matches is None or our_matches.empty:
            return out
        our_opps = set(our_matches.get("opponent", pd.Series(dtype=str)).astype(str).str.strip().str.lower().unique())
        common_games = [g for g in opponent_stats.get("games", []) if str(g.get("opponent"," ")).strip().lower() in our_opps]
        if not common_games:
            return out
        out["common"] = common_games
        out["wins"] = sum(1 for g in common_games if g["gf"] > g["ga"]) 
        out["losses"] = sum(1 for g in common_games if g["gf"] < g["ga"]) 
        out["draws"] = sum(1 for g in common_games if g["gf"] == g["ga"]) 
        out["gf"] = sum(g["gf"] for g in common_games)
        out["ga"] = sum(g["ga"] for g in common_games)
        return out
    except Exception:
        return out

def predict_vs_opponent(matches: pd.DataFrame, opponent_name: str) -> Dict[str, float]:
    """Simple prediction using available data only (our schedule):
    - Head-to-head averages vs opponent (if any)
    - Season averages
    - Recent 3 games averages
    Returns suggested expected GF/GA.
    """
    out = {"gf_pred": 0.0, "ga_pred": 0.0}
    if matches.empty:
        return out

    df = matches.copy().sort_values("date")
    df["GF"] = df.get("goals_for", 0)
    df["GA"] = df.get("goals_against", 0)

    # Season averages
    season_gf = float(df["GF"].mean()) if len(df) else 0.0
    season_ga = float(df["GA"].mean()) if len(df) else 0.0

    # Recent form (last 3)
    recent = df.tail(3)
    recent_gf = float(recent["GF"].mean()) if len(recent) else season_gf
    recent_ga = float(recent["GA"].mean()) if len(recent) else season_ga

    # Head-to-head
    h2h = df[df["opponent"].astype(str).str.contains(opponent_name, case=False, na=False)]
    h2h_gf = float(h2h["GF"].mean()) if not h2h.empty else None
    h2h_ga = float(h2h["GA"].mean()) if not h2h.empty else None

    # Blend: if H2H exists, 60% H2H, 40% split between season/recent; else 50/50 season/recent
    if h2h_gf is not None and h2h_ga is not None:
        gf_pred = 0.6 * h2h_gf + 0.2 * season_gf + 0.2 * recent_gf
        ga_pred = 0.6 * h2h_ga + 0.2 * season_ga + 0.2 * recent_ga
    else:
        gf_pred = 0.5 * season_gf + 0.5 * recent_gf
        ga_pred = 0.5 * season_ga + 0.5 * recent_ga

    out.update({
        "gf_pred": round(gf_pred, 2),
        "ga_pred": round(ga_pred, 2),
        "season_gf": round(season_gf, 2),
        "season_ga": round(season_ga, 2),
        "recent_gf": round(recent_gf, 2),
        "recent_ga": round(recent_ga, 2),
        "h2h_gf": round(h2h_gf, 2) if h2h_gf is not None else None,
        "h2h_ga": round(h2h_ga, 2) if h2h_ga is not None else None,
        "h2h_games": int(len(h2h)),
    })
    return out

def generate_ai_opponent_analysis(opponent_name: str,
                                 matches: pd.DataFrame,
                                 next_opponent_data: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Generate AI analysis of upcoming opponent."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Get historical data about opponent + simple prediction
        opponent_analysis = analyze_opponent_from_data(opponent_name, matches)
        prediction = predict_vs_opponent(matches, opponent_name)

        # Try to enrich with scraped opponent season and common-opponent stats
        opponent_slug = find_opponent_slug_from_our_schedule(opponent_name)
        opponent_stats = scrape_team_schedule_stats(opponent_slug) if opponent_slug else None
        common_vs = summarize_vs_common_opponents(opponent_stats, matches) if opponent_stats else {}
        
        # Get next opponent info
        if not next_opponent_data:
            next_opponent_data = get_next_opponent_from_schedule()

        system_prompt = (
            "You are an expert soccer analyst preparing a scouting report. Analyze the opponent data and provide "
            "strategic insights, key matchups, and tactical recommendations. Be specific and actionable."
        )

        context = {
            "opponent_name": opponent_name,
            "historical_data": opponent_analysis,
            "next_opponent_info": next_opponent_data,
            "team_record": _team_record_text(matches),
            "recent_form": matches.tail(3)[["opponent", "result", "goals_for", "goals_against"]].to_dict("records") if len(matches) >= 3 else [],
            "prediction": prediction,
            "opponent_stats": opponent_stats or {},
            "vs_common_opponents": common_vs or {}
        }

        user_prompt = f"""
        OPPONENT ANALYSIS REQUEST: {opponent_name}

        CONTEXT:
        {context}

        Provide a comprehensive opponent analysis including:
        1. Historical matchup summary
        2. Key tactical insights
        3. Strengths and weaknesses to exploit
        4. Recommended game plan
        5. Key players to watch (if available)
        6. Compare the opponent's overall and vs-common-opponents W-L-D and GF/GA to our season and recent form.
        7. Use the provided season/recent/head-to-head metrics and the simple prediction to give a likely score range and preparation focus.
        """

        resp = model.generate_content([system_prompt, user_prompt])
        return getattr(resp, "text", "").strip() or None
    except Exception:
        return None

# --- AI: set-piece analysis summary ---
def generate_ai_set_piece_summary(plays_df: pd.DataFrame,
                                  matches: pd.DataFrame,
                                  players: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        # Normalize data
        df = plays_df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        if "set_piece" not in df.columns:
            df["set_piece"] = ""
        if "goal_created" not in df.columns:
            df["goal_created"] = False
        if "play_call_id" not in df.columns:
            df["play_call_id"] = ""
        if "taker_id" not in df.columns:
            df["taker_id"] = ""
        
        df["set_piece"] = _normalize_set_piece(df["set_piece"]) 
        df["goal_created"] = _bool_col(df["goal_created"]) 

        # Get player names if available
        pl = players.set_index("player_id") if "player_id" in players.columns else pd.DataFrame()
        
        # Analyze by set piece type
        set_piece_stats = {}
        for sp_type in ["corner", "penalty", "fk_direct", "fk_indirect"]:
            sub = df[df["set_piece"] == sp_type]
            if not sub.empty:
                total = len(sub)
                goals = sub["goal_created"].sum()
                pct = (goals / total * 100) if total > 0 else 0.0
                set_piece_stats[sp_type] = {
                    "total": total,
                    "goals": goals,
                    "pct": pct
                }

        # Analyze by taker (if taker_id is available)
        taker_stats = {}
        if "taker_id" in df.columns and not df["taker_id"].isna().all():
            for taker_id in df["taker_id"].dropna().unique():
                if taker_id:
                    sub = df[df["taker_id"] == taker_id]
                    total = len(sub)
                    goals = sub["goal_created"].sum()
                    pct = (goals / total * 100) if total > 0 else 0.0
                    taker_name = ""
                    if not pl.empty and str(taker_id) in pl.index:
                        taker_name = pl.at[str(taker_id), "name"]
                    taker_stats[taker_id] = {
                        "name": taker_name or str(taker_id),
                        "total": total,
                        "goals": goals,
                        "pct": pct
                    }

        # Get top performers
        top_takers = sorted(taker_stats.items(), key=lambda x: x[1]["pct"], reverse=True)[:3]
        top_takers = [(data["name"], data) for _, data in top_takers if data["total"] >= 2]  # Only if 2+ attempts

        context = {
            "total_set_pieces": len(df),
            "set_piece_stats": set_piece_stats,
            "top_takers": top_takers,
            "total_takers": len(taker_stats)
        }

        prompt = (
            "You are a soccer set-piece specialist analyst. Review the set-piece performance data and give a brief, "
            "coach-friendly summary (120-160 words max) with 3-5 concrete actions. "
            "Focus on: which set-piece types are most effective, which takers are performing best, "
            "patterns in success rates, and specific training recommendations. "
            "Avoid jargon. Keep it practical and actionable.\n\n"
            f"DATA: {context}"
        )

        resp = model.generate_content(prompt)
        return getattr(resp, "text", "").strip() or None
    except Exception:
        return None

# ---------------------------------------------------------------------
# UI RENDERERS
# ---------------------------------------------------------------------
def _team_kpis(matches_view: pd.DataFrame, d2_rank: Optional[int]=None, compact: bool=False):
    # --- aggregate
    gf = int(matches_view.get("goals_for", pd.Series(dtype=int)).sum()) if not matches_view.empty else 0
    ga = int(matches_view.get("goals_against", pd.Series(dtype=int)).sum()) if not matches_view.empty else 0
    sh_for = int(matches_view.get("shots_for", pd.Series(dtype=int)).sum()) if not matches_view.empty else 0
    sh_ag  = int(matches_view.get("shots_against", pd.Series(dtype=int)).sum()) if not matches_view.empty else 0
    sv = int(matches_view.get("saves", pd.Series(dtype=int)).sum()) if not matches_view.empty else 0
    games = int(len(matches_view))

    save_denom = sv + ga
    save_pct = (sv / save_denom * 100.0) if save_denom > 0 else 0.0
    conv_for_pct = (gf / sh_for * 100.0) if sh_for > 0 else 0.0
    conv_agn_pct = (ga / sh_ag  * 100.0) if sh_ag  > 0 else 0.0
    record_str = _team_record_text(matches_view)

    if compact:
        # ---------- Mobile / Compact: card grid ----------
        items = [
            ("Games", games),
            ("Record", record_str),
            ("GF", gf),
            ("GA", ga),
            ("Shots (For)", sh_for),
            ("Shots (Agst)", sh_ag),
            ("Saves", sv),
            ("Save%", f"{save_pct:.1f}%"),
            ("Conv% (For)", f"{conv_for_pct:.1f}%"),
            ("Conv% (Agst)", f"{conv_agn_pct:.1f}%"),
        ]
        if d2_rank:
            items.append(("D2 Rank", f"{d2_rank}{_suffix(d2_rank)}"))

        html = "<div class='kpi-grid'>" + "".join(
            f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>"
            for label, value in items
        ) + "</div>"
        st.markdown(html, unsafe_allow_html=True)
        return

    # ---------- Desktop: keep classic metrics row ----------
    cols = st.columns(11)
    cols[0].metric("Games", games)
    cols[1].metric("GF", gf)
    cols[2].metric("GA", ga)
    cols[3].metric("Shots (For)", sh_for)
    cols[4].metric("Shots (Agst)", sh_ag)
    cols[5].metric("Saves", sv)
    cols[6].metric("Save%", f"{save_pct:.1f}%")
    cols[7].metric("Conv% (For)", f"{conv_for_pct:.1f}%")
    cols[8].metric("Conv% (Agst)", f"{conv_agn_pct:.1f}%")
    cols[9].metric("Record", record_str)
    if d2_rank:
        cols[10].metric("D2 Rank", f"{d2_rank}{_suffix(d2_rank)}")
    else:
        cols[10].metric("D2 Rank", "N/A")
        st.caption("Rank unavailable or not fetched. Click 'Open Rankings (D2)' in the sidebar.")

def render_games_table(matches: pd.DataFrame, compact: bool=False):
    st.subheader("Games")
    if matches.empty:
        st.info("No matches yet. Add rows to the 'matches' tab in your Google Sheet.")
        return

    view = matches.sort_values("date").copy()
    if {"goals_for","goals_against"}.issubset(view):
        view["GF-GA"] = view["goals_for"].astype(int).astype(str) + "-" + view["goals_against"].astype(int).astype(str)
    else:
        view["GF-GA"] = ""

    def _ha_pill(v: str) -> str:
        if str(v).upper() == "H": return "<span class='pill home'>Home</span>"
        if str(v).upper() == "A": return "<span class='pill away'>Away</span>"
        return "<span class='pill'>H/A</span>"

    def _div_pill(is_div: bool) -> str:
        return "<span class='pill div'>Division</span>" if bool(is_div) else "<span class='pill'>Non-division</span>"

    if compact:
        for idx, r in view.iterrows():
            date_html = _format_date(r.get("date",""))
            opp_html  = _color_opp(r.get("opponent",""), r.get("result",""))
            score     = r.get("GF-GA","")
            ha_html   = _ha_pill(r.get("home_away",""))
            div_html  = _div_pill(r.get("division_game", False))
            mid = str(r.get("match_id","") or f"row{idx}")

            card = f"""
            <a href='?match_id={mid}' style='text-decoration:none; color:inherit;'>
              <div class="game-card">
                <div class="gc-row">
                  <div>
                    <div class="gc-date">{date_html}</div>
                    <div class="gc-opp">{opp_html}</div>
                  </div>
                  <div class="gc-score">{score}</div>
                </div>
                <div class="gc-meta">
                  {ha_html}{div_html}
                  <span class="tiny-open">Open</span>
                </div>
              </div>
            </a>
            """
            st.markdown(card, unsafe_allow_html=True)
        return

    hdr = st.columns((0.3, 1.2, 2, 2.4, 0.9, 1.2, 1.0, 1.0, 0.9, 0.7))
    for c,t in zip(hdr, ["", "Date", "Match ID", "Opponent", "H/A", "Division", "GF-GA", "Shots", "Saves", ""]):
        c.markdown(f"**{t}**" if t else "")
    for idx, r in view.iterrows():
        cols = st.columns((0.3, 1.2, 2, 2.4, 0.9, 1.2, 1.0, 1.0, 0.9, 0.7))
        cols[0].markdown(_status_dot(r.get("result","")), unsafe_allow_html=True)
        cols[1].write(_format_date(r.get("date","")))
        cols[2].write(r.get("match_id",""))
        mid = str(r.get("match_id","") or f"row{idx}")
        cols[3].markdown(f"<a href='?match_id={mid}' style='text-decoration:none'>{_color_opp(r.get('opponent',''), r.get('result',''))}</a>", unsafe_allow_html=True)
        cols[4].write(r.get("home_away",""))
        cols[5].write("Yes" if r.get("division_game", False) else "No")
        cols[6].write(r.get("GF-GA",""))
        cols[7].write(r.get("shots_for", r.get("shots","")))
        cols[8].write(r.get("saves",""))
        cols[9].markdown(f"<a class='tiny-open' href='?match_id={mid}' title='Open game'>Open</a>", unsafe_allow_html=True)

    # CSV download of games
    try:
        export_cols = [c for c in ["date","match_id","opponent","home_away","division_game","GF-GA","shots_for","saves"] if c in view.columns]
        csv = view[export_cols].to_csv(index=False).encode('utf-8')
        st.download_button("Download games (CSV)", data=csv, file_name="games.csv", mime="text/csv")
    except Exception:
        pass

def render_points_leaderboard(events: pd.DataFrame, players: pd.DataFrame, top_n: int = 5, compact: bool=False):
    st.subheader("Points Leaderboard")
    if events.empty or players.empty:
        st.info("No events/players yet.")
        return

    ev = events.copy(); pl = players.copy()
    ev.columns = [c.strip().lower() for c in ev.columns]
    pl.columns = [c.strip().lower() for c in pl.columns]
    if "assist" in ev.columns and "assists" not in ev.columns:
        ev = ev.rename(columns={"assist": "assists"})
    for n in ["goals","assists","shots","fouls"]:
        if n not in ev.columns: ev[n] = 0
        ev[n] = pd.to_numeric(ev[n], errors="coerce").fillna(0).astype(int)
    if "player_id" in ev.columns: ev["player_id"] = ev["player_id"].astype(str)
    if "player_id" in pl.columns: pl["player_id"] = pl["player_id"].astype(str)

    num_cols = [c for c in ["goals","assists","shots","fouls"] if c in ev.columns]
    agg = ev.groupby("player_id", as_index=False)[num_cols].sum()
    pidx = pl.set_index("player_id")[["name","jersey"]].copy()
    pidx.index = pidx.index.astype(str)
    df = agg.set_index("player_id").join(pidx, how="left").fillna({"jersey":0,"name":"Unknown"})
    df["points"] = 2*df.get("goals", 0) + df.get("assists", 0)

    cols_full = ["jersey","name"] + num_cols + ["points"]
    full = df.reset_index()[cols_full] \
             .sort_values(["points","goals","assists","jersey"], ascending=[False,False,False,True])

    def _medal(i: int) -> str:
        return "1" if i == 0 else ("2" if i == 1 else ("3" if i == 2 else ""))
    top = full.head(top_n if top_n and top_n > 0 else 5).copy()
    top.insert(0, "", [ _medal(i) for i in range(len(top)) ])  # blank header for medal col

    if compact:
        show = top[["", "name", "points"]].rename(columns={"": " "})
        st.dataframe(show, use_container_width=True, hide_index=True, height=180)
    else:
        st.dataframe(
            top[["","jersey","name","goals","assists","points"]],
            use_container_width=True,
            hide_index=True,
            height=210
        )
        st.caption("Scoring = 2×Goals + 1×Assists")

    if not top.empty:
        top_for_chart = top.copy()
        top_for_chart["name"] = top_for_chart["name"].astype(str)

        label_axis = alt.Axis(labelAngle=-45) if compact else alt.Axis()
        h = 240 if compact else 280

        c1, c2 = st.columns(2)
        with c1:
            chart_pts = alt.Chart(top_for_chart).mark_bar().encode(
                x=alt.X("name:N", sort="-y", title="Player", axis=label_axis),
                y=alt.Y("points:Q", title="Points"),
                tooltip=["name","goals","assists","points"]
            ).properties(height=h)
            st.altair_chart(chart_pts, use_container_width=True)
        with c2:
            melted = top_for_chart.melt(
                id_vars=["name","points"],
                value_vars=["goals","assists"],
                var_name="Stat",
                value_name="Value"
            )
            chart_breakdown = alt.Chart(melted).mark_bar().encode(
                x=alt.X("name:N", sort="-y", title="Player", axis=label_axis),
                y=alt.Y("Value:Q", title="Goals / Assists"),
                color=alt.Color("Stat:N", title=""),
                tooltip=["name","Stat","Value"]
            ).properties(height=h)
            st.altair_chart(chart_breakdown, use_container_width=True)

    with st.expander("View full team leaderboard"):
        st.dataframe(
            full[cols_full],
            use_container_width=True,
            hide_index=True,
            height=420
        )
        try:
            csv = full[cols_full].to_csv(index=False).encode('utf-8')
            st.download_button("Download leaderboard (CSV)", data=csv, file_name="leaderboard.csv", mime="text/csv")
        except Exception:
            pass

def _set_piece_type_stats(df: pd.DataFrame, sp_type: str) -> tuple[int, float]:
    """Return (total_attempts, pct_scored) for a given set_piece type (normalized)."""
    if df.empty or "set_piece" not in df.columns:
        return 0, 0.0
    # Normalize before computing
    sp = _normalize_set_piece(df["set_piece"]) if "set_piece" in df.columns else pd.Series([], dtype=str)
    gc = _bool_col(df["goal_created"]) if "goal_created" in df.columns else pd.Series([], dtype=bool)
    sub_mask = (sp == sp_type)
    total = int(sub_mask.sum())
    pct = float(gc[sub_mask].mean() * 100) if total > 0 else 0.0
    return total, pct

def _set_piece_type_counts(df: pd.DataFrame, sp_type: str) -> tuple[int, int]:
    """Return (total_attempts, goals_scored) for a given set_piece type (normalized)."""
    if df.empty or "set_piece" not in df.columns:
        return 0, 0
    sp = _normalize_set_piece(df["set_piece"]) if "set_piece" in df.columns else pd.Series([], dtype=str)
    gc = _bool_col(df.get("goal_created", pd.Series([], dtype=bool)))
    mask = (sp == sp_type)
    total = int(mask.sum())
    goals = int(gc[mask].sum()) if total > 0 else 0
    return total, goals

def _set_piece_aggregate(df: pd.DataFrame, include_penalties: bool = True) -> tuple[int, int]:
    """Return (total_attempts, goals_scored) across set-piece types.
    Includes corners, direct FK, indirect FK, and optionally penalties.
    """
    if df.empty:
        return 0, 0
    sp = _normalize_set_piece(df.get("set_piece", pd.Series([], dtype=str)))
    gc = _bool_col(df.get("goal_created", pd.Series([], dtype=bool)))
    allowed = {"corner", "fk_direct", "fk_indirect"}
    if include_penalties:
        allowed.add("penalty")
    mask = sp.isin(list(allowed))
    total = int(mask.sum())
    goals = int(gc[mask].sum()) if total > 0 else 0
    return total, goals

def render_set_piece_analysis_from_plays(plays_df: pd.DataFrame, matches: pd.DataFrame, players: pd.DataFrame):
    st.subheader("Set-Piece Analysis")

    # ---- Guard + normalize ----
    if plays_df is None or plays_df.empty:
        st.session_state.pop("ai_set_piece_summary", None)
        st.session_state.pop("ai_set_piece_error", None)
        st.info("No set-play rows yet. Add data to the `plays` sheet.")
        return

    df = plays_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "set_piece" not in df.columns:
        df["set_piece"] = ""
    if "goal_created" not in df.columns:
        df["goal_created"] = False
    df["set_piece"] = _normalize_set_piece(df["set_piece"])
    df["goal_created"] = _bool_col(df["goal_created"])

    # ---- KPI tiles (mobile-friendly card grid) ----
    # Show values for current filters (df) and season totals for clarity
    season_df = load_plays_simple()
    season_df.columns = [c.strip().lower() for c in season_df.columns]
    if "set_piece" not in season_df.columns:
        season_df["set_piece"] = ""
    if "goal_created" not in season_df.columns:
        season_df["goal_created"] = False
    season_df["set_piece"] = _normalize_set_piece(season_df["set_piece"]) 
    season_df["goal_created"] = _bool_col(season_df["goal_created"]) 

    def build_row_kpi(label: str, key: str):
        sz_total, sz_goals = _set_piece_type_counts(season_df, key)
        sz_pct = (sz_goals / sz_total * 100) if sz_total > 0 else 0.0
        return (
            f"<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{sz_total}</div>"
            f"<div class='stat-sub'>Scored {sz_pct:.1f}%</div>"
            f"</div>"
        )

    def build_row(label: str, key: str):
        ft_total, ft_goals = _set_piece_type_counts(df, key)
        ft_pct = (ft_goals / ft_total * 100) if ft_total > 0 else 0.0
        sz_total, sz_goals = _set_piece_type_counts(season_df, key)
        sz_pct = (sz_goals / sz_total * 100) if sz_total > 0 else 0.0
        return (
            f"<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{ft_total}</div>"
            f"<div class='stat-sub'>Filtered: {ft_goals}/{ft_total} ({ft_pct:.1f}%) · Season: {sz_goals}/{sz_total} ({sz_pct:.1f}%)</div>"
            f"</div>"
        )

    def build_agg_row(label: str, include_pk: bool):
        ft_total, ft_goals = _set_piece_aggregate(df, include_penalties=include_pk)
        ft_pct = (ft_goals / ft_total * 100) if ft_total > 0 else 0.0
        sz_total, sz_goals = _set_piece_aggregate(season_df, include_penalties=include_pk)
        sz_pct = (sz_goals / sz_total * 100) if sz_total > 0 else 0.0
        return (
            f"<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{ft_total}</div>"
            f"<div class='stat-sub'>Filtered: {ft_goals}/{ft_total} ({ft_pct:.1f}%) · Season: {sz_goals}/{sz_total} ({sz_pct:.1f}%)</div>"
            f"</div>"
        )

    # Build single aggregate KPI row showing incl/no PK for filtered and season
    ft_total_incl, ft_goals_incl = _set_piece_aggregate(df, include_penalties=True)
    ft_total_no,   ft_goals_no   = _set_piece_aggregate(df, include_penalties=False)
    ft_pct_incl = (ft_goals_incl / ft_total_incl * 100) if ft_total_incl > 0 else 0.0
    ft_pct_no   = (ft_goals_no   / ft_total_no   * 100) if ft_total_no   > 0 else 0.0
    sz_total_incl, sz_goals_incl = _set_piece_aggregate(season_df, include_penalties=True)
    sz_total_no,   sz_goals_no   = _set_piece_aggregate(season_df, include_penalties=False)
    sz_pct_incl = (sz_goals_incl / sz_total_incl * 100) if sz_total_incl > 0 else 0.0
    sz_pct_no   = (sz_goals_no   / sz_total_no   * 100) if sz_total_no   > 0 else 0.0

    total_row_html2 = (
        "<div class='stat-card'>"
        "<div class='stat-label'>Total Set Pieces</div>"
        f"<div class='stat-value'>{ft_total_incl}</div>"
        f"<div class='stat-sub'>Incl PK — Values: {ft_goals_incl}/{ft_total_incl} ({ft_pct_incl:.1f}%) &middot; Season: {sz_goals_incl}/{sz_total_incl} ({sz_pct_incl:.1f}%)</div>"
        f"<div class='stat-sub'>No PK — Values: {ft_goals_no}/{ft_total_no} ({ft_pct_no:.1f}%) &middot; Season: {sz_goals_no}/{sz_total_no} ({sz_pct_no:.1f}%)</div>"
        "</div>"
    )

    total_row_html2 = (
        "<div class='stat-card'>"
        "<div class='stat-label'>Total Set Pieces</div>"
        f"<div class='stat-value'>{sz_total_incl}</div>"
        f"<div class='stat-sub'>Incl PK — Season: {sz_goals_incl}/{sz_total_incl} ({sz_pct_incl:.1f}%)</div>"
        f"<div class='stat-sub'>No PK — Season: {sz_goals_no}/{sz_total_no} ({sz_pct_no:.1f}%)</div>"
        "</div>"
    )

    # Order per-type cards by Season attempts (desc)
    type_labels = [("corner", "Corners"), ("penalty", "Penalties"), ("fk_direct", "Direct FK"), ("fk_indirect", "Indirect FK")]
    type_with_counts = []
    for key, label in type_labels:
        total, _g = _set_piece_type_counts(season_df, key)
        type_with_counts.append((total, key, label))
    type_with_counts.sort(key=lambda x: x[0], reverse=True)

    per_type_html = "".join([build_row_kpi(label, key) for (total, key, label) in type_with_counts])

    kpi_html = (
        "<div class='kpi-grid'>"
        + total_row_html2
        + per_type_html
        + "</div>"
    )
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ---- Table (unchanged) ----
    tbl = set_piece_leaderboard_from_plays(df)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ---- Chart (unchanged) ----
    if not tbl.empty:
        # Keep Goal% on Y, but order by attempts (desc) on X
        chart = alt.Chart(tbl).mark_bar().encode(
            x=alt.X("Play Call:N", sort=alt.SortField(field="attempts", order="descending"), title="Play Call"),
            y=alt.Y("Goal%:Q", title="Goal %"),
            color=alt.Color("set_piece:N", title="Type"),
            tooltip=list(tbl.columns),
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)

    # ---- AI Insights ----
    state_key = "ai_set_piece_summary"
    error_key = "ai_set_piece_error"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if error_key not in st.session_state:
        st.session_state[error_key] = None

    if st.button("Generate AI Insights on Set-Piece Performance", key="generate_ai_set_piece"):
        with st.spinner("Generating set-piece insights..."):
            ai_txt = generate_ai_set_piece_summary(plays_df, matches, players)
        if ai_txt:
            st.session_state[state_key] = ai_txt
            st.session_state[error_key] = None
        else:
            st.session_state[state_key] = None
            st.session_state[error_key] = "AI summary unavailable (no Gemini key set or not enough context)."

    summary_text = st.session_state.get(state_key)
    summary_error = st.session_state.get(error_key)
    if summary_text:
        st.markdown("**AI Set-Piece Analysis & Recommendations**")
        st.write(summary_text)
    elif summary_error:
        st.caption(summary_error)



def render_coach_notes_and_summary(match_id: str,
                                   matches: pd.DataFrame,
                                   summaries: pd.DataFrame,
                                   events: pd.DataFrame):
    st.subheader("Coach Notes & Summary")
    mrow = matches.loc[matches["match_id"] == match_id]
    m = mrow.iloc[0] if not mrow.empty else pd.Series(dtype=object)

    srow = None
    if not summaries.empty and "match_id" in summaries.columns:
        srow_df = summaries.loc[summaries["match_id"] == str(match_id)]
        if not srow_df.empty:
            srow = srow_df.iloc[0]

    if srow is not None:
        show = srow.drop(labels=[c for c in ["match_id"] if c in srow.index])
        nice = show.rename(index=lambda k: k.replace("_", " ").title())
        st.markdown("**Coach Notes (from sheet)**")
        st.dataframe(nice.to_frame("Value"), use_container_width=True, hide_index=False, height=280)
    else:
        st.info("No coach notes yet for this game. Add a row in the `summary` tab with this match_id.")

    ai_txt = generate_ai_game_summary(m, srow, events)
    if ai_txt:
        st.markdown("**AI Game Summary**")
        st.write(ai_txt)
    else:
        st.caption("AI summary unavailable (no Gemini key set or not enough context).")

def render_goals_allowed_analysis(ga_df: pd.DataFrame,
                                  matches: pd.DataFrame,
                                  players: pd.DataFrame,
                                  compact: bool=False):
    st.subheader("Goals Allowed (Season)")
    if ga_df.empty:
        st.session_state.pop("ai_conceded_summary", None)
        st.session_state.pop("ai_conceded_error", None)
        st.info("No rows in `goals_allowed` yet. Add columns: match_id, goal_id, description, goalie_player_id, minute, situation.")
        return

    pl = players.set_index("player_id") if "player_id" in players.columns else pd.DataFrame()
    mx = matches.set_index("match_id") if "match_id" in matches.columns else pd.DataFrame()

    view = ga_df.copy()
    if not pl.empty:
        view["goalie_name"] = view["goalie_player_id"].map(lambda pid: pl.at[str(pid), "name"] if str(pid) in pl.index else "")
    else:
        view["goalie_name"] = ""
    if not mx.empty:
        view["opponent"] = view["match_id"].map(lambda mid: mx.at[str(mid), "opponent"] if str(mid) in mx.index else "")
        view["date"] = view["match_id"].map(lambda mid: mx.at[str(mid), "date"] if str(mid) in mx.index else "")
        try: 
            view["date"] = pd.to_datetime(view["date"], errors="coerce")
            # Format date to remove time portion
            view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        except Exception: pass
    else:
        view["opponent"] = ""; view["date"] = pd.NaT

    view["minute_bucket"] = view["minute"].apply(_minute_bucket)

    cols_show = [c for c in ["date","opponent","minute","minute_bucket","situation","goalie_name","description","goal_id"] if c in view.columns]
    st.dataframe(view[cols_show].sort_values(["date","minute"], ascending=[True, True]),
                 use_container_width=True, hide_index=True, height=320)
    # CSV download for goals allowed table
    try:
        csv = view[cols_show].to_csv(index=False).encode('utf-8')
        st.download_button("Download goals allowed (CSV)", data=csv, file_name="goals_allowed.csv", mime="text/csv")
    except Exception:
        pass

    total_ga = len(view)
    games = len(matches) if not matches.empty else 0
    ga_per_game = (total_ga / games) if games > 0 else 0.0

    # Shutouts: matches with 0 goals against (based on current matches view)
    shutouts = 0
    if not matches.empty and "goals_against" in matches.columns:
        ga_series = pd.to_numeric(matches["goals_against"], errors="coerce").fillna(0)
        shutouts = int((ga_series == 0).sum())
    shutout_rate = (shutouts / games * 100) if games > 0 else 0.0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Conceded (Total)", total_ga)
    c2.metric("Games", games)
    c3.metric("GA / Game", f"{ga_per_game:.2f}")
    c4.metric("Shutouts", f"{shutouts}", delta=f"{shutout_rate:.0f}%", help="Matches with 0 goals against")

    label_axis = alt.Axis(labelAngle=-30) if compact else alt.Axis()
    h = 260 if compact else 300

    by_sit = view.groupby("situation", as_index=False).size().rename(columns={"size":"count"})
    by_sit["situation"] = by_sit["situation"].fillna("").replace({"": "Unspecified"}).str.title()
    chart_sit = alt.Chart(by_sit).mark_bar().encode(
        x=alt.X("situation:N", sort="-y", title="Situation", axis=label_axis),
        y=alt.Y("count:Q", title="Goals Conceded"),
        tooltip=["situation","count"]
    ).properties(height=h)

    order_buckets = ["0-15","16-30","31-45","46-60","61-75","76-90+","N/A"]
    by_min = view.groupby("minute_bucket", as_index=False).size().rename(columns={"size":"count"})
    by_min["minute_bucket"] = pd.Categorical(by_min["minute_bucket"], categories=order_buckets, ordered=True)
    chart_min = alt.Chart(by_min).mark_bar().encode(
        x=alt.X("minute_bucket:N", sort=order_buckets, title="Minute Window", axis=label_axis),
        y=alt.Y("count:Q", title="Goals Conceded"),
        tooltip=["minute_bucket","count"]
    ).properties(height=h)

    by_gk = view.groupby("goalie_name", as_index=False).size().rename(columns={"size":"count"})
    by_gk["goalie_name"] = by_gk["goalie_name"].replace({"": "Unspecified"})
    chart_gk = alt.Chart(by_gk).mark_bar().encode(
        x=alt.X("goalie_name:N", sort="-y", title="Goalie", axis=label_axis),
        y=alt.Y("count:Q", title="Goals Conceded"),
        tooltip=["goalie_name","count"]
    ).properties(height=h)

    st.altair_chart(chart_sit | chart_min, use_container_width=True)
    st.altair_chart(chart_gk, use_container_width=True)

    state_key = "ai_conceded_summary"
    error_key = "ai_conceded_error"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if error_key not in st.session_state:
        st.session_state[error_key] = None

    if st.button("Generate AI Insights on Conceded Goals", key="generate_ai_conceded"):
        with st.spinner("Analyzing conceded goals..."):
            ai_txt = generate_ai_conceded_summary(view, matches, players)
        if ai_txt:
            st.session_state[state_key] = ai_txt
            st.session_state[error_key] = None
        else:
            st.session_state[state_key] = None
            st.session_state[error_key] = "AI summary unavailable (no Gemini key set or not enough context)."

    conceded_summary = st.session_state.get(state_key)
    conceded_error = st.session_state.get(error_key)
    if conceded_summary:
        st.markdown("**AI Defensive Summary & Recommendations**")
        st.write(conceded_summary)
    elif conceded_error:
        st.caption(conceded_error)

def render_game_drilldown(match_id: str, matches: pd.DataFrame, players: pd.DataFrame, events: pd.DataFrame, plays_df: pd.DataFrame, summaries: pd.DataFrame):
    row = matches.loc[matches["match_id"] == match_id]
    if row.empty:
        st.error("Match not found.")
        if st.button("Back to Dashboard"):
            _qparams_set(); st.rerun()
        return
    m = row.iloc[0]
    st.header(f"Game View – {_format_date(m.get('date',''))} vs {m.get('opponent','')} ({m.get('home_away','')})")
    st.caption(f"Division: {'Yes' if m.get('division_game', False) else 'No'} | Result: {m.get('result','')} | Score: {m.get('goals_for','')}-{m.get('goals_against','')}")
    
    # ============================================================================
    # GAME RECORDING URL DISPLAY
    # ============================================================================
    # Check for game recording URL in multiple possible column names from Google Sheets
    # Supports: 'url', 'recording_url', 'game_url', 'video_url', 'link'
    url = None
    for url_col in ['url', 'recording_url', 'game_url', 'video_url', 'link']:
        if m.get(url_col) and str(m.get(url_col)).strip():
            url = str(m.get(url_col)).strip()
            break
    
    if url:
        # Auto-add https:// protocol if missing for proper link functionality
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Display styled recording link with blue info box design
        st.markdown(f"""
        <div style="
            background: #f0f8ff; 
            border: 1px solid #4a90e2; 
            border-radius: 8px; 
            padding: 12px; 
            margin: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size: 18px;">📹</span>
            <div>
                <strong style="color: #2c3e50;">Game Recording Available</strong><br>
                <a href="{url}" target="_blank" style="color: #4a90e2; text-decoration: none; font-weight: 500;" title="Click to open game recording in a new tab">
                    🎥 Watch Game Recording →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show info message when no recording URL is available
        # Note: Both cases show same message for consistency (removed technical details)
        st.info("📹 No game recording available for this match.")

    by_player = (events.query("match_id == @match_id").copy()
                 if "match_id" in events.columns else pd.DataFrame())
    if by_player.empty:
        base = players[["player_id","name","jersey","position"]].copy()
        base["shots"]=base["goals"]=base["assists"]=base["points"]=0
        view = base[["jersey","name","position","shots","goals","assists","points"]]
    else:
        sums = by_player.groupby("player_id", as_index=False)[["shots","goals","assists"]].sum()
        sums["points"] = 2*sums["goals"] + sums["assists"]
        view = sums.set_index("player_id").join(
            players.set_index("player_id")[["name","jersey","position"]], how="left"
        ).fillna({"name":"Unknown","position":"","jersey":0})
        view = view.reset_index()[["jersey","name","position","shots","goals","assists","points"]]
        view = view.sort_values(["points","goals","shots"], ascending=[False,False,False])

    st.subheader("Per-Player Breakdown")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("Set-Play Attempts (this game)")
    sp = plays_df.query("match_id == @match_id") if not plays_df.empty else pd.DataFrame()
    if sp.empty:
        st.info("No set-play rows for this match.")
    else:
        cols = [c for c in ["set_piece","play_call_id","play_type","taker_notes","goal_created"] if c in sp.columns]
        df_show = sp[cols].rename(columns={"play_call_id":"Play Call"})
        df_show = df_show[["set_piece","Play Call","play_type","taker_notes","goal_created"]]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.divider()
    render_coach_notes_and_summary(match_id, matches, summaries, events)

    st.divider()
    c1,c2 = st.columns([1,1])
    if c1.button("Back to Dashboard"): _qparams_set(); st.rerun()
    c2.markdown(f"[Open this game in a new tab](?match_id={match_id})")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
matches = load_matches()
players = load_players()
events = load_events()
plays_simple = load_plays_simple()
summaries = load_summaries()
goals_allowed = load_goals_allowed()


# Sidebar (clean labels)
with st.sidebar:
    st.title("HS Soccer")
    if st.button("Dashboard (Home)"):
        _qparams_set(); st.rerun()

    # Read qp for initial toggle values
    qp_init = _qparams_get()
    COMPACT_DEFAULT = True
    compact_init = _qp_bool(qp_init.get("compact"), COMPACT_DEFAULT)
    div_only_init = _qp_bool(qp_init.get("div_only"), False)

    compact = st.toggle("Compact mode", value=compact_init, help="Phone-friendly layout")
    div_only = st.checkbox("Division games only", value=div_only_init)

    # Global filters
    st.subheader("Filters")
    opponent_q = st.text_input("Opponent contains", value=str(qp_init.get("opp","")))
    ha_opt = st.selectbox("Home/Away", ["Any","Home","Away"], index={"any":0,"home":1,"away":2}.get(str(qp_init.get("ha","any")).lower(),0))

    st.link_button("Open Schedule", SBLIVE_SCHEDULE_URL)
    st.link_button("Open Rankings (D2)", SBLIVE_RANKINGS_URL)

    # Sync toggles/filters to query params only when they differ
    try:
        desired = {
            "compact": str(compact).lower(),
            "div_only": str(div_only).lower(),
            "opp": opponent_q.strip(),
            # Store full text so "Any" is not mistaken for Away
            "ha": ha_opt.lower() if ha_opt else "any",
        }

        # Only update if any difference
        diffs = []
        for k, v in desired.items():
            curv = qp_init.get(k)
            if isinstance(curv, list): curv = curv[0] if curv else None
            if (curv or "") != (v or ""):
                diffs.append(k)
        if diffs:
            _qparams_merge_update(**desired)
            st.rerun()
    except Exception:
        pass

# Apply filters (division/date/opponent/H-A)
matches_view = matches.copy()
if div_only and not matches_view.empty and "division_game" in matches_view:
    matches_view = matches_view.query("division_game == True")

qp = _qparams_get()
opp_filter = str(qp.get("opp",""))
if isinstance(opp_filter, list): opp_filter = opp_filter[0]
opp_filter = opp_filter.strip()
if opp_filter and not matches_view.empty and "opponent" in matches_view:
    matches_view = matches_view[matches_view["opponent"].astype(str).str.contains(opp_filter, case=False, na=False)]

ha_val = str(qp.get("ha","any")).lower()
if isinstance(ha_val, list): ha_val = ha_val[0]
if ha_val in ("h","home","a","away") and not matches_view.empty and "home_away" in matches_view:
    want = "H" if ha_val.startswith("h") else "A"
    matches_view = matches_view[matches_view["home_away"].astype(str).str.upper() == want]



# Derive related views by match_id
if not matches_view.empty and "match_id" in matches_view:
    keep = set(matches_view["match_id"].astype(str))
    events_view = events[events["match_id"].astype(str).isin(keep)] if "match_id" in events.columns else events
    plays_view  = plays_simple[plays_simple["match_id"].astype(str).isin(keep)] if not plays_simple.empty else plays_simple
    ga_view     = goals_allowed[goals_allowed["match_id"].astype(str).isin(keep)] if not goals_allowed.empty else goals_allowed
else:
    events_view, plays_view, ga_view = events, plays_simple, goals_allowed

# Drill-in param
qp = _qparams_get()
match_id: Optional[str] = None
try:
    raw = qp.get("match_id")
    match_id = raw[0] if isinstance(raw, list) else raw
except Exception:
    pass

# D2 rank (KPI only)
our_rank = None
try:
    si_html_rank = fetch_html(SBLIVE_RANKINGS_URL)
    ranks = parse_all_ranks_from_si(si_html_rank)
    our_rank = fuzzy_find_rank(ranks, TEAM_NAME_CANON)
except Exception:
    our_rank = None

# Routing
if match_id:
    render_game_drilldown(match_id, matches_view, players, events_view, plays_view, summaries)
else:
    st.header("Milton Varsity Boys Soccer Team 2025")

    # Data health panel
    with st.expander("Data Health", expanded=False):
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Matches", len(matches))
        c2.metric("Players", len(players))
        c3.metric("Events", len(events))
        c4.metric("Plays", len(plays_simple))
        c5.metric("Summaries", len(summaries))
        c6.metric("Goals Allowed", len(goals_allowed))
        st.caption("Sheets cached for up to 5 minutes. Use Refresh in sidebar to reload.")
        if "cache_cleared_at" in st.session_state:
            st.caption(f"Last manual refresh: {st.session_state['cache_cleared_at']}")
        if st.button("Refresh now"):
            st.cache_data.clear(); st.rerun()

    _team_kpis(matches_view, d2_rank=our_rank, compact=compact)


    tab_labels = ["Games","Trends","Leaders","Goals Allowed","Set Pieces"]
    if "main_tab_radio" not in st.session_state:
        st.session_state["main_tab_radio"] = tab_labels[0]
    selected_tab = st.radio("", tab_labels, horizontal=True, key="main_tab_radio", label_visibility="collapsed")

    if selected_tab == "Games":
        render_games_table(matches_view, compact=compact)
        # Place AI Chat Assistant under the game schedule
        st.divider()
        st.subheader("AI Assistant")
        st.caption("Ask questions about team performance and season trends")

        # (Removed next opponent preview; AI limited to known sheet data)

        # Initialize chat history in session state
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.ai_chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class='ai-chat-message ai-chat-user'>
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='ai-chat-message ai-chat-assistant'>
                        <strong>AI:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        c1, c2 = st.columns([4, 1])
        with c1:
            user_input = st.text_input(
                "Ask a question about the team:",
                placeholder="e.g., 'Summarize our season performance'",
                key="ai_chat_input_games",
            )
        with c2:
            send_button = st.button("Send", type="primary")

        # Quick action buttons (known data only)
        st.markdown("**Quick Actions:**")
        q2, q3 = st.columns(2)
        with q2:
            if st.button("Season Summary", help="Get a comprehensive overview of your season"):
                user_input = (
                    "Provide a comprehensive summary of our season performance including strengths, "
                    "weaknesses, and key insights"
                )
                send_button = True
        with q3:
            if st.button("Performance Trends", help="Analyze trends and identify improvement areas"):
                user_input = "Analyze our performance trends and identify areas for improvement"
                send_button = True

        # Process user input
        if send_button and user_input and user_input.strip():
            # Add user message to history
            st.session_state.ai_chat_history.append({"role": "user", "content": user_input})

            # Show loading spinner
            with st.spinner("AI is analyzing..."):
                # Known-data analysis only
                ai_response = generate_ai_team_analysis(
                    user_input,
                    matches_view,
                    players,
                    events_view,
                    plays_view,
                    ga_view,
                )

            # Add AI response to history
            if ai_response:
                st.session_state.ai_chat_history.append({"role": "assistant", "content": ai_response})
            else:
                st.session_state.ai_chat_history.append({
                    "role": "assistant",
                    "content": (
                        "I'm sorry, I couldn't generate a response. "
                        "Please make sure you have a Gemini API key configured and try again."
                    ),
                })

            # Clear input and rerun to show new message
            st.rerun()

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.ai_chat_history = []
            st.rerun()

    elif selected_tab == "Trends":
        if matches_view.empty:
            st.info("No games yet to build trends.")
        else:
            # Comparison between all games vs last 3 games
            comparison_df = build_comparison_trend_frame(matches_view)
            individual_df = build_individual_game_trends(matches_view)

            st.subheader("All Games vs Last 3 Games Comparison")

            # Display comparison table
            st.dataframe(
                comparison_df.round(2),
                use_container_width=True,
                hide_index=True,
                height=200
            )

            # Create comparison charts
            label_axis = alt.Axis(labelAngle=-45) if compact else alt.Axis()
            h = 220

            # Melt the comparison data for better charting
            comparison_melted = comparison_df.melt(
                id_vars=["Metric"],
                value_vars=["All Games", "Last 3 Games"],
                var_name="Period",
                value_name="Value"
            )

            # Comparison bar chart
            comparison_chart = alt.Chart(comparison_melted).mark_bar().encode(
                x=alt.X("Metric:N", title="Metric", axis=label_axis),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Period:N", title="Period"),
                tooltip=["Metric", "Period", "Value"]
            ).properties(height=h)
            st.altair_chart(comparison_chart, use_container_width=True)

            st.subheader("Individual Game Performance")

            # Individual game trends
            for col, title in [
                ("GF", "Goals For"),
                ("GA", "Goals Against"),
                ("Save%", "Save %"),
                ("GF Conv%", "Conversion % (For)"),
                ("GA Conv%", "Conversion % (Against)")
            ]:
                # Create chart with different colors for last 3 games
                chart = alt.Chart(individual_df).mark_circle(size=60).encode(
                    x=alt.X("Game #:O", title="Game Number"),
                    y=alt.Y(f"{col}:Q", title=title),
                    color=alt.Color("Last 3 Games:N",
                                  scale=alt.Scale(domain=[True, False], range=["#ff6b6b", "#4ecdc4"]),
                                  title="Last 3 Games"),
                    tooltip=["Game #", "Date", "Opponent", col, "Last 3 Games"]
                ).properties(height=h)

                # Add trend line
                trend_line = alt.Chart(individual_df).mark_line(color="gray", opacity=0.5).encode(
                    x=alt.X("Game #:O"),
                    y=alt.Y(f"{col}:Q")
                ).properties(height=h)

                final_chart = (chart + trend_line).resolve_scale(color="independent")
                st.altair_chart(final_chart, use_container_width=True)

    elif selected_tab == "Leaders":
        render_points_leaderboard(events_view, players, top_n=5, compact=compact)

    elif selected_tab == "Goals Allowed":
        render_goals_allowed_analysis(ga_view, matches_view, players, compact=compact)

    elif selected_tab == "Set Pieces":
        render_set_piece_analysis_from_plays(plays_view, matches_view, players)
