# app.py
import os
import re
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
        st.title("🔒 Coaches Only")
        entered = st.text_input("Enter password", type="password")
        if st.button("Enter"):
            st.session_state.authed = (entered == pwd)
        st.stop()

require_app_password()

# External (SI/SBLive) — used only for D2 Rank KPI and quick links
SI_BASE = "https://www.si.com/high-school/stats/vermont"
DIVISION_SLUGS = {2: "28806-division-2"}
SI_TEAM_SLUG = "397925-milton-yellowjackets"
SI_RANKINGS_URL = f"{SI_BASE}/{DIVISION_SLUGS[2]}/soccer/rankings?formula=DIVISION_POINT_INDEX"
SI_SCHEDULE_URL = f"{SI_BASE}/soccer/teams/{SI_TEAM_SLUG}/games"
TEAM_NAME_CANON = "Milton"

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def _bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true","1","yes","y"])

def _qparams_get():
    try: return st.query_params
    except Exception: return st.experimental_get_query_params()

def _qparams_set(**kwargs):
    try:
        st.query_params.clear()
        for k,v in kwargs.items(): st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)

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
    try:
        df = read_sheet_to_df(SPREADSHEET_KEY, "summary")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "match_id" in df.columns:
            df["match_id"] = df["match_id"].astype(str)
        return df
    except Exception:
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
# SI RANKINGS HELPERS (for D2 Rank KPI)
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_html(url: str) -> str:
    r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.text

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
    text = _clean_text(html)
    pairs = re.findall(r"\b(\d{1,2})\s+([A-Z][A-Za-z0-9.\-\' ]{2,})", text)
    ranks: Dict[str,int] = {}
    for num, name in pairs:
        try:
            rank = int(num)
        except Exception:
            continue
        name = name.strip()
        if rank not in ranks.values():
            ranks[name] = rank
    return ranks

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
        return pd.DataFrame(columns=["set_piece", "Play Call", "play_type", "attempts", "Goal%"])

    grp = (
        plays_df.groupby(["play_call_id", "set_piece", "play_type"], dropna=False)
        .agg(attempts=("play_call_id", "count"), goal_rate=("goal_created", "mean"))
        .reset_index()
    )
    grp["Goal%"] = (grp["goal_rate"] * 100).round(1)

    out = grp.rename(columns={"play_call_id": "Play Call"})
    return out[["set_piece", "Play Call", "play_type", "attempts", "Goal%"]].sort_values(
        ["Goal%", "attempts", "Play Call"], ascending=[False, False, True]
    )

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

# --- AI: match summary ---
def generate_ai_game_summary(match_row: pd.Series,
                             notes_row: Optional[pd.Series],
                             events: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

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
        model = genai.GenerativeModel("gemini-1.5-flash")

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
                <a class="tiny-open" href='?match_id={mid}' title='Open game'>↗ Open</a>
              </div>
            </div>
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
        cols[3].markdown(_color_opp(r.get("opponent",""), r.get("result","")), unsafe_allow_html=True)
        cols[4].write(r.get("home_away",""))
        cols[5].write("✅" if r.get("division_game", False) else "–")
        cols[6].write(r.get("GF-GA",""))
        cols[7].write(r.get("shots_for", r.get("shots","")))
        cols[8].write(r.get("saves",""))
        mid = str(r.get("match_id","") or f"row{idx}")
        cols[9].markdown(f"<a class='tiny-open' href='?match_id={mid}' title='Open game'>↗</a>", unsafe_allow_html=True)

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
        return "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else ""))
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

def _set_piece_type_stats(df: pd.DataFrame, sp_type: str) -> tuple[int, float]:
    """Return (total_attempts, pct_scored) for a given set_piece type (lowercase)."""
    if df.empty or "set_piece" not in df.columns:
        return 0, 0.0
    sub = df[df["set_piece"].astype(str).str.lower() == sp_type]
    total = int(len(sub))
    pct = float(sub["goal_created"].mean() * 100) if total > 0 and "goal_created" in sub.columns else 0.0
    return total, pct

def render_set_piece_analysis_from_plays(plays_df: pd.DataFrame):
    st.subheader("Set-Piece Analysis")

    # ---- Guard + normalize ----
    if plays_df is None or plays_df.empty:
        st.info("No set-play rows yet. Add data to the `plays` sheet.")
        return

    df = plays_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "set_piece" not in df.columns:
        df["set_piece"] = ""
    if "goal_created" not in df.columns:
        df["goal_created"] = False
    df["set_piece"] = df["set_piece"].astype(str).str.lower()
    df["goal_created"] = df["goal_created"].astype(bool)

    # ---- KPI tiles (mobile-friendly card grid) ----
    corners_total, corners_pct   = _set_piece_type_stats(df, "corner")
    pens_total, pens_pct         = _set_piece_type_stats(df, "penalty")
    fk_dir_total, fk_dir_pct     = _set_piece_type_stats(df, "fk_direct")
    fk_ind_total, fk_ind_pct     = _set_piece_type_stats(df, "fk_indirect")

    kpi_items = [
        ("Corners",      corners_total,  corners_pct),
        ("Penalties",    pens_total,     pens_pct),
        ("Direct FK",    fk_dir_total,   fk_dir_pct),
        ("Indirect FK",  fk_ind_total,   fk_ind_pct),
    ]

    kpi_html = "<div class='kpi-grid'>" + "".join(
        f"<div class='stat-card'><div class='stat-label'>{name}</div><div class='stat-value'>{total}</div><div class='stat-sub'>Scored {pct:.1f}%</div></div>"
        for (name, total, pct) in kpi_items
    ) + "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ---- Table (unchanged) ----
    tbl = set_piece_leaderboard_from_plays(df)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ---- Chart (unchanged) ----
    if not tbl.empty:
        chart = alt.Chart(tbl).mark_bar().encode(
            x=alt.X("Play Call:N", sort="-y", title="Play Call"),
            y=alt.Y("Goal%:Q", title="Goal %"),
            color=alt.Color("set_piece:N", title="Type"),
            tooltip=list(tbl.columns),
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)



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
        try: view["date"] = pd.to_datetime(view["date"], errors="coerce")
        except Exception: pass
    else:
        view["opponent"] = ""; view["date"] = pd.NaT

    view["minute_bucket"] = view["minute"].apply(_minute_bucket)

    cols_show = [c for c in ["date","opponent","minute","minute_bucket","situation","goalie_name","description","goal_id"] if c in view.columns]
    st.dataframe(view[cols_show].sort_values(["date","minute"], ascending=[True, True]),
                 use_container_width=True, hide_index=True, height=320)

    total_ga = len(view)
    games = len(matches) if not matches.empty else 0
    ga_per_game = (total_ga / games) if games > 0 else 0.0
    c1,c2,c3 = st.columns(3)
    c1.metric("Conceded (Total)", total_ga)
    c2.metric("Games", games)
    c3.metric("GA / Game", f"{ga_per_game:.2f}")

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

    if st.button("🔎 Generate AI Insights on Conceded Goals"):
        ai_txt = generate_ai_conceded_summary(view, matches, players)
        if ai_txt:
            st.markdown("**AI Defensive Summary & Recommendations**")
            st.write(ai_txt)
        else:
            st.caption("AI summary unavailable (no Gemini key set or not enough context).")

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

# Sidebar
with st.sidebar:
    st.title("HS Soccer")
    if st.button("🏠 Dashboard (Home)"):
        _qparams_set(); st.rerun()

    COMPACT_DEFAULT = True
    compact = st.toggle("📱 Compact mode", value=COMPACT_DEFAULT, help="Phone-friendly layout")

    div_only = st.checkbox("Division games only", value=False)

    st.link_button("📅 Open Schedule", SI_SCHEDULE_URL)
    st.link_button("🏆 Open Rankings (D2)", SI_RANKINGS_URL)

    st.divider()
    if st.button("🔄 Refresh Google Sheets"):
        st.cache_data.clear()
        st.success("Google Sheets cache cleared.")
        st.rerun()

# Filter if division-only
if div_only and not matches.empty:
    matches_view = matches.query("division_game == True")
    keep = set(matches_view["match_id"].astype(str))
    events_view = events[events["match_id"].astype(str).isin(keep)] if "match_id" in events.columns else events
    plays_view  = plays_simple[plays_simple["match_id"].astype(str).isin(keep)] if not plays_simple.empty else plays_simple
    ga_view     = goals_allowed[goals_allowed["match_id"].astype(str).isin(keep)] if not goals_allowed.empty else goals_allowed
else:
    matches_view, events_view, plays_view, ga_view = matches, events, plays_simple, goals_allowed

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
    si_html_rank = fetch_html(SI_RANKINGS_URL)
    ranks = parse_all_ranks_from_si(si_html_rank)
    our_rank = fuzzy_find_rank(ranks, TEAM_NAME_CANON)
except Exception:
    our_rank = None

# Routing
if match_id:
    render_game_drilldown(match_id, matches_view, players, events_view, plays_view, summaries)
else:
    st.header("Milton Varsity Boys Soccer Team 2025")
    _team_kpis(matches_view, d2_rank=our_rank, compact=compact)

    st.divider()
    render_games_table(matches_view, compact=compact)

    st.divider()
    exp = st.expander("Trends (Rolling 3 Games)", expanded=False)
    with exp:
        df_tr = build_trend_frame(matches_view)
        if df_tr.empty:
            st.info("No games yet to build trends.")
        else:
            label_axis = alt.Axis(labelAngle=-45) if compact else alt.Axis()
            h = 220
            for col, title in [
                ("R3 GF","Goals For (R3)"),
                ("R3 GA","Goals Against (R3)"),
                ("R3 Save%","Save% (R3)"),
                ("R3 GF Conv%","Conv% (For) (R3)"),
                ("R3 GA Conv%","Conv% (Agst) (R3)")
            ]:
                ch = alt.Chart(df_tr).mark_line(point=True).encode(
                    x=alt.X("Date:T", title="Date", axis=label_axis),
                    y=alt.Y(f"{col}:Q", title=title),
                    tooltip=["Date","GF","GA","Save%","GF Conv%","GA Conv%",col]
                ).properties(height=h)
                st.altair_chart(ch, use_container_width=True)

    st.divider()
    render_points_leaderboard(events_view, players, top_n=5, compact=compact)

    st.divider()
    render_goals_allowed_analysis(ga_view, matches_view, players, compact=compact)

    st.divider()
    render_set_piece_analysis_from_plays(plays_view)
