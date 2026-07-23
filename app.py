# app.py
import os
import re
from typing import Optional, Dict
from urllib.parse import urlencode, urljoin

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

from app_pages.home import HomeHandlers
from data.maxpreps import (
    MAXPREPS_D2_URL,
    MAXPREPS_RANKINGS_URL,
    MAXPREPS_SCHEDULE_URL,
    parse_maxpreps_division_rank,
    parse_maxpreps_next_opponent,
)
from data.metrics import calculate_shot_on_target_percentages
import requests
from dotenv import load_dotenv

# Centralized cached data loaders
from loaders import (
    load_seasons,
    load_matches,
    load_players,
    load_events,
    load_plays_simple,
    load_summaries,
    load_goals_allowed,
)

# Optional Groq import (guarded)
try:
    from groq import Groq
except Exception:
    Groq = None


def _groq_chat(system_prompt: str, user_prompt: str, *, temperature: float = 0.2) -> str:
    """Return a chat completion from Groq.

    Expects GROQ_API_KEY in env/secrets. Raises on failure.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    if Groq is None:
        raise RuntimeError("groq package not installed")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return (completion.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
st.set_page_config(page_title="HS Soccer Dashboard", layout="wide")

# Debug flag for AI issues (set DEBUG_AI=true in .env)
DEBUG_AI = os.getenv("DEBUG_AI", "").strip().lower() in ("1", "true", "yes", "on")

def _record_ai_error(context: str, err: Exception) -> None:
    """Store AI errors in session state when DEBUG_AI is enabled."""
    if not DEBUG_AI:
        return
    try:
        st.session_state["ai_last_error_context"] = context
        st.session_state["ai_last_error"] = str(err)
    except Exception:
        pass

def _ai_user_error_message(default_msg: str) -> str:
    """Return a simplified user-facing AI error message."""
    err = str(st.session_state.get("ai_last_error", "")).lower()
    if "quota" in err or "rate limit" in err or "429" in err:
        return "AI quota limit reached. Please try again later or check billing."
    return default_msg

def _render_ai_debug() -> None:
    """Render last AI error in the UI when DEBUG_AI is enabled."""
    if not DEBUG_AI:
        return
    err = st.session_state.get("ai_last_error")
    if err:
        ctx = st.session_state.get("ai_last_error_context", "unknown")
        st.caption(f"AI debug ({ctx}): {err}")

# Load local .env (for local dev)
load_dotenv()

# Streamlit Secrets fallback (for Cloud)
try:
    SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY") or st.secrets.get("SPREADSHEET_KEY", "YOUR_SPREADSHEET_KEY_OR_URL")
    for _k in ["GROQ_API_KEY", "APP_PASSWORD", "GOOGLE_SERVICE_ACCOUNT_JSON"]:
        if not os.getenv(_k) and _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    SPREADSHEET_KEY = os.getenv("SPREADSHEET_KEY", "YOUR_SPREADSHEET_KEY_OR_URL")

# Dashboard visual system + mobile polish
def _inject_css():
    st.markdown(
        """
        <style>
          :root {
            --milton-ink: #102235;
            --milton-navy: #0b1b2b;
            --milton-navy-soft: #152b40;
            --milton-yellow: #f4c542;
            --milton-canvas: #f5f6f3;
            --milton-surface: #ffffff;
            --milton-line: #dfe4e7;
            --milton-muted: #66727d;
          }

          [data-testid="stAppViewContainer"] {
            background: var(--milton-canvas);
          }
          [data-testid="stHeader"] {
            background: rgba(245, 246, 243, 0.88);
            backdrop-filter: blur(10px);
          }
          .block-container {
            max-width: 1260px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
          }

          /* ----- Team masthead ----- */
          .dashboard-masthead {
            position: relative;
            padding: 0.35rem 0 1.5rem 1.15rem;
            margin-bottom: 0.25rem;
            border-left: 5px solid var(--milton-yellow);
            animation: dashboard-enter 420ms ease-out both;
          }
          .dashboard-kicker {
            color: #53616d;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.14em;
            text-transform: uppercase;
          }
          .dashboard-masthead h1 {
            color: var(--milton-ink);
            font-size: clamp(2rem, 4vw, 3.25rem);
            font-weight: 780;
            letter-spacing: -0.045em;
            line-height: 1.02;
            margin: 0.45rem 0 0.55rem;
          }
          .dashboard-masthead p {
            color: var(--milton-muted);
            font-size: 0.98rem;
            margin: 0;
          }

          /* ----- Sidebar brand and controls ----- */
          section[data-testid="stSidebar"] {
            background: var(--milton-navy);
            border-right: 1px solid #20374b;
          }
          section[data-testid="stSidebar"] > div {
            background: transparent;
          }
          section[data-testid="stSidebar"] p,
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] h1,
          section[data-testid="stSidebar"] h2,
          section[data-testid="stSidebar"] h3 {
            color: #edf2f4;
          }
          .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.35rem 0 1.25rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.13);
          }
          .sidebar-brand-mark {
            display: grid;
            width: 2.55rem;
            height: 2.55rem;
            place-items: center;
            flex: 0 0 auto;
            border-radius: 0.72rem;
            background: var(--milton-yellow);
            color: var(--milton-navy);
            font-size: 1.05rem;
            font-weight: 850;
            letter-spacing: -0.04em;
          }
          .sidebar-brand-copy strong,
          .sidebar-brand-copy span {
            display: block;
          }
          .sidebar-brand-copy strong {
            color: #ffffff;
            font-size: 1.05rem;
            line-height: 1.15;
          }
          .sidebar-brand-copy span {
            color: #9fb0bf;
            font-size: 0.72rem;
            letter-spacing: 0.09em;
            margin-top: 0.2rem;
            text-transform: uppercase;
          }
          section[data-testid="stSidebar"] [data-baseweb="select"] > div,
          section[data-testid="stSidebar"] input {
            background: var(--milton-navy-soft);
            border-color: #385066;
            color: #ffffff;
          }
          section[data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #ffffff;
          }
          section[data-testid="stSidebar"] [data-testid="stButton"] button {
            border-color: #476077;
            background: transparent;
            color: #ffffff;
          }
          section[data-testid="stSidebar"] [data-testid="stLinkButton"] a {
            border-color: rgba(244,197,66,0.62);
            color: var(--milton-yellow);
            background: rgba(244,197,66,0.06);
          }
          section[data-testid="stSidebar"] button:hover,
          section[data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover {
            border-color: var(--milton-yellow);
            background: rgba(244,197,66,0.12);
          }

          /* ----- Data health + KPI hierarchy ----- */
          [data-testid="stExpander"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--milton-line);
            border-radius: 0.75rem;
          }
          [data-testid="stMetric"] {
            min-height: 6.4rem;
            padding: 0.85rem 0.9rem 0.75rem;
            border-top: 3px solid #d9dee1;
            background: rgba(255,255,255,0.78);
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
            animation: metric-enter 380ms ease-out both;
          }
          [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-top-color: var(--milton-yellow);
            background: #ffffff;
          }
          [data-testid="stMetricLabel"] p {
            color: var(--milton-muted);
            font-size: 0.7rem;
            font-weight: 720;
            letter-spacing: 0.045em;
            line-height: 1.2;
            overflow: visible;
            text-overflow: clip;
            text-transform: uppercase;
            white-space: normal;
          }
          [data-testid="stMetricLabel"],
          [data-testid="stMetricLabel"] > div {
            overflow: visible;
          }
          [data-testid="stMetricValue"] {
            color: var(--milton-ink);
            font-weight: 760;
            letter-spacing: -0.035em;
          }

          /* ----- Navigation + actions ----- */
          [data-testid="stRadio"] > div {
            gap: 0.25rem;
            border-bottom: 1px solid var(--milton-line);
          }
          [data-testid="stRadio"] label {
            padding: 0.6rem 0.72rem 0.7rem;
            border-bottom: 2px solid transparent;
            transition: color 150ms ease, border-color 150ms ease, background 150ms ease;
          }
          [data-testid="stRadio"] label[data-selected="true"] {
            border-bottom-color: var(--milton-yellow);
            color: var(--milton-ink);
            font-weight: 700;
          }
          [data-testid="stRadioOption"] > div > div > div:first-child {
            display: none;
          }
          [data-testid="stButton"] button,
          [data-testid="stLinkButton"] a,
          [data-testid="stDownloadButton"] button {
            border-radius: 0.55rem;
            transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
          }
          [data-testid="stButton"] button:hover,
          [data-testid="stLinkButton"] a:hover,
          [data-testid="stDownloadButton"] button:hover {
            transform: translateY(-1px);
          }
          button[kind="primary"] {
            background: var(--milton-navy) !important;
            border-color: var(--milton-navy) !important;
          }

          a.tiny-open {
            display:inline-block;
            padding:4px 8px;
            font-size:12px;
            line-height:1;
            border-radius:999px;
            background:#e9eef1;
            color:var(--milton-ink);
            text-decoration:none;
          }
          a.tiny-open:hover { background:var(--milton-yellow); }

          /* ----- Game cards (existing) ----- */
          .game-card {
            border:1px solid var(--milton-line);
            border-left:4px solid var(--milton-yellow);
            border-radius:10px;
            padding:12px 14px;
            margin:10px 0 14px;
            background:#ffffff;
            box-shadow:none;
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
            border:1px solid var(--milton-line);
            border-top:3px solid var(--milton-yellow);
            border-radius:10px;
            background:#fff;
            padding:12px 14px;
            box-shadow:none;
          }
          .stat-label { font-size:.85rem; color:#6b7280; margin-bottom:4px; }
          .stat-value { font-size:1.6rem; font-weight:700; line-height:1.1; }
          .stat-sub { font-size:.8rem; color:#6b7280; margin-top:2px; }

          @media (max-width: 480px) {
            .block-container { padding-top: 3.5rem; padding-left: 0.75rem; padding-right: 0.75rem; }
            .dashboard-masthead { padding-left: 0.85rem; padding-bottom: 1.1rem; }
            .dashboard-masthead h1 { font-size: 2rem; }
            [data-testid="stMetric"] { min-height: 5.5rem; padding: 0.7rem; }
            [data-testid="stRadio"] label { padding-left: 0.4rem; padding-right: 0.4rem; }
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
            background: #ffffff;
            border-left: 4px solid #8a98a4;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: none;
          }
          .ai-chat-user {
            background: #fff9e5;
            border-left-color: var(--milton-yellow);
          }
          .ai-chat-assistant {
            background: #eef2f4;
            border-left-color: var(--milton-navy-soft);
          }

          @keyframes dashboard-enter {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes metric-enter {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
              animation-duration: 0.01ms !important;
              animation-iteration-count: 1 !important;
              transition-duration: 0.01ms !important;
              scroll-behavior: auto !important;
            }
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

def _qparams_get() -> dict[str, str]:
    """Return the current URL query parameters as a plain dictionary."""
    return st.query_params.to_dict()

def _qparams_set(**kwargs):
    """Replace all URL query parameters in one frontend update."""
    st.query_params.from_dict(kwargs)

def _qparams_merge_update(**kwargs):
    """Merge update query params without dropping existing ones."""
    query_params = st.query_params.to_dict()
    query_params.update({key: value for key, value in kwargs.items() if value is not None})
    _qparams_set(**query_params)

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
# (moved to loaders.py)
# ---------------------------------------------------------------------
# RANKINGS HELPERS (for D2 Rank KPI)
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
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        if DEBUG_AI:
            _record_ai_error(
                "generate_ai_game_summary",
                Exception("Missing GROQ_API_KEY or groq import failed"),
            )
        return None
    try:
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

        
        text = _groq_chat(sys, str(user), temperature=0.2)
        return text or None
    except Exception as e:
        _record_ai_error("generate_ai_game_summary", e)
        return None

# --- AI: conceded goals summary ---
def generate_ai_conceded_summary(ga_df: pd.DataFrame,
                                 matches: pd.DataFrame,
                                 players: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        if DEBUG_AI:
            _record_ai_error(
                "generate_ai_conceded_summary",
                Exception("Missing GROQ_API_KEY or groq import failed"),
            )
        return None
    try:
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

        
        text = _groq_chat("You are a concise assistant that summarizes soccer match data for a coach. Use bullet points when helpful.", prompt, temperature=0.2)
        return text or None
    except Exception as e:
        _record_ai_error("generate_ai_conceded_summary", e)
        return None

# --- AI: General team analysis and Q&A ---
def generate_ai_team_analysis(query: str,
                             matches: pd.DataFrame,
                             players: pd.DataFrame,
                             events: pd.DataFrame,
                             plays_df: pd.DataFrame,
                             goals_allowed: pd.DataFrame) -> Optional[str]:
    """Generate AI analysis based on user query about team performance."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        if DEBUG_AI:
            _record_ai_error(
                "generate_ai_team_analysis",
                Exception("Missing GROQ_API_KEY or groq import failed"),
            )
        return None
    
    try:
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

        
        text = _groq_chat(system_prompt, user_prompt, temperature=0.2)
        return text or None
    except Exception as e:
        _record_ai_error("generate_ai_team_analysis", e)
        return None

def get_next_opponent_from_schedule() -> Optional[Dict[str, str]]:
    """Return next opponent using the Google Sheet schedule when possible.

    Falls back to MaxPreps scraping only if the sheet is unavailable.
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

    # 2) Fallback to MaxPreps' structured schedule data.
    try:
        html = fetch_html(MAXPREPS_SCHEDULE_URL)
        return parse_maxpreps_next_opponent(html)
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

def find_opponent_schedule_url(opponent_name: str) -> Optional[str]:
    """Find an opponent's MaxPreps schedule URL from Milton's schedule page."""
    try:
        html = fetch_html(MAXPREPS_SCHEDULE_URL)
        links = _extract_links_with_text(html)
        target = opponent_name.lower().strip()
        for href, text in links:
            if target in text.lower() and "/soccer/" in href:
                team_url = urljoin(MAXPREPS_SCHEDULE_URL, href).split("?")[0]
                if "/match/" in team_url:
                    continue
                if not team_url.endswith("/"):
                    team_url += "/"
                return team_url if team_url.endswith("/schedule/") else team_url + "schedule/"
    except Exception:
        return None
    return None

def scrape_team_schedule_stats(schedule_url: str) -> Optional[Dict[str, any]]:
    """Fetch a MaxPreps team schedule and derive rough W-L-D, GF, GA and opponents.
    This is a best-effort text parse; if it fails, returns None.
    """
    try:
        html = fetch_html(schedule_url)
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
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        if DEBUG_AI:
            _record_ai_error(
                "generate_ai_opponent_analysis",
                Exception("Missing GROQ_API_KEY or groq import failed"),
            )
        return None
    
    try:
        # Get historical data about opponent + simple prediction
        opponent_analysis = analyze_opponent_from_data(opponent_name, matches)
        prediction = predict_vs_opponent(matches, opponent_name)

        # Try to enrich with scraped opponent season and common-opponent stats
        opponent_schedule_url = find_opponent_schedule_url(opponent_name)
        opponent_stats = scrape_team_schedule_stats(opponent_schedule_url) if opponent_schedule_url else None
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

        
        text = _groq_chat(system_prompt, user_prompt, temperature=0.2)
        return text or None
    except Exception as e:
        _record_ai_error("generate_ai_opponent_analysis", e)
        return None

# --- AI: set-piece analysis summary ---
def generate_ai_set_piece_summary(plays_df: pd.DataFrame,
                                  matches: pd.DataFrame,
                                  players: pd.DataFrame) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or Groq is None:
        if DEBUG_AI:
            _record_ai_error(
                "generate_ai_set_piece_summary",
                Exception("Missing GROQ_API_KEY or groq import failed"),
            )
        return None
    
    try:
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

        
        text = _groq_chat("You are a concise assistant that summarizes soccer match data for a coach. Use bullet points when helpful.", prompt, temperature=0.2)
        return text or None
    except Exception as e:
        _record_ai_error("generate_ai_set_piece_summary", e)
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
    shots_target_pct, shots_against_target_pct = calculate_shot_on_target_percentages(matches_view)
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
            ("SOT% (For)", f"{shots_target_pct:.1f}%"),
            ("Shots (Agst)", sh_ag),
            ("SOT% (Agst)", f"{shots_against_target_pct:.1f}%"),
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

    # ---------- Desktop: separate volume from efficiency for legibility ----------
    volume_cols = st.columns(7)
    volume_cols[0].metric("Games", games)
    volume_cols[1].metric("Record", record_str)
    volume_cols[2].metric("GF", gf)
    volume_cols[3].metric("GA", ga)
    volume_cols[4].metric("Shots (For)", sh_for)
    volume_cols[5].metric("Shots (Agst)", sh_ag)
    if d2_rank:
        volume_cols[6].metric("D2 Rank", f"{d2_rank}{_suffix(d2_rank)}")
    else:
        volume_cols[6].metric("D2 Rank", "N/A")

    efficiency_cols = st.columns(6)
    efficiency_cols[0].metric("SOT% (For)", f"{shots_target_pct:.1f}%")
    efficiency_cols[1].metric("SOT% (Agst)", f"{shots_against_target_pct:.1f}%")
    efficiency_cols[2].metric("Conv% (For)", f"{conv_for_pct:.1f}%")
    efficiency_cols[3].metric("Conv% (Agst)", f"{conv_agn_pct:.1f}%")
    efficiency_cols[4].metric("Saves", sv)
    efficiency_cols[5].metric("Save%", f"{save_pct:.1f}%")
    if not d2_rank:
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
            season_id = str(r.get("season_id", "")).strip()
            game_url = "?" + urlencode({"season": season_id, "match_id": mid})

            card = f"""
            <a href='{game_url}' style='text-decoration:none; color:inherit;'>
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
        season_id = str(r.get("season_id", "")).strip()
        game_url = "?" + urlencode({"season": season_id, "match_id": mid})
        cols[3].markdown(f"<a href='{game_url}' style='text-decoration:none'>{_color_opp(r.get('opponent',''), r.get('result',''))}</a>", unsafe_allow_html=True)
        cols[4].write(r.get("home_away",""))
        cols[5].write("Yes" if r.get("division_game", False) else "No")
        cols[6].write(r.get("GF-GA",""))
        cols[7].write(r.get("shots_for", r.get("shots","")))
        cols[8].write(r.get("saves",""))
        cols[9].markdown(f"<a class='tiny-open' href='{game_url}' title='Open game'>Open</a>", unsafe_allow_html=True)

    # CSV download of games
    try:
        export_cols = [c for c in ["season_id","date","match_id","opponent","home_away","division_game","GF-GA","shots_for","saves"] if c in view.columns]
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
        st.dataframe(show, width="stretch", hide_index=True, height=180)
    else:
        st.dataframe(
            top[["","jersey","name","goals","assists","points"]],
            width="stretch",
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
            st.altair_chart(chart_pts, width="stretch")
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
            st.altair_chart(chart_breakdown, width="stretch")

    with st.expander("View full team leaderboard"):
        st.dataframe(
            full[cols_full],
            width="stretch",
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

def render_set_piece_analysis_from_plays(
    plays_df: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    *,
    season_plays_df: pd.DataFrame,
):
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
    season_df = season_plays_df.copy()
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
    st.dataframe(tbl, width="stretch", hide_index=True)

    # ---- Chart (unchanged) ----
    if not tbl.empty:
        # Keep Goal% on Y, but order by attempts (desc) on X
        chart = alt.Chart(tbl).mark_bar().encode(
            x=alt.X("Play Call:N", sort=alt.SortField(field="attempts", order="descending"), title="Play Call"),
            y=alt.Y("Goal%:Q", title="Goal %"),
            color=alt.Color("set_piece:N", title="Type"),
            tooltip=list(tbl.columns),
        ).properties(height=280)
        st.altair_chart(chart, width="stretch")

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
            st.session_state[error_key] = "AI summary unavailable (no Groq key set or not enough context)."

    summary_text = st.session_state.get(state_key)
    summary_error = st.session_state.get(error_key)
    if summary_text:
        st.markdown("**AI Set-Piece Analysis & Recommendations**")
        st.write(summary_text)
    elif summary_error:
        st.caption(_ai_user_error_message(summary_error))
        _render_ai_debug()



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
                 width="stretch", hide_index=True, height=320)
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

    st.altair_chart(chart_sit | chart_min, width="stretch")
    st.altair_chart(chart_gk, width="stretch")

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
            st.session_state[error_key] = "AI summary unavailable (no Groq key set or not enough context)."

    conceded_summary = st.session_state.get(state_key)
    conceded_error = st.session_state.get(error_key)
    if conceded_summary:
        st.markdown("**AI Defensive Summary & Recommendations**")
        st.write(conceded_summary)
    elif conceded_error:
        st.caption(_ai_user_error_message(conceded_error))
        _render_ai_debug()

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
all_seasons = load_seasons(SPREADSHEET_KEY)
all_matches = load_matches(SPREADSHEET_KEY)
all_players = load_players(SPREADSHEET_KEY)
all_events = load_events(SPREADSHEET_KEY)
all_plays_simple = load_plays_simple(SPREADSHEET_KEY)
all_summaries = load_summaries(SPREADSHEET_KEY)
all_goals_allowed = load_goals_allowed(SPREADSHEET_KEY)

from data.seasons import build_season_catalog, resolve_season_id, season_is_active, season_label
from data.views import (
    apply_match_filters,
    derive_related_views,
    filter_by_season,
    filter_players_for_season,
    get_match_id,
)
from ui.sidebar import render_sidebar

season_catalog = build_season_catalog(all_seasons, all_matches)
season_options = season_catalog["season_id"].astype(str).tolist()
season_labels = dict(zip(season_options, season_catalog["label"].astype(str)))
active_season_id = resolve_season_id(None, season_catalog)
requested_season = _qparams_get().get("season")
default_season = resolve_season_id(requested_season, season_catalog)

# Sidebar (clean labels)
compact, div_only, selected_season = render_sidebar(
    qparams_get=_qparams_get,
    qp_bool=_qp_bool,
    qparams_set=_qparams_set,
    qparams_merge_update=_qparams_merge_update,
    season_options=season_options,
    season_labels=season_labels,
    default_season=default_season,
    schedule_url=MAXPREPS_SCHEDULE_URL,
    rankings_url=MAXPREPS_D2_URL,
)

# Scope every table before applying match filters so IDs can safely repeat by season.
matches = filter_by_season(all_matches, selected_season)
players = filter_players_for_season(
    all_players,
    selected_season,
    active_season_id=active_season_id,
)
events = filter_by_season(all_events, selected_season)
plays_simple = filter_by_season(all_plays_simple, selected_season)
summaries = filter_by_season(all_summaries, selected_season)
goals_allowed = filter_by_season(all_goals_allowed, selected_season)

# Apply filters (division/date/opponent/H-A)
qp = _qparams_get()
opp_filter = str(qp.get("opp", ""))
if isinstance(opp_filter, list):
    opp_filter = opp_filter[0]

ha_val = str(qp.get("ha", "any")).lower()
if isinstance(ha_val, list):
    ha_val = ha_val[0]

matches_view = apply_match_filters(matches, div_only=div_only, opp_filter=opp_filter, ha_val=ha_val)

# Derive related views by match_id
events_view, plays_view, ga_view = derive_related_views(
    matches_view=matches_view,
    events=events,
    plays_simple=plays_simple,
    goals_allowed=goals_allowed,
)

# Drill-in param
qp = _qparams_get()
match_id = get_match_id(qp)

# D2 rank (KPI only)
our_rank = None
if season_is_active(season_catalog, selected_season):
    try:
        rankings_html = fetch_html(MAXPREPS_RANKINGS_URL)
        our_rank = parse_maxpreps_division_rank(rankings_html)
    except Exception:
        our_rank = None

from app_context import AppContext
from router import route

# Routing
ctx = AppContext(
    compact=compact,
    div_only=div_only,
    season_id=selected_season,
    season_label=season_label(season_catalog, selected_season),
    season_is_active=season_is_active(season_catalog, selected_season),
    matches=matches,
    players=players,
    events=events,
    plays_simple=plays_simple,
    summaries=summaries,
    goals_allowed=goals_allowed,
    matches_view=matches_view,
    events_view=events_view,
    plays_view=plays_view,
    ga_view=ga_view,
    match_id=match_id,
    our_rank=our_rank,
)

handlers = HomeHandlers(
    team_kpis=_team_kpis,
    render_games_table=render_games_table,
    render_points_leaderboard=render_points_leaderboard,
    render_goals_allowed_analysis=render_goals_allowed_analysis,
    render_set_piece_analysis_from_plays=render_set_piece_analysis_from_plays,
    qparams_set=_qparams_set,
    format_date=_format_date,
    generate_ai_game_summary=generate_ai_game_summary,
    build_comparison_trend_frame=build_comparison_trend_frame,
    build_individual_game_trends=build_individual_game_trends,
    generate_ai_team_analysis=generate_ai_team_analysis,
    ai_user_error_message=_ai_user_error_message,
    render_ai_debug=_render_ai_debug,
)

route(ctx=ctx, handlers=handlers)
