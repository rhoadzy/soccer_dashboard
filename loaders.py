"""Cached data loaders.

This module centralizes all Google Sheets reads and light cleaning.
The goal is to keep app.py focused on UI and analysis.

Behavior should match the original inline loader functions.
"""

import pandas as pd
import streamlit as st

from google_sheets_adapter import read_sheet_to_df


def _bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def _strip_and_alias_matches(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip()
    if "shots_for" not in df.columns and "shots" in df.columns:
        df = df.rename(columns={"shots": "shots_for"})
    return df


def _normalize_season_id(df: pd.DataFrame) -> pd.DataFrame:
    if "season_id" in df.columns:
        df["season_id"] = df["season_id"].astype(str).str.strip()
    return df


@st.cache_data(ttl=300)
def load_seasons(spreadsheet_key: str) -> pd.DataFrame:
    try:
        df = read_sheet_to_df(spreadsheet_key, "seasons")
    except Exception:
        return pd.DataFrame(columns=["season_id", "label", "active"])
    df.columns = [str(column).strip().lower() for column in df.columns]
    return _normalize_season_id(df)


@st.cache_data(ttl=300)
def load_matches(spreadsheet_key: str) -> pd.DataFrame:
    df = read_sheet_to_df(spreadsheet_key, "matches")
    df = _strip_and_alias_matches(df)
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "division_game" in df:
        df["division_game"] = _bool_col(df["division_game"])
    if "home_away" in df:
        df["home_away"] = (
            df["home_away"].astype(str).str.strip().str.lower().map({"h": "H", "home": "H", "a": "A", "away": "A"})
        )
    for c in [
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_target",
        "shots_against",
        "shots_against_target",
        "saves",
    ]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if {"goals_for", "goals_against"}.issubset(df):
        df["result"] = df.apply(
            lambda r: "W" if r.goals_for > r.goals_against else ("L" if r.goals_for < r.goals_against else "D"), axis=1
        )
    if "match_id" not in df:
        df["match_id"] = df.index.astype(str)
    else:
        df["match_id"] = df["match_id"].astype(str)
    return _normalize_season_id(df)


@st.cache_data(ttl=300)
def load_players(spreadsheet_key: str) -> pd.DataFrame:
    df = read_sheet_to_df(spreadsheet_key, "players")
    df.columns = [str(column).strip().lower() for column in df.columns]
    if "jersey" in df:
        df["jersey"] = pd.to_numeric(df["jersey"], errors="coerce").fillna(0).astype(int)
    if "player_id" in df:
        df["player_id"] = df["player_id"].astype(str)
    if "player_status" in df:
        df["player_status"] = df["player_status"].astype(str).str.strip().str.lower()
    return _normalize_season_id(df)


@st.cache_data(ttl=300)
def load_events(spreadsheet_key: str) -> pd.DataFrame:
    df = read_sheet_to_df(spreadsheet_key, "events")
    df.columns = [c.strip().lower() for c in df.columns]
    if "assist" in df.columns and "assists" not in df.columns:
        df = df.rename(columns={"assist": "assists"})
    for k in ["event_id", "match_id", "player_id"]:
        if k in df.columns:
            df[k] = df[k].astype(str)
    for n in ["goals", "assists", "shots", "fouls"]:
        if n not in df.columns:
            df[n] = 0
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0).astype(int)
    return _normalize_season_id(df)


@st.cache_data(ttl=300)
def load_plays_simple(spreadsheet_key: str) -> pd.DataFrame:
    try:
        raw = read_sheet_to_df(spreadsheet_key, "plays")
    except Exception:
        return pd.DataFrame()
    raw.columns = [c.lower().strip() for c in raw.columns]
    if "play type" in raw and "play_type" not in raw:
        raw = raw.rename(columns={"play type": "play_type"})
    if "set_piece" in raw:
        sp = raw["set_piece"].astype(str).str.strip().str.lower()
        raw["set_piece"] = sp.replace(
            {"direct": "fk_direct", "indirect": "fk_indirect", "fk direct": "fk_direct", "fk indirect": "fk_indirect"}
        )
    raw["taker_notes"] = raw.get("taker_id", "").astype(str).fillna("")
    if "goal_created" in raw:
        raw["goal_created"] = (
            raw["goal_created"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "yes": True, "y": True, "1": True, "no": False, "false": False, "0": False})
            .fillna(False)
        )
    for k in ["match_id", "play_call_id", "play_type"]:
        if k in raw:
            raw[k] = raw[k].astype(str).fillna("").str.strip()
    keep = [
        c
        for c in ["season_id", "match_id", "set_piece", "play_call_id", "play_type", "taker_notes", "goal_created"]
        if c in raw
    ]
    return _normalize_season_id(raw[keep].copy())


@st.cache_data(ttl=300)
def load_summaries(spreadsheet_key: str) -> pd.DataFrame:
    # Support both 'summary' and 'summaries'
    for tab in ("summary", "summaries"):
        try:
            df = read_sheet_to_df(spreadsheet_key, tab)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "match_id" in df.columns:
                df["match_id"] = df["match_id"].astype(str)
            return _normalize_season_id(df)
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_goals_allowed(spreadsheet_key: str) -> pd.DataFrame:
    """Read 'goals_allowed' tab.

    Expected columns (case-insensitive, flexible):
      - match_id (str)
      - goal_id (str)  [optional]
      - description or description_of_goal (str)  [optional]
      - goalie_player_id / goalkeeper_player_id / goalie (player_id as str)
      - minute (int) [optional]
      - situation (str) [optional]
    """

    try:
        df = read_sheet_to_df(spreadsheet_key, "goals_allowed")
    except Exception:
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "description_of_goal" in df.columns and "description" not in df.columns:
        df = df.rename(columns={"description_of_goal": "description"})
    for cand in ["goalie_player_id", "goalkeeper_player_id", "goalie"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "goalie_player_id"})
            break
    if "goalie_player_id" not in df.columns:
        df["goalie_player_id"] = ""

    for k in ["match_id", "goal_id", "goalie_player_id"]:
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

    return _normalize_season_id(df)
