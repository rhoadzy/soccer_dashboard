"""Home page (internal routing).

This module intentionally wires existing functions from app.py into a page-level
renderer, without changing behavior.

Over time we'll move individual renderers and helpers out of app.py into modules
under pages/, ui/, and ai/.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from app_pages.home_tabs.games import render_home_tab_games
from app_pages.home_tabs.goals_allowed import render_home_tab_goals_allowed
from app_pages.home_tabs.leaders import render_home_tab_leaders
from app_pages.home_tabs.set_pieces import render_home_tab_set_pieces
from app_pages.home_tabs.trends import render_home_tab_trends


@dataclass(frozen=True)
class HomeHandlers:
    team_kpis: Callable[..., None]
    render_games_table: Callable[..., None]
    render_points_leaderboard: Callable[..., None]
    render_goals_allowed_analysis: Callable[..., None]
    render_set_piece_analysis_from_plays: Callable[..., None]

    # Drilldown page injected helpers (router uses these)
    qparams_set: Callable[..., None]
    format_date: Callable[..., str]
    generate_ai_game_summary: Callable[..., str]

    build_comparison_trend_frame: Callable[..., pd.DataFrame]
    build_individual_game_trends: Callable[..., pd.DataFrame]

    generate_ai_team_analysis: Callable[..., str]
    ai_user_error_message: Callable[..., str]
    render_ai_debug: Callable[..., None]


def render_home(
    *,
    title: str,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    events: pd.DataFrame,
    plays_simple: pd.DataFrame,
    summaries: pd.DataFrame,
    goals_allowed: pd.DataFrame,
    matches_view: pd.DataFrame,
    events_view: pd.DataFrame,
    plays_view: pd.DataFrame,
    ga_view: pd.DataFrame,
    season_id: str,
    our_rank: Optional[int],
    compact: bool,
    handlers: HomeHandlers,
) -> None:
    st.markdown(
        f"""
        <section class="dashboard-masthead">
            <div class="dashboard-kicker">Milton soccer / team workspace</div>
            <h1>{escape(title)}</h1>
            <p>Match results, performance trends, player leaders, and defensive review.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Data health panel
    with st.expander("Data Health", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
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
            st.cache_data.clear()
            st.rerun()

    handlers.team_kpis(
        matches_view,
        d2_rank=our_rank,
        compact=compact,
        season_id=season_id,
    )

    tab_labels = ["Games", "Trends", "Leaders", "Goals Allowed", "Set Pieces"]
    if "main_tab_radio" not in st.session_state:
        st.session_state["main_tab_radio"] = tab_labels[0]

    selected_tab = st.radio(
        "Main tabs",
        tab_labels,
        horizontal=True,
        key="main_tab_radio",
        label_visibility="collapsed",
    )

    if selected_tab == "Games":
        render_home_tab_games(
            matches_view,
            players,
            events_view,
            plays_view,
            ga_view,
            compact=compact,
            render_games_table=handlers.render_games_table,
            generate_ai_team_analysis=handlers.generate_ai_team_analysis,
            ai_user_error_message=handlers.ai_user_error_message,
            render_ai_debug=handlers.render_ai_debug,
        )
    else:
        tab_renderers = {
            "Trends": lambda: render_home_tab_trends(
                matches_view,
                compact=compact,
                build_comparison_trend_frame=handlers.build_comparison_trend_frame,
                build_individual_game_trends=handlers.build_individual_game_trends,
            ),
            "Leaders": lambda: render_home_tab_leaders(
                events_view,
                players,
                compact=compact,
                render_points_leaderboard=handlers.render_points_leaderboard,
            ),
            "Goals Allowed": lambda: render_home_tab_goals_allowed(
                ga_view,
                matches_view,
                players,
                compact=compact,
                render_goals_allowed_analysis=handlers.render_goals_allowed_analysis,
            ),
            "Set Pieces": lambda: render_home_tab_set_pieces(
                plays_view,
                plays_simple,
                matches_view,
                players,
                render_set_piece_analysis_from_plays=handlers.render_set_piece_analysis_from_plays,
            ),
        }

        renderer = tab_renderers.get(selected_tab)
        if renderer:
            renderer()
