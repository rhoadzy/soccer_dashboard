"""Game drilldown page (match_id route).

Refactor-only extraction out of app.py.

Important: this module does NOT import app.py to avoid circular imports.
All app-local helpers (query param reset, AI helpers, formatting) are injected.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_coach_notes_and_summary(
    *,
    match_id: str,
    matches: pd.DataFrame,
    summaries: pd.DataFrame,
    events: pd.DataFrame,
    generate_ai_game_summary,
    ai_user_error_message,
    render_ai_debug,
) -> None:
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
        st.dataframe(nice.to_frame("Value"), width="stretch", hide_index=False, height=280)
    else:
        st.info("No coach notes yet for this game. Add a row in the `summary` tab with this match_id.")

    ai_txt = generate_ai_game_summary(m, srow, events)
    if ai_txt:
        st.markdown("**AI Game Summary**")
        st.write(ai_txt)
    else:
        st.caption(ai_user_error_message("AI summary unavailable (no Groq key set or not enough context)."))
        render_ai_debug()


def render_game_drilldown(
    *,
    match_id: str,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    events: pd.DataFrame,
    plays_df: pd.DataFrame,
    summaries: pd.DataFrame,
    qparams_set,
    format_date,
    generate_ai_game_summary,
    ai_user_error_message,
    render_ai_debug,
) -> None:
    row = matches.loc[matches["match_id"] == match_id]
    if row.empty:
        st.error("Match not found.")
        if st.button("Back to Dashboard"):
            qparams_set()
            st.rerun()
        return

    m = row.iloc[0]
    st.header(f"Game View – {format_date(m.get('date',''))} vs {m.get('opponent','')} ({m.get('home_away','')})")
    st.caption(
        f"Division: {'Yes' if m.get('division_game', False) else 'No'} | Result: {m.get('result','')} | Score: {m.get('goals_for','')}-{m.get('goals_against','')}"
    )

    # =========================================================================
    # GAME RECORDING URL DISPLAY
    # =========================================================================
    url = None
    for url_col in ["url", "recording_url", "game_url", "video_url", "link"]:
        if m.get(url_col) and str(m.get(url_col)).strip():
            url = str(m.get(url_col)).strip()
            break

    if url:
        if not url.startswith("http"):
            url = "https://" + url

        st.markdown(
            f"""
        <div style="
            background-color: #e8f4f8;
            border-left: 4px solid #4a90e2;
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
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info("📹 No game recording available for this match.")

    by_player = events.query("match_id == @match_id").copy() if "match_id" in events.columns else pd.DataFrame()
    if by_player.empty:
        base = players[["player_id", "name", "jersey", "position"]].copy()
        base["shots"] = base["goals"] = base["assists"] = base["points"] = 0
        view = base[["jersey", "name", "position", "shots", "goals", "assists", "points"]]
    else:
        sums = by_player.groupby("player_id", as_index=False)[["shots", "goals", "assists"]].sum()
        sums["points"] = 2 * sums["goals"] + sums["assists"]
        view = (
            sums.set_index("player_id")
            .join(players.set_index("player_id")[["name", "jersey", "position"]], how="left")
            .fillna({"name": "Unknown", "position": "", "jersey": 0})
        )
        view = view.reset_index()[["jersey", "name", "position", "shots", "goals", "assists", "points"]]
        view = view.sort_values(["points", "goals", "shots"], ascending=[False, False, False])

    st.subheader("Per-Player Breakdown")
    st.dataframe(view, width="stretch", hide_index=True)

    st.subheader("Set-Play Attempts (this game)")
    sp = plays_df.query("match_id == @match_id") if not plays_df.empty else pd.DataFrame()
    if sp.empty:
        st.info("No set-play rows for this match.")
    else:
        cols = [c for c in ["set_piece", "play_call_id", "play_type", "taker_notes", "goal_created"] if c in sp.columns]
        df_show = sp[cols].rename(columns={"play_call_id": "Play Call"})
        df_show = df_show[["set_piece", "Play Call", "play_type", "taker_notes", "goal_created"]]
        st.dataframe(df_show, width="stretch", hide_index=True)

    st.divider()
    render_coach_notes_and_summary(
        match_id=match_id,
        matches=matches,
        summaries=summaries,
        events=events,
        generate_ai_game_summary=generate_ai_game_summary,
        ai_user_error_message=ai_user_error_message,
        render_ai_debug=render_ai_debug,
    )

    st.divider()
    c1, c2 = st.columns([1, 1])
    if c1.button("Back to Dashboard"):
        qparams_set()
        st.rerun()
    c2.markdown(f"[Open this game in a new tab](?match_id={match_id})")
