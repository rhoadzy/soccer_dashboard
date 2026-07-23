def render_home_tab_set_pieces(
    plays_view,
    season_plays,
    matches_view,
    players,
    *,
    render_set_piece_analysis_from_plays,
) -> None:
    # Refactor-only extraction: preserve behavior by delegating to existing renderer.
    render_set_piece_analysis_from_plays(
        plays_view,
        matches_view,
        players,
        season_plays_df=season_plays,
    )
