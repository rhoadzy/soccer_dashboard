from __future__ import annotations

from app_pages.game import render_game_drilldown
from app_pages.home import HomeHandlers, render_home


def route(*, ctx, handlers: HomeHandlers) -> None:
    """Route between drilldown and home.

    Behavior should remain identical to the inline router previously in app.py.
    """

    if ctx.match_id:
        render_game_drilldown(
            season_id=ctx.season_id,
            match_id=ctx.match_id,
            matches=ctx.matches_view,
            players=ctx.players,
            events=ctx.events_view,
            plays_df=ctx.plays_view,
            summaries=ctx.summaries,
            qparams_set=handlers.qparams_set,
            format_date=handlers.format_date,
            generate_ai_game_summary=handlers.generate_ai_game_summary,
            ai_user_error_message=handlers.ai_user_error_message,
            render_ai_debug=handlers.render_ai_debug,
        )
        return

    render_home(
        title=f"Milton Varsity Boys Soccer Team {ctx.season_id}",
        matches=ctx.matches,
        players=ctx.players,
        events=ctx.events,
        plays_simple=ctx.plays_simple,
        summaries=ctx.summaries,
        goals_allowed=ctx.goals_allowed,
        matches_view=ctx.matches_view,
        events_view=ctx.events_view,
        plays_view=ctx.plays_view,
        ga_view=ctx.ga_view,
        our_rank=ctx.our_rank,
        compact=ctx.compact,
        handlers=handlers,
    )
