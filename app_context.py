from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class AppContext:
    # Flags / UI
    compact: bool
    div_only: bool
    season_id: str
    season_label: str
    season_is_active: bool

    # Raw tables
    matches: pd.DataFrame
    players: pd.DataFrame
    events: pd.DataFrame
    plays_simple: pd.DataFrame
    summaries: pd.DataFrame
    goals_allowed: pd.DataFrame

    # Filtered views
    matches_view: pd.DataFrame
    events_view: pd.DataFrame
    plays_view: pd.DataFrame
    ga_view: pd.DataFrame

    # Routing
    match_id: Optional[str]

    # Enrichment
    our_rank: Optional[int]
