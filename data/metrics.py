from __future__ import annotations

import pandas as pd


def _numeric_total(dataframe: pd.DataFrame, column: str) -> float:
    if dataframe is None or dataframe.empty or column not in dataframe.columns:
        return 0.0
    return float(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())


def calculate_shot_on_target_percentages(matches: pd.DataFrame) -> tuple[float, float]:
    """Return team and opponent shots-on-target percentages for a match set."""

    total_shots = _numeric_total(matches, "shots_for")
    shots_on_target = _numeric_total(matches, "shots_target")
    total_shots_against = _numeric_total(matches, "shots_against")
    shots_on_target_against = _numeric_total(matches, "shots_against_target")

    shots_on_target_pct = shots_on_target / total_shots * 100.0 if total_shots > 0 else 0.0
    shots_on_target_against_pct = (
        shots_on_target_against / total_shots_against * 100.0
        if total_shots_against > 0
        else 0.0
    )
    return shots_on_target_pct, shots_on_target_against_pct
