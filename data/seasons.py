from __future__ import annotations

import pandas as pd


LEGACY_SEASON_ID = "2025"
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}


def build_season_catalog(
    seasons: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    legacy_season_id: str = LEGACY_SEASON_ID,
) -> pd.DataFrame:
    """Return normalized season metadata, newest season first.

    The matches table is also inspected so an existing season remains
    available if the optional ``seasons`` worksheet is incomplete.
    """

    rows: dict[str, dict[str, object]] = {}
    if seasons is not None and not seasons.empty:
        source = seasons.copy()
        source.columns = [str(column).strip().lower() for column in source.columns]
        for _, row in source.iterrows():
            season_id = str(row.get("season_id", "")).strip()
            if not season_id:
                continue
            label = str(row.get("label", "")).strip()
            # A mismatched copied label is more confusing than a derived one.
            if not label or season_id not in label:
                label = f"{season_id} season"
            active = str(row.get("active", "")).strip().lower() in _TRUE_VALUES
            rows[season_id] = {"season_id": season_id, "label": label, "active": active}

    if matches is not None and not matches.empty and "season_id" in matches.columns:
        for value in matches["season_id"].dropna().astype(str):
            season_id = value.strip()
            if season_id and season_id not in rows:
                rows[season_id] = {
                    "season_id": season_id,
                    "label": f"{season_id} season",
                    "active": False,
                }

    if not rows:
        rows[legacy_season_id] = {
            "season_id": legacy_season_id,
            "label": f"{legacy_season_id} season",
            "active": True,
        }

    catalog = pd.DataFrame(rows.values())
    catalog["_numeric_sort"] = pd.to_numeric(catalog["season_id"], errors="coerce").fillna(-1)
    catalog = catalog.sort_values(["_numeric_sort", "season_id"], ascending=[False, False])
    return catalog.drop(columns="_numeric_sort").reset_index(drop=True)


def resolve_season_id(requested: object, catalog: pd.DataFrame) -> str:
    """Resolve a requested season, falling back to active then newest."""

    requested_id = str(requested or "").strip()
    valid_ids = set(catalog["season_id"].astype(str))
    if requested_id in valid_ids:
        return requested_id

    active = catalog.loc[catalog["active"]]
    if not active.empty:
        return str(active.iloc[0]["season_id"])
    return str(catalog.iloc[0]["season_id"])


def season_label(catalog: pd.DataFrame, season_id: str) -> str:
    match = catalog.loc[catalog["season_id"].astype(str) == str(season_id)]
    return str(match.iloc[0]["label"]) if not match.empty else f"{season_id} season"


def season_is_active(catalog: pd.DataFrame, season_id: str) -> bool:
    match = catalog.loc[catalog["season_id"].astype(str) == str(season_id)]
    return bool(match.iloc[0]["active"]) if not match.empty else False
