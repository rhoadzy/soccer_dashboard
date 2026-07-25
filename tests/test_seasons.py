import unittest

import pandas as pd

from data.seasons import (
    build_season_catalog,
    resolve_season_id,
    season_is_active,
    season_label,
    supports_shot_on_target_kpis,
)
from data.views import derive_related_views, filter_by_season, filter_players_for_season


class SeasonCatalogTests(unittest.TestCase):
    def setUp(self):
        self.seasons = pd.DataFrame(
            [
                {"season_id": "2025", "label": "2025 season", "active": "FALSE"},
                {"season_id": "2026", "label": "2026 season", "active": "TRUE"},
            ]
        )
        self.matches = pd.DataFrame(
            [{"season_id": "2025", "match_id": "0", "opponent": "Legacy"}]
        )

    def test_active_season_is_the_default(self):
        catalog = build_season_catalog(self.seasons, self.matches)

        self.assertEqual(resolve_season_id(None, catalog), "2026")
        self.assertTrue(season_is_active(catalog, "2026"))
        self.assertEqual(season_label(catalog, "2026"), "2026 season")

    def test_valid_historical_request_is_preserved(self):
        catalog = build_season_catalog(self.seasons, self.matches)

        self.assertEqual(resolve_season_id("2025", catalog), "2025")

    def test_mismatched_label_falls_back_to_season_id(self):
        seasons = self.seasons.copy()
        seasons.loc[seasons["season_id"] == "2026", "label"] = "2025 season"

        catalog = build_season_catalog(seasons, self.matches)

        self.assertEqual(season_label(catalog, "2026"), "2026 season")

    def test_shot_on_target_kpis_start_with_2026_season(self):
        self.assertFalse(supports_shot_on_target_kpis("2025"))
        self.assertTrue(supports_shot_on_target_kpis("2026"))
        self.assertTrue(supports_shot_on_target_kpis("2027"))


class SeasonFilteringTests(unittest.TestCase):
    def test_rows_are_isolated_by_season(self):
        rows = pd.DataFrame(
            [
                {"season_id": "2025", "match_id": "0", "value": "old"},
                {"season_id": "2026", "match_id": "0", "value": "new"},
            ]
        )

        selected = filter_by_season(rows, "2026")

        self.assertEqual(selected["value"].tolist(), ["new"])

    def test_legacy_table_does_not_leak_into_new_season(self):
        legacy_rows = pd.DataFrame([{"match_id": "0", "value": "old"}])

        self.assertEqual(len(filter_by_season(legacy_rows, "2025")), 1)
        self.assertTrue(filter_by_season(legacy_rows, "2026").empty)

    def test_active_roster_excludes_graduated_players(self):
        players = pd.DataFrame(
            [
                {"player_id": "1", "name": "Returning", "player_status": "current"},
                {"player_id": "2", "name": "Graduate", "player_status": "graduated"},
            ]
        )

        active = filter_players_for_season(players, "2026", active_season_id="2026")
        historical = filter_players_for_season(players, "2025", active_season_id="2026")

        self.assertEqual(active["name"].tolist(), ["Returning"])
        self.assertEqual(historical["name"].tolist(), ["Returning", "Graduate"])

    def test_empty_match_selection_produces_empty_related_views(self):
        empty_matches = pd.DataFrame(columns=["match_id"])
        events = pd.DataFrame([{"match_id": "0", "goals": 1}])
        plays = pd.DataFrame([{"match_id": "0", "set_piece": "corner"}])
        goals_allowed = pd.DataFrame([{"match_id": "0", "goal_id": "1"}])

        events_view, plays_view, goals_view = derive_related_views(
            matches_view=empty_matches,
            events=events,
            plays_simple=plays,
            goals_allowed=goals_allowed,
        )

        self.assertTrue(events_view.empty)
        self.assertTrue(plays_view.empty)
        self.assertTrue(goals_view.empty)


if __name__ == "__main__":
    unittest.main()
