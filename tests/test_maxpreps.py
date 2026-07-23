import json
import unittest
from datetime import datetime

from data.maxpreps import parse_maxpreps_division_rank, parse_maxpreps_next_opponent


def _page_html(rankings_data):
    payload = {"props": {"pageProps": {"rankingsData": rankings_data}}}
    return (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(payload)
        + "</script></html>"
    )


class MaxPrepsRankTests(unittest.TestCase):
    def test_extracts_milton_rank_from_division_rankings_page(self):
        payload = {
            "lastUpdated": "2026-10-01T09:35:22",
            "rankings": [
                {
                    "rank": 1,
                    "schoolId": "another-school",
                    "schoolName": "U-32",
                },
                {
                    "rank": 2,
                    "schoolId": "5aee9a87-4784-4552-9902-7fecbbf920d0",
                    "schoolName": "Milton",
                },
            ],
        }

        self.assertEqual(parse_maxpreps_division_rank(json.dumps(payload)), 2)

    def test_supports_team_rankings_payload_as_fallback(self):
        rankings_data = {
            "contexts": [
                {
                    "contextName": "Vermont",
                    "entries": [{"schoolName": "Milton", "rank": 2}],
                },
                {
                    "contextName": "Vermont Division II",
                    "entries": [
                        {
                            "schoolId": "5aee9a87-4784-4552-9902-7fecbbf920d0",
                            "schoolName": "Milton",
                            "rank": 1,
                        }
                    ],
                },
            ]
        }

        self.assertEqual(parse_maxpreps_division_rank(_page_html(rankings_data)), 1)

    def test_returns_none_before_rankings_are_published(self):
        self.assertIsNone(parse_maxpreps_division_rank(_page_html(None)))
        self.assertIsNone(parse_maxpreps_division_rank(""))

    def test_does_not_substitute_state_rank_for_division_rank(self):
        rankings_data = {
            "contexts": [
                {
                    "contextName": "Vermont",
                    "entries": [{"schoolName": "Milton", "rank": 2}],
                }
            ]
        }

        self.assertIsNone(parse_maxpreps_division_rank(_page_html(rankings_data)))


class MaxPrepsScheduleTests(unittest.TestCase):
    @staticmethod
    def _team(name):
        team = [None] * 15
        team[14] = name
        return team

    def test_returns_earliest_upcoming_opponent(self):
        later = [[self._team("Milton"), self._team("Rice Memorial")]] + [None] * 11
        later[11] = "2026-09-05T16:00:00"
        earlier = [[self._team("U-32"), self._team("Milton")]] + [None] * 11
        earlier[11] = "2026-08-15T12:00:00"
        payload = {"props": {"pageProps": {"contests": [later, earlier]}}}
        html = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(payload)
            + "</script>"
        )

        result = parse_maxpreps_next_opponent(html, now=datetime(2026, 7, 22))

        self.assertEqual(result["opponent"], "U-32")
        self.assertEqual(result["date"], "2026-08-15T12:00:00")
        self.assertEqual(result["source"], "MaxPreps")

    def test_returns_none_when_schedule_is_not_published(self):
        self.assertIsNone(parse_maxpreps_next_opponent(_page_html(None)))


if __name__ == "__main__":
    unittest.main()
