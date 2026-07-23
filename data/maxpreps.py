from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional


MAXPREPS_TEAM_URL = "https://www.maxpreps.com/vt/milton/milton-yellowjackets/soccer/"
MAXPREPS_SCHEDULE_URL = f"{MAXPREPS_TEAM_URL}schedule/"
MAXPREPS_D2_URL = (
    "https://www.maxpreps.com/vt/soccer/26-27/division/division-ii/"
    "?statedivisionid=b872c7a0-0488-4524-bdf7-2806555e2942"
)
MAXPREPS_RANKINGS_URL = (
    "https://www.maxpreps.com/vt/soccer/26-27/division/division-ii/rankings/1/"
    "?statedivisionid=b872c7a0-0488-4524-bdf7-2806555e2942"
)
MAXPREPS_MILTON_SCHOOL_ID = "5aee9a87-4784-4552-9902-7fecbbf920d0"
MAXPREPS_D2_CONTEXT = "Vermont Division II"

_NEXT_DATA_PATTERN = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    flags=re.IGNORECASE | re.DOTALL,
)


def _page_props(page_html: str) -> dict:
    if not page_html:
        return {}
    match = _NEXT_DATA_PATTERN.search(page_html)
    if not match:
        return {}
    try:
        next_data = json.loads(match.group(1))
        page_props = next_data.get("props", {}).get("pageProps", {})
        return page_props if isinstance(page_props, dict) else {}
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def parse_maxpreps_division_rank(
    page_html: str,
    *,
    school_id: str = MAXPREPS_MILTON_SCHOOL_ID,
    school_name: str = "Milton",
    division_name: str = MAXPREPS_D2_CONTEXT,
) -> Optional[int]:
    """Extract a team's rank from MaxPreps' structured D2 rankings data."""

    target_school_name = school_name.strip().casefold()

    # Division rankings pages serialize the complete table into the rendered
    # document rather than exposing it through ``__NEXT_DATA__``.
    marker = re.search(r'"rankings"\s*:\s*', page_html or "")
    if marker:
        try:
            entries, _ = json.JSONDecoder().raw_decode(page_html, marker.end())
        except (TypeError, ValueError, json.JSONDecodeError):
            entries = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_school_id = str(entry.get("schoolId", "")).strip()
                entry_school_name = str(entry.get("schoolName", "")).strip().casefold()
                if entry_school_id != school_id and entry_school_name != target_school_name:
                    continue
                try:
                    rank = int(entry.get("rank"))
                except (TypeError, ValueError):
                    return None
                return rank if rank > 0 else None

    # Retain support for MaxPreps' team-ranking payload shape as a defensive
    # fallback if the division page layout changes.
    rankings_data = _page_props(page_html).get("rankingsData") or {}
    if not isinstance(rankings_data, dict):
        return None
    contexts = rankings_data.get("contexts") or []

    target_division = division_name.strip().casefold()
    for context in contexts:
        if not isinstance(context, dict):
            continue
        context_name = str(context.get("contextName", "")).strip().casefold()
        if context_name != target_division:
            continue

        for entry in context.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            entry_school_id = str(entry.get("schoolId", "")).strip()
            entry_school_name = str(entry.get("schoolName", "")).strip().casefold()
            if entry_school_id != school_id and entry_school_name != target_school_name:
                continue
            try:
                rank = int(entry.get("rank"))
            except (TypeError, ValueError):
                return None
            return rank if rank > 0 else None

    return None


def parse_maxpreps_next_opponent(
    page_html: str,
    *,
    team_name: str = "Milton",
    now: Optional[datetime] = None,
) -> Optional[dict[str, str]]:
    """Return the earliest upcoming opponent from MaxPreps schedule data."""

    contests = _page_props(page_html).get("contests") or []
    if not isinstance(contests, list):
        return None

    today = (now or datetime.now()).date()
    upcoming: list[tuple[datetime, str]] = []
    target_team = team_name.strip().casefold()
    for contest in contests:
        if not isinstance(contest, list) or len(contest) <= 11:
            continue
        try:
            contest_date = datetime.fromisoformat(str(contest[11]).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if contest_date.date() < today:
            continue

        teams = contest[0] if contest else []
        if not isinstance(teams, list):
            continue
        team_names = [
            str(team[14]).strip()
            for team in teams
            if isinstance(team, list) and len(team) > 14 and str(team[14]).strip()
        ]
        opponent = next((name for name in team_names if name.casefold() != target_team), "")
        if opponent:
            upcoming.append((contest_date, opponent))

    if not upcoming:
        return None
    contest_date, opponent = min(upcoming, key=lambda item: item[0])
    return {
        "opponent": opponent,
        "date": contest_date.isoformat(),
        "source": "MaxPreps",
    }
