# HS Soccer Dashboard

A Streamlit dashboard for high school soccer teams. Data is loaded from a Google Sheet (via a service account) and the app can generate optional AI summaries (via Groq).

## Features

- Team KPIs: goals for/against, shots, saves, conversion rates
- Trends: rolling 3-game metrics
- Set-piece analysis: corners, free kicks, penalties + taker effectiveness
- Defensive analysis: goals conceded patterns by situation/minute + keeper breakdowns
- Game drill-down: per-match views + coach notes + recording links
- AI summaries (optional): coach-friendly recaps and recommendations

## Quick start (local)

### Prereqs
- Python 3.10+ recommended
- A Google Cloud service account with access to your Google Sheet

### 1) Clone + install

```bash
git clone https://github.com/rhoadzy-labs/soccer_dashboard.git
cd soccer_dashboard
pip install -r requirements.txt
```

### 2) Create `.env`

Create a `.env` in the repo root:

```env
# Required
SPREADSHEET_KEY=your_google_sheet_id_or_full_url
GOOGLE_SERVICE_ACCOUNT_JSON={...full service account json...}

# Optional (AI)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional (simple password gate)
APP_PASSWORD=choose_a_password
```

Notes:
- `SPREADSHEET_KEY` can be the long ID in the Sheet URL or the full URL.
- `GOOGLE_SERVICE_ACCOUNT_JSON` should be the **entire JSON contents** of your service account key.
  - If you prefer using a file locally, you can instead set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json`.
- AI features will quietly disable themselves if `GROQ_API_KEY` is not set.

### 3) Share the Sheet with the service account

In Google Sheets → Share → add the service account email (from the JSON `client_email`) as at least **Viewer**.

### 4) Run

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Create a new Streamlit app pointing at this repo and the `main` branch, with `app.py` as the entrypoint.
2. In **Manage app → Settings → Secrets**, set:

```toml
SPREADSHEET_KEY = "your_sheet_id_or_url"
GOOGLE_SERVICE_ACCOUNT_JSON = '''{ ...full json... }'''

# Optional
GROQ_API_KEY = "gsk_..."
GROQ_MODEL = "llama-3.3-70b-versatile"
APP_PASSWORD = "..."
```

Then reboot/redeploy.

## Data model (Google Sheet tabs)

Your spreadsheet should contain these worksheets:
- `seasons` (`season_id`, display `label`, and one `active=TRUE` row)
- `matches` (game results and stats; optional `url` column for recordings)
- `players` (stable `player_id` plus `player_status` of `current` or `graduated`)
- `events`
- `plays` (set pieces)
- `goals_allowed`
- `summary` or `summaries` (coach notes / summaries)

Add `season_id` to matches, events, plays, goals allowed, and summaries. Match IDs may restart at `0` each season because the app resolves them together with `season_id`. The active season displays only current players; historical seasons retain graduated players so old records continue to resolve correctly.

For shot accuracy KPIs, the `matches` worksheet uses `shots` (aliased to `shots_for` by the app), `shots_target`, `shots_against`, and `shots_against_target`. The dashboard calculates `SOT% (For)` as `shots_target / shots` and `SOT% (Agst)` as `shots_against_target / shots_against`.

A snapshot of the expected schema lives in `docs/SHEET_SCHEMA_SNAPSHOT.md`.

## MaxPreps schedule and rankings

The sidebar links to Milton's current MaxPreps schedule and the Vermont Division II hub. The app uses MaxPreps' structured schedule data as a fallback when the Google Sheet has no upcoming match, and shows the published Division II rank when MaxPreps makes the season rankings available. Until then, the rank is displayed as `N/A`.

- Schedule: <https://www.maxpreps.com/vt/milton/milton-yellowjackets/soccer/schedule/>
- Division II: <https://www.maxpreps.com/vt/soccer/26-27/division/division-ii/?statedivisionid=b872c7a0-0488-4524-bdf7-2806555e2942>

## Recording links

The game detail view will show a “Game Recording” link when a URL is present. Supported column names include:
- `url`, `recording_url`, `game_url`, `video_url`, `link`

## AI summaries (Groq)

AI is used for small-data summarization (coach-friendly recaps and recommendations). Defaults:
- Model: `llama-3.3-70b-versatile` (override via `GROQ_MODEL`)

If AI fails and you have `DEBUG_AI=true`, the app will show a debug hint in the UI.

## Troubleshooting

**FileNotFoundError: Service account JSON not found at 'service_account.json'**
- On Streamlit Cloud: you must set `GOOGLE_SERVICE_ACCOUNT_JSON` in Secrets.
- Locally: ensure `.env` is present and you start Streamlit from the repo root, or set `GOOGLE_APPLICATION_CREDENTIALS`.

**AI debug: Missing GROQ_API_KEY or groq import failed**
- Ensure `GROQ_API_KEY` is set and the `groq` package is installed (`pip install -r requirements.txt`).

## Contributing

PRs welcome. Keep changes small and tested; update schema docs if you change the sheet contract.

## License

MIT (see `LICENSE`).
