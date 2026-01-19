# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Streamlit UI, dashboard pages, and AI insights entry points.
- `google_sheets_adapter.py`: Google Sheets data access and transformations.
- `requirements.txt`: Python dependencies for local and cloud runs.
- `README.md`: Setup, data schema, and deployment notes.
- `__pycache__/`: Local bytecode artifacts (do not edit or commit).

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: Install Python dependencies.
- `streamlit run app.py`: Launch the dashboard locally.
- (Optional) `python -m venv .venv` then `source .venv/bin/activate`: Isolate local deps if needed.

## Coding Style & Naming Conventions
- Python 3.8+ codebase; keep indentation at 4 spaces.
- Favor explicit, descriptive names (`matches_df`, `set_piece_summary`) over abbreviations.
- Keep Streamlit sections grouped by feature (Overview, Games, Trends, Set-Pieces, Defense).
- If you add new tabs/metrics, keep column names consistent with the Google Sheets schema in `README.md`.

## Architecture Overview
- Data flow: Google Sheets → `google_sheets_adapter.py` → Pandas DataFrames → Streamlit UI + AI summaries.
- Core sheets: `matches`, `players`, `events`, `plays`, `goals_allowed`, `summaries` (or `summary`).
- External fetches: SBLive rankings/schedule are optional enrichments and should not block the core UI.

## Testing Guidelines
- No automated test suite is present in this repo.
- If you add tests, place them under `tests/` and name files `test_*.py`.
- Suggested frameworks: `pytest` for unit tests and `streamlit`-oriented smoke tests.

## Commit & Pull Request Guidelines
- Recent commits use short, lowercase summaries (e.g., “kpi fixes”, “fix home game bug”).
- Keep commit messages concise and action-oriented; avoid long scopes or tags unless the team adopts them.
- PRs should include:
  - A clear description of user-visible changes.
  - Data schema changes (if any) with updated `README.md` tables.
  - Screenshots or GIFs for UI updates (Streamlit pages change often).

## Security & Configuration Tips
- Do not commit secrets. Keep `service_account.json` and other credentials local only.
- Configure `SPREADSHEET_KEY`, `GOOGLE_SERVICE_ACCOUNT_JSON`, and optional `GEMINI_API_KEY` via `.env` or Streamlit secrets.
- Validate new columns/tabs in Sheets before relying on them in the UI.
