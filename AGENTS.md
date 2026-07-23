# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: App entrypoint/wiring (loads data, calls sidebar, builds context, routes).
- `app_context.py`: `AppContext` dataclass (loaded tables + filtered views + routing info).
- `router.py`: `route(ctx, handlers)` selects drilldown vs home.
- `app_pages/`: Internal page modules (NOT Streamlit multipage).
  - `app_pages/home.py`: Home page renderer + tab dispatch.
  - `app_pages/home_tabs/`: Individual home tabs (Games/Trends/Leaders/Goals Allowed/Set Pieces).
  - `app_pages/game.py`: Match drilldown page.
- `ui/sidebar.py`: Sidebar + query-param sync.
- `data/views.py`: Match filtering + derived views.
- `google_sheets_adapter.py`: Google Sheets data access and transformations.
- `loaders.py`: Cached loaders for each sheet.
- `requirements.txt`: Python dependencies for local and cloud runs.
- `README.md`: Setup, data schema, and deployment notes.
- `docs/`: Docs and schema snapshots.
- `__pycache__/`: Local bytecode artifacts (do not edit or commit).

### Important (Streamlit)
- Do **not** create a top-level folder named `pages/`. Streamlit treats it as multipage navigation and will add unexpected sidebar items. Use `app_pages/` for internal routing modules.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: Create/activate a venv (recommended).
- `pip install -r requirements.txt`: Install Python dependencies.
- `streamlit run app.py`: Launch the dashboard locally.
- Smoke test checklist (recommended before merging to `main`):
  - Home loads
  - All tabs render
  - Filters update + persist in query params
  - Drilldown via `?match_id=...` works + Back to Dashboard works

## Coding Style & Naming Conventions
- Python 3.10+ codebase; keep indentation at 4 spaces.
- Favor explicit, descriptive names (`matches_df`, `set_piece_summary`) over abbreviations.
- Keep Streamlit sections grouped by feature (Overview, Games, Trends, Set-Pieces, Defense).
- If you add new tabs/metrics, keep column names consistent with the Google Sheets schema in `README.md`.

## Architecture Overview
- Data flow: Google Sheets → `google_sheets_adapter.py` → Pandas DataFrames → Streamlit UI + AI summaries.
- Core sheets: `matches`, `players`, `events`, `plays`, `goals_allowed`, `summaries` (or `summary`).
- External fetches: SBLive rankings/schedule are optional enrichments and should not block the core UI.

## Testing Guidelines
- CI is intentionally lightweight (GitHub Actions): installs deps + `python -m py_compile ...`.
- No automated runtime/UI test suite is present in this repo yet.
- If you add tests, place them under `tests/` and name files `test_*.py`.
- Suggested frameworks: `pytest` for unit tests.
- Streamlit interactions are validated via the smoke test checklist above (Jetson is the preferred environment).

## Commit & Pull Request Guidelines
- Recent commits use short, lowercase summaries (e.g., “kpi fixes”, “fix home game bug”).
- Keep commit messages concise and action-oriented; avoid long scopes or tags unless the team adopts them.
- PRs should include:
  - A clear description of user-visible changes.
  - Data schema changes (if any) with updated `README.md` tables.
  - Screenshots or GIFs for UI updates (Streamlit pages change often).

## Security & Configuration Tips
- Do not commit secrets. Keep `service_account.json` and other credentials local only.
- Configure `SPREADSHEET_KEY`, `GOOGLE_SERVICE_ACCOUNT_JSON`, and optional `GROQ_API_KEY` via `.env` or Streamlit secrets.
- Validate new columns/tabs in Sheets before relying on them in the UI.
