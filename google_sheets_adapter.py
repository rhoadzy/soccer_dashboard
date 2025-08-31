# google_sheets_adapter.py
import os
import io
import json
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def _gspread_client():
    """
    Build a gspread client using a service-account JSON loaded from:
    - GOOGLE_SERVICE_ACCOUNT_JSON (stringified JSON from Streamlit Secrets), or
    - service_account.json on disk (local dev).
    """
    json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        info = json.loads(json_str)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        return gspread.authorize(creds)

    if os.path.exists("service_account.json"):
        creds = Credentials.from_service_account_file("service_account.json", scopes=SCOPES)
        return gspread.authorize(creds)

    return None  # No creds found

def _maybe_public_csv(spreadsheet_key_or_url: str, sheet_name: str) -> pd.DataFrame:
    """
    Fallback for public/anyone-with-link sheets (no creds). Requires the sheet to be viewable.
    """
    # Try to extract the key if URL
    key = None
    if "spreadsheets/d/" in spreadsheet_key_or_url:
        # URL form
        try:
            key = spreadsheet_key_or_url.split("spreadsheets/d/")[1].split("/")[0]
        except Exception:
            key = None
    else:
        key = spreadsheet_key_or_url

    if not key:
        raise RuntimeError("Spreadsheet key not found and no credentials provided.")

    # CSV export endpoint
    url = f"https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def read_sheet_to_df(spreadsheet_key_or_url: str, sheet_name: str) -> pd.DataFrame:
    """
    Read a Google Sheet tab into a DataFrame.
    Supports:
      - Service account auth (preferred)
      - Public CSV export (fallback)
    `spreadsheet_key_or_url` can be either a key or full URL.
    """
    client = _gspread_client()
    if client:
        try:
            if "spreadsheets/d/" in spreadsheet_key_or_url:
                sh = client.open_by_url(spreadsheet_key_or_url)
            else:
                sh = client.open_by_key(spreadsheet_key_or_url)
            ws = sh.worksheet(sheet_name)
            return pd.DataFrame(ws.get_all_records())
        except Exception as e:
            # If authenticated path fails (e.g., no access), re-raise;
            # caller can decide to handle it or rely on public fallback.
            raise

    # No creds: try public CSV fallback
    return _maybe_public_csv(spreadsheet_key_or_url, sheet_name)
