# google_sheets_adapter.py
import os
import json
import time
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _build_credentials() -> Credentials:
    """
    Prefer GOOGLE_SERVICE_ACCOUNT_JSON (Streamlit Cloud).
    Fallback to GOOGLE_APPLICATION_CREDENTIALS or ./service_account.json (local).
    """
    json_blob = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if json_blob:
        info = json.loads(json_blob)
        return Credentials.from_service_account_info(info, scopes=_SCOPES)

    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"Service account JSON not found at '{key_path}'. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON in secrets or provide service_account.json locally."
        )
    return Credentials.from_service_account_file(key_path, scopes=_SCOPES)

def _authorize_client() -> gspread.Client:
    creds = _build_credentials()
    if not creds.valid:
        creds.refresh(Request())
    return gspread.authorize(creds)

def _retry(fn, *, tries=3, delay=0.8, backoff=2.0):
    last_exc = None
    _delay = delay
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(_delay)
            _delay *= backoff
    raise last_exc

def _open_spreadsheet(client: gspread.Client, key_or_url: str):
    if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
        return _retry(lambda: client.open_by_url(key_or_url))
    else:
        return _retry(lambda: client.open_by_key(key_or_url))

def read_sheet_to_df(spreadsheet_key_or_url: str, worksheet_name: str) -> pd.DataFrame:
    client = _authorize_client()
    sh = _open_spreadsheet(client, spreadsheet_key_or_url)
    ws = _retry(lambda: sh.worksheet(worksheet_name))
    values = _retry(lambda: ws.get_all_values())

    if not values:
        return pd.DataFrame()

    header, *rows = values
    if not header:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[str(c).strip() for c in header])
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df
