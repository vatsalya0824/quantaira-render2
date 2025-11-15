import os
import requests
import pandas as pd

# Your backend API base URL
API_BASE = os.getenv("API_BASE", "https://quantaira-backend.onrender.com/api").rstrip("/")

def fetch_data(hours: int, patient_id: str | None = None):
    """
    Call the backend /api/measurements?hours=... endpoint.
    Returns a pandas DataFrame with measurement rows.
    """
    params = {"hours": hours}
    if patient_id:
        params["patient_id"] = patient_id

    try:
        r = requests.get(f"{API_BASE}/measurements", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data)
    except Exception as e:
        print("Fetcher error:", e)
        return pd.DataFrame()
