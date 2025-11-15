# streamlit_app/fetcher.py
from __future__ import annotations
import os, time
from typing import Any, Dict, Optional, Iterable

import pandas as pd
import requests
import streamlit as st

def _get_base_url() -> str:
    env_val = os.getenv("BACKEND_URL")
    secrets_val = None
    try:
        secrets_obj = getattr(st, "secrets", None)
        if isinstance(secrets_obj, dict) and "BACKEND_URL" in secrets_obj:
            secrets_val = secrets_obj["BACKEND_URL"]
    except Exception:
        secrets_val = None
    url = env_val or secrets_val or "http://localhost:8000"
    return str(url).rstrip("/")

BASE_URL = _get_base_url()
_FAKE_MODE = os.getenv("FAKE_MODE", "").lower() in {"1", "true", "yes"}
_DEFAULT_TIMEOUT = 20

def _url(path: str) -> str:
    if not path.startswith("/"): path = "/" + path
    return BASE_URL + path

def _should_retry(status: Optional[int]) -> bool:
    return status is None or 500 <= status < 600 or status in {408, 429}

def _request_json(method: str, path: str, *, params: Optional[Dict[str, Any]] = None,
                  json: Optional[Dict[str, Any]] = None, retries: int = 2, timeout: int = _DEFAULT_TIMEOUT,
                  extra_ok_statuses: Iterable[int] = ()) -> Any:
    session = requests.Session()
    session.headers.update({"User-Agent": "quantaira-dashboard/1.0 (+streamlit)", "Accept": "application/json"})
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        status_for_retry: Optional[int] = None
        try:
            resp = session.request(method.upper(), _url(path), params=params, json=json, timeout=timeout)
            status_for_retry = resp.status_code
            if resp.status_code >= 400 and resp.status_code not in set(extra_ok_statuses):
                resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as e:
                raise RuntimeError(f"{method} {_url(path)} did not return JSON") from e
        except (requests.RequestException, RuntimeError) as e:
            last_exc = e
            if attempt < retries and _should_retry(status_for_retry):
                time.sleep(0.6 * (attempt + 1))
                continue
            break
    if _FAKE_MODE and path.strip("/") in {"patients", "vitals"}:
        return _fake_response(path, params or {})
    raise last_exc  # type: ignore[misc]

def cache_fn(ttl: int = 20):
    def deco(fn):
        return st.cache_data(show_spinner=False, ttl=ttl)(fn)
    return deco

@cache_fn(ttl=20)
def fetch_patients() -> pd.DataFrame:
    data = _request_json("GET", "/patients")
    if not isinstance(data, list):
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            data = data["items"]
        else:
            data = []
    df = pd.DataFrame(data)
    if "id" in df.columns:
        df["id"] = df["id"].astype(str)
    return df

@cache_fn(ttl=8)
def fetch_data(*, hours: int = 24, patient_id: Optional[str] = None) -> pd.DataFrame:
    """
    Live data ONLY for patient_id == 'todd'.
    Everyone else returns high-quality fake series so the UI is always full.
    """
    params: Dict[str, Any] = {"hours": int(hours)}
    if patient_id:
        params["patient_id"] = str(patient_id)
    pid = (patient_id or "").strip().lower()

    if pid and pid != "todd":
        data = _fake_response("vitals", params)
    else:
        data = _request_json("GET", "/vitals", params=params)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            data = data["items"]

    if not isinstance(data, list):
        data = []
    df = pd.DataFrame(data)

    # normalize
    if "timestamp_utc" not in df.columns:
        for cand in ("ts", "timestamp", "time_utc", "created_at_utc"):
            if cand in df.columns:
                df["timestamp_utc"] = df[cand]
                break
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    if "timestamp_utc" in df.columns:
        df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    return df.reset_index(drop=True)

def _fake_response(path: str, params: Dict[str, Any]) -> Any:
    if path.strip("/") == "patients":
        return [
            {"id": "todd", "name": "Todd Carter", "age": 47, "gender": "Male"},
            {"id": "jane", "name": "Jane Wilson", "age": 53, "gender": "Female"},
            {"id": "54321", "name": "Webhook Test", "age": 40, "gender": "—"},
        ]
    if path.strip("/") == "vitals":
        import numpy as np
        from datetime import datetime, timedelta, timezone
        hours = int(params.get("hours", 24))
        pid = params.get("patient_id") or "todd"
        now = datetime.now(timezone.utc)
        ts = [now - timedelta(minutes=15 * i) for i in range(hours * 4)]
        ts = list(reversed(ts))
        hr  = 72 + 8 * np.sin(np.linspace(0, 8, len(ts)))
        spo2= 97 + np.sin(np.linspace(0, 6, len(ts))) * 0.6
        sbp = 120 + 10 * np.sin(np.linspace(0, 5, len(ts)))
        dbp = 78 + 6  * np.cos(np.linspace(0, 5, len(ts)))
        out = []
        for i, t in enumerate(ts):
            iso = t.isoformat()
            out.append({"patient_id": pid, "timestamp_utc": iso, "metric": "pulse",        "value": float(hr[i])})
            out.append({"patient_id": pid, "timestamp_utc": iso, "metric": "spo2",         "value": float(spo2[i])})
            out.append({"patient_id": pid, "timestamp_utc": iso, "metric": "systolic_bp",  "value": float(sbp[i])})
            out.append({"patient_id": pid, "timestamp_utc": iso, "metric": "diastolic_bp", "value": float(dbp[i])})
        return out
    return {"ok": False, "error": "unknown path in fake mode"}

@cache_fn(ttl=10)
def backend_health() -> dict:
    try:
        data = _request_json("GET", "/")
        return {"ok": True, "base_url": BASE_URL, "data": data}
    except Exception as e:
        return {"ok": False, "base_url": BASE_URL, "error": str(e)}
