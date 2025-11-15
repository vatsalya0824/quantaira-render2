# dashboard/main.py
# Quantaira Dashboard — iOS-style teal pills, colored segments, stats card on the right,
# notes + meals + recent-meals section, 24h/3d/7d/30d windows (using backend `hours`).

from datetime import datetime
from pathlib import Path
import json
from importlib import reload
from string import Template
import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

from fetcher import fetch_data   # ✅ our local fetcher, no secrets needed

import common
common = reload(common)
from common import best_ts_col, convert_tz, split_blood_pressure  # type: ignore

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Quantaira Dashboard", layout="wide")
BUILD_TAG = "quantaira-dashboard v2 (hours + teal layout)"
st.markdown(
    f"<div style='opacity:.45;font:12px/1.2 ui-sans-serif,system-ui'>build {BUILD_TAG}</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# USDA key (for meals)
# ─────────────────────────────────────────────
USDA_API_KEY = os.getenv("USDA_API_KEY") or st.secrets.get("USDA_API_KEY", "")
if not USDA_API_KEY:
    st.warning("USDA_API_KEY not set (env var or .streamlit/secrets.toml)")

# ─────────────────────────────────────────────
# Colors + constants
# ─────────────────────────────────────────────
P = {
    "bg": "#F6FBFD", "ink": "#0F172A", "muted": "#667085",
    "chip": "#F3F6F8", "chipBrd": "rgba(2,6,23,.08)",
    "tealA": "#48C9C3", "tealB": "#3FB7B2", "glow": "rgba(68,194,189,.32)",
    # GREEN=above, YELLOW=normal, RED=below
    "segGreen": "#10B981", "segYellow": "#FACC15", "segRed": "#EF4444",
    "refLine": "rgba(15,23,42,.45)",
    "pillDot": "#0F172A", "mealDot": "#f472b6", "noteDot": "#14b8a6",
}
UNITS = {
    "pulse": "bpm",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "spo2": "%",
}

DEFAULT_LIMITS = {
    "pulse": (60, 100),
    "systolic_bp": (90, 130),
    "diastolic_bp": (60, 85),
    "spo2": (94, 100),
}

# ─────────────────────────────────────────────
# Per-patient persistence (CSV) for meals + notes
# ─────────────────────────────────────────────
DATA_DIR = Path(".user_state")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _meals_path(pid: str) -> Path:
    return DATA_DIR / f"meals_{pid}.csv"

def _notes_path(pid: str) -> Path:
    return DATA_DIR / f"notes_{pid}.csv"

MEAL_COLS = [
    "timestamp_utc", "food", "kcal", "protein_g",
    "carbs_g", "fat_g", "sodium_mg", "fdc_id"
]
NOTE_COLS = ["timestamp_utc", "note"]

def load_meals(pid: str) -> pd.DataFrame:
    p = _meals_path(pid)
    if not p.exists():
        return pd.DataFrame(columns=MEAL_COLS)
    df = pd.read_csv(p, dtype={"fdc_id": "string"})
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df[MEAL_COLS].dropna(subset=["timestamp_utc"])

def load_notes(pid: str) -> pd.DataFrame:
    p = _notes_path(pid)
    if not p.exists():
        return pd.DataFrame(columns=NOTE_COLS)
    df = pd.read_csv(p)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df[NOTE_COLS].dropna(subset=["timestamp_utc"])

def save_meals(pid: str, df: pd.DataFrame):
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out[MEAL_COLS].sort_values("timestamp_utc").reset_index(drop=True)
    out.to_csv(_meals_path(pid), index=False)

def save_notes(pid: str, df: pd.DataFrame):
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out[NOTE_COLS].sort_values("timestamp_utc").reset_index(drop=True)
    out.to_csv(_notes_path(pid), index=False)

# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────
def _get_param(key: str, default: str):
    try:
        qp = st.query_params
        if key in qp and qp[key]:
            return qp[key]
    except Exception:
        pass
    return st.session_state.get(key, default)

# You can pass ?pid=54321&name=Todd in the URL
pid = str(_get_param("pid", "quantaira"))
name = str(_get_param("name", "Quantaira Dashboard"))

if "win" not in st.session_state:
    st.session_state.win = "24h"
if "metric_sel" not in st.session_state:
    st.session_state.metric_sel = "systolic_bp"

HOURS_LOOKUP = {
    "24h": 24,
    "3d": 72,
    "7d": 7 * 24,
    "30d": 30 * 24,
}

if "limits" not in st.session_state:
    st.session_state.limits = {}
if "global_limits" not in st.session_state:
    st.session_state.global_limits = DEFAULT_LIMITS.copy()
else:
    for k, v in DEFAULT_LIMITS.items():
        st.session_state.global_limits.setdefault(k, v)

if "limit_mode" not in st.session_state:
    st.session_state.limit_mode = "Global defaults"

# Initialize per-patient meals/notes state
if "persist_loaded_for" not in st.session_state or st.session_state.persist_loaded_for != pid:
    st.session_state["meals"] = load_meals(pid)
    st.session_state["notes"] = load_notes(pid)
    st.session_state["usda_hits"] = []
    st.session_state.persist_loaded_for = pid

# ─────────────────────────────────────────────
# CSS — teal iOS-style pills + cards
# ─────────────────────────────────────────────
st.markdown(
    f"""
<style>
  .stApp {{
      background:{P['bg']};
      color:{P['ink']};
  }}
  section[data-testid="stSidebar"] {{
      background:#ECF7F6;
      border-right:1px solid rgba(2,6,23,.06);
  }}
  .h-title {{
      font-weight:900;
      font-size:34px;
      margin:2px 0;
      color:{P['ink']};
  }}
  .h-sub {{
      color:{P['muted']};
      margin:0 0 12px;
  }}

  .pillrow {{
      display:flex;
      gap:12px;
      flex-wrap:wrap;
      align-items:center;
      margin:6px 0 14px;
  }}
  .pillrow .stButton {{
      margin:0 !important;
  }}

  .stButton > button {{
      appearance:none !important;
      border:1px solid {P['chipBrd']} !important;
      background:{P['chip']} !important;
      color:{P['ink']} !important;
      border-radius:999px !important;
      padding:12px 20px !important;
      font-weight:900 !important;
      font-size:15px !important;
      line-height:1 !important;
      box-shadow:0 10px 24px rgba(17,24,39,.08) !important;
      transition: transform .18s cubic-bezier(.22,.61,.36,1),
                  box-shadow .22s ease,
                  filter .18s linear,
                  background-color .18s linear,
                  color .18s linear,
                  border-color .18s linear;
  }}
  .stButton > button:hover {{
      transform: translateY(-2px);
      filter: brightness(.99);
      box-shadow:0 14px 30px rgba(17,24,39,.10) !important;
  }}
  .stButton > button:active {{
      transform: translateY(0);
      box-shadow:0 8px 16px rgba(17,24,39,.10) !important;
  }}

  /* active pills */
  .stButton > button#tw_{st.session_state.win}-button,
  .stButton > button#metric_{st.session_state.metric_sel}-button {{
      background:linear-gradient(180deg,{P['tealA']},{P['tealB']}) !important;
      color:#fff !important;
      border-color:transparent !important;
      box-shadow:0 18px 38px {P['glow']} !important;
      filter:none !important;
  }}

  .chart-wrap {{
      background:#fff;
      border-radius:18px;
      padding:12px 14px;
      box-shadow:0 18px 44px rgba(17,24,39,.10);
  }}
  .stats {{
      background:#fff;
      border-radius:14px;
      padding:14px 16px;
      box-shadow:0 10px 26px rgba(0,0,0,.08);
      width:260px;
      font-size:13px;
      color:#374151;
  }}
  .stats h4 {{
      margin:0 0 6px;
      font-weight:800;
      font-size:14px;
      color:{P['ink']};
  }}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.sidebar.header("Settings")
tz_choice = st.sidebar.selectbox(
    "Timezone",
    ["UTC", "America/New_York", "Europe/London", "Asia/Kolkata"],
    index=0,
    key="tz_sel",
)
line_w = st.sidebar.slider("Line width", 1, 6, 4)
marker_size = st.sidebar.slider("Marker size (dots)", 6, 20, 10)
show_ref = st.sidebar.checkbox("Show LSL/USL dashed lines", True)

st.sidebar.markdown("---")
st.sidebar.subheader("Limits mode")
limit_mode = st.sidebar.radio(
    "How to pick LSL/USL?",
    ["Auto (μ±0.5σ)", "Global defaults", "Patient override"],
    index=["Auto (μ±0.5σ)", "Global defaults", "Patient override"].index(
        st.session_state.limit_mode
    ),
    key="limit_mode_radio",
)
st.session_state.limit_mode = limit_mode

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(f"<div class='h-title'>{name}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='h-sub'>Green = above USL • Yellow = normal • Red = below LSL. "
    "Dots: pill (dark), meal (pink), note (teal).</div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Time window + metric pills (top row)
# ─────────────────────────────────────────────
st.markdown('<div class="pillrow">', unsafe_allow_html=True)
tw_cols = st.columns(4, gap="small")
for i, lbl in enumerate(["24h", "3d", "7d", "30d"]):
    if tw_cols[i].button(lbl, key=f"tw_{lbl}", type="secondary"):
        st.session_state.win = lbl
st.markdown("</div>", unsafe_allow_html=True)

METRIC_LABELS = {
    "pulse": "Heart Rate",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "spo2": "SpO₂",
    # "bp_both": "BP (both)",  # add later if you want dual-line BP
}

st.markdown('<div class="pillrow">', unsafe_allow_html=True)
mcols = st.columns(len(METRIC_LABELS), gap="small")
for i, m in enumerate(METRIC_LABELS.keys()):
    if mcols[i].button(METRIC_LABELS[m], key=f"metric_{m}", type="secondary"):
        st.session_state.metric_sel = m
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data loading (using `hours`)
# ─────────────────────────────────────────────
def load_window(hours: int) -> pd.DataFrame:
    """
    Pull data from backend and attach a `timestamp_utc` column.
    """
    df = fetch_data(hours=hours, patient_id=pid)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 0) explicit known timestamp col names from backend
    if "timestamp_utc" in df.columns:
        ts_col = "timestamp_utc"
    elif "created_utc" in df.columns:  # from your backend SELECT
        ts_col = "created_utc"
    else:
        # fall back to best guess
        ts_col = best_ts_col(df)

    if ts_col is None or ts_col not in df.columns:
        st.write("Could not find timestamp column. Columns from API:", list(df.columns))
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp_utc"])

raw = load_window(HOURS_LOOKUP[st.session_state.win])
raw = split_blood_pressure(raw)

if raw.empty:
    st.info("No data to display for this window.")
    st.stop()

try:
    newest = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce").max()
    st.caption(
        "Newest data point in source: "
        + newest.tz_convert(tz_choice).strftime("%b %d, %H:%M %Z")
    )
except Exception:
    pass

# ─────────────────────────────────────────────
# Prepare + pill events
# ─────────────────────────────────────────────
def prepare(df: pd.DataFrame, tz_name: str):
    df = df.copy()
    df["value"] = pd.to_numeric(df.get("value", df.get("value_1")), errors="coerce")
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()

    # pillbox detection
    is_pill = df["metric"].eq("pillbox_opened")
    if "device_name" in df.columns:
        is_pill |= df["device_name"].astype(str).str.lower().str.contains(
            "pillbox", na=False
        )

    pill_events = df.loc[is_pill, "timestamp_utc"].dropna().sort_values().unique().tolist()

    plot_df = df.loc[~is_pill].copy()
    plot_df["local_time"] = convert_tz(plot_df["timestamp_utc"], tz_name)
    return plot_df, pill_events

plot_df, pill_events = prepare(raw, tz_choice)

# ─────────────────────────────────────────────
# Limits helpers
# ─────────────────────────────────────────────
def suggest_limits(values: pd.Series):
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return None, None
    mu = float(s.mean())
    sd = float(s.std(ddof=0) or 0.0)
    return mu - 0.5 * sd, mu + 0.5 * sd

def get_limits_for_mode(mode: str, pid: str, metric: str, values: pd.Series):
    if mode == "Patient override":
        pmap = st.session_state.limits.get(pid, {})
        if metric in pmap:
            lsl, usl = pmap[metric]
            return float(lsl), float(usl)
        return suggest_limits(values)

    if mode == "Global defaults":
        g = st.session_state.global_limits.get(metric)
        if g and len(g) == 2:
            return float(g[0]), float(g[1])
        return suggest_limits(values)

    # Auto
    return suggest_limits(values)

# ─────────────────────────────────────────────
# UTC matching for markers (pills, meals, notes)
# ─────────────────────────────────────────────
def nearest_indices_utc(x_ts, event_ts_list):
    if not x_ts or not event_ts_list:
        return []
    x_utc = (
        pd.to_datetime(pd.Series(x_ts), errors="coerce")
        .dt.tz_convert("UTC")
        .view("int64")
        .values
    )
    out = []
    for e in event_ts_list:
        e_i64 = pd.Timestamp(e).tz_convert("UTC").value
        out.append(int(np.argmin(np.abs(x_utc - e_i64))))
    return sorted(set(out))

# ─────────────────────────────────────────────
# Chart.js helper — single metric with markers
# ─────────────────────────────────────────────
def chartjs_single_with_markers(
    x, y, pill_idx, meal_idx, note_idx, lsl, usl, key="cj_single", height=460
):
    labels = [pd.to_datetime(t).strftime("%b %d %H:%M") for t in x]
    data = [
        None if pd.isna(v) else float(v) for v in pd.to_numeric(y, errors="coerce")
    ]

    def mask_points(idxs, arr):
        out = [None] * len(arr)
        for i in idxs:
            if 0 <= i < len(arr) and arr[i] is not None:
                out[i] = arr[i]
        return out

    pill_points = mask_points(pill_idx, data)
    meal_points = mask_points(meal_idx, data)
    note_points = mask_points(note_idx, data)

    ref_datasets = []
    if show_ref and len(data) > 0:
        if lsl is not None:
            ref_datasets.append(
                {
                    "label": "LSL",
                    "data": [
                        None if v is None else float(lsl) for v in data
                    ],
                    "borderColor": P["refLine"],
                    "borderWidth": 1.2,
                    "borderDash": [6, 4],
                    "pointRadius": 0,
                }
            )
        if usl is not None:
            ref_datasets.append(
                {
                    "label": "USL",
                    "data": [
                        None if v is None else float(usl) for v in data
                    ],
                    "borderColor": P["refLine"],
                    "borderWidth": 1.2,
                    "borderDash": [6, 4],
                    "pointRadius": 0,
                }
            )

    html_tpl = Template(
        """
    <div class="chart-wrap" style="height:${height}px"><canvas id="${cid}"></canvas></div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      (function(){
        const LSL = ${lsl}, USL = ${usl};
        const C_GREEN='${c_green}', C_YELLOW='${c_yellow}', C_RED='${c_red}';

        const baseDatasets = [
          { data: ${series_data}, borderWidth: ${line_w}, tension: 0.55,
            cubicInterpolationMode: 'monotone', pointRadius: 0, spanGaps: true,
            segment: { borderColor: s => {
              const y0 = s.p0.parsed.y, y1 = s.p1.parsed.y;
              if (y0==null || y1==null) return C_YELLOW;
              const m = (y0 + y1)/2;
              if (USL!=null && m>USL) return C_GREEN;
              if (LSL!=null && m<LSL) return C_RED;
              return C_YELLOW;
            }}
          },
          { data:${pill_points}, showLine:false, borderColor:'{pill}', backgroundColor:'{pill}',
            pointBorderColor:'#FFFFFF', pointBorderWidth:2, pointRadius:${marker_sz} },
          { data:${meal_points}, showLine:false, borderColor:'{meal}', backgroundColor:'{meal}',
            pointBorderColor:'#FFFFFF', pointBorderWidth:2, pointRadius:${marker_sz} },
          { data:${note_points}, showLine:false, borderColor:'{note}', backgroundColor:'{note}',
            pointBorderColor:'#FFFFFF', pointBorderWidth:2, pointRadius:${marker_sz} }
        ];
        const refDatasets = ${ref_datasets};

        const ctx = document.getElementById('${cid}').getContext('2d');
        new Chart(ctx,{
          type:'line',
          data:{ labels:${labels}, datasets: baseDatasets.concat(refDatasets) },
          options:{
            responsive:true, maintainAspectRatio:false,
            plugins:{ legend:{display:false}, tooltip:{intersect:false, mode:'index'} },
            interaction:{ intersect:false, mode:'index' },
            scales:{
              x:{ grid:{color:'rgba(120,120,180,0.18)'},
                  ticks:{autoSkip:true,maxTicksLimit:6,maxRotation:0,minRotation:0}},
              y:{ grid:{color:'rgba(0,0,0,0.06)'} }
            }
          }
        });
      })();
    </script>
    """.replace("{pill}", P["pillDot"])
        .replace("{meal}", P["mealDot"])
        .replace("{note}", P["noteDot"])
    )

    html = html_tpl.substitute(
        height=int(height),
        cid=key,
        labels=json.dumps(labels),
        series_data=json.dumps(data),
        pill_points=json.dumps(pill_points),
        meal_points=json.dumps(meal_points),
        note_points=json.dumps(note_points),
        marker_sz=max(6, int(marker_size)),
        line_w=int(line_w),
        lsl=("null" if lsl is None else f"{float(lsl):.6f}"),
        usl=("null" if usl is None else f"{float(usl):.6f}"),
        c_green=P["segGreen"],
        c_yellow=P["segYellow"],
        c_red=P["segRed"],
        ref_datasets=json.dumps(ref_datasets),
    )
    st_html(html, height=height, scrolling=False)

# ─────────────────────────────────────────────
# Render main chart + stats
# ─────────────────────────────────────────────
metric = st.session_state.metric_sel

sub = plot_df[plot_df["metric"] == metric].copy().sort_values("local_time")
if sub.empty:
    st.info("No data for this metric.")
else:
    x = sub["local_time"].tolist()
    y = pd.to_numeric(sub["value"], errors="coerce")
    lsl, usl = get_limits_for_mode(st.session_state.limit_mode, pid, metric, y)

    meals_ts = (
        st.session_state["meals"]["timestamp_utc"].tolist()
        if not st.session_state["meals"].empty
        else []
    )
    notes_ts = (
        st.session_state["notes"]["timestamp_utc"].tolist()
        if not st.session_state["notes"].empty
        else []
    )
    pill_idx = nearest_indices_utc(x, pill_events)
    meal_idx = nearest_indices_utc(x, meals_ts)
    note_idx = nearest_indices_utc(x, notes_ts)

    chart_col, stats_col = st.columns([9, 3], gap="large")
    with chart_col:
        chartjs_single_with_markers(
            x,
            y.tolist(),
            pill_idx,
            meal_idx,
            note_idx,
            lsl,
            usl,
            key=f"cj_{metric}_{st.session_state.win}",
        )
    with stats_col:
        s = pd.to_numeric(sub["value"], errors="coerce").dropna()
        latest = (
            f"{float(s.iloc[-1]):.1f} {UNITS.get(metric, '')}" if not s.empty else "—"
        )
        lsl_s = "—" if lsl is None else f"{lsl:.1f}"
        usl_s = "—" if usl is None else f"{usl:.1f}"
        mean_s = "—" if s.empty else f"{s.mean():.1f}"
        std_s = "—" if s.empty else f"{s.std(ddof=0):.1f}"
        min_s = "—" if s.empty else f"{s.min():.1f}"
        max_s = "—" if s.empty else f"{s.max():.1f}"

        st.markdown(
            "<div class='stats'><h4>Stats</h4>"
            + f"<div><b>LSL/USL:</b> {lsl_s} / {usl_s} {UNITS.get(metric,'')}</div>"
            + f"<div><b>Latest</b> {latest}</div>"
            + f"<div><b>μ Mean:</b> {mean_s}</div>"
            + f"<div><b>σ Std:</b> {std_s}</div>"
            + f"<div><b>Min:</b> {min_s}</div>"
            + f"<div><b>Max:</b> {max_s}</div></div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Add Note & Add Meal (USDA)
# ─────────────────────────────────────────────
st.markdown("### Add Note & Add Meal")
note_col, meal_col = st.columns([1, 2], gap="large")

with note_col:
    st.subheader("📝 Add Note")
    with st.form("note_form", clear_on_submit=True):
        note_text = st.text_input(
            "Note", key="note_text_input", placeholder="e.g., felt dizzy after a walk"
        )
        use_now_note = st.checkbox("Use current time", value=False, key="use_now_note")
        note_date = st.date_input(
            "When? (date)",
            value=datetime.now().date(),
            disabled=use_now_note,
            key="note_date_input",
        )
        note_time = st.time_input(
            "Time",
            value=datetime.now().time().replace(second=0, microsecond=0),
            disabled=use_now_note,
            key="note_time_input",
        )
        submitted = st.form_submit_button("➕ Add Note")
    if submitted:
        if use_now_note:
            ts_utc = pd.Timestamp.now(tz=tz_choice).tz_convert("UTC")
        else:
            local = pd.Timestamp.combine(note_date, note_time).tz_localize(tz_choice)
            ts_utc = local.tz_convert("UTC")
        new = pd.DataFrame(
            [{"timestamp_utc": ts_utc, "note": (note_text or "").strip()}]
        )
        st.session_state["notes"] = (
            pd.concat([st.session_state["notes"], new], ignore_index=True)
            .dropna(subset=["timestamp_utc"])
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        save_notes(pid, st.session_state["notes"])
        st.success("Note added.")
        st.rerun()

with meal_col:
    import requests  # local import so backend doesn’t care

    st.subheader("🍽️ Add Meal (USDA)")
    with st.form("usda_search_form"):
        q = st.text_input(
            "Search food (USDA)",
            placeholder="grilled chicken salad, oatmeal, …",
            key="usda_query_input",
        )
        use_now_meal = st.checkbox("Use current time", value=False, key="use_now_meal")
        mdate = st.date_input(
            "When was it eaten? (date)",
            value=datetime.now().date(),
            disabled=use_now_meal,
            key="meal_date_input",
        )
        mtime = st.time_input(
            "Time",
            value=datetime.now().time().replace(second=0, microsecond=0),
            disabled=use_now_meal,
            key="meal_time_input",
        )
        do_search = st.form_submit_button("🔎 Search")
    if do_search:
        hits = []
        if q.strip():
            try:
                r = requests.get(
                    "https://api.nal.usda.gov/fdc/v1/foods/search",
                    params={
                        "query": q.strip(),
                        "pageSize": 10,
                        "api_key": USDA_API_KEY,
                    },
                    timeout=12,
                )
                r.raise_for_status()
                hits = r.json().get("foods", []) or []
            except Exception as e:
                st.error(f"USDA search failed: {e}")
        st.session_state["usda_hits"] = hits

    def parse_nutrients(hit: dict):
        kcal = prot = carbs = fat = sodium = 0.0
        ln = hit.get("labelNutrients") or {}
        if ln:
            kcal = float((ln.get("calories") or {}).get("value") or 0)
            prot = float((ln.get("protein") or {}).get("value") or 0)
            carbs = float((ln.get("carbohydrates") or {}).get("value") or 0)
            fat = float((ln.get("fat") or {}).get("value") or 0)
            sodium = float((ln.get("sodium") or {}).get("value") or 0)
        else:
            for n in hit.get("foodNutrients", []) or []:
                nname = (n.get("nutrientName") or "").lower()
                unit = (n.get("unitName") or "").lower()
                val = float(n.get("value") or 0)
                if "energy" in nname and unit.startswith("kcal"):
                    kcal = val
                elif "protein" in nname:
                    prot = val
                elif "carbo" in nname:
                    carbs = val
                elif "fat" in nname:
                    fat = val
                elif "sodium" in nname:
                    sodium = val
        return int(round(kcal)), round(prot, 1), round(carbs, 1), round(fat, 1), int(
            round(sodium)
        )

    hits = st.session_state.get("usda_hits", [])
    if hits:
        st.markdown("**Results**")
        for i, hit in enumerate(hits):
            fdc_id = hit.get("fdcId")
            desc = hit.get("description", "Food")
            brand = hit.get("brandOwner") or hit.get("brandName")
            title = f"{desc}" + (f" — {brand}" if brand else "")
            kcal, prot, carbs, fat, sodium = parse_nutrients(hit)

            left_col, right_col = st.columns([5, 2])
            with left_col:
                st.markdown(
                    f"**{title}**  \n"
                    f"_{kcal} kcal · P {prot}g · C {carbs}g · F {fat}g · Na {sodium}mg_"
                )
            with right_col:
                with st.form(f"add_meal_{fdc_id}_{i}", clear_on_submit=False):
                    add_clicked = st.form_submit_button("➕ Add")
                if add_clicked:
                    if st.session_state.get("use_now_meal", False):
                        ts_utc = pd.Timestamp.now(tz=tz_choice).tz_convert("UTC")
                    else:
                        local = pd.Timestamp.combine(
                            st.session_state.get("meal_date_input"),
                            st.session_state.get("meal_time_input"),
                        ).tz_localize(tz_choice)
                        ts_utc = local.tz_convert("UTC")
                    new_row = {
                        "timestamp_utc": ts_utc,
                        "food": title,
                        "kcal": kcal,
                        "protein_g": prot,
                        "carbs_g": carbs,
                        "fat_g": fat,
                        "sodium_mg": sodium,
                        "fdc_id": fdc_id,
                    }
                    st.session_state["meals"] = (
                        pd.concat(
                            [st.session_state["meals"], pd.DataFrame([new_row])],
                            ignore_index=True,
                        )
                        .dropna(subset=["timestamp_utc"])
                        .sort_values("timestamp_utc")
                        .reset_index(drop=True)
                    )
                    save_meals(pid, st.session_state["meals"])
                    st.success("Meal added.")
                    st.rerun()

# ─────────────────────────────────────────────
# Recent Meals
# ─────────────────────────────────────────────
st.markdown("### 🍽️ Recent Meals")
if st.session_state["meals"].empty:
    st.info("No meals added yet.")
else:
    meals_sorted = (
        st.session_state["meals"]
        .sort_values("timestamp_utc", ascending=False)
        .head(10)
    )
    for _, row in meals_sorted.iterrows():
        with st.container(border=True):
            top = st.columns([6, 2])
            with top[0]:
                try:
                    ts_local = (
                        pd.Timestamp(row["timestamp_utc"])
                        .tz_convert(tz_choice)
                        .strftime("%Y-%m-%d %H:%M %Z")
                    )
                except Exception:
                    ts_local = (
                        pd.to_datetime(row["timestamp_utc"]).strftime(
                            "%Y-%m-%d %H:%M UTC"
                        )
                    )
                st.markdown(
                    f"**{row['food']}**  \n"
                    f"<span style='opacity:0.7'>{ts_local}</span>",
                    unsafe_allow_html=True,
                )
            with top[1]:
                st.markdown(
                    f"<div style='text-align:right;font-weight:600;font-size:1.05rem'>"
                    f"{int(row['kcal'])} kcal</div>",
                    unsafe_allow_html=True,
                )
            cols = st.columns(4)
            cols[0].metric("Protein", f"{row['protein_g']} g")
            cols[1].metric("Carbs", f"{row['carbs_g']} g")
            cols[2].metric("Fat", f"{row['fat_g']} g")
            cols[3].metric("Sodium", f"{int(row['sodium_mg'])} mg")
