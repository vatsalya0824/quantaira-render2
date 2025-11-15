import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from dateutil import tz

# ───────────────────────── Config & CSS ─────────────────────────

st.set_page_config(page_title="Quantaira Dashboard", layout="wide")

CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-render2.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY", "")  # set this in Render env vars

METRIC_COLORS = {
    "pulse": "#6F52ED",
    "systolic_bp": "#22C55E",   # green-ish
    "diastolic_bp": "#0EA5E9",  # blue-ish
    "spo2": "#F97316",          # orange
}

STATUS_COLORS = {
    "low": "#EF4444",     # red
    "ok": "#22C55E",      # green
    "high": "#FACC15",    # yellow
}


# ───────────────────────── Helpers: Time & Limits ─────────────────────────

def to_tz(ts: pd.Series, tz_name: str) -> pd.Series:
    """Convert UTC timestamps to selected timezone."""
    if tz_name == "UTC":
        return ts.dt.tz_convert(timezone.utc)
    zone = tz.gettz(tz_name)
    return ts.dt.tz_convert(zone)


def default_limits(metric: str) -> Tuple[float, float]:
    """
    Very simple default LSL / USL.
    These are just demo values, not clinical!
    """
    m = metric.lower()
    if m in ("pulse", "heart_rate", "hr"):
        return 60.0, 100.0
    if m == "systolic_bp":
        return 90.0, 130.0
    if m == "diastolic_bp":
        return 60.0, 85.0
    if m == "spo2":
        return 92.0, 100.0
    # fallback
    return 0.0, 1.0


def classify_status(v: float, lsl: float, usl: float) -> str:
    if v < lsl:
        return "low"
    if v > usl:
        return "high"
    return "ok"


def build_colored_segments(sub: pd.DataFrame,
                           lsl: float,
                           usl: float) -> Tuple[List[Dict], List[str]]:
    """
    Take a (timestamp_utc, value) subset and build:
      - segments: list of dicts with x, y, color
      - pt_colors: color per point for markers
    Handles 0 or 1 points safely.
    """
    segments: List[Dict] = []
    pt_colors: List[str] = []

    if sub.empty:
        return segments, pt_colors

    xs = list(sub["timestamp"])
    ys = list(sub["value"])

    statuses = [classify_status(v, lsl, usl) for v in ys]
    pt_colors = [STATUS_COLORS[s] for s in statuses]

    # only one point → no line segments, just markers
    if len(xs) == 1:
        return segments, pt_colors

    # build contiguous segments where status doesn't change
    start_idx = 0
    for i in range(1, len(xs)):
        if statuses[i] != statuses[i - 1]:
            seg_status = statuses[i - 1]
            segments.append({
                "x": xs[start_idx:i + 1],
                "y": ys[start_idx:i + 1],
                "color": STATUS_COLORS[seg_status],
            })
            start_idx = i

    # last segment
    seg_status = statuses[-1]
    segments.append({
        "x": xs[start_idx:],
        "y": ys[start_idx:],
        "color": STATUS_COLORS[seg_status],
    })

    return segments, pt_colors


# ───────────────────────── Data Fetch ─────────────────────────

def fetch_measurements(hours: int) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{API_BASE}/measurements",
            params={"hours": hours},
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc", "metric", "value_1", "value_2"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    # normalize metrics
    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    bp_mask = mlow.isin({"blood_pressure", "bp"})
    if "value_2" in df.columns and bp_mask.any():
        bp = df[bp_mask].copy()
        sys = bp.assign(metric="systolic_bp",
                        value=pd.to_numeric(bp["value_1"], errors="coerce"))
        dia = bp.assign(metric="diastolic_bp",
                        value=pd.to_numeric(bp["value_2"], errors="coerce"))
        other = df[~bp_mask].copy()
        other["value"] = pd.to_numeric(other.get("value_1"), errors="coerce")
        df2 = pd.concat([other, sys, dia], ignore_index=True)
    else:
        df["value"] = pd.to_numeric(df.get("value_1"), errors="coerce")
        df2 = df

    df2 = df2.dropna(subset=["created_utc", "metric", "value"])
    return df2


# ───────────────────────── Plotting ─────────────────────────

def stats_box(sub: pd.DataFrame, metric_label: str, lsl: float, usl: float):
    if sub.empty:
        st.info("No data to summarize.")
        return

    latest_row = sub.sort_values("timestamp").iloc[-1]
    latest_val = float(latest_row["value"])
    latest_time = latest_row["timestamp"]

    mu = float(sub["value"].mean())
    sigma = float(sub["value"].std(ddof=0) or 0.0)
    vmin = float(sub["value"].min())
    vmax = float(sub["value"].max())

    st.markdown(
        f"""
        <div class="stats-footer">
          <div>LSL/USL: <b>{lsl:.1f}</b> / <b>{usl:.1f}</b></div>
          <div>Latest: <b>{latest_val:.1f}</b> at {latest_time.strftime('%Y-%m-%d %H:%M')}</div>
          <div>µ Mean: <b>{mu:.1f}</b></div>
          <div>σ Std: <b>{sigma:.1f}</b></div>
          <div>Min: <b>{vmin:.1f}</b></div>
          <div>Max: <b>{vmax:.1f}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_metric_with_limits(df: pd.DataFrame,
                            metric: str,
                            tz_name: str,
                            line_width: int,
                            dot_size: int,
                            show_limits: bool):
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        st.warning(f"No data for {metric}")
        return

    sub = sub.sort_values("created_utc")
    sub["timestamp"] = to_tz(sub["created_utc"], tz_name)

    lsl, usl = default_limits(metric)
    segments, pt_colors = build_colored_segments(sub, lsl, usl)

    fig = go.Figure()

    # line segments
    for seg in segments:
        fig.add_trace(go.Scatter(
            x=seg["x"],
            y=seg["y"],
            mode="lines",
            line=dict(color=seg["color"], width=line_width),
            showlegend=False,
            hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
        ))

    # markers on every point
    fig.add_trace(go.Scatter(
        x=sub["timestamp"],
        y=sub["value"],
        mode="markers",
        marker=dict(
            size=dot_size,
            color=pt_colors,
            line=dict(width=1, color="#FFFFFF"),
        ),
        showlegend=False,
        hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
    ))

    if show_limits:
        fig.add_hline(y=lsl, line_dash="dash", line_color="#9CA3AF")
        fig.add_hline(y=usl, line_dash="dash", line_color="#9CA3AF")

    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=420,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.30)",
        griddash="dot",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(203, 213, 225, 0.60)",
    )

    st.plotly_chart(fig, use_container_width=True)
    stats_box(sub, metric, lsl, usl)


# ───────────────────────── USDA Meals ─────────────────────────

def usda_search(query: str, when_utc: datetime) -> List[Dict]:
    if not USDA_API_KEY:
        st.warning("USDA_API_KEY is not set in environment variables.")
        return []

    try:
        r = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={
                "api_key": USDA_API_KEY,
                "query": query,
                "pageSize": 10,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"USDA search failed: {e}")
        return []

    foods = []
    for f in data.get("foods", []):
        nutrients = {n["nutrientName"].lower(): n for n in f.get("foodNutrients", [])}

        def val(name: str, default: float = 0.0) -> float:
            for k, n in nutrients.items():
                if name in k:
                    return float(n.get("value", default) or default)
            return default

        foods.append({
            "description": f.get("description", "Unknown item"),
            "brand": f.get("brandName", ""),
            "kcal": val("energy"),
            "protein": val("protein"),
            "carbs": val("carbohydrate"),
            "fat": val("fat, total"),
            "sodium": val("sodium"),
            "time_utc": when_utc,
        })
    return foods


def ensure_meal_state():
    if "meals" not in st.session_state:
        st.session_state["meals"] = []
    if "meal_results" not in st.session_state:
        st.session_state["meal_results"] = []


def render_meal_panel(tz_name: str):
    ensure_meal_state()
    st.markdown("### 🍽️ Add Meal (USDA)")

    col_note, col_meal = st.columns(2)

    with col_meal:
        q = st.text_input("Search food (USDA)", key="meal_query", value="oatmeal")
        use_now = st.checkbox("Use current time", value=True, key="meal_use_now")

        if use_now:
            when_utc = datetime.now(timezone.utc)
            when_str = when_utc.strftime("%Y-%m-%d %H:%M")
        else:
            date = st.date_input("When was it eaten?")
            time_ = st.time_input("Time", value=datetime.now().time())
            when_utc = datetime.combine(date, time_).replace(tzinfo=timezone.utc)
            when_str = when_utc.strftime("%Y-%m-%d %H:%M")

        st.caption(f"Eating time (UTC): {when_str}")

        if st.button("🔍 Search"):
            st.session_state["meal_results"] = usda_search(q, when_utc)

        for idx, item in enumerate(st.session_state["meal_results"]):
            desc = item["description"]
            brand = item["brand"]
            kcal = item["kcal"]
            p = item["protein"]
            c = item["carbs"]
            f = item["fat"]
            na = item["sodium"]

            box = st.container()
            with box:
                st.markdown(
                    f"**{desc}**  "
                    f"{'— ' + brand if brand else ''}  \n"
                    f"{kcal:.0f} kcal · P {p:.1f} g · C {c:.1f} g · F {f:.1f} g · Na {na:.0f} mg"
                )
                if st.button("➕ Add", key=f"add_meal_{idx}"):
                    st.session_state["meals"].append(item)
                    st.success(f"Added meal: {desc}")

    # Recent meals list
    st.markdown("### 🧾 Recent Meals")
    if not st.session_state["meals"]:
        st.info("No meals added yet.")
        return

    # show newest first
    meals_sorted = sorted(
        st.session_state["meals"],
        key=lambda m: m["time_utc"],
        reverse=True,
    )

    for m in meals_sorted:
        local_ts = m["time_utc"].astimezone(
            tz.gettz(tz_name) if tz_name != "UTC" else timezone.utc
        )
        st.markdown(
            f"**{m['description']}** — "
            f"{local_ts.strftime('%Y-%m-%d %H:%M')} {tz_name}  \n"
            f"{m['kcal']:.0f} kcal · "
            f"P {m['protein']:.1f} g · "
            f"C {m['carbs']:.1f} g · "
            f"F {m['fat']:.1f} g · "
            f"Na {m['sodium']:.0f} mg"
        )


# ───────────────────────── Main UI ─────────────────────────

with st.sidebar:
    st.markdown("<h2 class='sb-title'>Home</h2>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Patient"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<h2 class='sb-title'>Settings</h2>", unsafe_allow_html=True)

    tz_choice = st.selectbox(
        "Timezone",
        ["UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific"],
        index=0,
    )
    hours = st.selectbox("Time window (hours)", [8, 24, 72, 168, 720], index=1)
    line_width = st.slider("Line width", 1, 6, 4)
    dot_size = st.slider("Marker size (dots)", 6, 20, 10)
    show_limits = st.checkbox("Show LSL/USL dashed lines", value=True)

st.markdown(
    "<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>",
    unsafe_allow_html=True,
)

if page == "Patient":
    df = fetch_measurements(hours)

    if df.empty:
        st.warning("No data yet from Tenovi / backend.")
    else:
        metric_tabs = st.tabs(["❤️ Heart Rate", "💧 Systolic BP",
                               "💜 Diastolic BP", "🫁 SpO₂", "📊 Combined"])

        with metric_tabs[0]:
            plot_metric_with_limits(df, "pulse", tz_choice,
                                    line_width, dot_size, show_limits)

        with metric_tabs[1]:
            plot_metric_with_limits(df, "systolic_bp", tz_choice,
                                    line_width, dot_size, show_limits)

        with metric_tabs[2]:
            plot_metric_with_limits(df, "diastolic_bp", tz_choice,
                                    line_width, dot_size, show_limits)

        with metric_tabs[3]:
            plot_metric_with_limits(df, "spo2", tz_choice,
                                    line_width, dot_size, show_limits)

        # Combined normalized overlay
        with metric_tabs[4]:
            show_cols = ["pulse", "systolic_bp", "diastolic_bp", "spo2"]
            combo = df[df["metric"].isin(show_cols)].copy()
            if combo.empty:
                st.warning("No data for combined view.")
            else:
                combo["timestamp"] = to_tz(combo["created_utc"], tz_choice)
                combo["value_norm"] = combo.groupby("metric")["value"].transform(
                    lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)
                )
                figc = px.line(
                    combo,
                    x="timestamp",
                    y="value_norm",
                    color="metric",
                    category_orders={"metric": show_cols},
                    color_discrete_map=METRIC_COLORS,
                )
                figc.update_traces(line=dict(width=line_width))
                figc.update_layout(
                    margin=dict(l=20, r=20, t=10, b=40),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    hovermode="x unified",
                    height=420,
                )
                figc.update_xaxes(
                    showgrid=True,
                    gridcolor="rgba(148, 163, 184, 0.30)",
                    griddash="dot",
                )
                figc.update_yaxes(
                    showgrid=True,
                    gridcolor="rgba(203, 213, 225, 0.60)",
                )
                st.plotly_chart(figc, use_container_width=True)

    st.markdown("---")
    render_meal_panel(tz_choice)
