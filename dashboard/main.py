# dashboard/main.py
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ───────────────────────── Setup
st.set_page_config(page_title="Quantaira Dashboard", layout="wide")

CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-realtime.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY", "")  # set in Render
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

COLORS = {
    "pulse":        "#6F52ED",  # purple
    "spo2":         "#FF6B8A",  # pink
    "systolic_bp":  "#2D9CDB",  # blue
    "diastolic_bp": "#9B51E0",  # violet
}

# Default LSL / USL for vitals (you can tweak)
LIMITS = {
    "pulse":        (71.3, 73.7),
    "systolic_bp":  (110.0, 130.0),
    "diastolic_bp": (70.0, 90.0),
    "spo2":         (94.0, 100.0),
}

# ───────────────────────── Tenovi helpers
def fetch_vitals(hours: int = 72) -> pd.DataFrame:
    """Fetch measurement rows from backend and normalize metric names."""
    try:
        r = requests.get(f"{API_BASE}/measurements", params={"hours": hours}, timeout=15)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame(columns=["created_utc", "metric", "value_1", "value_2"])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    bp_mask = mlow.isin({"blood_pressure", "bp"})
    if "value_2" in df.columns and bp_mask.any():
        bp = df[bp_mask].copy()
        sys = bp.assign(
            metric="systolic_bp",
            value=pd.to_numeric(bp["value_1"], errors="coerce"),
        )
        dia = bp.assign(
            metric="diastolic_bp",
            value=pd.to_numeric(bp["value_2"], errors="coerce"),
        )
        df = pd.concat([df[~bp_mask], sys, dia], ignore_index=True)
    else:
        df["value"] = pd.to_numeric(df.get("value_1"), errors="coerce")

    return df[["timestamp_utc", "metric", "value"]].dropna(subset=["timestamp_utc", "metric"])


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def nearest_y(sub: pd.DataFrame, t: pd.Timestamp) -> float | None:
    if sub.empty:
        return None
    diffs = np.abs(sub["timestamp_utc"].view("int64") - t.value)
    i = int(diffs.argmin())
    try:
        return float(sub.iloc[i]["value"])
    except Exception:
        return None


def stats_block(sub: pd.DataFrame) -> str:
    mu = float(sub["value"].mean()) if not sub.empty else float("nan")
    sigma = float(sub["value"].std(ddof=0)) if not sub.empty else float("nan")
    return (
        "<div class='stats-footer'>"
        f"<span>Mean (µ): {mu:.2f}</span>"
        f"<span>Sigma (σ): {sigma:.2f}</span>"
        "</div>"
    )


def plot_with_limits(
    df: pd.DataFrame,
    metric: str,
    pill_events: List[pd.Timestamp],
    lsl: float | None,
    usl: float | None,
):
    """Draw one metric with red / yellow / green segments + LSL / USL + pill markers."""
    sub = df[df["metric"] == metric].sort_values("timestamp_utc")
    if sub.empty:
        st.warning(f"No data for {metric}")
        return

    fig = go.Figure()

    # Decide colors per point
    if lsl is not None and usl is not None:
        below = sub["value"] < lsl
        normal = (sub["value"] >= lsl) & (sub["value"] <= usl)
        above = sub["value"] > usl
    else:
        below = sub["value"] < -1e9  # none
        normal = pd.Series(True, index=sub.index)
        above = sub["value"] > 1e9   # none

    def add_segment(mask, color):
        seg = sub[mask]
        if seg.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=seg["timestamp_utc"],
                y=seg["value"],
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color),
                hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
                showlegend=False,
            )
        )

    add_segment(below, "red")
    add_segment(normal, "yellow")
    add_segment(above, "green")

    # Shaded underlay for style
    fig.add_trace(
        go.Scatter(
            x=sub["timestamp_utc"],
            y=sub["value"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor=hex_to_rgba("#cccccc", 0.08),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # LSL / USL dashed lines
    if lsl is not None:
        fig.add_hline(
            y=lsl,
            line=dict(color="red", dash="dash", width=1),
            annotation_text="LSL",
            annotation_position="top left",
        )
    if usl is not None:
        fig.add_hline(
            y=usl,
            line=dict(color="green", dash="dash", width=1),
            annotation_text="USL",
            annotation_position="bottom left",
        )

    # Pill markers
    xs, ys = [], []
    for e in pill_events:
        v = nearest_y(sub, e)
        if v is not None:
            xs.append(e)
            ys.append(v)
    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=12,
                    color="black",
                    symbol="circle",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate=(
                    "Pill opened<br>Value: %{y}"
                    "<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=400,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor="rgba(120,120,180,0.20)",
        griddash="dot",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(stats_block(sub), unsafe_allow_html=True)


# ───────────────────────── USDA meal helpers
def usda_search_food(query: str) -> List[Dict[str, Any]]:
    if not USDA_API_KEY:
        st.warning("USDA_API_KEY not set – meals search disabled.")
        return []
    if not query.strip():
        return []

    try:
        params = {
            "api_key": USDA_API_KEY,
            "query": query,
            "pageSize": 10,
            "dataType": ["Survey (FNDDS)", "SR Legacy"],
        }
        r = requests.get(USDA_SEARCH_URL, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"USDA search failed: {e}")
        return []

    foods = data.get("foods", []) or []
    results: List[Dict[str, Any]] = []
    for f in foods:
        desc = f.get("description", "")
        brand = f.get("brandOwner") or ""
        nutrients = {n.get("nutrientName"): n.get("value") for n in f.get("foodNutrients", [])}

        def g(name: str) -> float:
            return float(nutrients.get(name, 0.0) or 0.0)

        results.append(
            {
                "fdcId": f.get("fdcId"),
                "description": desc,
                "brand": brand,
                "kcal": g("Energy"),
                "protein": g("Protein"),
                "carbs": g("Carbohydrate, by difference"),
                "fat": g("Total lipid (fat)"),
                "sodium": g("Sodium, Na"),
            }
        )
    return results


def ensure_session_lists():
    if "notes" not in st.session_state:
        st.session_state["notes"] = []
    if "meals" not in st.session_state:
        st.session_state["meals"] = []
    if "meal_search_results" not in st.session_state:
        st.session_state["meal_search_results"] = []


# ───────────────────────── UI – sidebar
with st.sidebar:
    st.markdown("<h2 class='sb-title'>Controls</h2>", unsafe_allow_html=True)
    hours = st.selectbox("Time window (hours)", [8, 24, 72, 168, 720], index=1)
    normalize = st.checkbox("Normalize combined overlay", True)

st.markdown(
    "<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>",
    unsafe_allow_html=True,
)

# ───────────────────────── Pull vitals
df = fetch_vitals(hours)
if df.empty:
    st.warning("No data yet.")
    st.stop()

pill_events = df.loc[
    df["metric"].str.contains("pillbox", case=False, na=False),
    "timestamp_utc",
].tolist()

# ───────────────────────── Tabs – vitals
tabs = st.tabs(
    ["❤️ Heart Rate", "💧 Systolic BP", "💜 Diastolic BP", "🫁 SpO₂", "📊 Combined"]
)

with tabs[0]:
    lsl, usl = LIMITS["pulse"]
    plot_with_limits(df, "pulse", pill_events, lsl, usl)

with tabs[1]:
    lsl, usl = LIMITS["systolic_bp"]
    plot_with_limits(df, "systolic_bp", pill_events, lsl, usl)

with tabs[2]:
    lsl, usl = LIMITS["diastolic_bp"]
    plot_with_limits(df, "diastolic_bp", pill_events, lsl, usl)

with tabs[3]:
    lsl, usl = LIMITS["spo2"]
    plot_with_limits(df, "spo2", pill_events, lsl, usl)

# Combined normalized overlay
with tabs[4]:
    show_cols = ["pulse", "spo2", "systolic_bp", "diastolic_bp"]
    combo = df[df["metric"].isin(show_cols)].sort_values("timestamp_utc").copy()
    if combo.empty:
        st.warning("No data for combined view")
    else:
        if normalize:
            combo["value"] = combo.groupby("metric")["value"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)
            )

        figc = px.line(
            combo,
            x="timestamp_utc",
            y="value",
            color="metric",
            category_orders={"metric": show_cols},
            color_discrete_map=COLORS,
        )
        figc.update_traces(line=dict(width=2))
        figc.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            hovermode="x unified",
            height=420,
        )
        figc.update_xaxes(
            rangeslider=dict(visible=True),
            showgrid=True,
            gridcolor="rgba(120,120,180,0.20)",
            griddash="dot",
        )
        figc.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

        st.plotly_chart(figc, use_container_width=True)
        st.markdown(stats_block(combo), unsafe_allow_html=True)

# ───────────────────────── Add Note + Add Meal (USDA)
st.markdown("### Add Note & Add Meal")

ensure_session_lists()
col_note, col_meal = st.columns([1, 2])

with col_note:
    st.subheader("📝 Add Note")
    note_text = st.text_area("Note", height=80, label_visibility="collapsed")
    note_time = st.time_input("When did this happen?", datetime.now().time())
    note_date = st.date_input("Date", datetime.now().date())
    if st.button("➕ Add Note"):
        dt = datetime.combine(note_date, note_time).replace(tzinfo=timezone.utc)
        st.session_state["notes"].append({"timestamp_utc": dt.isoformat(), "text": note_text})
        st.success("Note added.")

with col_meal:
    st.subheader("🍽️ Add Meal (USDA)")
    query = st.text_input("Search food (USDA)", value="oatmeal")
    meal_time = st.time_input("Meal time", datetime.now().time(), key="meal_time")
    meal_date = st.date_input("Meal date", datetime.now().date(), key="meal_date")

    if st.button("🔍 Search"):
        st.session_state["meal_search_results"] = usda_search_food(query)

    results = st.session_state["meal_search_results"]
    if results:
        for i, food in enumerate(results):
            desc = food["description"]
            brand = f" ({food['brand']})" if food["brand"] else ""
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"**{desc}**{brand}<br>"
                    f"{food['kcal']:.0f} kcal · "
                    f"{food['protein']:.1f} g protein · "
                    f"{food['carbs']:.1f} g carbs · "
                    f"{food['fat']:.1f} g fat · "
                    f"{food['sodium']:.0f} mg sodium",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("+ Add", key=f"add_meal_{i}"):
                    dt = datetime.combine(meal_date, meal_time).replace(tzinfo=timezone.utc)
                    meal_entry = {"timestamp_utc": dt.isoformat(), **food}
                    st.session_state["meals"].append(meal_entry)
                    st.success("Meal added.")

# ───────────────────────── Recent Meals summary
if st.session_state["meals"]:
    st.markdown("### Recent Meals (last 10)")
    meals_df = pd.DataFrame(st.session_state["meals"]).sort_values("timestamp_utc", ascending=False).head(10)

    # Nutrient totals
    totals = {
        "protein": meals_df["protein"].sum(),
        "carbs": meals_df["carbs"].sum(),
        "fat": meals_df["fat"].sum(),
        "sodium": meals_df["sodium"].sum(),
        "kcal": meals_df["kcal"].sum(),
    }

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Protein", f"{totals['protein']:.1f} g")
    c2.metric("Carbs", f"{totals['carbs']:.1f} g")
    c3.metric("Fat", f"{totals['fat']:.1f} g")
    c4.metric("Sodium", f"{totals['sodium']:.0f} mg")
    c5.metric("Energy", f"{totals['kcal']:.0f} kcal")

    st.dataframe(
        meals_df[["timestamp_utc", "description", "kcal", "protein", "carbs", "fat", "sodium"]],
        use_container_width=True,
        hide_index=True,
    )
