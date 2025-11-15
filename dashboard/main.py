import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ───────────────────────── Setup ─────────────────────────
st.set_page_config(page_title="Quantaira Dashboard", layout="wide")
CSS_FILE = os.path.join("assets", "custom.css")
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

API_BASE = os.getenv("API_BASE", "https://quantaira-render2.onrender.com/api")
USDA_API_KEY = os.getenv("USDA_API_KEY")  # optional

COLORS = {
    "pulse":        "#6F52ED",  # purple
    "spo2":         "#FF6B8A",  # pink
    "systolic_bp":  "#2D9CDB",  # blue
    "diastolic_bp": "#9B51E0",  # violet,
}

# simple default limits (you can tweak)
LIMITS = {
    "pulse":        (60, 100),   # bpm
    "systolic_bp":  (90, 130),   # mmHg
    "diastolic_bp": (60, 85),    # mmHg
    "spo2":         (94, 100),   # %
}

TIMEZONES = {
    "UTC": "UTC",
    "US/Eastern": "US/Eastern",
    "US/Central": "US/Central",
    "US/Pacific": "US/Pacific",
}

# ───────────────────────── Helpers: vitals ─────────────────────────
def fetch_vitals(hours: int = 72) -> pd.DataFrame:
    """Pull raw measurement rows from your FastAPI backend."""
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

    df["timestamp_utc"] = pd.to_datetime(df["created_utc"], utc=True)

    mlow = df["metric"].astype(str).str.lower()
    df.loc[mlow.isin({"pulse", "heart_rate", "hr"}), "metric"] = "pulse"
    df.loc[mlow.isin({"spo2", "sp02", "oxygen"}), "metric"] = "spo2"

    # split BP into systolic & diastolic
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

    return df[["timestamp_utc", "metric", "value"]].dropna(
        subset=["timestamp_utc", "metric", "value"]
    )


def convert_tz(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        return df.assign(timestamp=df["timestamp_utc"].dt.tz_convert(tz_name))
    except Exception:
        # fallback to UTC if timezone is weird
        return df.assign(timestamp=df["timestamp_utc"])


def classify_value(v: float, lsl: float, usl: float) -> str:
    """
    low / warn / ok / high based on LSL/USL.
    - below LSL or above USL => 'alert'
    - near the edges => 'warn' (yellow)
    - middle => 'ok' (green)
    """
    if v < lsl or v > usl:
        return "alert"
    mid = (lsl + usl) / 2.0
    band = (usl - lsl) * 0.25  # 25% band around middle is green
    if abs(v - mid) <= band:
        return "ok"
    return "warn"


def color_for_class(cls: str) -> str:
    if cls == "ok":
        return "#00B894"  # green
    if cls == "warn":
        return "#F1C40F"  # yellow
    return "#E74C3C"      # red


def build_colored_segments(sub: pd.DataFrame, lsl: float, usl: float):
    """
    Turn a time series into segments with different colors.
    Returns list of (x, y, color) segments.
    """
    xs = list(sub["timestamp"])
    ys = list(sub["value"])
    if len(xs) < 2:
        return []

    classes = [classify_value(v, lsl, usl) for v in ys]
    colors = [color_for_class(c) for c in classes]

    segments = []
    start = 0
    current_color = colors[0]

    for i in range(1, len(xs)):
        if colors[i] != current_color:
            # end previous segment at i
            segments.append(
                (xs[start:i + 1], ys[start:i + 1], current_color)
            )
            # start new segment at i-1 so lines join cleanly
            start = i - 1
            current_color = colors[i]

    segments.append((xs[start:], ys[start:], current_color))
    return segments, colors


def plot_metric_with_limits(df: pd.DataFrame, metric: str, tz: str, line_width: int, dot_size: int, show_limits: bool):
    sub = df[df["metric"] == metric].sort_values("timestamp_utc")
    if sub.empty:
        st.warning(f"No data for {metric}")
        return

    sub = convert_tz(sub, tz)
    lsl, usl = LIMITS.get(metric, (None, None))

    fig = go.Figure()

    if lsl is not None and usl is not None:
        segments, pt_colors = build_colored_segments(sub, lsl, usl)
        # colored line segments
        for xs, ys, color in segments:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=line_width, color=color),
                    hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
                    showlegend=False,
                )
            )
        # markers with same colors
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["value"],
                mode="markers",
                marker=dict(size=dot_size, color=pt_colors, line=dict(width=1, color="white")),
                hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
                showlegend=False,
            )
        )
    else:
        # fallback: single color line
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"],
                y=sub["value"],
                mode="lines+markers",
                line=dict(width=line_width, color=COLORS.get(metric, "#555")),
                marker=dict(size=dot_size),
                hovertemplate="%{y}<br>%{x|%Y-%m-%d %H:%M:%S %Z}<extra></extra>",
                showlegend=False,
            )
        )

    # LSL / USL dashed lines
    if show_limits and lsl is not None and usl is not None:
        tmin, tmax = sub["timestamp"].min(), sub["timestamp"].max()
        fig.add_hline(
            y=lsl,
            line=dict(color="rgba(231, 76, 60, 0.6)", dash="dash"),
            annotation_text=f"LSL {lsl}",
            annotation_position="bottom left",
        )
        fig.add_hline(
            y=usl,
            line=dict(color="rgba(231, 76, 60, 0.6)", dash="dash"),
            annotation_text=f"USL {usl}",
            annotation_position="top left",
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        height=420,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        showgrid=True,
        gridcolor="rgba(120,120,180,0.20)",
        griddash="dot",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")

    st.plotly_chart(fig, use_container_width=True)

    # stats card
    latest = float(sub["value"].iloc[-1])
    mu = float(sub["value"].mean())
    sigma = float(sub["value"].std(ddof=0) or 0.0)
    vmin = float(sub["value"].min())
    vmax = float(sub["value"].max())

    stats_html = f"""
    <div class="stats-footer">
      <span><b>LSL/USL:</b> {lsl if lsl is not None else "-"} / {usl if usl is not None else "-"}</span>
      <span><b>Latest:</b> {latest:.1f}</span>
      <span><b>µ Mean:</b> {mu:.1f}</span>
      <span><b>σ Std:</b> {sigma:.1f}</span>
      <span><b>Min:</b> {vmin:.1f}</span>
      <span><b>Max:</b> {vmax:.1f}</span>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

# ───────────────────────── Helpers: meals / USDA ─────────────────────────
def usda_search(query: str, page_size: int = 10):
    if not USDA_API_KEY:
        st.error("USDA_API_KEY not set – add it in Render env vars to enable meal search.")
        return []
    if not query:
        return []

    try:
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={
                "api_key": USDA_API_KEY,
                "query": query,
                "pageSize": page_size,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"USDA search failed: {e}")
        return []

    foods = []
    for item in data.get("foods", []):
        name = item.get("description", "Unknown")
        brand = item.get("brandName")
        full_name = f"{name}" + (f" — {brand}" if brand else "")

        # pull basic nutrients if present
        nutrients = {n["nutrientName"]: n["value"] for n in item.get("foodNutrients", [])}
        kcal = nutrients.get("Energy", 0)
        protein = nutrients.get("Protein", 0)
        carbs = nutrients.get("Carbohydrate, by difference", 0)
        fat = nutrients.get("Total lipid (fat)", 0)
        sodium = nutrients.get("Sodium, Na", 0)

        foods.append(
            dict(
                fdc_id=item.get("fdcId"),
                name=full_name,
                kcal=kcal,
                protein=protein,
                carbs=carbs,
                fat=fat,
                sodium=sodium,
            )
        )
    return foods


def init_session():
    if "meals" not in st.session_state:
        st.session_state["meals"] = []
    if "notes" not in st.session_state:
        st.session_state["notes"] = []


def add_meal_entry(food, when: datetime):
    entry = dict(
        name=food["name"],
        time=when.isoformat(),
        kcal=food["kcal"],
        protein=food["protein"],
        carbs=food["carbs"],
        fat=food["fat"],
        sodium=food["sodium"],
    )
    st.session_state["meals"].append(entry)


def add_note_entry(text: str, when: datetime):
    st.session_state["notes"].append({"text": text, "time": when.isoformat()})


# ───────────────────────── Layout ─────────────────────────
init_session()

with st.sidebar:
    st.markdown("<h2 class='sb-title'>Settings</h2>", unsafe_allow_html=True)
    tz_label = st.selectbox("Timezone", list(TIMEZONES.keys()), index=0)
    tz = TIMEZONES[tz_label]
    hours = st.selectbox("Time window (hours)", [8, 24, 72, 168, 720], index=1)
    line_width = st.slider("Line width", 1, 6, 4)
    dot_size = st.slider("Marker size (dots)", 6, 20, 10)
    show_limits = st.checkbox("Show LSL/USL dashed lines", True)

st.markdown("<div class='top-bar'><span class='title'>Quantaira Dashboard</span></div>", unsafe_allow_html=True)

# Tabs like in Todd’s UI – Home + Patient
tabs = st.tabs(["🏠 Home", "🧍 Patient"])

# ───────────────────────── Home tab (very simple) ─────────────────────────
with tabs[0]:
    st.subheader("Home")
    st.write("Use the **Patient** tab to view vitals, limits and meals.")

# ───────────────────────── Patient tab ─────────────────────────
with tabs[1]:
    df = fetch_vitals(hours)
    if df.empty:
        st.warning("No data yet.")
        st.stop()

    # metric selector pills
    metric_tabs = st.tabs(["❤️ Heart Rate", "💧 Systolic BP", "💜 Diastolic BP", "🫁 SpO₂", "📊 Combined"])

    with metric_tabs[0]:
        plot_metric_with_limits(df, "pulse", tz, line_width, dot_size, show_limits)

    with metric_tabs[1]:
        plot_metric_with_limits(df, "systolic_bp", tz, line_width, dot_size, show_limits)

    with metric_tabs[2]:
        plot_metric_with_limits(df, "diastolic_bp", tz, line_width, dot_size, show_limits)

    with metric_tabs[3]:
        plot_metric_with_limits(df, "spo2", tz, line_width, dot_size, show_limits)

    # Combined overlay (normalized)
    with metric_tabs[4]:
        show_cols = ["pulse", "spo2", "systolic_bp", "diastolic_bp"]
        combo = df[df["metric"].isin(show_cols)].copy()
        combo = convert_tz(combo, tz)
        if combo.empty:
            st.warning("No data for combined view")
        else:
            combo["value_norm"] = combo.groupby("metric")["value"].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
            )
            figc = px.line(
                combo,
                x="timestamp_utc",
                y="value_norm",
                color="metric",
                category_orders={"metric": show_cols},
                color_discrete_map=COLORS,
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
                rangeslider=dict(visible=True),
                showgrid=True,
                gridcolor="rgba(120,120,180,0.20)",
                griddash="dot",
            )
            figc.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
            st.plotly_chart(figc, use_container_width=True)

    st.markdown("---")

    # ───── Add Note / Add Meal (USDA) forms ─────
    left, right = st.columns(2)

    # Add Note
    with left:
        st.subheader("✏️ Add Note")
        note_text = st.text_area("Note", placeholder="e.g., felt dizzy after a walk", key="note_text")
        use_now_note = st.checkbox("Use current time for note", value=True, key="note_now")
        if use_now_note:
            note_when = datetime.now(timezone.utc)
        else:
            note_date = st.date_input("When? (date)", datetime.now().date(), key="note_date")
            note_time = st.time_input("Time", datetime.now().time(), key="note_time")
            note_when = datetime.combine(note_date, note_time).replace(tzinfo=timezone.utc)

        if st.button("➕ Add Note"):
            if note_text.strip():
                add_note_entry(note_text.strip(), note_when)
                st.success("Note added.")
            else:
                st.warning("Please type a note first.")

    # Add Meal (USDA)
    with right:
        st.subheader("🍽️ Add Meal (USDA)")
        meal_query = st.text_input("Search food (USDA)", placeholder="oatmeal, muffin, salad…")
        use_now_meal = st.checkbox("Use current time for meal", value=True, key="meal_now")
        if use_now_meal:
            meal_when = datetime.now(timezone.utc)
        else:
            meal_date = st.date_input("When was it eaten? (date)", datetime.now().date(), key="meal_date")
            meal_time = st.time_input("Time", datetime.now().time(), key="meal_time")
            meal_when = datetime.combine(meal_date, meal_time).replace(tzinfo=timezone.utc)

        search_clicked = st.button("🔍 Search")
        foods = []
        if search_clicked:
            foods = usda_search(meal_query)

        for idx, food in enumerate(foods):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"**{food['name']}**  \n"
                    f"{food['kcal']} kcal • P {food['protein']} g • "
                    f"C {food['carbs']} g • F {food['fat']} g • Na {food['sodium']} mg"
                )
            with col2:
                if st.button("Add", key=f"add_meal_{idx}"):
                    add_meal_entry(food, meal_when)
                    st.success("Meal added.")

    st.markdown("---")

    # ───── Recent Meals card ─────
    st.subheader("🍽️ Recent Meals")
    if not st.session_state["meals"]:
        st.info("No meals logged yet.")
    else:
        for m in reversed(st.session_state["meals"][-10:]):
            mt = datetime.fromisoformat(m["time"]).astimezone(timezone.utc)
            st.markdown(
                f"**{m['name']}** — {m['kcal']} kcal  \n"
                f"{mt:%Y-%m-%d %H:%M UTC}  \n"
                f"Protein **{m['protein']} g** • Carbs **{m['carbs']} g** • "
                f"Fat **{m['fat']} g** • Sodium **{m['sodium']} mg**"
            )
            st.write("---")
