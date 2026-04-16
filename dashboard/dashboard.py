"""
Smart Water Leakage Detection Dashboard
========================================
A professional analytics dashboard for monitoring smart city water infrastructure.
Built with Streamlit + Plotly for interactive data exploration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Water Leakage Detection",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — professional dark-blue theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Hide Streamlit Branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0f1e;
        color: #e0e8f8;
        font-family: 'Segoe UI', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1526;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] * { color: #c8d8f0 !important; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4fc3f7;
        border-left: 4px solid #0288d1;
        padding-left: 12px;
        margin: 1.5rem 0 0.8rem 0;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #90a4c4;
        margin-bottom: 1rem;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #0d1f3c 0%, #112240 100%);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px 22px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #7090b0;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #4fc3f7;
        line-height: 1.1;
    }
    .metric-card .delta {
        font-size: 0.82rem;
        color: #81c784;
        margin-top: 6px;
    }
    .metric-card.danger .value { color: #ef5350; }
    .metric-card.warning .value { color: #ffa726; }
    .metric-card.success .value { color: #66bb6a; }

    /* ── Insight boxes ── */
    .insight-box {
        background: linear-gradient(135deg, #0d2137 0%, #0f2a47 100%);
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.93rem;
        color: #c8ddf2;
    }
    .insight-box.warning { border-left-color: #ffa726; }
    .insight-box.danger  { border-left-color: #ef5350; }
    .insight-box.success { border-left-color: #66bb6a; }

    /* ── Alert badge ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-critical { background:#b71c1c; color:#fff; }
    .badge-high     { background:#e65100; color:#fff; }
    .badge-moderate { background:#f9a825; color:#000; }
    .badge-low      { background:#1b5e20; color:#fff; }

    /* ── Divider ── */
    hr { border-color: #1e3a5f; }

    /* ── Plotly chart containers ── */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING & CACHING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    """Load and preprocess the smart meter dataset (parquet or CSV)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data", "processed")

    # Try parquet first (smaller, faster), then CSV
    parquet_path = os.path.join(data_dir, "leakage_intelligence_dataset.parquet")
    csv_path = os.path.join(data_dir, "leakage_intelligence_dataset.csv")
    csv_fallback = r"C:\Users\khakh\OneDrive\Desktop\water_leakage_bigdata_project\data\processed\leakage_intelligence_dataset.csv"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(csv_fallback):
        df = pd.read_csv(csv_fallback)
    else:
        st.error("❌ Dataset not found! Please place 'leakage_intelligence_dataset.parquet' or '.csv' in data/processed/")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    df["month"]     = df["timestamp"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # Ordinal encodings for sorting
    sev_order  = {"Normal": 0, "Low Risk": 1, "Moderate Risk": 2, "High Risk": 3, "Critical Leak": 4}
    risk_order = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2, "Critical": 3}
    df["sev_order"]  = df["leak_severity"].map(sev_order)
    df["risk_order"] = df["risk_level"].map(risk_order)
    return df


# ──────────────────────────────────────────────
# PLOTLY THEME DEFAULTS
# ──────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="#0d1526",
    plot_bgcolor="#0d1526",
    font_color="#c8d8f0",
    
)
PALETTE = px.colors.sequential.Blues_r
COLOR_MAP_RISK = {
    "Low Risk":      "#66bb6a",
    "Moderate Risk": "#ffa726",
    "High Risk":     "#ef5350",
    "Critical":      "#b71c1c",
}
COLOR_MAP_SEV = {
    "Normal":        "#4fc3f7",
    "Low Risk":      "#66bb6a",
    "Moderate Risk": "#ffa726",
    "High Risk":     "#ef5350",
    "Critical Leak": "#b71c1c",
}


def apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply consistent dark theme to every Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#4fc3f7"), x=0.02),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(13,21,38,0.8)",
            bordercolor="#1e3a5f",
            borderwidth=1,
            font=dict(color="#c8d8f0"),
        ),
        **CHART_THEME,
    )
    fig.update_xaxes(gridcolor="#1a2f4a", zerolinecolor="#1a2f4a", tickfont_color="#90a4c4")
    fig.update_yaxes(gridcolor="#1a2f4a", zerolinecolor="#1a2f4a", tickfont_color="#90a4c4")
    return fig


# ──────────────────────────────────────────────
# SIDEBAR — global filters
# ──────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 💧 Smart Water Monitor")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Filters")

    # Risk level
    risk_options = sorted(df["risk_level"].unique(), key=lambda x: {"Low Risk":0,"Moderate Risk":1,"High Risk":2,"Critical":3}.get(x, 99))
    sel_risk = st.sidebar.multiselect("Risk Level", risk_options, default=risk_options)

    # Leak severity
    sev_options = sorted(df["leak_severity"].unique(), key=lambda x: {"Normal":0,"Low Risk":1,"Moderate Risk":2,"High Risk":3,"Critical Leak":4}.get(x, 99))
    sel_sev = st.sidebar.multiselect("Leak Severity", sev_options, default=sev_options)

    # Hour range
    h_min, h_max = int(df["hour"].min()), int(df["hour"].max())
    sel_hours = st.sidebar.slider("Hour Range (0–23)", h_min, h_max, (h_min, h_max), help="Filter records by time of day. Note: Continuous usage between hours 0 - 5 is a strong indicator of a persistent leak.")

    # Spike ratio threshold
    spike_thresh = st.sidebar.slider("Min Spike Ratio", 0.0, float(df["spike_ratio"].max()), 0.0, step=0.1, help="Show only records where usage was X times higher than the 7-day rolling average. Target > 2.0 to find severe anomalies.")

    # Household search
    hh_options = sorted(df["household_id"].unique())
    sel_hh = st.sidebar.multiselect("Household IDs (leave blank = all)", hh_options, default=[])

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📊 Dataset Info")
    st.sidebar.caption(f"Total records loaded: **{len(df):,}**")
    st.sidebar.caption(f"Date range: **{df['timestamp'].min().date()}** → **{df['timestamp'].max().date()}**")

    # Apply filters
    mask = (
        df["risk_level"].isin(sel_risk) &
        df["leak_severity"].isin(sel_sev) &
        df["hour"].between(*sel_hours) &
        (df["spike_ratio"] >= spike_thresh)
    )
    if sel_hh:
        mask &= df["household_id"].isin(sel_hh)

    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='text-align: center; color: #4fc3f7; font-size: 0.8rem; background-color: #112240; padding:10px; border-radius:8px;'>🌊 Sentinel Water Intelligence<br>🟢 Core AI System Online</div>", unsafe_allow_html=True)

    return df[mask].copy()


# ──────────────────────────────────────────────
# PAGE 1 — SYSTEM OVERVIEW
# ──────────────────────────────────────────────
def page_overview(df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 System Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time smart water infrastructure monitoring summary</div>', unsafe_allow_html=True)

    total_records    = len(df)
    total_households = df["household_id"].nunique()
    avg_usage        = df["water_usage_liters"].mean()
    total_leaks      = df["leak_flag_detected"].sum()
    high_risk_hh     = df[df["risk_level"].isin(["High Risk", "Critical"])]["household_id"].nunique()
    critical_hh      = df[df["risk_level"] == "Critical"]["household_id"].nunique()
    avg_spike        = df["spike_ratio"].mean()
    leak_pct         = (total_leaks / total_records * 100) if total_records else 0

    cols = st.columns(5)
    cards = [
        ("Total Records",        f"{total_records:,}",       f"Filtered dataset",              ""),
        ("Households",           f"{total_households:,}",    f"Unique smart meters",           ""),
        ("Avg Water Usage",      f"{avg_usage:.1f} L",       f"Per reading",                   ""),
        ("Leak Events",          f"{total_leaks:,}",         f"{leak_pct:.1f}% of records",    "danger"),
        ("High-Risk Households", f"{high_risk_hh:,}",        f"{critical_hh} critical",        "warning"),
    ]
    for col, (label, value, delta, cls) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Secondary KPI row ──
    c1, c2, c3 = st.columns(3)

    with c1:
        risk_counts = df["risk_level"].value_counts()
        fig = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=COLOR_MAP_RISK,
            labels={"x": "Risk Level", "y": "Records"},
        )
        apply_theme(fig, "Risk Level Distribution", 300)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        spike_bins = pd.cut(df["spike_ratio"], bins=[0, 1, 2, 3, 4, 99],
                            labels=["<1", "1–2", "2–3", "3–4", ">4"])
        spike_dist = spike_bins.value_counts().sort_index()
        fig2 = px.bar(x=spike_dist.index.astype(str), y=spike_dist.values,
                      color=spike_dist.index.astype(str),
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      labels={"x": "Spike Ratio Bucket", "y": "Records"})
        apply_theme(fig2, "Spike Ratio Distribution", 300)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        daily_leaks = df[df["leak_flag_detected"] == 1].groupby("date").size().reset_index(name="leaks")
        fig3 = px.area(daily_leaks, x="date", y="leaks",
                       color_discrete_sequence=["#ef5350"],
                       labels={"date": "Date", "leaks": "Leak Events"})
        fig3.update_traces(fillcolor="rgba(239,83,80,0.2)")
        apply_theme(fig3, "Daily Leak Event Trend", 300)
        st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 2 — LEAKAGE SEVERITY ANALYSIS
# ──────────────────────────────────────────────
def page_severity(df: pd.DataFrame):
    st.markdown('<div class="section-header">🔴 Leakage Severity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Breakdown of leak events by severity level and temporal trends</div>', unsafe_allow_html=True)

    sev_counts = df["leak_severity"].value_counts().sort_values(ascending=False)
    sev_order  = ["Normal", "Low Risk", "Moderate Risk", "High Risk", "Critical Leak"]
    sev_counts = sev_counts.reindex([s for s in sev_order if s in sev_counts.index]).dropna()

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            x=sev_counts.index, y=sev_counts.values,
            color=sev_counts.index,
            color_discrete_map=COLOR_MAP_SEV,
            text=sev_counts.values,
            labels={"x": "Severity", "y": "Count"},
        )
        fig.update_traces(textposition="outside", textfont_color="#c8d8f0")
        apply_theme(fig, "Leak Severity Distribution (Bar)", 380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            names=sev_counts.index, values=sev_counts.values,
            color=sev_counts.index,
            color_discrete_map=COLOR_MAP_SEV,
            hole=0.45,
        )
        fig2.update_traces(textinfo="percent+label", textfont_color="#fff")
        apply_theme(fig2, "Severity Share (Donut)", 380)
        st.plotly_chart(fig2, use_container_width=True)

    # Trend over time by severity
    sev_time = (
        df[df["leak_severity"] != "Normal"]
        .groupby(["month", "leak_severity"])
        .size()
        .reset_index(name="count")
    )
    fig3 = px.line(
        sev_time, x="month", y="count",
        color="leak_severity",
        color_discrete_map=COLOR_MAP_SEV,
        markers=True,
        labels={"month": "Month", "count": "Events", "leak_severity": "Severity"},
    )
    apply_theme(fig3, "Leak Events Over Time by Severity", 360)
    st.plotly_chart(fig3, use_container_width=True)

    # Severity vs avg water usage
    sev_usage = df.groupby("leak_severity")["water_usage_liters"].mean().reindex(sev_order).dropna()
    fig4 = px.bar(
        x=sev_usage.index, y=sev_usage.values,
        color=sev_usage.index,
        color_discrete_map=COLOR_MAP_SEV,
        labels={"x": "Severity", "y": "Avg Usage (L)"},
        text=[f"{v:.1f} L" for v in sev_usage.values],
    )
    fig4.update_traces(textposition="outside", textfont_color="#c8d8f0")
    apply_theme(fig4, "Average Water Usage by Severity Level", 360)
    st.plotly_chart(fig4, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 3 — HOUSEHOLD RISK INTELLIGENCE
# ──────────────────────────────────────────────
def page_risk(df: pd.DataFrame):
    st.markdown('<div class="section-header">🏠 Household Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Risk profiling and ranking of smart meter households</div>', unsafe_allow_html=True)

    # Risk distribution
    risk_order = ["Low Risk", "Moderate Risk", "High Risk", "Critical"]
    risk_counts = df["risk_level"].value_counts().reindex(risk_order).dropna()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=COLOR_MAP_RISK,
            text=risk_counts.values,
            labels={"x": "Risk Level", "y": "Households"},
        )
        fig.update_traces(textposition="outside", textfont_color="#c8d8f0")
        apply_theme(fig, "Household Count by Risk Level", 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            names=risk_counts.index, values=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map=COLOR_MAP_RISK,
            hole=0.4,
        )
        fig2.update_traces(textinfo="percent+label", textfont_color="#fff")
        apply_theme(fig2, "Risk Level Proportions", 360)
        st.plotly_chart(fig2, use_container_width=True)

    # Top high-risk households
    st.markdown('<div class="section-header" style="font-size:1.1rem">🚨 Top High-Risk Households</div>', unsafe_allow_html=True)
    top_risk = (
        df[df["risk_level"].isin(["High Risk", "Critical"])]
        .groupby("household_id")
        .agg(
            risk_level=("risk_level", "first"),
            avg_leak_prob=("leak_probability", "mean"),
            avg_spike=("spike_ratio", "mean"),
            leak_events=("leak_flag_detected", "sum"),
            avg_usage=("water_usage_liters", "mean"),
        )
        .sort_values("avg_leak_prob", ascending=False)
        .reset_index()
        .head(20)
    )
    top_risk["avg_leak_prob"] = top_risk["avg_leak_prob"].map("{:.4f}".format)
    top_risk["avg_spike"]     = top_risk["avg_spike"].map("{:.2f}".format)
    top_risk["avg_usage"]     = top_risk["avg_usage"].map("{:.1f} L".format)

    st.dataframe(
        top_risk.rename(columns={
            "household_id": "Household",
            "risk_level": "Risk",
            "avg_leak_prob": "Leak Probability",
            "avg_spike": "Avg Spike Ratio",
            "leak_events": "Leak Events",
            "avg_usage": "Avg Usage",
        }),
        use_container_width=True,
        height=420,
    )

    # Interactive search
    st.markdown('<div class="section-header" style="font-size:1.1rem">🔎 Household Search</div>', unsafe_allow_html=True)
    search = st.text_input("Search household ID (partial match)", "")
    filtered = df[df["household_id"].str.contains(search, case=False)] if search else df
    summary = (
        filtered.groupby("household_id")
        .agg(
            risk_level=("risk_level", "first"),
            leak_severity=("leak_severity", lambda x: x.mode()[0]),
            avg_leak_prob=("leak_probability", "mean"),
            avg_spike=("spike_ratio", "mean"),
            leak_events=("leak_flag_detected", "sum"),
        )
        .sort_values("avg_leak_prob", ascending=False)
        .reset_index()
    )
    summary["avg_leak_prob"] = summary["avg_leak_prob"].map("{:.4f}".format)
    summary["avg_spike"]     = summary["avg_spike"].map("{:.2f}".format)
    st.dataframe(summary.head(100), use_container_width=True, height=380)


# ──────────────────────────────────────────────
# PAGE 4 — WATER CONSUMPTION BEHAVIOR
# ──────────────────────────────────────────────
def page_consumption(df: pd.DataFrame):
    st.markdown('<div class="section-header">💧 Water Consumption Behavior</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Temporal and distributional analysis of household water usage</div>', unsafe_allow_html=True)

    # Hourly average
    hourly = df.groupby("hour")["water_usage_liters"].mean().reset_index()
    fig = px.line(
        hourly, x="hour", y="water_usage_liters",
        markers=True,
        color_discrete_sequence=["#4fc3f7"],
        labels={"hour": "Hour of Day", "water_usage_liters": "Avg Usage (L)"},
    )
    fig.add_vline(x=hourly.loc[hourly["water_usage_liters"].idxmax(), "hour"],
                  line_dash="dash", line_color="#ffa726",
                  annotation_text="Peak hour", annotation_font_color="#ffa726")
    apply_theme(fig, "Hourly Average Water Consumption", 360)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.histogram(
            df, x="water_usage_liters", nbins=60,
            color_discrete_sequence=["#4fc3f7"],
            labels={"water_usage_liters": "Usage (L)", "count": "Frequency"},
        )
        apply_theme(fig2, "Water Usage Distribution", 360)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_usage = df.groupby("day_of_week")["water_usage_liters"].mean().reindex(dow_order).dropna()
        fig3 = px.bar(
            x=dow_usage.index, y=dow_usage.values,
            color=dow_usage.values,
            color_continuous_scale="Blues",
            labels={"x": "Day", "y": "Avg Usage (L)", "color": "Usage"},
        )
        apply_theme(fig3, "Average Consumption by Day of Week", 360)
        st.plotly_chart(fig3, use_container_width=True)

    # Box plot per risk level
    fig4 = px.box(
        df, x="risk_level", y="water_usage_liters",
        color="risk_level",
        color_discrete_map=COLOR_MAP_RISK,
        category_orders={"risk_level": ["Low Risk","Moderate Risk","High Risk","Critical"]},
        labels={"risk_level": "Risk Level", "water_usage_liters": "Usage (L)"},
    )
    apply_theme(fig4, "Water Usage Distribution by Risk Level", 400)
    st.plotly_chart(fig4, use_container_width=True)

    # Heatmap: hour × day
    heat_data = df.groupby(["day_of_week","hour"])["water_usage_liters"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="day_of_week", columns="hour", values="water_usage_liters")
    heat_pivot = heat_pivot.reindex([d for d in dow_order if d in heat_pivot.index])
    fig5 = px.imshow(
        heat_pivot,
        color_continuous_scale="Blues",
        labels={"x": "Hour", "y": "Day", "color": "Avg L"},
        aspect="auto",
    )
    apply_theme(fig5, "Consumption Heatmap: Hour × Day", 380)
    st.plotly_chart(fig5, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 5 — ABNORMAL PATTERN DETECTION
# ──────────────────────────────────────────────
def page_anomalies(df: pd.DataFrame):
    st.markdown('<div class="section-header">⚠️ Abnormal Pattern Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Spike ratio–based anomaly identification across the network</div>', unsafe_allow_html=True)

    # Summary badges
    n2 = (df["spike_ratio"] > 2).sum()
    n3 = (df["spike_ratio"] > 3).sum()
    pct2 = n2 / len(df) * 100 if len(df) else 0
    pct3 = n3 / len(df) * 100 if len(df) else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card warning">
            <div class="label">Spike Ratio &gt; 2</div>
            <div class="value">{n2:,}</div>
            <div class="delta">{pct2:.1f}% of records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card danger">
            <div class="label">Spike Ratio &gt; 3</div>
            <div class="value">{n3:,}</div>
            <div class="delta">{pct3:.1f}% of records</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        max_spike = df["spike_ratio"].max()
        worst_hh  = df.loc[df["spike_ratio"].idxmax(), "household_id"]
        st.markdown(f"""<div class="metric-card danger">
            <div class="label">Max Spike Ratio</div>
            <div class="value">{max_spike:.2f}</div>
            <div class="delta">{worst_hh}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Scatter: spike_ratio vs water_usage, coloured by anomaly level
    df_plot = df.copy()
    df_plot["anomaly"] = "Normal"
    df_plot.loc[df_plot["spike_ratio"] > 2, "anomaly"] = "Spike > 2"
    df_plot.loc[df_plot["spike_ratio"] > 3, "anomaly"] = "Spike > 3"
    anom_colors = {"Normal": "#4fc3f7", "Spike > 2": "#ffa726", "Spike > 3": "#ef5350"}

    fig = px.scatter(
        df_plot.sample(min(3000, len(df_plot))),
        x="water_usage_liters", y="spike_ratio",
        color="anomaly", color_discrete_map=anom_colors,
        opacity=0.7, size_max=6,
        labels={"water_usage_liters": "Usage (L)", "spike_ratio": "Spike Ratio"},
        hover_data=["household_id", "risk_level"],
    )
    fig.add_hline(y=2, line_dash="dash", line_color="#ffa726",
                  annotation_text="Threshold 2", annotation_font_color="#ffa726")
    fig.add_hline(y=3, line_dash="dash", line_color="#ef5350",
                  annotation_text="Threshold 3", annotation_font_color="#ef5350")
    apply_theme(fig, "Spike Ratio vs Water Usage (Anomaly Scatter)", 440)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Spike > 2 over time
        spike2_time = df[df["spike_ratio"] > 2].groupby("date").size().reset_index(name="count")
        fig2 = px.bar(spike2_time, x="date", y="count",
                      color_discrete_sequence=["#ffa726"],
                      labels={"date": "Date", "count": "Anomaly Count"})
        apply_theme(fig2, "Daily Anomalies (Spike > 2)", 340)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        spike3_time = df[df["spike_ratio"] > 3].groupby("date").size().reset_index(name="count")
        fig3 = px.bar(spike3_time, x="date", y="count",
                      color_discrete_sequence=["#ef5350"],
                      labels={"date": "Date", "count": "Critical Anomaly Count"})
        apply_theme(fig3, "Daily Critical Anomalies (Spike > 3)", 340)
        st.plotly_chart(fig3, use_container_width=True)

    # Anomaly table
    st.markdown('<div class="section-header" style="font-size:1.1rem">📋 Anomalous Records (Spike > 2)</div>', unsafe_allow_html=True)
    anomaly_df = df[df["spike_ratio"] > 2][
        ["household_id","timestamp","water_usage_liters","spike_ratio","leak_severity","risk_level","leak_probability"]
    ].sort_values("spike_ratio", ascending=False).head(200)
    st.dataframe(anomaly_df, use_container_width=True, height=400)


# ──────────────────────────────────────────────
# PAGE 6 — HOUSEHOLD EXPLORER
# ──────────────────────────────────────────────
def page_explorer(df: pd.DataFrame):
    st.markdown('<div class="section-header">🔬 Household Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep-dive into individual household water behavior and risk profile</div>', unsafe_allow_html=True)

    households = sorted(df["household_id"].unique())
    selected_hh = st.selectbox("Select Household ID", households)

    hh_df = df[df["household_id"] == selected_hh].sort_values("timestamp")

    if hh_df.empty:
        st.warning("No data found for this household with the current filters.")
        return

    # KPI row
    risk    = hh_df["risk_level"].iloc[-1]
    prob    = hh_df["leak_probability"].mean()
    leaks   = hh_df["leak_flag_detected"].sum()
    avg_u   = hh_df["water_usage_liters"].mean()
    max_sp  = hh_df["spike_ratio"].max()
    risk_color = {"Low Risk":"success","Moderate Risk":"","High Risk":"warning","Critical":"danger"}.get(risk,"")

    cols = st.columns(5)
    for col, (lbl, val, cls) in zip(cols, [
        ("Risk Level",       risk,            risk_color),
        ("Leak Probability", f"{prob:.4f}",   "danger" if prob > 0.1 else ""),
        ("Leak Events",      str(leaks),      "danger" if leaks > 0 else "success"),
        ("Avg Usage",        f"{avg_u:.1f} L", ""),
        ("Max Spike Ratio",  f"{max_sp:.2f}", "danger" if max_sp > 3 else "warning" if max_sp > 2 else ""),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card {cls}">
                <div class="label">{lbl}</div>
                <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Water usage over time
    fig = px.line(hh_df, x="timestamp", y="water_usage_liters",
                  color_discrete_sequence=["#4fc3f7"],
                  labels={"timestamp": "Time", "water_usage_liters": "Usage (L)"})
    # Overlay leak events
    leaks_df = hh_df[hh_df["leak_flag_detected"] == 1]
    if not leaks_df.empty:
        fig.add_scatter(x=leaks_df["timestamp"], y=leaks_df["water_usage_liters"],
                        mode="markers", marker=dict(color="#ef5350", size=8, symbol="x"),
                        name="Leak Detected")
    apply_theme(fig, f"Water Usage Over Time — {selected_hh}", 380)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.line(hh_df, x="timestamp", y="spike_ratio",
                       color_discrete_sequence=["#ffa726"],
                       labels={"timestamp": "Time", "spike_ratio": "Spike Ratio"})
        fig2.add_hline(y=2, line_dash="dash", line_color="#ffa726",
                       annotation_text="Alert Threshold 2")
        fig2.add_hline(y=3, line_dash="dash", line_color="#ef5350",
                       annotation_text="Critical Threshold 3")
        apply_theme(fig2, "Spike Ratio Trend", 340)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        sev_dist = hh_df["leak_severity"].value_counts()
        fig3 = px.pie(names=sev_dist.index, values=sev_dist.values,
                      color=sev_dist.index, color_discrete_map=COLOR_MAP_SEV, hole=0.4)
        fig3.update_traces(textinfo="percent+label", textfont_color="#fff")
        apply_theme(fig3, "Severity Distribution", 340)
        st.plotly_chart(fig3, use_container_width=True)

    # Leak events table
    if not leaks_df.empty:
        st.markdown('<div class="section-header" style="font-size:1.1rem">🚨 Detected Leak Events</div>', unsafe_allow_html=True)
        st.dataframe(
            leaks_df[["timestamp","water_usage_liters","spike_ratio","leak_severity","leak_probability","risk_level"]],
            use_container_width=True,
        )
    else:
        st.markdown('<div class="insight-box success">✅ No leak events detected for this household in the filtered window.</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 6B — ML RISK PREDICTION (RF + XGBoost)
# ──────────────────────────────────────────────
def _load_model(name: str):
    """Load a model pipeline from the models directory. Returns pipeline or None."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", name)
    if not os.path.exists(model_path):
        return None
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception:
        return None


def page_ml_prediction(df: pd.DataFrame):
    st.markdown('<div class="section-header">🤖 Machine Learning Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-driven predictive analytics using Random Forest &amp; XGBoost — side-by-side</div>', unsafe_allow_html=True)

    # Load both models
    rf_pipeline  = _load_model("household_risk_model.pkl")
    xgb_pipeline = _load_model("household_xgboost_model.pkl")

    if rf_pipeline is None and xgb_pipeline is None:
        st.warning("⚠️ No models found! Please run `train_xgboost_model.py` to train both Random Forest and XGBoost models.")
        return

    rf_model  = rf_pipeline['model']  if rf_pipeline  else None
    xgb_model = xgb_pipeline['model'] if xgb_pipeline else None
    req_features = (rf_pipeline or xgb_pipeline).get('features',
        ['avg_usage', 'max_usage', 'std_usage', 'night_avg_usage',
         'avg_spike_ratio', 'max_spike_ratio', 'std_spike_ratio'])

    households = sorted(df["household_id"].unique())
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="insight-box"><b>1. Select Household</b></div>', unsafe_allow_html=True)
        selected_hh = st.selectbox("Select Household ID ", households)

        hh_data = df[df["household_id"] == selected_hh]
        night_data = hh_data[hh_data['hour'].isin([0, 1, 2, 3, 4, 5])] if 'hour' in hh_data.columns else hh_data
        day_data   = hh_data[~hh_data['hour'].isin([0, 1, 2, 3, 4, 5])] if 'hour' in hh_data.columns else hh_data
        total_readings = len(hh_data) if len(hh_data) > 0 else 1

        real_features = {
            'avg_usage': hh_data['water_usage_liters'].mean(),
            'max_usage': hh_data['water_usage_liters'].max(),
            'std_usage': hh_data['water_usage_liters'].std() if len(hh_data) > 1 else 0.0,
            'night_avg_usage': night_data['water_usage_liters'].mean() if len(night_data) > 0 else 0.0,
            'avg_spike_ratio': hh_data['spike_ratio'].mean(),
            'max_spike_ratio': hh_data['spike_ratio'].max(),
            'std_spike_ratio': hh_data['spike_ratio'].std() if len(hh_data) > 1 else 0.0,
            'night_day_ratio': (night_data['water_usage_liters'].mean() / max(day_data['water_usage_liters'].mean(), 0.01)) if len(night_data) > 0 and len(day_data) > 0 else 0.0,
            'usage_range': hh_data['water_usage_liters'].max() - hh_data['water_usage_liters'].mean(),
            'cv_usage': (hh_data['water_usage_liters'].std() / max(hh_data['water_usage_liters'].mean(), 0.01)) if len(hh_data) > 1 else 0.0,
        }

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="insight-box warning"><b>2. What-If Analysis (Behavioral Simulation)</b></div>', unsafe_allow_html=True)

        with st.expander("🧪 Tweak Household Behaviors", expanded=True):
            val_avg_u = st.slider("Average Usage (L)", 0.0, float(max(100.0, real_features['avg_usage']*2)), float(real_features['avg_usage']))
            val_max_u = st.slider("Max Usage (L)", 0.0, float(max(200.0, real_features['max_usage']*2)), float(real_features['max_usage']))
            val_std_u = st.slider("Usage Volatility (Std Dev)", 0.0, float(max(50.0, real_features['std_usage']*2)), float(real_features['std_usage']))
            val_nigh = st.slider("Night Avg Usage", 0.0, float(max(50.0, real_features['night_avg_usage']*2)), float(real_features['night_avg_usage']))
            val_avg_s = st.slider("Avg Spike Ratio", 0.0, 8.0, float(real_features['avg_spike_ratio']))
            val_max_s = st.slider("Max Spike Ratio", 0.0, 20.0, float(real_features['max_spike_ratio']))
            val_std_s = st.slider("Spike Volatility", 0.0, float(max(5.0, real_features['std_spike_ratio']*2)), float(real_features['std_spike_ratio']))

        # Compute derived features from slider values
        day_avg_est = max(val_avg_u, 0.01)
        val_night_day_ratio = val_nigh / day_avg_est
        val_usage_range = val_max_u - val_avg_u
        val_cv_usage = val_std_u / day_avg_est

    with c2:
        # Build feature vector — only behavioral features (no target leakage)
        input_row = {
            'avg_usage': val_avg_u,
            'max_usage': val_max_u,
            'std_usage': val_std_u,
            'night_avg_usage': val_nigh,
            'avg_spike_ratio': val_avg_s,
            'max_spike_ratio': val_max_s,
            'std_spike_ratio': val_std_s,
            'night_day_ratio': val_night_day_ratio,
            'usage_range': val_usage_range,
            'cv_usage': val_cv_usage,
        }
        # Only use features the model expects
        input_data = pd.DataFrame([{f: input_row.get(f, 0.0) for f in req_features}])

        # ── Side-by-side gauge charts ──
        st.markdown('<div class="insight-box"><b>🎯 Model Predictions — Side by Side</b></div>', unsafe_allow_html=True)
        gauge_cols = st.columns(2)

        models_info = []
        if rf_model is not None:
            rf_prob = rf_model.predict_proba(input_data)[0][1] if len(rf_model.classes_) > 1 else (0.0 if rf_model.classes_[0] == 0 else 1.0)
            rf_pred = rf_model.predict(input_data)[0]
            models_info.append(("🌲 Random Forest", rf_prob, rf_pred, rf_model, "#4fc3f7"))
        if xgb_model is not None:
            xgb_prob = xgb_model.predict_proba(input_data)[0][1] if len(xgb_model.classes_) > 1 else (0.0 if xgb_model.classes_[0] == 0 else 1.0)
            xgb_pred = xgb_model.predict(input_data)[0]
            models_info.append(("⚡ XGBoost", xgb_prob, xgb_pred, xgb_model, "#ab47bc"))

        for idx, (name, prob, pred, model, accent) in enumerate(models_info):
            with gauge_cols[idx] if len(models_info) > 1 else gauge_cols[0]:
                prob_color = "#66bb6a" if prob < 0.3 else "#ffa726" if prob < 0.7 else "#ef5350"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{name}", 'font': {'size': 16, 'color': accent}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#c8d8f0"},
                        'bar': {'color': prob_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "#1e3a5f",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(102, 187, 106, 0.15)'},
                            {'range': [30, 70], 'color': 'rgba(255, 167, 38, 0.15)'},
                            {'range': [70, 100], 'color': 'rgba(239, 83, 80, 0.15)'}],
                    }
                ))
                fig.update_layout(paper_bgcolor="#0d1526", plot_bgcolor="#0d1526", font_color="#c8d8f0",
                                  margin=dict(t=60, b=10, l=30, r=30), height=270)
                st.plotly_chart(fig, use_container_width=True)

                if pred == 1:
                    st.markdown(f'<div class="metric-card danger"><div class="value">🚨 HIGH RISK</div><div class="delta">{name}: Immediate inspection needed</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card success"><div class="value">✅ NORMAL</div><div class="delta">{name}: Operating properly</div></div>', unsafe_allow_html=True)

        # ── Ensemble Verdict ──
        if len(models_info) == 2:
            avg_prob = (models_info[0][1] + models_info[1][1]) / 2
            both_agree = models_info[0][2] == models_info[1][2]
            st.markdown("<br>", unsafe_allow_html=True)
            if both_agree:
                verdict_icon = "🚨" if models_info[0][2] == 1 else "✅"
                verdict_txt = "HIGH RISK" if models_info[0][2] == 1 else "NORMAL"
                st.markdown(f"""<div class="insight-box {'danger' if models_info[0][2] == 1 else 'success'}">
                    <b>🤝 Ensemble Verdict:</b> Both models agree — <b>{verdict_icon} {verdict_txt}</b> (Consensus probability: {avg_prob*100:.1f}%)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="insight-box warning">
                    <b>⚖️ Ensemble Verdict:</b> Models disagree — RF says <b>{"HIGH RISK" if models_info[0][2] == 1 else "NORMAL"}</b>, 
                    XGBoost says <b>{"HIGH RISK" if models_info[1][2] == 1 else "NORMAL"}</b>. 
                    Average probability: {avg_prob*100:.1f}%. <i>Manual review recommended.</i>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature importance comparison ──
        if len(models_info) == 2:
            rf_imp = models_info[0][3].feature_importances_
            xgb_imp = models_info[1][3].feature_importances_
            imp_df = pd.DataFrame({
                'Feature': req_features,
                'Random Forest': rf_imp,
                'XGBoost': xgb_imp,
            })
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='🌲 Random Forest', x=imp_df['Feature'], y=imp_df['Random Forest'],
                                  marker_color='#4fc3f7', opacity=0.85))
            fig2.add_trace(go.Bar(name='⚡ XGBoost', x=imp_df['Feature'], y=imp_df['XGBoost'],
                                  marker_color='#ab47bc', opacity=0.85))
            fig2.update_layout(barmode='group',
                               paper_bgcolor="#0d1526", plot_bgcolor="#0d1526", font_color="#c8d8f0",
                               title=dict(text="Feature Importance — RF vs XGBoost", font=dict(color="#4fc3f7", size=15)),
                               margin=dict(t=50, b=40, l=20, r=20), height=300,
                               legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f", borderwidth=1))
            fig2.update_xaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
            fig2.update_yaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4", title_text="Importance")
            st.plotly_chart(fig2, use_container_width=True)
        elif len(models_info) == 1:
            imp = models_info[0][3].feature_importances_
            imp_df = pd.DataFrame({'Feature': req_features, 'Importance': imp}).sort_values('Importance')
            fig2 = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues_r',
                         labels={'Importance': 'Influence', 'Feature': 'Driver'})
            fig2.update_layout(paper_bgcolor="#0d1526", plot_bgcolor="#0d1526", font_color="#c8d8f0",
                               title=dict(text="Key Risk Drivers (Feature Importance)", font=dict(color="#4fc3f7")),
                               margin=dict(t=50, b=0, l=20, r=20), height=250)
            fig2.update_xaxes(showgrid=False)
            st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE — MODEL COMPARISON (RF vs XGBoost)
# ──────────────────────────────────────────────
def page_model_comparison(df: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Model Comparison — Random Forest vs XGBoost</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive evaluation and comparison of both machine learning models used in the system</div>', unsafe_allow_html=True)

    # Load comparison metrics
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_comparison_metrics.pkl")
    if not os.path.exists(metrics_path):
        metrics_path = r"C:\Games\water_leakage_bigdata_project_2\models\model_comparison_metrics.pkl"

    if not os.path.exists(metrics_path):
        st.warning("⚠️ Model comparison metrics not found. Please run `train_xgboost_model.py` first to train both models and generate comparison data.")
        st.code("python train_xgboost_model.py", language="bash")
        return

    try:
        metrics = joblib.load(metrics_path)
    except Exception as e:
        st.error(f"Error loading comparison metrics: {e}")
        return

    rf_m  = metrics['random_forest']
    xgb_m = metrics['xgboost']
    features = metrics.get('features', [])

    # ════════════════════════════════════════════
    # Section 1: Algorithm Overview
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">🧬 Algorithm Overview</div>', unsafe_allow_html=True)

    algo_c1, algo_c2 = st.columns(2)
    with algo_c1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0d2137 0%, #0f2a47 100%);
                    border: 1px solid #1e3a5f; border-radius: 14px; padding: 22px;
                    border-top: 3px solid #4fc3f7;">
            <h3 style="color:#4fc3f7; margin-top:0; font-size:1.15rem;">🌲 Random Forest</h3>
            <p style="color:#90a4c4; font-size:0.88rem; line-height:1.6;">
                An <b>ensemble</b> of multiple decision trees, each trained on a random subset of the data.
                Final prediction is made by <b>majority vote</b> (classification) across all trees.
            </p>
            <hr style="border-color:#1e3a5f;">
            <p style="color:#c8ddf2; font-size:0.83rem; margin-bottom:4px;"><b>✅ Strengths:</b></p>
            <ul style="color:#90a4c4; font-size:0.82rem; margin-top:0;">
                <li>Resistant to overfitting due to bagging</li>
                <li>Handles noisy data and outliers well</li>
                <li>Provides reliable feature importance</li>
                <li>Minimal hyperparameter tuning needed</li>
            </ul>
            <p style="color:#c8ddf2; font-size:0.83rem; margin-bottom:4px;"><b>⚠️ Limitations:</b></p>
            <ul style="color:#90a4c4; font-size:0.82rem; margin-top:0;">
                <li>Can be slower for very large datasets</li>
                <li>Less effective on highly imbalanced data</li>
                <li>Models can be large in size</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with algo_c2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a0d2e 0%, #231447 100%);
                    border: 1px solid #3a1e5f; border-radius: 14px; padding: 22px;
                    border-top: 3px solid #ab47bc;">
            <h3 style="color:#ab47bc; margin-top:0; font-size:1.15rem;">⚡ XGBoost</h3>
            <p style="color:#90a4c4; font-size:0.88rem; line-height:1.6;">
                <b>Extreme Gradient Boosting</b> — builds trees <i>sequentially</i>, where each new tree
                corrects the errors of the previous ones using <b>gradient descent optimization</b>.
            </p>
            <hr style="border-color:#3a1e5f;">
            <p style="color:#c8ddf2; font-size:0.83rem; margin-bottom:4px;"><b>✅ Strengths:</b></p>
            <ul style="color:#90a4c4; font-size:0.82rem; margin-top:0;">
                <li>Often achieves higher accuracy via boosting</li>
                <li>Built-in L1/L2 regularization prevents overfitting</li>
                <li>Handles imbalanced classes with scale_pos_weight</li>
                <li>Extremely fast with hardware optimizations</li>
            </ul>
            <p style="color:#c8ddf2; font-size:0.83rem; margin-bottom:4px;"><b>⚠️ Limitations:</b></p>
            <ul style="color:#90a4c4; font-size:0.82rem; margin-top:0;">
                <li>More sensitive to hyperparameter tuning</li>
                <li>Can overfit on small/noisy datasets</li>
                <li>Sequential nature makes training slower</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # Section 2: Head-to-Head Metrics
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">🏆 Performance Head-to-Head</div>', unsafe_allow_html=True)

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    rf_vals  = [rf_m['accuracy'], rf_m['precision'], rf_m['recall'], rf_m['f1'], rf_m['auc_roc']]
    xgb_vals = [xgb_m['accuracy'], xgb_m['precision'], xgb_m['recall'], xgb_m['f1'], xgb_m['auc_roc']]

    metric_cols = st.columns(5)
    for i, (name, rv, xv) in enumerate(zip(metric_names, rf_vals, xgb_vals)):
        with metric_cols[i]:
            winner = "rf" if rv > xv else "xgb" if xv > rv else "tie"
            rf_color = "#4fc3f7" if winner == "rf" else "#90a4c4"
            xgb_color = "#ab47bc" if winner == "xgb" else "#90a4c4"
            crown = "👑" if winner != "tie" else "🤝"
            st.markdown(f"""
            <div class="metric-card" style="padding:16px 12px;">
                <div class="label" style="font-size:0.72rem;">{name} {crown}</div>
                <div style="display:flex; justify-content:center; gap:18px; margin-top:8px;">
                    <div>
                        <div style="font-size:0.65rem; color:#4fc3f7; text-transform:uppercase; letter-spacing:1px;">RF</div>
                        <div style="font-size:1.5rem; font-weight:800; color:{rf_color};">{rv:.3f}</div>
                    </div>
                    <div style="border-left:1px solid #1e3a5f;"></div>
                    <div>
                        <div style="font-size:0.65rem; color:#ab47bc; text-transform:uppercase; letter-spacing:1px;">XGB</div>
                        <div style="font-size:1.5rem; font-weight:800; color:{xgb_color};">{xv:.3f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Winner summary
    rf_wins = sum(1 for rv, xv in zip(rf_vals, xgb_vals) if rv > xv)
    xgb_wins = sum(1 for rv, xv in zip(rf_vals, xgb_vals) if xv > rv)
    ties = 5 - rf_wins - xgb_wins

    if rf_wins > xgb_wins:
        st.markdown(f"""<div class="insight-box"><b>🌲 Random Forest wins {rf_wins}/5 metrics</b> | XGBoost wins {xgb_wins}/5 | Ties: {ties}
        <br><span style="color:#90a4c4; font-size:0.85rem;">Random Forest shows stronger overall performance on this dataset. 
        Its bagging approach and resistance to noise give it an edge for household risk classification.</span></div>""", unsafe_allow_html=True)
    elif xgb_wins > rf_wins:
        st.markdown(f"""<div class="insight-box"><b>⚡ XGBoost wins {xgb_wins}/5 metrics</b> | Random Forest wins {rf_wins}/5 | Ties: {ties}
        <br><span style="color:#90a4c4; font-size:0.85rem;">XGBoost's gradient boosting approach and regularization provide superior performance on this dataset, 
        particularly strong in precision and recall trade-offs.</span></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="insight-box success"><b>🤝 It's a tie!</b> Both models perform equally across the evaluation metrics.
        <br><span style="color:#90a4c4; font-size:0.85rem;">Using both models simultaneously (ensemble) provides the most reliable predictions.</span></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # Section 3: ROC Curves + Confusion Matrices
    # ════════════════════════════════════════════
    roc_col, cm_col = st.columns(2)

    with roc_col:
        st.markdown('<div class="section-header" style="font-size:1.05rem">📈 ROC Curve Comparison</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=rf_m['fpr'], y=rf_m['tpr'], mode='lines',
                                     name=f"RF (AUC={rf_m['auc_roc']:.3f})",
                                     line=dict(color='#4fc3f7', width=2.5)))
        fig_roc.add_trace(go.Scatter(x=xgb_m['fpr'], y=xgb_m['tpr'], mode='lines',
                                     name=f"XGB (AUC={xgb_m['auc_roc']:.3f})",
                                     line=dict(color='#ab47bc', width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name="Random Baseline",
                                     line=dict(color='#3a5a80', width=1, dash='dash')))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380, margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f", borderwidth=1,
                       font=dict(color="#c8d8f0"), x=0.55, y=0.05),
            **CHART_THEME,
        )
        fig_roc.update_xaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
        fig_roc.update_yaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown("""<div class="insight-box" style="font-size:0.82rem;">
            <b>📖 How to read the ROC curve:</b> The closer the curve follows the top-left corner, the better.
            AUC = 1.0 is perfect; AUC = 0.5 is random guessing (dashed line). A higher AUC means the model 
            is better at distinguishing high-risk from normal households.
        </div>""", unsafe_allow_html=True)

    with cm_col:
        st.markdown('<div class="section-header" style="font-size:1.05rem">🔢 Confusion Matrices</div>', unsafe_allow_html=True)

        cm_tabs = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])
        for tab, (label, cm, accent) in zip(cm_tabs, [
            ("Random Forest", rf_m['confusion_matrix'], 'Blues'),
            ("XGBoost", xgb_m['confusion_matrix'], 'Purples'),
        ]):
            with tab:
                cm_arr = np.array(cm)
                fig_cm = px.imshow(
                    cm_arr,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Normal", "High Risk"], y=["Normal", "High Risk"],
                    color_continuous_scale=accent,
                    text_auto=True,
                    aspect="equal",
                )
                fig_cm.update_layout(
                    height=310, margin=dict(l=20, r=20, t=30, b=20),
                    title=dict(text=f"{label} Confusion Matrix", font=dict(color="#4fc3f7", size=13)),
                    **CHART_THEME,
                )
                fig_cm.update_traces(textfont_size=18, textfont_color="#fff")
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""<div class="insight-box" style="font-size:0.82rem;">
            <b>📖 Reading the matrix:</b> Top-left = correct normals (TN), bottom-right = correct high-risk (TP).
            Top-right = false alarms (FP), bottom-left = missed risks (FN). Fewer FN is critical for safety.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # Section 4: Feature Importance Comparison
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">🔬 Feature Importance Analysis</div>', unsafe_allow_html=True)

    rf_imp  = rf_m.get('feature_importances', {})
    xgb_imp = xgb_m.get('feature_importances', {})

    imp_df = pd.DataFrame({
        'Feature': features,
        'Random Forest': [rf_imp.get(f, 0) for f in features],
        'XGBoost': [xgb_imp.get(f, 0) for f in features],
    })
    imp_df['Difference'] = imp_df['XGBoost'] - imp_df['Random Forest']
    imp_df = imp_df.sort_values('Random Forest', ascending=True)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(name='🌲 Random Forest', y=imp_df['Feature'], x=imp_df['Random Forest'],
                             orientation='h', marker_color='#4fc3f7', opacity=0.85))
    fig_imp.add_trace(go.Bar(name='⚡ XGBoost', y=imp_df['Feature'], x=imp_df['XGBoost'],
                             orientation='h', marker_color='#ab47bc', opacity=0.85))
    fig_imp.update_layout(
        barmode='group', height=380,
        margin=dict(l=20, r=20, t=50, b=40),
        title=dict(text="Feature Influence — Which Behaviors Drive Risk?", font=dict(color="#4fc3f7", size=14)),
        xaxis_title="Importance Score",
        legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f", borderwidth=1),
        **CHART_THEME,
    )
    fig_imp.update_xaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
    fig_imp.update_yaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Feature descriptions
    feature_desc = {
        'avg_usage':       '📊 Average water consumption per reading',
        'max_usage':       '📈 Maximum single reading — detects burst events',
        'std_usage':       '📉 Volatility of consumption — erratic usage signals leaks',
        'night_avg_usage': '🌙 Average usage at night (12 AM – 5 AM) — key leak indicator',
        'avg_spike_ratio': '⚡ Average anomaly spike ratio over time',
        'max_spike_ratio': '🔺 Maximum spike ratio — detects the worst anomaly',
        'std_spike_ratio': '🎲 Spike ratio volatility — unstable patterns',
    }
    desc_items = "".join([f"<li><code>{f}</code> — {feature_desc.get(f, 'Behavioral metric')}</li>" for f in features])
    st.markdown(f"""<div class="insight-box" style="font-size:0.85rem;">
        <b>📖 Feature Descriptions:</b>
        <ul style="margin-top:6px; line-height:1.7;">{desc_items}</ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # Section 5: Training Configuration Details
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">⚙️ Model Configuration &amp; Training Details</div>', unsafe_allow_html=True)

    cfg_c1, cfg_c2 = st.columns(2)
    with cfg_c1:
        rf_params = rf_m.get('model_params', {})
        params_html = "".join([f"<tr><td style='color:#90a4c4; padding:6px 12px; border-bottom:1px solid #1a2f4a;'>{k}</td><td style='color:#4fc3f7; padding:6px 12px; border-bottom:1px solid #1a2f4a; font-weight:600;'>{v}</td></tr>" for k, v in rf_params.items()])
        st.markdown(f"""
        <div style="background:#0d1f3c; border:1px solid #1e3a5f; border-radius:12px; padding:18px; border-top:3px solid #4fc3f7;">
            <h4 style="color:#4fc3f7; margin-top:0;">🌲 Random Forest Configuration</h4>
            <table style="width:100%; border-collapse:collapse;">{params_html}</table>
            <p style="color:#7090b0; font-size:0.78rem; margin-top:12px;">Training samples: {rf_m.get('training_samples', 'N/A')} | Test samples: {rf_m.get('test_samples', 'N/A')}</p>
        </div>""", unsafe_allow_html=True)

    with cfg_c2:
        xgb_params = xgb_m.get('model_params', {})
        params_html = "".join([f"<tr><td style='color:#90a4c4; padding:6px 12px; border-bottom:1px solid #2a1a4a;'>{k}</td><td style='color:#ab47bc; padding:6px 12px; border-bottom:1px solid #2a1a4a; font-weight:600;'>{v}</td></tr>" for k, v in xgb_params.items()])
        st.markdown(f"""
        <div style="background:#1a0d2e; border:1px solid #3a1e5f; border-radius:12px; padding:18px; border-top:3px solid #ab47bc;">
            <h4 style="color:#ab47bc; margin-top:0;">⚡ XGBoost Configuration</h4>
            <table style="width:100%; border-collapse:collapse;">{params_html}</table>
            <p style="color:#7090b0; font-size:0.78rem; margin-top:12px;">Training samples: {xgb_m.get('training_samples', 'N/A')} | Test samples: {xgb_m.get('test_samples', 'N/A')}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # Section 6: Detailed Metrics Table
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">📋 Complete Metrics Summary</div>', unsafe_allow_html=True)

    summary_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC-ROC',
                   'True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
        'Random Forest': [
            f"{rf_m['accuracy']:.4f}", f"{rf_m['precision']:.4f}", f"{rf_m['recall']:.4f}",
            f"{rf_m['f1']:.4f}", f"{rf_m['auc_roc']:.4f}",
            str(rf_m['confusion_matrix'][0][0]), str(rf_m['confusion_matrix'][0][1]),
            str(rf_m['confusion_matrix'][1][0]), str(rf_m['confusion_matrix'][1][1]),
        ],
        'XGBoost': [
            f"{xgb_m['accuracy']:.4f}", f"{xgb_m['precision']:.4f}", f"{xgb_m['recall']:.4f}",
            f"{xgb_m['f1']:.4f}", f"{xgb_m['auc_roc']:.4f}",
            str(xgb_m['confusion_matrix'][0][0]), str(xgb_m['confusion_matrix'][0][1]),
            str(xgb_m['confusion_matrix'][1][0]), str(xgb_m['confusion_matrix'][1][1]),
        ],
    })

    # Best indicator
    best = []
    for i in range(5):
        rv = float(summary_data['Random Forest'].iloc[i])
        xv = float(summary_data['XGBoost'].iloc[i])
        best.append("🌲 RF" if rv > xv else "⚡ XGB" if xv > rv else "🤝 Tie")
    best += ["—", "—", "—", "—"]
    summary_data['Winner'] = best

    st.dataframe(summary_data, use_container_width=True, height=380)

    # ════════════════════════════════════════════
    # Section 7: Key Takeaways
    # ════════════════════════════════════════════
    st.markdown('<div class="section-header" style="font-size:1.15rem">💡 Key Takeaways</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d2137 0%, #0f2a47 100%);
                border: 1px solid #1e3a5f; border-radius: 12px; padding: 22px; margin-bottom: 16px;">
        <h4 style="color:#4fc3f7; margin-top:0;">Why Use Two Models?</h4>
        <ul style="color:#c8ddf2; font-size:0.9rem; line-height:1.8;">
            <li><b>Ensemble Confidence:</b> When both models agree on a prediction, we have much higher 
                confidence in the result. Disagreements flag borderline cases for manual review.</li>
            <li><b>Different Perspectives:</b> Random Forest uses <i>bagging</i> (parallel trees, majority vote) 
                while XGBoost uses <i>boosting</i> (sequential error correction). They capture different patterns.</li>
            <li><b>Robustness:</b> If one model is affected by data drift or noise, the other provides a safety net.
                This is critical for infrastructure monitoring where missed leaks are costly.</li>
            <li><b>Interpretability vs Performance:</b> Random Forest feature importance is intuitive and stable; 
                XGBoost often squeezes out slightly better accuracy with gradient optimization.</li>
        </ul>
    </div>

    <div style="background: linear-gradient(135deg, #1e1e0d 0%, #2a2a14 100%);
                border: 1px solid #5f5f1e; border-radius: 12px; padding: 22px;">
        <h4 style="color:#ffa726; margin-top:0;">📊 Metric Explanations</h4>
        <ul style="color:#c8ddf2; font-size:0.88rem; line-height:1.8;">
            <li><b>Accuracy:</b> Overall correctness — what % of all predictions were right.</li>
            <li><b>Precision:</b> Of households flagged as high-risk, what % truly are. High precision = few false alarms.</li>
            <li><b>Recall (Sensitivity):</b> Of all actual high-risk households, what % did we catch. 
                <i>Critical for safety — missed leaks are expensive.</i></li>
            <li><b>F1 Score:</b> Harmonic mean of precision &amp; recall — balances both concerns.</li>
            <li><b>AUC-ROC:</b> Model's ability to rank high-risk above normal across all thresholds. 
                Higher = better discrimination ability.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# PAGE 7 — SMART INSIGHTS PANEL
# ──────────────────────────────────────────────
def page_insights(df: pd.DataFrame):
    st.markdown('<div class="section-header">🧠 Smart Insights Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automatically generated analytics intelligence from the dataset</div>', unsafe_allow_html=True)

    total = len(df)

    # ── Peak hours ──
    hourly_avg = df.groupby("hour")["water_usage_liters"].mean()
    peak_hour  = int(hourly_avg.idxmax())
    low_hour   = int(hourly_avg.idxmin())
    peak_val   = hourly_avg.max()

    # ── Spike stats ──
    pct_spike2 = (df["spike_ratio"] > 2).sum() / total * 100
    pct_spike3 = (df["spike_ratio"] > 3).sum() / total * 100
    avg_spike  = df["spike_ratio"].mean()

    # ── Leak stats ──
    leak_rate  = df["leak_flag_detected"].mean() * 100
    top5_leak  = df.groupby("household_id")["leak_flag_detected"].sum().nlargest(5)

    # ── Risk ──
    critical_pct = (df["risk_level"] == "Critical").sum() / total * 100
    high_pct     = (df["risk_level"] == "High Risk").sum() / total * 100

    insights = [
        ("info",    f"🕐 Peak consumption occurs at **{peak_hour:02d}:00** with avg {peak_val:.1f} L. "
                    f"Lowest demand is at **{low_hour:02d}:00**."),
        ("warning", f"⚠️ **{pct_spike2:.1f}%** of readings show abnormal spikes (ratio > 2). "
                    f"**{pct_spike3:.1f}%** are critically high (ratio > 3)."),
        ("info",    f"📊 Average spike ratio across the entire dataset is **{avg_spike:.3f}**. "
                    f"Values above 2.0 require immediate inspection."),
        ("danger",  f"🚨 Overall leak detection rate: **{leak_rate:.1f}%** of all records triggered a leak event."),
        ("warning", f"🔴 **{critical_pct:.1f}%** of records are in the Critical risk tier; "
                    f"**{high_pct:.1f}%** in High Risk."),
        ("success", f"✅ {(100 - pct_spike2):.1f}% of readings are within normal consumption bounds (spike ratio ≤ 2)."),
    ]

    col_cls = {"info": "", "warning": "warning", "danger": "danger", "success": "success"}
    for kind, text in insights:
        st.markdown(f'<div class="insight-box {col_cls[kind]}">{text}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Top households by leak events
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Top 10 Households — Leak Events")
        top_leaks = df.groupby("household_id")["leak_flag_detected"].sum().nlargest(10).reset_index()
        fig = px.bar(top_leaks, x="household_id", y="leak_flag_detected",
                     color="leak_flag_detected", color_continuous_scale="Reds",
                     labels={"household_id": "Household", "leak_flag_detected": "Leak Events"})
        apply_theme(fig, "", 360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### 📈 Top 10 Households — Avg Spike Ratio")
        top_spikes = df.groupby("household_id")["spike_ratio"].mean().nlargest(10).reset_index()
        fig2 = px.bar(top_spikes, x="household_id", y="spike_ratio",
                      color="spike_ratio", color_continuous_scale="Oranges",
                      labels={"household_id": "Household", "spike_ratio": "Avg Spike Ratio"})
        apply_theme(fig2, "", 360)
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend
    monthly = df.groupby("month").agg(
        avg_usage=("water_usage_liters", "mean"),
        leak_events=("leak_flag_detected", "sum"),
        avg_spike=("spike_ratio", "mean"),
    ).reset_index()

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(x=monthly["month"], y=monthly["avg_usage"],
                          name="Avg Usage (L)", marker_color="#4fc3f7", opacity=0.7), secondary_y=False)
    fig3.add_trace(go.Scatter(x=monthly["month"], y=monthly["leak_events"],
                              name="Leak Events", mode="lines+markers",
                              line=dict(color="#ef5350", width=2)), secondary_y=True)
    fig3.update_layout(height=360, paper_bgcolor="#0d1526", plot_bgcolor="#0d1526",
                       font_color="#c8d8f0", title=dict(text="Monthly Usage vs Leak Events", font=dict(color="#4fc3f7")),
                       legend=dict(bgcolor="rgba(13,21,38,0.8)", bordercolor="#1e3a5f"))
    fig3.update_xaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
    fig3.update_yaxes(gridcolor="#1a2f4a", tickfont_color="#90a4c4")
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 8 — DATA FILTERS / RAW EXPLORER
# ──────────────────────────────────────────────
def page_data(df: pd.DataFrame):
    st.markdown('<div class="section-header">🗃️ Filtered Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Full interactive data table reflecting all active sidebar filters</div>', unsafe_allow_html=True)

    st.info(f"**{len(df):,}** records match the current filter criteria.")

    # Column selector
    all_cols = list(df.columns)
    visible  = st.multiselect(
        "Select columns to display",
        all_cols,
        default=["household_id","timestamp","hour","water_usage_liters","spike_ratio",
                 "leak_flag_detected","leak_severity","leak_probability","risk_level"],
    )

    # Sort
    sort_col = st.selectbox("Sort by", visible, index=0)
    sort_asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"

    display_df = df[visible].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=500)

    # Download
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Filtered CSV",
        data=csv_bytes,
        file_name="filtered_water_data.csv",
        mime="text/csv",
    )

    # Quick stats
    st.markdown("#### 📊 Quick Statistics")
    numeric_cols = display_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.dataframe(display_df[numeric_cols].describe().T.style.background_gradient(cmap="Blues"),
                     use_container_width=True)


# ──────────────────────────────────────────────
# PAGE 8 — METHODOLOGY & DEFINITIONS
# ──────────────────────────────────────────────
def page_methodology(df: pd.DataFrame):
    st.markdown('<div class="section-header">📖 Methodology & Data Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Technical transparency for formulas, thresholds, and risk profiling logic.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #4fc3f7; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">1. Spike Ratio (Anomaly Detection)</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;"><b>Formula:</b> <code>Spike Ratio = Current Hourly Usage / 7-Day Rolling Average</code></p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>< 1.0:</b> Below average usage (Normal).</li>
            <li><b>1.0 - 1.5:</b> Slightly elevated usage (Expected variance).</li>
            <li><b>1.5 - 3.0:</b> High Variance (Moderate Anomaly).</li>
            <li><b>> 3.0:</b> Severe Spike (Significant Anomaly/Potential Burst).</li>
        </ul>
    </div>
    
    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #ffa726; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">2. Leak Probability (Bayesian/Heuristic)</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;">Calculated dynamically as a weighted score incorporating continuous prolonged usage (especially at night) and severe volume spikes.</p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>Threshold:</b> Probability > 0.8 automatically triggers a <code>leak_flag</code>.</li>
        </ul>
    </div>

    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #ef5350; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">3. Household Risk Tiers</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;">Each record is ranked. A household's overarching priority is typically dictated by its highest historical risk tier.</p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>🟢 Normal:</b> No sustained anomalies.</li>
            <li><b>🟡 Low Risk:</b> Occasional minor spikes.</li>
            <li><b>🟠 Moderate Risk:</b> Frequent spikes or minor continuous flow (Prob: 0.4 - 0.7).</li>
            <li><b>🔴 High Risk:</b> High probability of hidden minor leak (Prob: 0.7 - 0.9).</li>
            <li><b>🚨 Critical:</b> Active burst pipe or massive continuous flow (Prob > 0.9).</li>
        </ul>
    </div>

    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #4fc3f7; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">4. Random Forest Classifier</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;">An <b>ensemble learning</b> method that operates by constructing <b>100 decision trees</b> during training. Each tree is trained on a random subset of data (bagging) and features. Final prediction = majority vote.</p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>Inputs:</b> <i>Avg Usage, Max Usage, Night Usage (12 AM–5 AM), Std Dev of Usage, Spike Ratios (avg, max, std).</i></li>
            <li><b>Method:</b> Bagging (Bootstrap Aggregating) — parallel independent trees → majority vote.</li>
            <li><b>Target:</b> Predicts if a household is "High Risk" (1) or "Normal" (0) based on aggregated behavioral patterns.</li>
            <li><b>Advantage:</b> Robust to noise and outliers, requires minimal tuning, and provides interpretable feature importances.</li>
        </ul>
    </div>

    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #ab47bc; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">5. XGBoost Classifier (Extreme Gradient Boosting)</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;">A <b>gradient boosting</b> algorithm that builds <b>150 trees sequentially</b>, where each tree corrects the residual errors of the previous ones using <b>gradient descent</b>.</p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>Inputs:</b> Same 7 behavioral features as Random Forest for fair comparison.</li>
            <li><b>Method:</b> Boosting — sequential dependent trees → additive error correction.</li>
            <li><b>Regularization:</b> Built-in L1/L2 regularization prevents overfitting; <code>scale_pos_weight</code> handles class imbalance.</li>
            <li><b>Key Params:</b> learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8.</li>
            <li><b>Advantage:</b> Often achieves higher accuracy, hardware-optimized, handles imbalanced data well.</li>
        </ul>
    </div>

    <div style="background-color: #1e2638; padding: 20px; border-radius: 8px; border-left: 5px solid #66bb6a; margin-bottom: 20px;">
        <h4 style="color:#c8d8f0; margin-top:0;">6. Dual-Model Ensemble Strategy</h4>
        <p style="color:#90a4c4; font-size:14px; margin-bottom:5px;">This system uses <b>both models simultaneously</b> to provide higher-confidence predictions.</p>
        <ul style="color:#90a4c4; font-size:14px;">
            <li><b>Agreement (Both predict same class):</b> High confidence — proceed with automated action.</li>
            <li><b>Disagreement (Models conflict):</b> Flags the case for manual review — the household likely falls in a borderline risk zone.</li>
            <li><b>Bagging vs Boosting:</b> Different learning paradigms capture complementary patterns, reducing overall prediction error.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():
    # Header banner
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0d1f3c,#0a3060,#0d1f3c);
                padding:22px 32px;border-radius:14px;margin-bottom:8px;
                border:1px solid #1e3a5f;">
        <h1 style="color:#4fc3f7;margin:0;font-size:1.9rem;font-weight:800;">
            💧 Smart Water Leakage Detection System
        </h1>
        <p style="color:#90a4c4;margin:6px 0 0;font-size:0.92rem;">
            Big Data Analytics Platform · Smart City Infrastructure Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar filters → returns filtered dataframe
    filtered_df = render_sidebar(df)

    # Navigation
    pages = {
        "📊 System Overview":           page_overview,
        "🔴 Leakage Severity Analysis": page_severity,
        "🏠 Household Risk Intelligence": page_risk,
        "💧 Water Consumption Behavior": page_consumption,
        "⚠️ Abnormal Pattern Detection": page_anomalies,
        "🔬 Household Explorer":         page_explorer,
        "🤖 ML Risk Prediction":         page_ml_prediction,
        "📊 Model Comparison (RF vs XGB)": page_model_comparison,
        "🧠 Smart Insights Panel":       page_insights,
        "🗃️ Data Explorer":             page_data,
        "📖 Methodology & Formulas":     page_methodology,
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📑 Navigation")
    selected_page = st.sidebar.radio("Navigate to", list(pages.keys()), label_visibility="collapsed")

    # Warn if filters remove too much data
    if len(filtered_df) == 0:
        st.error("⚠️ No records match the current filters. Please adjust the sidebar filters.")
        return

    pages[selected_page](filtered_df)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#3a5a80;font-size:0.78rem;">'
        'Smart Water Leakage Detection · Big Data Analytics Platform · '
        'Built with Apache Spark + Streamlit + Plotly</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
