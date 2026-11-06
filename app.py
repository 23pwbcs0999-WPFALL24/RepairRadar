import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# RepairRadar — clean, responsive, top-navigation Streamlit interface
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_PATH = ARTIFACT_DIR / "dataset_for_app.csv"
HERO_PATH = BASE_DIR / "repairradar_machine_banner.jpg"
BEARING_PATH = BASE_DIR / "repairradar_bearing_detail.jpg"

SAGE = "#788B78"
DUSTY_ROSE = "#B87980"
WARM_CREAM = "#F7F3EA"
DARK_GREEN = "#26352C"
MUTED = "#657568"
PALE_SAGE = "#E7EDE4"
PALE_ROSE = "#F3E3E3"
CARD = "#FFFCF7"
PALETTE = [SAGE, DUSTY_ROSE, "#C99F68", "#91A79D", "#92777B"]

st.set_page_config(
    page_title="RepairRadar | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit's default sidebar and dark top toolbar. Navigation is in our header.
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {
  --rr-cream: #F7F3EA;
  --rr-card: #FFFCF7;
  --rr-sage: #788B78;
  --rr-rose: #B87980;
  --rr-dark: #26352C;
  --rr-muted: #657568;
  --rr-border: #E5DDD1;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--rr-dark) !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background: var(--rr-cream) !important;
  color: var(--rr-dark) !important;
}
/* Remove the built-in navigation sidebar completely. */
section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
  display: none !important;
  width: 0 !important;
}
/* Remove the black Streamlit toolbar strip shown in some Colab previews. */
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], #MainMenu, footer {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1480px !important;
  padding-top: 1.05rem !important;
  padding-bottom: 3rem !important;
  padding-left: clamp(1rem, 3vw, 2.7rem) !important;
  padding-right: clamp(1rem, 3vw, 2.7rem) !important;
}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption,
[data-testid="stCaptionContainer"], label, .stText, .stAlert,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
  color: var(--rr-dark) !important;
}
h1, h2, h3, h4, h5, h6, p, li {
  color: var(--rr-dark) !important;
}
h1, h2, h3 { letter-spacing: -0.035em; }
small, .rr-muted { color: var(--rr-muted) !important; }

/* Top brand and navigation header */
.rr-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1.2rem; padding: .35rem 0 1rem 0;
}
.rr-brand { display: flex; align-items: center; gap: .8rem; min-width: 230px; }
.rr-mark {
  width: 46px; height: 46px; border-radius: 15px;
  display:flex; align-items:center; justify-content:center;
  color: white; font-size: 1.35rem; font-weight: 800;
  background: linear-gradient(145deg, var(--rr-sage), #5F735F);
  box-shadow: 0 7px 16px rgba(65,85,67,.16);
}
.rr-brand-name { font-size: 1.28rem; font-weight: 800; line-height: 1.1; color: var(--rr-dark); }
.rr-brand-tag { font-size: .78rem; color: var(--rr-muted); margin-top: .2rem; }
.rr-header-note {
  padding: .48rem .78rem; border: 1px solid var(--rr-border);
  border-radius: 999px; background: var(--rr-card); color: var(--rr-muted);
  font-size: .78rem; white-space: nowrap;
}
.rr-rule { height: 1px; background: #DED7CB; margin: .15rem 0 1.35rem 0; }
/* Horizontal Streamlit radio is the replacement for the sidebar menu. */
div[data-testid="stRadio"] { margin: 0 !important; }
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] [role="radiogroup"] {
  display: flex !important; flex-direction: row !important; flex-wrap: wrap !important;
  justify-content: flex-end !important; gap: .38rem !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label {
  background: var(--rr-card) !important; border: 1px solid var(--rr-border) !important;
  border-radius: 999px !important; padding: .43rem .78rem !important;
  margin: 0 !important; min-height: 38px !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: var(--rr-dark) !important; border-color: var(--rr-dark) !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
  color: #FFFFFF !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label p {
  font-size: .82rem !important; font-weight: 700 !important; margin: 0 !important;
}
div[data-testid="stRadio"] input { accent-color: var(--rr-rose) !important; }

/* Intro and cards */
.rr-eyebrow { color: var(--rr-sage) !important; font-size: .72rem; font-weight: 800;
  letter-spacing: .14em; text-transform: uppercase; }
.rr-title { color: var(--rr-dark) !important; font-size: clamp(2rem, 3.3vw, 3.15rem);
  line-height: 1.04; font-weight: 800; letter-spacing: -.055em; margin: .25rem 0 .45rem 0; }
.rr-title span { color: var(--rr-rose) !important; }
.rr-subtitle { color: var(--rr-muted) !important; font-size: 1.02rem; line-height: 1.55; }
.rr-hero-note {
  background: linear-gradient(110deg, #E7EDE4 0%, #F7F3EA 56%, #F3E3E3 100%);
  border: 1px solid var(--rr-border); border-radius: 18px;
  padding: 1rem 1.25rem; margin: .8rem 0 1.35rem 0; color: var(--rr-dark) !important;
}
.rr-hero-note b { color: var(--rr-dark) !important; }
.rr-section-title { color: var(--rr-dark) !important; font-size: 1.35rem;
  font-weight: 800; letter-spacing: -.035em; margin: .35rem 0 .9rem 0; }
.rr-section-sub { color: var(--rr-muted) !important; margin-top: -.45rem; margin-bottom: 1rem; }
.rr-card {
  background: var(--rr-card); border: 1px solid var(--rr-border);
  border-radius: 18px; padding: 1rem 1.1rem;
  box-shadow: 0 5px 18px rgba(45,57,46,.035); height: 100%;
}
.rr-card-label { color: var(--rr-muted) !important; font-size: .82rem; font-weight: 600; }
.rr-card-value { color: var(--rr-dark) !important; font-size: 1.9rem; font-weight: 800;
  letter-spacing: -.045em; margin-top: .32rem; }
.rr-card-foot { color: var(--rr-muted) !important; font-size: .75rem; margin-top: .15rem; }
.rr-note {
  background: var(--rr-pale-sage, #E7EDE4); border-left: 4px solid var(--rr-sage);
  border-radius: 10px; padding: .85rem 1rem; color: var(--rr-dark) !important;
}
.rr-risk-high { background: #F3E3E3; border: 1px solid #E6C8CB; }
.rr-risk-medium { background: #F5EBDD; border: 1px solid #E8D6B9; }
.rr-risk-low { background: #E7EDE4; border: 1px solid #D4DED0; }
.rr-risk-value { color: var(--rr-dark) !important; font-size: 2.35rem; font-weight: 800; }

/* Streamlit native widgets */
[data-testid="stMetric"] {
  background: var(--rr-card) !important; border: 1px solid var(--rr-border) !important;
  border-radius: 17px !important; padding: 14px 16px !important;
  box-shadow: 0 4px 14px rgba(45,57,46,.035);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: var(--rr-muted) !important; font-size: .82rem !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
  color: var(--rr-dark) !important; font-weight: 800 !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
  background: var(--rr-card) !important; color: var(--rr-dark) !important;
  border-color: #D9D1C5 !important; border-radius: 10px !important;
}
[data-testid="stNumberInput"] button { color: var(--rr-dark) !important; }
.stButton > button, [data-testid="stFormSubmitButton"] button, .stDownloadButton button {
  background: var(--rr-rose) !important; color: #FFFFFF !important;
  border: 0 !important; border-radius: 11px !important;
  font-weight: 700 !important; min-height: 2.65rem !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
.stDownloadButton button:hover { background: #9F646C !important; color: #FFFFFF !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--rr-border); border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 13px !important; }
hr { border-color: var(--rr-border) !important; }

/* Responsive layout */
@media (max-width: 900px) {
  .rr-header { align-items: flex-start; flex-direction: column; }
  div[data-testid="stRadio"] [role="radiogroup"] { justify-content: flex-start !important; }
  .rr-header-note { display: none; }
}
@media (max-width: 600px) {
  .rr-title { font-size: 2rem; }
  div[data-testid="stRadio"] [role="radiogroup"] label { padding: .35rem .58rem !important; }
}

/* Final layout hardening: no sidebar, no dark toolbar, no leftover top gap. */
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarContent"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: none !important;
  visibility: hidden !important;
  width: 0 !important;
  min-width: 0 !important;
}
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  margin-left: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
}
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
#MainMenu, footer {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  background: var(--rr-cream) !important;
}
body, html, #root, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--rr-cream) !important;
  color: var(--rr-dark) !important;
}
[data-testid="stMainBlockContainer"] {
  padding-top: .8rem !important;
}
/* Header navigation looks like tabs rather than a vertical radio list. */
div[data-testid="stRadio"] [role="radiogroup"] label {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 40px !important;
  padding: .48rem .88rem !important;
  border-radius: 999px !important;
  background: #FFFCF7 !important;
  border: 1px solid #E5DDD1 !important;
  color: #26352C !important;
}
div[data-testid="stRadio"] [role="radiogroup"] input[type="radio"] {
  display: none !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label p,
div[data-testid="stRadio"] [role="radiogroup"] label span {
  color: #26352C !important;
  font-weight: 700 !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: #26352C !important;
  border-color: #26352C !important;
}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span {
  color: #FFFFFF !important;
}
.rr-header-shell {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; width: 100%; padding: .25rem 0 .8rem 0;
}
.rr-status {
  display: inline-flex; align-items: center; gap: .42rem;
  padding: .48rem .78rem; border-radius: 999px;
  background: #E7EDE4; color: #405844 !important;
  border: 1px solid #D4DED0; font-size: .77rem; font-weight: 700;
  white-space: nowrap;
}
.rr-status-dot {
  display: inline-block; width: .5rem; height: .5rem;
  background: #64866B; border-radius: 50%;
}
.rr-intro-panel {
  display: flex; align-items: center; justify-content: space-between;
  gap: 1.5rem; padding: .9rem 0 .5rem 0;
}
.rr-intro-copy { flex: 1 1 55%; min-width: 0; }
.rr-image-frame {
  overflow: hidden; border-radius: 18px; border: 1px solid #E5DDD1;
  box-shadow: 0 8px 24px rgba(45,57,46,.07);
}
.rr-image-frame img { display: block; width: 100%; height: auto; object-fit: cover; }
[data-testid="stPlotlyChart"] { background: transparent !important; }
[data-testid="stPlotlyChart"] iframe { background: transparent !important; }
@media (max-width: 900px) {
  .rr-header-shell, .rr-intro-panel { display: block; }
  .rr-status { margin-top: .75rem; }
  .rr-intro-copy { margin-bottom: .7rem; }
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    failure = joblib.load(ARTIFACT_DIR / "failure_model.joblib")
    fault = joblib.load(ARTIFACT_DIR / "fault_model.joblib")
    with open(ARTIFACT_DIR / "metadata.json", "r", encoding="utf-8") as file:
        meta = json.load(file)
    return failure, fault, meta


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def model_classes(model):
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    return list(estimator.classes_)


def style_figure(fig, height=350):
    """Apply readable text, axis, grid, and background colors to every chart."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="DM Sans, sans-serif", size=12, color=DARK_GREEN),
        title_font=dict(family="DM Sans, sans-serif", size=17, color=DARK_GREEN),
        legend=dict(font=dict(color=DARK_GREEN), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD,
        margin=dict(l=18, r=18, t=58, b=18),
        hoverlabel=dict(bgcolor=CARD, font=dict(color=DARK_GREEN)),
    )
    fig.update_xaxes(
        color=DARK_GREEN,
        title_font=dict(color=DARK_GREEN),
        tickfont=dict(color=MUTED),
        gridcolor="#ECE5DA",
        linecolor="#D9D1C5",
        zerolinecolor="#E6DED2",
    )
    fig.update_yaxes(
        color=DARK_GREEN,
        title_font=dict(color=DARK_GREEN),
        tickfont=dict(color=MUTED),
        gridcolor="#ECE5DA",
        linecolor="#D9D1C5",
        zerolinecolor="#E6DED2",
    )
    return fig


def card(label, value, foot=""):
    st.markdown(
        f"<div class='rr-card'><div class='rr-card-label'>{label}</div>"
        f"<div class='rr-card-value'>{value}</div>"
        f"<div class='rr-card-foot'>{foot}</div></div>",
        unsafe_allow_html=True,
    )


try:
    failure_model, fault_model, metadata = load_assets()
    df = load_data()
except Exception as error:
    st.error("RepairRadar could not load its saved project files.")
    st.info("Run the notebook's model-training and save section before launching the app.")
    st.exception(error)
    st.stop()

features = metadata["features"]

# -----------------------------------------------------------------------------
# Main top navigation — sidebar is intentionally not used.
# -----------------------------------------------------------------------------
brand_col, nav_col, status_col = st.columns([1.0, 3.1, 0.85], vertical_alignment="center")
with brand_col:
    st.markdown(
        """
        <div class="rr-brand">
          <div class="rr-mark">⚙</div>
          <div>
            <div class="rr-brand-name">RepairRadar</div>
            <div class="rr-brand-tag">Predict · Prevent · Perform</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with nav_col:
    page = st.radio(
        "Main navigation",
        ["Overview", "Predict", "Dataset", "Model results", "About"],
        horizontal=True,
        label_visibility="collapsed",
        key="repairradar_top_navigation_v2",
    )
with status_col:
    st.markdown(
        '<div class="rr-status"><span class="rr-status-dot"></span>System ready</div>',
        unsafe_allow_html=True,
    )

st.markdown("<div class='rr-rule'></div>", unsafe_allow_html=True)

page_intro = {
    "Overview": ("Intelligent machine care", "Welcome to <span>RepairRadar</span>", "Understand machine conditions. Predict possible faults. Plan maintenance earlier."),
    "Predict": ("Predictive assessment", "Check machine <span>health</span>", "Enter current sensor readings to estimate short-term failure risk and identify possible fault types."),
    "Dataset": ("Synthetic data explorer", "Explore the <span>dataset</span>", "Inspect machine observations, compare fault categories, and understand the simulated sensor patterns."),
    "Model results": ("Model evaluation", "Measure model <span>performance</span>", "Review validation comparisons and results from machines held out during training."),
    "About": ("Project information", "About <span>RepairRadar</span>", "A student-built predictive-maintenance prototype using an original synthetic industrial dataset."),
}
eyebrow, page_title, page_subtitle = page_intro[page]
intro_col, image_col = st.columns([1.35, 1], gap="large", vertical_alignment="center")
with intro_col:
    st.markdown(f"<div class='rr-eyebrow'>{eyebrow}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='rr-title'>{page_title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='rr-subtitle'>{page_subtitle}</div>", unsafe_allow_html=True)
with image_col:
    if HERO_PATH.exists():
        st.image(str(HERO_PATH), use_container_width=True)

st.markdown(
    """
    <div class="rr-hero-note">
      <b>Keep your machines healthy.</b><br>
      RepairRadar analyzes operating conditions and sensor readings to demonstrate
      early machine-failure prediction and support proactive maintenance planning.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown("<div class='rr-section-title'>System overview</div>", unsafe_allow_html=True)
    machine_count = int(df["machine_id"].nunique()) if "machine_id" in df.columns else 0
    row_count = len(df)
    failure_rate = float(pd.to_numeric(df["failure_within_7_days"], errors="coerce").mean())
    fault_count = int(df["fault_type"].nunique())

    metric_cols = st.columns(4)
    with metric_cols[0]:
        card("Machines represented", f"{machine_count:,}", "Unique simulated machines")
    with metric_cols[1]:
        card("Sensor observations", f"{row_count:,}", "Synthetic operating records")
    with metric_cols[2]:
        card("7-day failure labels", f"{failure_rate:.1%}", "Share of positive labels")
    with metric_cols[3]:
        card("Fault categories", str(fault_count), "Machine condition classes")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        fault_counts = (
            df["fault_type"].value_counts().rename_axis("Fault type")
            .reset_index(name="Observations")
        )
        fig = px.bar(
            fault_counts, x="Fault type", y="Observations", color="Fault type",
            color_discrete_sequence=PALETTE, title="Machine condition distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig, 370), use_container_width=True, theme=None)
    with right:
        fail_counts = (
            df["failure_within_7_days"].value_counts().sort_index()
            .rename(index={0: "No simulated failure", 1: "Failure within 7 days"})
            .rename_axis("Outcome").reset_index(name="Observations")
        )
        fig = px.pie(
            fail_counts, names="Outcome", values="Observations", hole=.62,
            color_discrete_sequence=[SAGE, DUSTY_ROSE], title="Seven-day failure labels",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont=dict(color=DARK_GREEN))
        st.plotly_chart(style_figure(fig, 370), use_container_width=True, theme=None)

    left, right = st.columns(2, gap="large")
    with left:
        fig = px.box(
            df, x="fault_type", y="vibration_rms_mm_s", color="fault_type",
            color_discrete_sequence=PALETTE, title="Vibration by machine condition",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig, 360), use_container_width=True, theme=None)
    with right:
        fig = px.scatter(
            df.sample(min(2500, len(df)), random_state=42),
            x="operating_temperature_c", y="motor_current_a", color="fault_type",
            color_discrete_sequence=PALETTE, opacity=.72,
            title="Operating temperature vs motor current",
        )
        st.plotly_chart(style_figure(fig, 360), use_container_width=True, theme=None)

    st.markdown(
        "<div class='rr-note'>All figures describe the student-generated synthetic dataset. "
        "They are not live readings from real industrial machines.</div>",
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# PREDICT
# -----------------------------------------------------------------------------
elif page == "Predict":
    st.markdown("<div class='rr-section-title'>Machine health assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rr-section-sub'>Enter a machine's current sensor readings. "
        "The trained models will estimate short-term failure risk and classify its condition.</div>",
        unsafe_allow_html=True,
    )

    defaults = {
        "vibration_rms_mm_s": 4.5,
        "operating_temperature_c": 75.0,
        "motor_current_a": 12.0,
        "rotational_speed_rpm": 1500.0,
        "machine_load_pct": 80.0,
        "operating_hours": 4500.0,
        "machine_age_years": 4.0,
        "hours_since_maintenance": 60.0,
        "lubrication_quality": 3.0,
    }
    labels = {
        "vibration_rms_mm_s": "Vibration RMS (mm/s)",
        "operating_temperature_c": "Operating temperature (°C)",
        "motor_current_a": "Motor current (A)",
        "rotational_speed_rpm": "Rotational speed (RPM)",
        "machine_load_pct": "Machine load (%)",
        "operating_hours": "Total operating hours",
        "machine_age_years": "Machine age (years)",
        "hours_since_maintenance": "Hours since maintenance",
        "lubrication_quality": "Lubrication quality (1–5)",
    }

    form_col, info_col = st.columns([1.45, 1], gap="large")
    with form_col:
        st.markdown("<div class='rr-card-label'>MACHINE SENSOR INPUTS</div>", unsafe_allow_html=True)
        with st.form("machine_input_form"):
            input_columns = st.columns(3)
            input_values = {}
            for index, feature in enumerate(features):
                with input_columns[index % 3]:
                    median_value = float(pd.to_numeric(df[feature], errors="coerce").median())
                    initial_value = defaults.get(feature, median_value)
                    input_values[feature] = st.number_input(
                        labels.get(feature, feature.replace("_", " ").title()),
                        value=float(initial_value), format="%.2f", key=f"input_{feature}",
                    )
            submitted = st.form_submit_button("Analyze machine health", use_container_width=True)
    with info_col:
        if BEARING_PATH.exists():
            st.image(str(BEARING_PATH), use_container_width=True)
        st.markdown(
            "<div class='rr-note'><b>How to use</b><br>Enter the latest available machine readings, "
            "then select <b>Analyze machine health</b>. The displayed predictions come from your "
            "trained models.</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        input_frame = pd.DataFrame([[input_values[column] for column in features]], columns=features)
        failure_prediction = failure_model.predict(input_frame)[0]
        fault_prediction = fault_model.predict(input_frame)[0]

        failure_probability = None
        if hasattr(failure_model, "predict_proba"):
            probabilities = failure_model.predict_proba(input_frame)[0]
            failure_classes = model_classes(failure_model)
            positive_index = next(
                (i for i, value in enumerate(failure_classes)
                 if value == 1 or str(value).strip().lower() in {"1", "true", "yes"}),
                None,
            )
            if positive_index is not None:
                failure_probability = float(probabilities[positive_index])
        if failure_probability is None:
            failure_probability = float(
                str(failure_prediction).strip().lower() in {"1", "true", "yes"}
            )

        fault_probabilities = None
        fault_classes = model_classes(fault_model)
        fault_confidence = None
        if hasattr(fault_model, "predict_proba"):
            fault_probabilities = fault_model.predict_proba(input_frame)[0]
            predicted_index = fault_classes.index(fault_prediction)
            fault_confidence = float(fault_probabilities[predicted_index])

        if failure_probability >= 0.70:
            priority, risk_class = "High", "rr-risk-high"
        elif failure_probability >= 0.35:
            priority, risk_class = "Medium", "rr-risk-medium"
        else:
            priority, risk_class = "Low", "rr-risk-low"

        st.markdown("<div class='rr-section-title'>Prediction results</div>", unsafe_allow_html=True)
        result_left, result_right = st.columns([1.05, 1], gap="large")
        with result_left:
            st.markdown(
                f"<div class='rr-card {risk_class}'>"
                f"<div class='rr-card-label'>PREDICTED MAINTENANCE PRIORITY</div>"
                f"<div class='rr-card-value'>{priority} priority</div>"
                f"<div class='rr-card-foot'>Estimated failure probability within the next seven days</div>"
                f"<div class='rr-risk-value'>{failure_probability:.1%}</div></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(failure_probability, 0.0), 1.0))
            st.metric(
                "Predicted fault type", str(fault_prediction),
                None if fault_confidence is None else f"{fault_confidence:.1%} model confidence",
            )
        with result_right:
            if fault_probabilities is not None:
                fault_chart = pd.DataFrame({
                    "Fault type": [str(value) for value in fault_classes],
                    "Model probability": fault_probabilities,
                })
                fig = px.bar(
                    fault_chart, x="Fault type", y="Model probability", color="Fault type",
                    color_discrete_sequence=PALETTE, title="Fault-type model scores",
                )
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_figure(fig, 330), use_container_width=True, theme=None)

        st.markdown(
            "<div class='rr-note'>Priority levels are illustrative display bands for this educational prototype. "
            "They are not certified industrial safety thresholds.</div>",
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
elif page == "Dataset":
    st.markdown("<div class='rr-section-title'>Explore the synthetic dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rr-section-sub'>Filter machine observations, inspect sensor values, "
        "and compare operating conditions across fault categories.</div>",
        unsafe_allow_html=True,
    )
    fault_options = sorted(df["fault_type"].dropna().unique().tolist())
    selected_faults = st.multiselect("Filter by fault type", fault_options, default=fault_options)
    filtered = df[df["fault_type"].isin(selected_faults)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} observations")
    st.dataframe(filtered.head(500), use_container_width=True, height=340)

    if not filtered.empty:
        left, right = st.columns(2, gap="large")
        with left:
            fig = px.histogram(
                filtered, x="vibration_rms_mm_s", color="fault_type", barmode="overlay",
                opacity=.72, color_discrete_sequence=PALETTE, title="Vibration distribution",
            )
            st.plotly_chart(style_figure(fig, 350), use_container_width=True, theme=None)
        with right:
            fig = px.scatter(
                filtered.sample(min(2500, len(filtered)), random_state=42),
                x="operating_temperature_c", y="motor_current_a", color="fault_type",
                color_discrete_sequence=PALETTE, title="Temperature vs motor current",
            )
            st.plotly_chart(style_figure(fig, 350), use_container_width=True, theme=None)
    else:
        st.info("No observations match the selected fault filters.")

# -----------------------------------------------------------------------------
# MODEL RESULTS
# -----------------------------------------------------------------------------
elif page == "Model results":
    st.markdown("<div class='rr-section-title'>Model performance</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rr-section-sub'>Compare the selected classifiers using the held-out test machines.</div>",
        unsafe_allow_html=True,
    )
    task_choice = st.selectbox("Prediction task", ["Failure within 7 days", "Fault type"])
    result = metadata["failure_result"] if task_choice == "Failure within 7 days" else metadata["fault_result"]
    metrics = result["test_metrics"]
    st.markdown(f"**Selected algorithm:** {result['best_model']} &nbsp; · &nbsp; **Test set:** unseen machines")

    metric_cols = st.columns(4)
    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision_weighted"),
        ("Recall", "recall_weighted"),
        ("F1 score", "f1_weighted"),
    ]
    for col, (label, key) in zip(metric_cols, metric_labels):
        with col:
            card(label, f"{metrics[key]:.1%}", "Held-out test performance")

    st.markdown("<br>", unsafe_allow_html=True)
    comparison = pd.DataFrame(result["validation_comparison"])
    st.markdown("#### Validation model comparison")
    st.dataframe(
        comparison.style.format({
            "accuracy": "{:.3f}", "precision_weighted": "{:.3f}",
            "recall_weighted": "{:.3f}", "f1_weighted": "{:.3f}", "f1_macro": "{:.3f}",
        }),
        use_container_width=True, hide_index=True,
    )

    cm = np.array(result["confusion_matrix"])
    class_labels = result["classes"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=class_labels, y=class_labels,
        colorscale=[[0, WARM_CREAM], [.5, "#B9C8B7"], [1, SAGE]],
        text=cm, texttemplate="%{text}",
        textfont=dict(color=DARK_GREEN),
        colorbar=dict(tickfont=dict(color=DARK_GREEN)),
    ))
    fig.update_layout(title="Test-set confusion matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(style_figure(fig, 430), use_container_width=True, theme=None)
    st.caption("Validation results select the model. The displayed metrics and confusion matrix use the held-out test set.")

# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
else:
    st.markdown("<div class='rr-section-title'>About RepairRadar</div>", unsafe_allow_html=True)
    st.markdown(
        "RepairRadar is a student predictive-maintenance prototype. It analyzes machine operating "
        "conditions and sensor readings to estimate simulated failure risk within seven days and "
        "classify the current machine condition."
    )
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### Project workflow")
        st.markdown(
            "1. Generate an original synthetic industrial-machine dataset.\n"
            "2. Explore and prepare sensor data.\n"
            "3. Compare classification algorithms.\n"
            "4. Evaluate using held-out machines.\n"
            "5. Use trained models in this Streamlit application."
        )
        st.markdown("#### Fault categories")
        st.write("Normal · Bearing Wear · Misalignment · Overheating · Imbalance")
    with right:
        if BEARING_PATH.exists():
            st.image(str(BEARING_PATH), use_container_width=True)
    st.warning(
        "The models were trained on student-generated synthetic data. This prototype has not been "
        "validated for real factories or safety-critical maintenance decisions. Real deployment "
        "would require real sensor data, engineering validation, monitoring, and domain-expert review."
    )

st.markdown("<div class='rr-rule'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#657568;font-size:.78rem;'>"
    "RepairRadar · Educational predictive-maintenance prototype · Synthetic data only</div>",
    unsafe_allow_html=True,
)
