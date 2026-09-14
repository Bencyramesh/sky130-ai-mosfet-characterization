import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="SKY130 MOSFET AI Characterization",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM UI
# -------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #8a8a8a;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load(
    "models/sky130_vth_model.pkl"
)

# -------------------------------------------------
# LOAD METRICS
# -------------------------------------------------

with open(
    "models/sky130_vth_metrics.json",
    "r"
) as f:
    metrics = json.load(f)

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------

df = pd.read_csv(
    "data/sky130_pvt_dataset.csv"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'SKY130 MOSFET AI Characterization'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI based MOSFET PVT characterization using '
    'ngspice generated SKY130 semiconductor data'
    '</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("Device Conditions")

corner = st.sidebar.selectbox(
    "Process Corner",
    ["tt", "ff", "ss", "fs", "sf"]
)

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0,
    max_value=125,
    value=25,
    step=25
)

vds = st.sidebar.select_slider(
    "Drain Source Voltage VDS (V)",
    options=[0.6, 0.9, 1.2, 1.5, 1.8],
    value=1.2
)

# -------------------------------------------------
# MODEL INPUT
# -------------------------------------------------

input_data = pd.DataFrame({
    "Corner": [corner],
    "Temperature_C": [temperature],
    "VDS_V": [vds]
})

predicted_vth = model.predict(
    input_data
)[0]

# -------------------------------------------------
# OPERATING POINT
# -------------------------------------------------

st.subheader("Selected Operating Condition")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Process Corner",
    corner.upper()
)

c2.metric(
    "Temperature",
    f"{temperature} °C"
)

c3.metric(
    "VDS",
    f"{vds:.1f} V"
)

st.divider()

# -------------------------------------------------
# AI RESULT
# -------------------------------------------------

st.subheader("AI Prediction")

r1, r2 = st.columns(2)

r1.metric(
    "Predicted Threshold Voltage",
    f"{predicted_vth:.4f} V"
)

r2.metric(
    "Technology",
    "SKY130 NMOS"
)

# -------------------------------------------------
# TABS
# -------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "PVT Characteristics",
    "AI Model Performance",
    "Project Architecture"
])

# =================================================
# TAB 1
# =================================================

with tab1:

    st.subheader(
        "Threshold Voltage vs Temperature"
    )

    fig1, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    for c in df["Corner"].unique():

        data = df[
            (df["Corner"] == c) &
            (df["VDS_V"] == vds)
        ]

        ax1.plot(
            data["Temperature_C"],
            data["Vth_V"],
            marker="o",
            label=c.upper()
        )

    ax1.set_xlabel(
        "Temperature (°C)"
    )

    ax1.set_ylabel(
        "Threshold Voltage VTH (V)"
    )

    ax1.set_title(
        f"SKY130 NMOS VTH vs Temperature at VDS = {vds} V"
    )

    ax1.grid(True)
    ax1.legend()

    st.pyplot(fig1)

    plt.close(fig1)

    st.subheader(
        "Threshold Voltage vs VDS"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    for c in df["Corner"].unique():

        data = df[
            (df["Corner"] == c) &
            (df["Temperature_C"] == temperature)
        ]

        ax2.plot(
            data["VDS_V"],
            data["Vth_V"],
            marker="o",
            label=c.upper()
        )

    ax2.set_xlabel(
        "Drain Source Voltage VDS (V)"
    )

    ax2.set_ylabel(
        "Threshold Voltage VTH (V)"
    )

    ax2.set_title(
        f"SKY130 NMOS VTH vs VDS at {temperature} °C"
    )

    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig2)

    plt.close(fig2)

# =================================================
# TAB 2
# =================================================

with tab2:

    st.subheader(
        "Machine Learning Performance"
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{metrics['MAE_V']:.6f} V"
    )

    m2.metric(
        "RMSE",
        f"{metrics['RMSE_V']:.6f} V"
    )

    m3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )

    st.write(
        """
        The Random Forest model was trained using
        150 SKY130 ngspice simulation points.

        **Inputs**
        - Process corner
        - Temperature
        - Drain Source Voltage

        **Output**
        - Threshold Voltage VTH
        """
    )

# =================================================
# TAB 3
# =================================================

with tab3:

    st.subheader(
        "Project Architecture"
    )

    st.code(
"""
SKY130 PDK
      ↓
SKY130 1.8 V NMOS
      ↓
ngspice Simulation
      ↓
Process Corners
TT / FF / SS / FS / SF
      ↓
Temperature Sweep
      ↓
VDS Sweep
      ↓
Automatic VTH Extraction
      ↓
150 Point PVT Dataset
      ↓
Machine Learning
Random Forest Regressor
      ↓
AI VTH Prediction
      ↓
Streamlit Dashboard
"""
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "SKY130 PDK | ngspice | Python | "
    "Machine Learning | Streamlit"
)
