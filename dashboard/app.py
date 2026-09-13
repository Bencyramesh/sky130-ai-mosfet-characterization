import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import subprocess
import tempfile
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MOSFET AI Characterization",
    layout="wide"
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #808080;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 15px;
        border-radius: 10px;
    }

    div[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD AI MODEL
# =========================================================

model = joblib.load(
    "models/mosfet_advanced_rf.pkl"
)


# =========================================================
# LOAD MODEL METRICS
# =========================================================

with open(
    "models/model_metrics.json",
    "r"
) as f:

    metrics = json.load(f)


# =========================================================
# NGSPICE ACTUAL CURRENT FUNCTION
# =========================================================

def run_ngspice_actual(
    vgs,
    vds,
    temperature,
    width,
    length
):

    with tempfile.TemporaryDirectory() as temp_dir:

        circuit_file = os.path.join(
            temp_dir,
            "mosfet_point.cir"
        )

        output_file = os.path.join(
            temp_dir,
            "mosfet_point.dat"
        )

        spice_code = f"""
* MOSFET Operating Point

.temp {temperature}

Vd drain 0 {vds}
Vg gate 0 {vgs}

M1 drain gate 0 0 NMOS_MODEL
+ W={width}u
+ L={length}u

.model NMOS_MODEL NMOS (
+ LEVEL=1
+ VTO=0.7
+ KP=200u
+ LAMBDA=0.02
+ )

.control

op

set wr_singlescale
set wr_vecnames

wrdata {output_file} i(Vd)

quit

.endc

.end
"""

        # Create temporary ngspice file
        with open(
            circuit_file,
            "w"
        ) as f:

            f.write(spice_code)

        # Run ngspice
        result = subprocess.run(
            [
                "ngspice",
                "-b",
                circuit_file
            ],
            capture_output=True,
            text=True
        )

        # Check simulation
        if result.returncode != 0:

            raise RuntimeError(
                "ngspice simulation failed:\n"
                + result.stderr
            )

        # Read ngspice output
        data = np.loadtxt(
            output_file,
            skiprows=1
        )

        # Last value = current through Vd
        current_A = -float(
            np.ravel(data)[-1]
        )

        # Convert A to mA
        current_mA = (
            current_A * 1000
        )

        return current_mA


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'MOSFET AI Characterization'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI based MOSFET characterization using '
    'ngspice generated semiconductor data'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Device Parameters"
)

st.sidebar.caption(
    "Configure MOSFET operating conditions"
)


# ---------------------------------------------------------
# VGS
# ---------------------------------------------------------

vgs = st.sidebar.slider(
    "Gate Source Voltage VGS (V)",
    min_value=0.0,
    max_value=1.8,
    value=1.4,
    step=0.1
)


# ---------------------------------------------------------
# VDS
# ---------------------------------------------------------

vds = st.sidebar.slider(
    "Drain Source Voltage VDS (V)",
    min_value=0.0,
    max_value=1.8,
    value=1.2,
    step=0.1
)


# ---------------------------------------------------------
# TEMPERATURE
# ---------------------------------------------------------

temperature = st.sidebar.selectbox(
    "Temperature (°C)",
    [
        0,
        27,
        75,
        125
    ],
    index=1
)


# ---------------------------------------------------------
# WIDTH
# ---------------------------------------------------------

width = st.sidebar.selectbox(
    "MOSFET Width W (µm)",
    [
        2.0,
        5.0,
        10.0
    ],
    index=1
)


# ---------------------------------------------------------
# LENGTH
# ---------------------------------------------------------

length = st.sidebar.selectbox(
    "MOSFET Length L (µm)",
    [
        0.18,
        0.5,
        1.0
    ],
    index=1
)


# ---------------------------------------------------------
# W/L
# ---------------------------------------------------------

wl_ratio = (
    width / length
)

st.sidebar.divider()

st.sidebar.metric(
    "W / L Ratio",
    f"{wl_ratio:.2f}"
)


# =========================================================
# AI INPUT DATA
# =========================================================

input_data = pd.DataFrame({

    "VGS_V": [
        vgs
    ],

    "VDS_V": [
        vds
    ],

    "Temperature_C": [
        temperature
    ],

    "Width_um": [
        width
    ],

    "Length_um": [
        length
    ]

})


# =========================================================
# AI PREDICTION
# =========================================================

predicted_id = model.predict(
    input_data
)[0]


# =========================================================
# RUN NGSPICE ACTUAL SIMULATION
# =========================================================

ngspice_error = None

try:

    actual_id = run_ngspice_actual(
        vgs,
        vds,
        temperature,
        width,
        length
    )

except Exception as error:

    actual_id = None

    ngspice_error = str(error)


# =========================================================
# CALCULATE ERROR
# =========================================================

if actual_id is not None:

    absolute_error = abs(
        predicted_id
        - actual_id
    )

    if abs(actual_id) > 1e-6:

        percentage_error = (
            absolute_error
            / abs(actual_id)
        ) * 100

    else:

        percentage_error = None

else:

    absolute_error = None

    percentage_error = None


# =========================================================
# MOSFET OPERATING REGION
# =========================================================

VTH = 0.7

if vgs <= VTH:

    region = "OFF"

elif vds < (
    vgs - VTH
):

    region = "TRIODE / LINEAR"

else:

    region = "SATURATION"


# =========================================================
# DEVICE OPERATING POINT
# =========================================================

st.markdown(
    '<div class="section-title">'
    'Device Operating Point'
    '</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)


c1.metric(
    "VGS",
    f"{vgs:.2f} V"
)

c2.metric(
    "VDS",
    f"{vds:.2f} V"
)

c3.metric(
    "Temperature",
    f"{temperature} °C"
)

c4.metric(
    "Width",
    f"{width:.2f} µm"
)

c5.metric(
    "Length",
    f"{length:.2f} µm"
)


st.divider()


# =========================================================
# AI + NGSPICE COMPARISON
# =========================================================

st.markdown(
    '<div class="section-title">'
    'AI Prediction and ngspice Validation'
    '</div>',
    unsafe_allow_html=True
)


r1, r2, r3 = st.columns(3)


# ---------------------------------------------------------
# AI PREDICTION
# ---------------------------------------------------------

r1.metric(
    "AI Predicted Drain Current",
    f"{predicted_id:.5f} mA"
)


# ---------------------------------------------------------
# NGSPICE ACTUAL
# ---------------------------------------------------------

if actual_id is not None:

    r2.metric(
        "ngspice Actual Drain Current",
        f"{actual_id:.5f} mA"
    )

else:

    r2.metric(
        "ngspice Actual Drain Current",
        "Simulation Error"
    )


# ---------------------------------------------------------
# ABSOLUTE ERROR
# ---------------------------------------------------------

if absolute_error is not None:

    r3.metric(
        "Absolute Error",
        f"{absolute_error:.6f} mA"
    )

else:

    r3.metric(
        "Absolute Error",
        "N/A"
    )


r4, r5, r6 = st.columns(3)


# ---------------------------------------------------------
# PERCENTAGE ERROR
# ---------------------------------------------------------

if percentage_error is not None:

    r4.metric(
        "Percentage Error",
        f"{percentage_error:.2f} %"
    )

else:

    r4.metric(
        "Percentage Error",
        "N/A"
    )


# ---------------------------------------------------------
# REGION
# ---------------------------------------------------------

r5.metric(
    "Operating Region",
    region
)


# ---------------------------------------------------------
# W/L
# ---------------------------------------------------------

r6.metric(
    "W / L Ratio",
    f"{wl_ratio:.2f}"
)


# ---------------------------------------------------------
# NGSPICE ERROR MESSAGE
# ---------------------------------------------------------

if ngspice_error is not None:

    st.error(
        "ngspice simulation failed. "
        "Check the Ubuntu terminal for details."
    )

    with st.expander(
        "Show ngspice error"
    ):

        st.code(
            ngspice_error
        )


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([

    "MOSFET Characteristics",

    "AI Model Performance",

    "System Architecture"

])


# =========================================================
# TAB 1 — MOSFET CHARACTERISTICS
# =========================================================

with tab1:

    st.subheader(
        "AI Predicted MOSFET Characteristics"
    )


    graph1, graph2 = st.columns(2)


    # =====================================================
    # ID-VGS CURVE
    # =====================================================

    vgs_sweep = np.linspace(
        0,
        1.8,
        100
    )


    id_vgs_input = pd.DataFrame({

        "VGS_V":
        vgs_sweep,

        "VDS_V":
        np.full(
            len(vgs_sweep),
            vds
        ),

        "Temperature_C":
        np.full(
            len(vgs_sweep),
            temperature
        ),

        "Width_um":
        np.full(
            len(vgs_sweep),
            width
        ),

        "Length_um":
        np.full(
            len(vgs_sweep),
            length
        )

    })


    id_vgs_prediction = model.predict(
        id_vgs_input
    )


    with graph1:

        fig1, ax1 = plt.subplots(
            figsize=(7, 5)
        )

        ax1.plot(
            vgs_sweep,
            id_vgs_prediction,
            linewidth=2
        )

        ax1.scatter(
            vgs,
            predicted_id,
            s=70,
            zorder=5
        )

        ax1.axvline(
            VTH,
            linestyle="--",
            alpha=0.6
        )

        ax1.set_title(
            "ID vs VGS"
        )

        ax1.set_xlabel(
            "Gate Source Voltage VGS (V)"
        )

        ax1.set_ylabel(
            "Drain Current ID (mA)"
        )

        ax1.grid(
            True,
            alpha=0.3
        )

        st.pyplot(
            fig1
        )

        plt.close(fig1)

        st.caption(
            "Shows how drain current changes "
            "with gate source voltage."
        )


    # =====================================================
    # ID-VDS CURVE
    # =====================================================

    vds_sweep = np.linspace(
        0,
        1.8,
        100
    )


    id_vds_input = pd.DataFrame({

        "VGS_V":
        np.full(
            len(vds_sweep),
            vgs
        ),

        "VDS_V":
        vds_sweep,

        "Temperature_C":
        np.full(
            len(vds_sweep),
            temperature
        ),

        "Width_um":
        np.full(
            len(vds_sweep),
            width
        ),

        "Length_um":
        np.full(
            len(vds_sweep),
            length
        )

    })


    id_vds_prediction = model.predict(
        id_vds_input
    )


    with graph2:

        fig2, ax2 = plt.subplots(
            figsize=(7, 5)
        )

        ax2.plot(
            vds_sweep,
            id_vds_prediction,
            linewidth=2
        )

        ax2.scatter(
            vds,
            predicted_id,
            s=70,
            zorder=5
        )


        saturation_boundary = max(
            vgs - VTH,
            0
        )


        ax2.axvline(
            saturation_boundary,
            linestyle="--",
            alpha=0.6
        )


        ax2.set_title(
            "ID vs VDS"
        )

        ax2.set_xlabel(
            "Drain Source Voltage VDS (V)"
        )

        ax2.set_ylabel(
            "Drain Current ID (mA)"
        )

        ax2.grid(
            True,
            alpha=0.3
        )

        st.pyplot(
            fig2
        )

        plt.close(fig2)

        st.caption(
            "Shows the linear and saturation "
            "behaviour of the MOSFET."
        )


# =========================================================
# TAB 2 — AI MODEL PERFORMANCE
# =========================================================

with tab2:

    st.subheader(
        "Machine Learning Model Performance"
    )


    m1, m2, m3 = st.columns(3)


    m1.metric(
        "Mean Absolute Error",
        f"{metrics['MAE_mA']:.6f} mA"
    )


    m2.metric(
        "Root Mean Square Error",
        f"{metrics['RMSE_mA']:.6f} mA"
    )


    m3.metric(
        "R² Score",
        f"{metrics['R2']:.4f}"
    )


    st.divider()


    st.write(
        """
        The Random Forest regression model was trained
        using MOSFET operating data generated using ngspice.

        **AI Inputs**

        - Gate Source Voltage VGS
        - Drain Source Voltage VDS
        - Temperature
        - MOSFET Width W
        - MOSFET Length L

        **AI Output**

        - Drain Current ID

        The trained AI model acts as a surrogate model
        for predicting MOSFET drain current.
        """
    )


    st.subheader(
        "Validation Method"
    )


    st.write(
        """
        The AI predicted drain current is compared against
        a fresh ngspice simulation at the selected operating
        point.

        The dashboard calculates:

        **Absolute Error**

        | AI Prediction − ngspice Result |

        and

        **Percentage Error**

        Absolute Error / ngspice Result × 100
        """
    )


# =========================================================
# TAB 3 — SYSTEM ARCHITECTURE
# =========================================================

with tab3:

    st.subheader(
        "Project Architecture"
    )


    st.code(
"""
MOSFET PARAMETERS
VGS | VDS | Temperature | W | L
                |
                v
        NGSPICE SIMULATION
                |
                v
      SEMICONDUCTOR DATASET
                |
                v
       DATA PREPROCESSING
                |
                v
      RANDOM FOREST TRAINING
                |
                v
        TRAINED AI MODEL
                |
                v
           AI INFERENCE
                |
                v
       PREDICTED DRAIN CURRENT
                |
                v
      NGSPICE VALIDATION RUN
                |
                v
      ACTUAL DRAIN CURRENT
                |
                v
      ERROR CALCULATION
                |
        +-------+-------+
        |               |
        v               v
 Absolute Error   Percentage Error
"""
    )


    st.info(
        "The current prototype uses an educational "
        "LEVEL 1 MOSFET model. The next development "
        "stage can integrate a realistic BSIM / SKY130 "
        "technology model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Semiconductor Device Characterization | "
    "ngspice + Python + Machine Learning"
)
