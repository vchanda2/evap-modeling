"""
Streamlit app for evaporator temperature and compressor power prediction.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from src.config import DEFAULT_SEQ_LENGTH, DEFAULT_N_STEPS
from src.models.system import RefrigerationSystem
from src.predictors.decision_tree_predictor import DecisionTreePredictor


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Evaporator Modeling",
    layout="wide",
)

st.title("Evaporator Temperature & Compressor Power Predictor")

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.header("Settings")

seq_length = st.sidebar.slider("Look-back window (steps)", 1, 30, DEFAULT_SEQ_LENGTH)
n_steps = st.sidebar.slider("Forecast horizon (steps)", 1, 48, DEFAULT_N_STEPS)
max_depth = st.sidebar.slider("Decision tree max depth", 2, 30, 10)

model_path = Path("models/dt.pkl")

# ------------------------------------------------------------------
# Load system data (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading system data...")
def get_system():
    system = RefrigerationSystem()
    system.load()
    return system


@st.cache_resource(show_spinner="Training model...")
def get_trained_predictor(seq_length, n_steps, max_depth):
    system = get_system()
    predictor = DecisionTreePredictor(
        seq_length=seq_length, n_steps=n_steps, max_depth=max_depth
    )
    predictor.fit(system)
    return predictor


try:
    system = get_system()
except FileNotFoundError as e:
    st.error(
        f"Data files not found: {e}\n\n"
        "Place `evap_raw_data.csv` and `comp_and_other_raw_data.csv` in the `data/` folder."
    )
    st.stop()

# ------------------------------------------------------------------
# Historical data browser
# ------------------------------------------------------------------
st.subheader("Historical Data")

df = system.get_dataframe()

date_min = df.index.min().date()
date_max = df.index.max().date()
date_range = st.date_input(
    "Date range",
    value=(date_max - pd.Timedelta(days=7), date_max),
    min_value=date_min,
    max_value=date_max,
)

if len(date_range) == 2:
    start, end = date_range
    df_view = df.loc[str(start): str(end)]
else:
    df_view = df

evap_options = [f"{e.id}_temp" for e in system.evaporators]
selected_evaps = st.multiselect(
    "Evaporators to display",
    options=evap_options,
    default=evap_options[:3],
)

if selected_evaps:
    fig_hist = go.Figure()
    for col in selected_evaps:
        fig_hist.add_trace(go.Scatter(x=df_view.index, y=df_view[col], name=col, mode="lines"))
    fig_hist.update_layout(
        xaxis_title="Time", yaxis_title="Temperature", height=350, margin=dict(t=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# Compressor power history
fig_power = go.Figure()
fig_power.add_trace(
    go.Scatter(
        x=df_view.index,
        y=df_view["total_compressor_power"],
        name="Compressor Power (kW)",
        line=dict(color="orange"),
    )
)
fig_power.update_layout(
    xaxis_title="Time", yaxis_title="Power (kW)", height=250, margin=dict(t=20)
)
st.plotly_chart(fig_power, use_container_width=True)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
st.subheader("N-Step Forecast")

if st.button("Train & Predict"):
    with st.spinner("Training decision tree..."):
        # Clear cache so re-training picks up new slider values
        get_trained_predictor.clear()
        predictor = get_trained_predictor(seq_length, n_steps, max_depth)

    # Use the last seq_length rows as the look-back window
    input_cols = system.input_feature_cols
    output_cols = system.output_feature_cols
    window = df[input_cols].iloc[-seq_length:].to_numpy(dtype="float32")

    predictions = predictor.predict(window)  # (n_steps, n_out)

    # Build a time index for future steps (assume uniform spacing)
    last_ts = df.index[-1]
    freq = df.index[-1] - df.index[-2]
    future_index = pd.date_range(start=last_ts + freq, periods=n_steps, freq=freq)

    pred_df = pd.DataFrame(predictions, index=future_index, columns=output_cols)

    # Plot evap temps
    evap_pred_cols = [c for c in output_cols if "_temp" in c]
    selected_pred = st.multiselect(
        "Evaporators to forecast",
        options=evap_pred_cols,
        default=evap_pred_cols[:3],
        key="pred_evap_select",
    )

    if selected_pred:
        fig_pred = go.Figure()
        # Historical tail for context
        tail = df[selected_pred].iloc[-min(50, len(df)):]
        for col in selected_pred:
            fig_pred.add_trace(
                go.Scatter(x=tail.index, y=tail[col], name=f"{col} (hist)", mode="lines", opacity=0.5)
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=pred_df.index,
                    y=pred_df[col],
                    name=f"{col} (forecast)",
                    mode="lines+markers",
                    line=dict(dash="dash"),
                )
            )
        fig_pred.update_layout(
            xaxis_title="Time", yaxis_title="Temperature", height=400, margin=dict(t=20)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # Plot compressor power forecast
    fig_pow_pred = go.Figure()
    pow_hist = df["total_compressor_power"].iloc[-min(50, len(df)):]
    fig_pow_pred.add_trace(
        go.Scatter(x=pow_hist.index, y=pow_hist, name="Power (hist)", opacity=0.5, line=dict(color="orange"))
    )
    fig_pow_pred.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df["total_compressor_power"],
            name="Power (forecast)",
            line=dict(color="red", dash="dash"),
            mode="lines+markers",
        )
    )
    fig_pow_pred.update_layout(
        xaxis_title="Time", yaxis_title="Power (kW)", height=300, margin=dict(t=20)
    )
    st.plotly_chart(fig_pow_pred, use_container_width=True)

    st.success(f"Predicted {n_steps} steps ahead using a decision tree (seq_length={seq_length}).")
