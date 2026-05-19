import streamlit as st
import pandas as pd
import numpy as np
import sys, os, copy
import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, PLOTLY_LAYOUT, COLORS
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="DL Time Series · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("4c", "Deep Learning — Time Series", "LSTM, Bidirectional LSTM, and GRU forecasting with Keras/TensorFlow.")

# ── TF availability check ──────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    TF_VERSION = tf.__version__
    TF_OK = True
except ImportError:
    TF_OK = False

if not TF_OK:
    st.markdown('<div class="ml-error">❌ TensorFlow not installed. Run: <code>pip install tensorflow</code></div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<span class="ml-badge ml-badge-success">✓ TensorFlow {TF_VERSION}</span>', unsafe_allow_html=True)

# ── Guards ─────────────────────────────────────────────────────────────────────
task_type = st.session_state.get("task_type","")
df = st.session_state.get("df")
date_col = st.session_state.get("date_col")
ts_target = st.session_state.get("ts_target")

if task_type != "Time Series":
    st.markdown('<div class="ml-info">ℹ Set task type to <strong>Time Series</strong> in <strong>Preprocess</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

if df is None or not date_col or not ts_target:
    st.markdown('<div class="ml-warn">⚠ Configure <strong>Time Series</strong> settings in <strong>Preprocess</strong> first (date column + target column).</div>', unsafe_allow_html=True)
    st.stop()

# Check date col parseable
try:
    df_ts = df[[date_col, ts_target]].copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col], infer_format=True, errors="coerce")
    df_ts = df_ts.dropna(subset=[date_col, ts_target]).sort_values(date_col).reset_index(drop=True)
    n_points = len(df_ts)
except Exception as e:
    st.markdown(f'<div class="ml-error">❌ Cannot parse time series: {e}</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<span class="ml-badge">Target: {ts_target}</span> <span class="ml-badge">Date: {date_col}</span> <span class="ml-badge">{n_points:,} time steps</span>', unsafe_allow_html=True)

# Quick TS preview
with st.expander("📈 Time series preview", expanded=False):
    fig = px.line(df_ts, x=date_col, y=ts_target, color_discrete_sequence=[COLORS["accent"]])
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    layout["height"] = 220
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
section("Architecture")

ARCH_INFO = {
    "LSTM": "Long Short-Term Memory — captures long-range dependencies. Solid general-purpose choice.",
    "BiLSTM": "Bidirectional LSTM — processes sequence forward and backward. Better for pattern detection (not pure autoregression).",
    "GRU": "Gated Recurrent Unit — faster than LSTM with fewer parameters. Good for shorter sequences.",
}

c1, c2 = st.columns([1, 2])
with c1:
    arch = st.radio("Model architecture", list(ARCH_INFO.keys()), label_visibility="collapsed")
with c2:
    st.markdown(f'<div class="ml-info" style="margin-top:0">{ARCH_INFO[arch]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
section("Hyperparameters")

c1, c2, c3 = st.columns(3)
with c1:
    seq_len = st.slider(
        "Sequence length (look-back window)", 3, min(100, n_points//4), min(10, n_points//6),
        help="How many past timesteps the model sees to predict the next one."
    )
    units = st.select_slider("Units per RNN layer", [16, 32, 64, 128, 256], value=64)

with c2:
    n_layers = st.radio("Number of RNN layers", [1, 2, 3], index=1, horizontal=True)
    dropout = st.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05)

with c3:
    learning_rate = st.select_slider(
        "Learning rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )
    epochs = st.slider("Max epochs", 10, 500, 100)
    batch_size = st.select_slider("Batch size", [8, 16, 32, 64, 128], value=32)

c1, c2 = st.columns(2)
with c1:
    test_size = st.slider("Test split", 0.1, 0.4, 0.2, 0.05)
with c2:
    patience = st.slider("Early stopping patience", 3, 50, 10,
                         help="Stop training if val_loss doesn't improve for this many epochs.")

min_required = seq_len * 4
if n_points < min_required:
    st.markdown(f'<div class="ml-warn">⚠ Dataset has {n_points} points but sequence length {seq_len} requires ~{min_required} for a meaningful split. Reduce sequence length or use more data.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🧠 Architecture diagram", expanded=False):
    cell_label = "BiLSTM" if arch == "BiLSTM" else ("LSTM" if arch == "LSTM" else "GRU")
    layer_boxes = []
    for i in range(n_layers):
        rs = "return_sequences=True" if i < n_layers - 1 else "return_sequences=False"
        layer_boxes.append(f"{cell_label}({units}) — {rs}")

    diagram_md = f"""
```
Input: (batch, {seq_len}, 1)  ← {seq_len} timesteps
      ↓
{"".join(chr(10) + "      ↓" + chr(10) + f"[{lb}] + Dropout({dropout})" for lb in layer_boxes)}
      ↓
Dense({max(units//2,16)}, relu) + Dropout({dropout/2:.2f})
      ↓
Dense(1)  ← predicted next value
      ↓
Output: (batch, 1)
```
Optimiser: Adam(lr={learning_rate:.0e}) · Loss: MSE
"""
    st.code(diagram_md.strip(), language=None)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════
section("Train")
st.markdown(f'<div class="ml-info">Training {n_layers}-layer {arch} · {units} units · seq_len={seq_len} · up to {epochs} epochs (early stopping patience={patience}).<br>No GPU detected — training on CPU. This may take 1–5 minutes depending on data size and epochs.</div>', unsafe_allow_html=True)

train_col, _ = st.columns([1, 3])
with train_col:
    train_btn = st.button("Train DL Model", type="primary", use_container_width=True)

if train_btn:
    from utils.ts_deep import train_dl_timeseries

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    with st.spinner("Building and training model..."):
        try:
            results = train_dl_timeseries(
                df_ts=df_ts,
                date_col=date_col,
                target_col=ts_target,
                arch=arch,
                seq_len=seq_len,
                units=units,
                n_layers=n_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                test_size=test_size,
                bidirectional=(arch == "BiLSTM"),
                early_stopping_patience=patience,
            )

            # Store results — tag as DL so Results page can handle it
            results["is_dl"] = True
            results["dl_arch"] = arch
            st.session_state["model_results"] = results
            st.session_state["trained_model"] = results["model"]
            st.session_state["trained_algo"] = f"{arch} (DL)"
            st.session_state["dl_scaler"] = results["scaler"]
            st.session_state["dl_seq_len"] = seq_len

        except Exception as e:
            st.markdown(f'<div class="ml-error">❌ Training failed: {e}</div>', unsafe_allow_html=True)
            st.exception(e)
            st.stop()

    results = st.session_state["model_results"]
    m = results["metrics"]
    st.markdown(f'<div class="ml-success">✓ {arch} trained for {results["epochs_ran"]} epochs (early stopping).</div>', unsafe_allow_html=True)

    # ── Metric cards ───────────────────────────────────────────────────────────
    metric_cols = st.columns(4)
    CLRS = ["#10b981","#f59e0b","#ef4444","#ef4444"]
    for col, (k,v), clr in zip(metric_cols, m.items(), CLRS):
        col.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:{clr}">{v}</div><div class="ml-metric-label">{k}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Training curves ────────────────────────────────────────────────────────
    section("Training curves")
    hist = results["history"]
    ep_range = list(range(1, len(hist["loss"])+1))
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ep_range, y=hist["loss"], name="Train loss", line=dict(color=COLORS["primary"], width=2)))
        if "val_loss" in hist:
            fig.add_trace(go.Scatter(x=ep_range, y=hist["val_loss"], name="Val loss", line=dict(color=COLORS["error"], width=2, dash="dot")))
        layout = copy.deepcopy(PLOTLY_LAYOUT)
        layout.update({"title": "Loss (MSE) per epoch", "xaxis_title": "Epoch", "height": 280})
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        if "mae" in hist:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=ep_range, y=hist["mae"], name="Train MAE", line=dict(color=COLORS["accent"], width=2)))
            if "val_mae" in hist:
                fig2.add_trace(go.Scatter(x=ep_range, y=hist["val_mae"], name="Val MAE", line=dict(color=COLORS["warning"], width=2, dash="dot")))
            layout2 = copy.deepcopy(PLOTLY_LAYOUT)
            layout2.update({"title": "MAE per epoch", "xaxis_title": "Epoch", "height": 280})
            fig2.update_layout(**layout2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    # ── Forecast vs actual ─────────────────────────────────────────────────────
    section("Forecast vs actual (test set)")
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    dates_test = results["dates_test"]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, name="Actual",
                              line=dict(color=COLORS["accent"], width=2)))
    fig3.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, name=f"{arch} Forecast",
                              line=dict(color=COLORS["primary"], width=2, dash="dot")))
    layout3 = copy.deepcopy(PLOTLY_LAYOUT)
    layout3.update({"title": f"{arch} — Test Set Forecast", "xaxis_title": "Step", "yaxis_title": ts_target, "height": 320})
    fig3.update_layout(**layout3)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

    # ── Future forecast ────────────────────────────────────────────────────────
    section("Future forecast")
    n_future = st.slider("Steps to forecast into the future", 1, min(100, n_points//5), 20)

    if st.button("Generate future forecast", key="future_fc"):
        from utils.ts_deep import forecast_future
        values = df_ts[ts_target].values.astype(float)
        last_seq = values[-seq_len:]
        scaler = results["scaler"]
        future_preds = forecast_future(results["model"], last_seq, scaler, n_future, seq_len)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=list(range(len(values))), y=values, name="Historical",
                                  line=dict(color=COLORS["muted"], width=1.5)))
        future_x = list(range(len(values), len(values)+n_future))
        fig4.add_trace(go.Scatter(x=future_x, y=future_preds, name="Future forecast",
                                  line=dict(color=COLORS["primary"], width=2.5, dash="dash"),
                                  marker=dict(size=5)))
        layout4 = copy.deepcopy(PLOTLY_LAYOUT)
        layout4.update({"title": f"{arch} — {n_future}-step future forecast", "height": 350})
        fig4.update_layout(**layout4)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar":False})

        future_df = pd.DataFrame({"Step": range(1, n_future+1), "Forecast": np.round(future_preds, 4)})
        st.dataframe(future_df, use_container_width=True, height=200)
        st.download_button("⬇ Download forecast CSV", future_df.to_csv(index=False).encode(), f"{arch.lower()}_forecast.csv")

    st.markdown('<div class="ml-success" style="margin-top:1rem;">✓ Results also available in the <strong>Results</strong> page. Model saved to session — use <strong>Save Model</strong> to export.</div>', unsafe_allow_html=True)
