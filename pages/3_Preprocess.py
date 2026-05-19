import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box
from utils.preprocessor import apply_imputer, apply_label_encoding, apply_onehot_encoding, encode_target, apply_scaler

st.set_page_config(page_title="Preprocess · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("03", "Preprocess", "Configure your data pipeline: task type, target, features, imputation, encoding.")

df = st.session_state.get("df")
if df is None:
    st.markdown('<div class="ml-warn">⚠ No dataset loaded. Go to <strong>Upload Data</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

all_cols = df.columns.tolist()

# ── Task type ──────────────────────────────────────────────────────────────────
section("Task type")
task_type = st.radio(
    "task_type", ["Classification", "Regression", "NLP Classification", "Time Series"],
    horizontal=True,
    index=["Classification","Regression","NLP Classification","Time Series"].index(
        st.session_state.get("task_type","Classification")
    ),
    label_visibility="collapsed"
)
st.session_state["task_type"] = task_type

# ── Branch: NLP ────────────────────────────────────────────────────────────────
if task_type == "NLP Classification":
    section("Text column")
    text_candidates = [c for c in all_cols if df[c].dtype == object]
    if not text_candidates:
        st.markdown('<div class="ml-error">❌ No string/object columns found for NLP.</div>', unsafe_allow_html=True)
        st.stop()
    text_col = st.selectbox("Select the column containing raw text", text_candidates,
                            index=text_candidates.index(st.session_state["text_col"]) if st.session_state.get("text_col") in text_candidates else 0)
    st.session_state["text_col"] = text_col

    section("Label column")
    label_candidates = [c for c in all_cols if c != text_col]
    target = st.selectbox("Select the target (label) column", label_candidates,
                          index=label_candidates.index(st.session_state["target"]) if st.session_state.get("target") in label_candidates else 0)
    st.session_state["target"] = target

    section("Preview")
    st.dataframe(df[[text_col, target]].head(8), use_container_width=True)
    vc = df[target].value_counts()
    st.markdown(f'**Class distribution:** ' + "  ·  ".join(f'`{k}` → {v}' for k,v in vc.items()))

    if st.button("✓ Confirm NLP Config", type="primary"):
        # Store minimal processed df for NLP
        nlp_df = df[[text_col, target]].dropna()
        st.session_state["df_processed"] = nlp_df
        st.session_state["features"] = [text_col]
        st.markdown('<div class="ml-success">✓ NLP config saved. Continue to <strong>Train Model</strong>.</div>', unsafe_allow_html=True)
    st.stop()

# ── Branch: Time Series ────────────────────────────────────────────────────────
if task_type == "Time Series":
    section("Date / time column")
    date_candidates = [c for c in all_cols if df[c].dtype == object or "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
    if not date_candidates:
        date_candidates = all_cols
    date_col = st.selectbox("Select the date/time column", date_candidates,
                            index=0 if st.session_state.get("date_col") not in date_candidates else date_candidates.index(st.session_state["date_col"]))
    st.session_state["date_col"] = date_col

    section("Target (value to forecast)")
    ts_candidates = [c for c in all_cols if c != date_col and pd.to_numeric(df[c], errors="coerce").notna().sum() > len(df)*0.5]
    ts_target = st.selectbox("Select the target numeric column", ts_candidates,
                             index=0 if st.session_state.get("ts_target") not in ts_candidates else ts_candidates.index(st.session_state["ts_target"]))
    st.session_state["ts_target"] = ts_target
    st.session_state["target"] = ts_target

    if st.button("✓ Confirm Time Series Config", type="primary"):
        st.session_state["df_processed"] = df.copy()
        st.session_state["features"] = [ts_target]
        st.markdown('<div class="ml-success">✓ Time series config saved. Continue to <strong>Train Model</strong>.</div>', unsafe_allow_html=True)
    st.stop()

# ── Standard tabular flow ──────────────────────────────────────────────────────

# Target
section("Target column")
default_target_idx = all_cols.index(st.session_state["target"]) if st.session_state.get("target") in all_cols else len(all_cols)-1
target = st.selectbox("Select your target (label) column", all_cols, index=default_target_idx, label_visibility="collapsed")
st.session_state["target"] = target

# Drop columns
section("Drop columns")
st.caption("Remove IDs, leaky columns, or anything irrelevant.")
avail_drop = [c for c in all_cols if c != target]
drop_cols = st.multiselect("Columns to drop", avail_drop,
                           default=[d for d in st.session_state.get("drop_cols",[]) if d in avail_drop],
                           label_visibility="collapsed")
st.session_state["drop_cols"] = drop_cols

df_work = df.drop(columns=drop_cols, errors="ignore")
remaining = [c for c in df_work.columns if c != target]

# Features
section("Feature columns")
prev_feat = [f for f in st.session_state.get("features",[]) if f in remaining]
features = st.multiselect("Feature columns (X)", remaining,
                          default=prev_feat or remaining,
                          label_visibility="collapsed")
st.session_state["features"] = features

if not features:
    st.markdown('<div class="ml-error">⚠ Select at least one feature column.</div>', unsafe_allow_html=True)
    st.stop()

# Imputer
section("Missing value imputation")
total_nulls = df_work[features + [target]].isnull().sum().sum()
c1, c2 = st.columns([2,3])
with c1:
    imputer_strategy = st.selectbox(
        "SimpleImputer strategy",
        ["mean", "median", "most_frequent"],
        index=["mean","median","most_frequent"].index(st.session_state.get("imputer_strategy","mean")),
        label_visibility="collapsed"
    )
    st.session_state["imputer_strategy"] = imputer_strategy
with c2:
    st.markdown(f'<div class="ml-info" style="margin-top:0">Missing cells in selection: <strong>{total_nulls}</strong><br><small>Numeric → {imputer_strategy} · Categorical → most_frequent</small></div>', unsafe_allow_html=True)

# Encoding
cat_in_features = df_work[features].select_dtypes(exclude="number").columns.tolist()
section("Categorical encoding")
if cat_in_features:
    st.markdown(" ".join(f'<span class="ml-badge ml-badge-warn">{c}</span>' for c in cat_in_features), unsafe_allow_html=True)
    encode_method = st.radio("Encoding method", ["Label Encoding","One-Hot Encoding"], horizontal=True,
                             index=0 if st.session_state.get("encode_method","Label Encoding")=="Label Encoding" else 1,
                             label_visibility="collapsed")
    st.session_state["encode_method"] = encode_method
else:
    st.markdown('<span class="ml-badge ml-badge-success">✓ No categorical features — no encoding needed</span>', unsafe_allow_html=True)
    st.session_state["encode_method"] = None

# Scaler
section("Feature scaling")
scaler_method = st.radio("Scaler", ["none","standard (z-score)","minmax (0–1)"], horizontal=True,
                         index=["none","standard (z-score)","minmax (0–1)"].index(st.session_state.get("scaler_method","none")),
                         label_visibility="collapsed")
st.session_state["scaler_method"] = scaler_method.split(" ")[0]

# Target info
if task_type == "Classification" and df_work[target].dtype == object:
    uniq = df_work[target].nunique()
    st.markdown(f'<div class="ml-info">🏷 Target <code>{target}</code> is categorical ({uniq} classes) — LabelEncoder will be applied automatically.</div>', unsafe_allow_html=True)

# ── Apply pipeline ─────────────────────────────────────────────────────────────
section("Apply pipeline")
apply_btn = st.button("Apply Preprocessing", type="primary")

if apply_btn:
    with st.spinner("Running pipeline..."):
        X_df = df_work[features].copy()
        y_series = df_work[target].copy()

        # Impute
        X_df = apply_imputer(X_df, strategy=imputer_strategy)
        if y_series.dtype == object:
            y_series = y_series.fillna(y_series.mode().iloc[0] if y_series.notna().any() else "unknown")
        else:
            y_series = y_series.fillna(y_series.median())

        # Encode features
        label_encoders = {}
        cat_present = X_df.select_dtypes(exclude="number").columns.tolist()
        if cat_present:
            if st.session_state["encode_method"] == "Label Encoding":
                X_df, label_encoders = apply_label_encoding(X_df, cat_present)
            else:
                X_df = apply_onehot_encoding(X_df, cat_present)
        st.session_state["label_encoders"] = label_encoders

        # Encode target
        target_encoder = None
        if task_type == "Classification" and y_series.dtype == object:
            y_series, target_encoder = encode_target(y_series)
        st.session_state["target_encoder"] = target_encoder

        # Scale
        from utils.preprocessor import apply_scaler
        X_arr, scaler_obj = apply_scaler(X_df.values, st.session_state["scaler_method"])
        X_df = pd.DataFrame(X_arr, columns=X_df.columns)
        st.session_state["scaler_obj"] = scaler_obj

        # Store
        proc_df = X_df.copy()
        proc_df[target] = y_series.values
        st.session_state["df_processed"] = proc_df
        st.session_state["features"] = X_df.columns.tolist()
        st.session_state["feature_names"] = X_df.columns.tolist()

    rem_nulls = proc_df.isnull().sum().sum()
    st.markdown(f'<div class="ml-success">✓ Done — shape {proc_df.shape[0]} × {proc_df.shape[1]}, {rem_nulls} nulls remaining.</div>', unsafe_allow_html=True)
    st.dataframe(proc_df.head(8), use_container_width=True)
    st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-top:0.5rem;">Continue to <strong>Train Model</strong> →</div>', unsafe_allow_html=True)

elif st.session_state.get("df_processed") is not None:
    st.markdown('<div class="ml-info">Preprocessing already applied. Change settings above and re-apply if needed.</div>', unsafe_allow_html=True)
    st.dataframe(st.session_state["df_processed"].head(5), use_container_width=True)
