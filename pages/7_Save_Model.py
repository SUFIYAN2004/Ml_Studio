import streamlit as st
import pandas as pd
import numpy as np
import sys, os, io, pickle, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section

st.set_page_config(page_title="Save · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("07", "Save Model", "Export your trained model and pipeline for deployment.")

model = st.session_state.get("trained_model")
task_type = st.session_state.get("task_type","Classification")
algo = st.session_state.get("trained_algo","Model")
results = st.session_state.get("model_results")
feature_names = st.session_state.get("feature_names",[])
target = st.session_state.get("target","target")
label_encoders = st.session_state.get("label_encoders",{})
target_encoder = st.session_state.get("target_encoder")
scaler = st.session_state.get("scaler_obj")

if model is None:
    st.markdown('<div class="ml-warn">⚠ No trained model. Complete <strong>Train Model</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<span class="ml-badge ml-badge-accent">🤖 {algo}</span> <span class="ml-badge">{task_type}</span>', unsafe_allow_html=True)

# ── Metrics summary ────────────────────────────────────────────────────────────
if results and results.get("metrics"):
    section("Model performance")
    metrics = results["metrics"]
    COLORS_MAP = {
        "Accuracy":"#10b981","F1 Score":"#6366f1","Precision":"#06b6d4","Recall":"#f59e0b",
        "R² Score":"#10b981","MAE":"#f59e0b","MSE":"#ef4444","RMSE":"#ef4444",
    }
    cols = st.columns(len(metrics))
    for col, (k,v) in zip(cols, metrics.items()):
        clr = COLORS_MAP.get(k,"#6366f1")
        col.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:{clr}">{v}</div><div class="ml-metric-label">{k}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Export options ─────────────────────────────────────────────────────────────
section("Export format")
export_format = st.radio("Format", ["Pickle (.pkl)", "Joblib (.joblib)", "Full Pipeline (.pkl)"],
                         horizontal=True, label_visibility="collapsed")

section("What to save")
save_model = st.checkbox("Model", value=True)
save_encoders = st.checkbox("Label encoders (for categorical features)", value=len(label_encoders) > 0)
save_target_enc = st.checkbox("Target encoder", value=target_encoder is not None)
save_scaler = st.checkbox("Scaler", value=scaler is not None)
save_features = st.checkbox("Feature names list", value=True)
save_metadata = st.checkbox("Training metadata (JSON)", value=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
algo_slug = algo.lower().replace(" ","_").replace("(","").replace(")","").replace("+","_").replace("/","_")
base_name = f"mlstudio_{algo_slug}_{timestamp}"

# ── Serialize and download ─────────────────────────────────────────────────────
section("Download")

def make_bundle():
    bundle = {}
    if save_model:
        bundle["model"] = model
    if save_encoders and label_encoders:
        bundle["label_encoders"] = label_encoders
    if save_target_enc and target_encoder:
        bundle["target_encoder"] = target_encoder
    if save_scaler and scaler:
        bundle["scaler"] = scaler
    if save_features:
        bundle["feature_names"] = feature_names
    bundle["task_type"] = task_type
    bundle["target"] = target
    bundle["algo"] = algo
    bundle["timestamp"] = timestamp
    return bundle


col1, col2, col3 = st.columns(3)

# ── PKL download ───────────────────────────────────────────────────────────────
with col1:
    if st.button("⬇ Download .pkl", type="primary", use_container_width=True):
        bundle = make_bundle()
        buf = io.BytesIO()
        pickle.dump(bundle, buf)
        buf.seek(0)
        st.download_button(
            "Click to save",
            data=buf,
            file_name=f"{base_name}.pkl",
            mime="application/octet-stream",
            key="dl_pkl"
        )

# ── Joblib download ────────────────────────────────────────────────────────────
with col2:
    if st.button("⬇ Download .joblib", use_container_width=True):
        try:
            import joblib
            bundle = make_bundle()
            buf = io.BytesIO()
            joblib.dump(bundle, buf)
            buf.seek(0)
            st.download_button(
                "Click to save",
                data=buf,
                file_name=f"{base_name}.joblib",
                mime="application/octet-stream",
                key="dl_joblib"
            )
        except ImportError:
            st.markdown('<div class="ml-error">joblib not installed: <code>pip install joblib</code></div>', unsafe_allow_html=True)

# ── Metadata JSON ──────────────────────────────────────────────────────────────
with col3:
    if save_metadata and st.button("⬇ Metadata JSON", use_container_width=True):
        meta = {
            "algo": algo,
            "task_type": task_type,
            "target": target,
            "feature_names": feature_names,
            "timestamp": timestamp,
            "metrics": results.get("metrics", {}) if results else {},
            "has_label_encoders": len(label_encoders) > 0,
            "has_target_encoder": target_encoder is not None,
            "has_scaler": scaler is not None,
        }
        st.download_button(
            "Click to save",
            data=json.dumps(meta, indent=2).encode(),
            file_name=f"{base_name}_meta.json",
            mime="application/json",
            key="dl_meta"
        )

# ── Instant download (always available) ───────────────────────────────────────
section("Quick save")
st.markdown('<div class="ml-info">Click below to immediately prepare and download all selected components as a <strong>.pkl bundle</strong>.</div>', unsafe_allow_html=True)

bundle = make_bundle()
buf = io.BytesIO()
pickle.dump(bundle, buf)
buf.seek(0)
st.download_button(
    f"⬇  Save {algo} bundle (.pkl)",
    data=buf,
    file_name=f"{base_name}.pkl",
    mime="application/octet-stream",
    type="primary",
    use_container_width=True,
)

# ── How to load ────────────────────────────────────────────────────────────────
section("How to load this model")
st.code(f"""import pickle
import numpy as np

# Load the bundle
with open("{base_name}.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_names = bundle["feature_names"]
label_encoders = bundle.get("label_encoders", {{}})
target_encoder = bundle.get("target_encoder", None)
scaler = bundle.get("scaler", None)

# Prepare new data (pandas DataFrame with same columns)
import pandas as pd
X_new = pd.DataFrame([{{ feat: value for feat, value in zip(feature_names, your_values) }}])

# Apply label encoding to categorical columns
for col, le in label_encoders.items():
    if col in X_new.columns:
        X_new[col] = le.transform(X_new[col].astype(str))

# Apply scaler if present
if scaler is not None:
    X_arr = scaler.transform(X_new.values)
else:
    X_arr = X_new.values

# Predict
prediction = model.predict(X_arr)

# Decode target label (classification only)
if target_encoder is not None:
    prediction = target_encoder.inverse_transform(prediction.astype(int))

print("Prediction:", prediction)
""", language="python")

# ── NLP pipeline note ──────────────────────────────────────────────────────────
if task_type == "NLP Classification":
    st.markdown('<div class="ml-info">For NLP tasks, the bundle contains a <code>sklearn.pipeline.Pipeline</code> (vectorizer + model). Call <code>model.predict(["your text here"])</code> directly.</div>', unsafe_allow_html=True)
    st.code("""import pickle

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["model"]  # sklearn Pipeline
texts = ["This is a great movie!", "Terrible experience."]
predictions = pipeline.predict(texts)
print(predictions)
""", language="python")
