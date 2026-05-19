import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box
from utils.model_trainer import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS, NLP_MODELS,
    train_tabular, train_nlp, train_timeseries,
    XGBOOST_AVAILABLE, CATBOOST_AVAILABLE
)

try:
    LGBM_AVAILABLE = "LightGBM" in REGRESSION_MODELS
except:
    LGBM_AVAILABLE = False

st.set_page_config(page_title="Train · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("04", "Train Model", "Select algorithm, configure hyperparameters, and train.")

task_type = st.session_state.get("task_type", "Classification")
df_processed = st.session_state.get("df_processed")
target = st.session_state.get("target")
features = st.session_state.get("features", [])

if df_processed is None:
    st.markdown('<div class="ml-warn">⚠ Complete <strong>Preprocess</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f"""
<div class="ml-card" style="display:flex; gap:2rem; align-items:center; flex-wrap:wrap;">
    <span class="ml-badge ml-badge-accent">Task: {task_type}</span>
    <span class="ml-badge">Target: {target or "—"}</span>
    <span class="ml-badge">Features: {len(features)}</span>
    <span class="ml-badge">Samples: {len(df_processed):,}</span>
</div>
""", unsafe_allow_html=True)

# ── NLP flow ───────────────────────────────────────────────────────────────────
if task_type == "NLP Classification":
    df = st.session_state.get("df")
    text_col = st.session_state.get("text_col")
    if df is None or text_col is None:
        st.markdown('<div class="ml-error">❌ Configure NLP settings in Preprocess first.</div>', unsafe_allow_html=True)
        st.stop()

    texts = df[text_col].fillna("").astype(str).tolist()
    raw_labels = df[target].fillna("unknown").astype(str).tolist()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(raw_labels)
    st.session_state["target_encoder"] = le

    section("Vectorizer")
    c1, c2, c3 = st.columns(3)
    with c1:
        vec_type = st.selectbox("Vectorizer", ["TF-IDF","Bag of Words (Count)"])
    with c2:
        ngram_max = st.selectbox("N-gram max", [1, 2, 3], index=1)
    with c3:
        max_features = st.number_input("Max vocabulary size", 1000, 100000, 10000, step=1000)

    section("Algorithm")
    algo_names = list(NLP_MODELS.keys())
    if not XGBOOST_AVAILABLE and "XGBoost" in algo_names:
        algo_names.remove("XGBoost")
    algo = st.selectbox("Model", algo_names, label_visibility="collapsed")

    section("Hyperparameters")
    hp = {}
    c1, c2 = st.columns(2)
    with c1:
        if algo == "Logistic Regression":
            hp["C"] = st.number_input("C (inverse regularization)", 0.001, 1000.0, 1.0)
            hp["max_iter"] = st.number_input("max_iter", 100, 5000, 1000, step=100)
        elif algo.startswith("Naive Bayes"):
            hp["alpha"] = st.number_input("alpha (smoothing)", 0.0, 10.0, 1.0, step=0.1)
        elif algo == "SVM (Linear)":
            hp["C"] = st.number_input("C", 0.001, 100.0, 1.0)
        elif algo in ("Random Forest","Gradient Boosting","XGBoost"):
            hp["n_estimators"] = st.number_input("n_estimators", 10, 500, 100, step=10)
    with c2:
        if algo in ("Random Forest","Gradient Boosting","XGBoost"):
            hp["max_depth"] = st.number_input("max_depth", 1, 20, 5)

    section("Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train NLP Model", type="primary"):
        with st.spinner(f"Training {algo} with {vec_type}..."):
            try:
                results = train_nlp(texts, labels, algo, vec_type, hp, test_size, ngram_max=ngram_max, max_features=int(max_features))
                results["label_encoder"] = le
                results["label_names"] = le.classes_
                st.session_state["model_results"] = results
                st.session_state["trained_model"] = results["pipeline"]
                st.session_state["trained_algo"] = f"{algo} + {vec_type}"
                st.markdown('<div class="ml-success">✓ NLP model trained! Go to <strong>Results</strong>.</div>', unsafe_allow_html=True)
                m = results["metrics"]
                cols = st.columns(4)
                for col, (k,v) in zip(cols, m.items()):
                    col.metric(k, v)
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ {e}</div>', unsafe_allow_html=True)
                st.exception(e)
    st.stop()

# ── Time Series flow ───────────────────────────────────────────────────────────
if task_type == "Time Series":
    df = st.session_state.get("df")
    date_col = st.session_state.get("date_col")
    ts_target = st.session_state.get("ts_target")
    if df is None or not date_col or not ts_target:
        st.markdown('<div class="ml-error">❌ Configure Time Series settings in Preprocess first.</div>', unsafe_allow_html=True)
        st.stop()

    section("Lag features")
    c1, c2 = st.columns(2)
    with c1:
        lag_n = st.slider("Number of lag features", 2, 30, 7)
    with c2:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    TS_MODELS = {k: v for k, v in REGRESSION_MODELS.items()
                 if k in ("Random Forest","Gradient Boosting","Linear Regression","Ridge Regression","XGBoost","LightGBM","Extra Trees")}
    section("Algorithm")
    ts_algo_names = list(TS_MODELS.keys())
    if not XGBOOST_AVAILABLE and "XGBoost" in ts_algo_names:
        ts_algo_names.remove("XGBoost")
    algo = st.selectbox("Model", ts_algo_names, label_visibility="collapsed")

    section("Hyperparameters")
    hp = {}
    c1, c2 = st.columns(2)
    with c1:
        if algo in ("Random Forest","Gradient Boosting","XGBoost","LightGBM","Extra Trees"):
            hp["n_estimators"] = st.number_input("n_estimators", 10, 1000, 100, step=10)
    with c2:
        if algo in ("Random Forest","Gradient Boosting","XGBoost","LightGBM","Extra Trees"):
            use_max = st.checkbox("Limit max_depth", value=True)
            if use_max:
                hp["max_depth"] = st.number_input("max_depth", 1, 20, 5)

    if st.button("Train Time Series Model", type="primary"):
        with st.spinner(f"Engineering lag features and training {algo}..."):
            try:
                results = train_timeseries(df, date_col, ts_target, algo, hp, lag_n, test_size)
                st.session_state["model_results"] = results
                st.session_state["trained_model"] = results["model"]
                st.session_state["trained_algo"] = algo
                st.session_state["feature_names"] = results["feature_cols"]
                st.markdown('<div class="ml-success">✓ Time series model trained! Go to <strong>Results</strong>.</div>', unsafe_allow_html=True)
                m = results["metrics"]
                cols = st.columns(4)
                for col, (k,v) in zip(cols, m.items()):
                    col.metric(k, v)
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ {e}</div>', unsafe_allow_html=True)
                st.exception(e)
    st.stop()

# ── Tabular (Regression / Classification) ─────────────────────────────────────
valid_features = [f for f in features if f in df_processed.columns]
if not valid_features or target not in df_processed.columns:
    st.markdown('<div class="ml-error">❌ Column mismatch — please re-run Preprocess.</div>', unsafe_allow_html=True)
    st.stop()

X = df_processed[valid_features].values
y = df_processed[target].values

registry = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
algo_names = list(registry.keys())
for lib, flag in [("XGBoost", XGBOOST_AVAILABLE), ("CatBoost", CATBOOST_AVAILABLE)]:
    if not flag and lib in algo_names:
        algo_names.remove(lib)
        st.markdown(f'<span class="ml-badge ml-badge-warn">⚠ {lib} not installed</span>', unsafe_allow_html=True)

ALGO_DESC = {
    "Linear Regression": "Fast linear baseline. Interpretable coefficients.",
    "Ridge Regression": "L2 regularization — reduces overfitting.",
    "Lasso Regression": "L1 regularization — automatic feature selection.",
    "ElasticNet": "Combines L1 + L2 regularization.",
    "Logistic Regression": "Linear classifier with probabilistic output.",
    "Decision Tree": "Interpretable tree-based model.",
    "Random Forest": "Robust ensemble — great default choice.",
    "Extra Trees": "Randomized trees — often faster than RF.",
    "Gradient Boosting": "Sequential boosting — strong performance.",
    "XGBoost": "State-of-the-art gradient boosting.",
    "CatBoost": "Native categorical support, minimal tuning.",
    "LightGBM": "Fast gradient boosting — large datasets.",
    "SVM (SVR)": "Margin-based regression.",
    "SVM (SVC)": "Powerful margin-based classifier.",
    "KNN": "Prediction from nearest neighbors.",
    "Neural Network (MLP)": "Multi-layer perceptron — flexible.",
    "Naive Bayes (Multinomial)": "Fast probabilistic classifier.",
    "Naive Bayes (Complement)": "Complement NB — imbalanced data.",
}

section("Algorithm")
algo = st.selectbox("Algorithm", algo_names, label_visibility="collapsed")
if algo in ALGO_DESC:
    st.markdown(f'<div style="font-size:0.8rem;color:#6b7280;margin-top:-0.5rem;margin-bottom:0.75rem;">{ALGO_DESC[algo]}</div>', unsafe_allow_html=True)

section("Hyperparameters")
hp = {}
c1, c2, c3 = st.columns(3)

with c1:
    if algo in ("Random Forest","Gradient Boosting","XGBoost","CatBoost","LightGBM","Extra Trees"):
        hp["n_estimators"] = st.number_input("n_estimators", 10, 2000, 100, step=10)
    if algo in ("Ridge Regression","Lasso Regression","ElasticNet"):
        hp["alpha"] = st.number_input("alpha", 0.0001, 1000.0, 1.0, format="%.4f")
    if algo == "Logistic Regression":
        hp["C"] = st.number_input("C", 0.001, 1000.0, 1.0)
        hp["max_iter"] = st.number_input("max_iter", 100, 5000, 1000, step=100)
    if algo == "KNN":
        hp["n_neighbors"] = st.number_input("n_neighbors", 1, 100, 5)

with c2:
    if algo in ("Decision Tree","Random Forest","Gradient Boosting","XGBoost","CatBoost","LightGBM","Extra Trees"):
        unlim = st.checkbox("max_depth = unlimited", value=(algo in ("Random Forest","Extra Trees")))
        if not unlim:
            hp["max_depth"] = st.number_input("max_depth", 1, 50, 5)
    if algo in ("SVM (SVR)","SVM (SVC)"):
        hp["C"] = st.number_input("C", 0.01, 1000.0, 1.0)
        hp["kernel"] = st.selectbox("kernel", ["rbf","linear","poly","sigmoid"])
    if algo == "Neural Network (MLP)":
        hl = st.text_input("hidden_layer_sizes", "128,64")
        try:
            hp["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hl.split(","))
        except:
            hp["hidden_layer_sizes"] = (100,)

with c3:
    if algo in ("Gradient Boosting","XGBoost","LightGBM"):
        hp["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.1, format="%.3f")
    if algo == "Neural Network (MLP)":
        hp["max_iter"] = st.number_input("max_iter", 100, 2000, 500, step=50)
        hp["learning_rate_init"] = st.number_input("learning_rate_init", 0.0001, 0.1, 0.001, format="%.4f")
    if algo == "KNN":
        hp["weights"] = st.selectbox("weights", ["uniform","distance"])
        hp["metric"] = st.selectbox("metric", ["minkowski","euclidean","manhattan"])
    if algo == "Logistic Regression":
        hp["solver"] = st.selectbox("solver", ["lbfgs","saga","liblinear"])

section("Train / test split")
c1, c2 = st.columns(2)
with c1:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with c2:
    random_state = st.number_input("Random seed", 0, 9999, 42)
st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;">Train ~{int((1-test_size)*len(X)):,} · Test ~{int(test_size*len(X)):,} samples</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Train Model", type="primary", use_container_width=True):
    with st.spinner(f"Training {algo}..."):
        try:
            clean_hp = {k: v for k, v in hp.items() if v is not None}
            results = train_tabular(X, y, algo, task_type, clean_hp, test_size, int(random_state))
            st.session_state.update({
                "model_results": results,
                "trained_model": results["model"],
                "trained_algo": algo,
                "feature_names": valid_features,
            })
            st.markdown('<div class="ml-success">✓ Model trained successfully! Go to <strong>Results</strong>.</div>', unsafe_allow_html=True)
            m = results["metrics"]
            cols = st.columns(len(m))
            for col, (k,v) in zip(cols, m.items()):
                col.metric(k, v)
        except Exception as e:
            st.markdown(f'<div class="ml-error">❌ Training failed: {e}</div>', unsafe_allow_html=True)
            st.exception(e)
