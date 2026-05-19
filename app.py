import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assets.theme import GLOBAL_CSS

st.set_page_config(
    page_title="ML Studio",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
DEFAULTS = {
    "df": None,
    "df_processed": None,
    "load_info": {},
    "target": None,
    "features": [],
    "drop_cols": [],
    "task_type": "Classification",
    "imputer_strategy": "mean",
    "encode_method": "Label Encoding",
    "scaler_method": "none",
    "model_results": None,
    "trained_model": None,
    "trained_algo": None,
    "feature_names": [],
    "label_encoders": {},
    "target_encoder": None,
    "uploaded_filename": None,
    # NLP specific
    "text_col": None,
    "nlp_vectorizer": "TF-IDF",
    # TS specific
    "date_col": None,
    "ts_target": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar branding ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 1rem 1rem; border-bottom: 1px solid #1e1e2e; margin-bottom: 0.5rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.02em;">
            ⬡ ML Studio
        </div>
        <div style="font-size: 0.72rem; color: #4b5563; margin-top: 0.2rem;">
            End-to-end machine learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    df = st.session_state.get("df")
    if df is not None:
        fname = st.session_state.get("uploaded_filename", "dataset")
        task = st.session_state.get("task_type", "—")
        st.markdown(f"""
        <div style="margin: 0.5rem 0.5rem 1rem; padding: 0.75rem; background: #0f0f1a; border: 1px solid #1e1e2e; border-radius: 8px;">
            <div style="font-size: 0.7rem; color: #4b5563; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.5rem;">ACTIVE DATASET</div>
            <div style="font-size: 0.8rem; color: #e5e7eb; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{fname}</div>
            <div style="font-size: 0.72rem; color: #6b7280; margin-top: 0.2rem;">{df.shape[0]:,} rows · {df.shape[1]} cols · {task}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="padding: 0 1rem; font-size: 0.68rem; font-weight: 600; color: #374151; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem;">WORKFLOW</div>', unsafe_allow_html=True)

# ── Home content ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="max-width: 680px; margin: 3rem auto 0; text-align: center; padding: 0 1rem;">
    <div style="font-size: 2.5rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.04em; line-height: 1.1; margin-bottom: 0.75rem;">
        Machine Learning,<br>end to end.
    </div>
    <div style="font-size: 1rem; color: #6b7280; font-weight: 400; line-height: 1.6; margin-bottom: 3rem;">
        Upload a CSV. Explore. Preprocess. Train. Evaluate. Test. Export.<br>
        No code required.
    </div>
</div>
""", unsafe_allow_html=True)

# Task type cards
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("📊", "Regression", "Predict continuous values — prices, scores, quantities."),
    ("🏷️", "Classification", "Categorize data — binary or multi-class labels."),
    ("📝", "NLP Classification", "Classify text — sentiment, topics, intent."),
    ("📈", "Time Series", "Forecast temporal data with lag-based features."),
]
for col, (icon, title, desc) in zip([c1,c2,c3,c4], cards):
    col.markdown(f"""
    <div class="ml-card" style="text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center; cursor: default;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 0.875rem; font-weight: 600; color: #e5e7eb; margin-bottom: 0.3rem;">{title}</div>
        <div style="font-size: 0.75rem; color: #6b7280; line-height: 1.4;">{desc}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Workflow steps
st.markdown('<div style="max-width: 800px; margin: 0 auto;">', unsafe_allow_html=True)
steps = [
    ("1", "Upload Data", "CSV with any encoding — UTF-8, Latin-1, CP1252. Handles double headers, missing headers, mixed separators."),
    ("2", "Explore (EDA)", "head(), info(), null analysis, distributions, correlation heatmap."),
    ("3", "Preprocess", "Drop columns, impute missing values, encode categoricals, scale features."),
    ("4", "Train Model", "Pick from 15+ algorithms with configurable hyperparameters. Regression, Classification, NLP, Time Series."),
    ("5", "Results", "Metrics, confusion matrix, residual plots, feature importance."),
    ("6", "Test Model", "Run predictions on new data — single row form or batch CSV upload."),
    ("7", "Save Model", "Export trained model as .pkl, .joblib, or full pipeline."),
]

cols_l, cols_r = st.columns([1, 2])
with cols_l:
    st.markdown('<div class="ml-section">7-step workflow</div>', unsafe_allow_html=True)
    for num, title, _ in steps:
        active = "active" if num == "1" else ""
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem;">
            <div class="ml-step-num {active}">{num}</div>
            <div style="font-size: 0.85rem; font-weight: 500; color: #9ca3af;">{title}</div>
        </div>""", unsafe_allow_html=True)

with cols_r:
    st.markdown('<div class="ml-section">Supported algorithms</div>', unsafe_allow_html=True)
    algos = [
        ("Regression", ["Linear","Ridge","Lasso","ElasticNet","Decision Tree","Random Forest","Extra Trees","Gradient Boosting","SVR","KNN","MLP","XGBoost","CatBoost","LightGBM"]),
        ("Classification", ["Logistic Reg.","Decision Tree","Random Forest","Gradient Boosting","SVM","KNN","MLP","XGBoost","CatBoost","LightGBM","Naive Bayes"]),
        ("NLP", ["Logistic Reg.","Naive Bayes","SVM Linear","Random Forest","Gradient Boosting","XGBoost"]),
        ("Time Series", ["Linear","Ridge","Random Forest","Gradient Boosting","XGBoost","LightGBM"]),
    ]
    for group, names in algos:
        st.markdown(f'<div style="font-size: 0.72rem; color: #4b5563; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin: 0.5rem 0 0.3rem;">{group}</div>', unsafe_allow_html=True)
        st.markdown(" ".join(f'<span class="ml-badge">{n}</span>' for n in names), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center; font-size: 0.75rem; color: #374151;">← Use the sidebar to navigate · Start with <strong style="color:#6366f1">Upload Data</strong></div>', unsafe_allow_html=True)
