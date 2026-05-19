import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box
from utils.data_handler import smart_load_csv

st.set_page_config(page_title="Upload · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

page_header("01", "Upload Data", "Upload any CSV file — we handle encoding, double headers, and formatting automatically.")

uploaded = st.file_uploader(
    "Drop your CSV here",
    type=["csv", "txt"],
    label_visibility="collapsed",
    help="Supports UTF-8, Latin-1, CP1252, UTF-16. Handles single & double header rows."
)

if uploaded:
    with st.spinner("Analysing file..."):
        df, info = smart_load_csv(uploaded)

    if df is None:
        st.markdown('<div class="ml-error">❌ Could not parse this file. Try saving it as UTF-8 CSV and re-uploading.</div>', unsafe_allow_html=True)
        st.stop()

    # Store
    st.session_state["df"] = df
    st.session_state["df_processed"] = df.copy()
    st.session_state["load_info"] = info
    st.session_state["uploaded_filename"] = uploaded.name
    # Reset downstream
    for k in ["target","features","drop_cols","model_results","trained_model","trained_algo","feature_names"]:
        st.session_state[k] = [] if k in ("features","drop_cols","feature_names") else None

    # ── Load alerts ────────────────────────────────────────────────────────────
    if info.get("issues"):
        for issue in info["issues"]:
            st.markdown(f'<div class="ml-warn">⚠ {issue}</div>', unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────────────
    section("Dataset overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    stats = [
        ("#0d0d1a", "#6366f1", str(df.shape[0]), "Rows"),
        ("#0d0d1a", "#8b5cf6", str(df.shape[1]), "Columns"),
        ("#0d0d1a", "#06b6d4", str(df.isnull().sum().sum()), "Missing cells"),
        ("#0d0d1a", "#10b981", str(len(df.select_dtypes(include="number").columns)), "Numeric cols"),
        ("#0d0d1a", "#f59e0b", str(len(df.select_dtypes(exclude="number").columns)), "Categorical cols"),
    ]
    for col, (bg, clr, val, lbl) in zip([c1,c2,c3,c4,c5], stats):
        col.markdown(f"""
        <div class="ml-metric">
            <div class="ml-metric-value" style="color:{clr}">{val}</div>
            <div class="ml-metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ── Load info ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.markdown(f'<span class="ml-badge">Encoding: {info.get("encoding","auto")}</span>', unsafe_allow_html=True)
    col_b.markdown(f'<span class="ml-badge">Separator: {repr(info.get("sep",","))}</span>', unsafe_allow_html=True)
    col_c.markdown(f'<span class="ml-badge">Header row: {info.get("header_row",0)}</span>', unsafe_allow_html=True)

    # ── Data preview ───────────────────────────────────────────────────────────
    section("Preview")
    st.dataframe(df.head(10), use_container_width=True, height=280)

    # ── Column info ────────────────────────────────────────────────────────────
    section("Column summary")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.count().values,
        "Null": df.isnull().sum().values,
        "Null %": (df.isnull().sum() / len(df) * 100).round(1).values,
        "Unique": [df[c].nunique() for c in df.columns],
        "Sample": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—" for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True, height=320)

    st.markdown('<div class="ml-success">✓ Dataset loaded successfully. Continue to <strong>Explore Data</strong> →</div>', unsafe_allow_html=True)

elif st.session_state.get("df") is not None:
    df = st.session_state["df"]
    info = st.session_state.get("load_info", {})
    fname = st.session_state.get("uploaded_filename", "dataset")
    st.markdown(f'<div class="ml-info">📌 <strong>{fname}</strong> is already loaded ({df.shape[0]:,} rows × {df.shape[1]} cols). Upload a new file to replace it.</div>', unsafe_allow_html=True)
    section("Current dataset preview")
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.markdown("""
    <div style="border: 1px dashed #1e1e2e; border-radius: 16px; padding: 4rem 2rem; text-align: center; margin-top: 1rem;">
        <div style="font-size: 2rem; margin-bottom: 1rem; opacity: 0.4;">⬆</div>
        <div style="font-size: 0.95rem; font-weight: 500; color: #4b5563;">Drop a CSV file above</div>
        <div style="font-size: 0.8rem; color: #374151; margin-top: 0.5rem; line-height: 1.6;">
            Handles UTF-8 · Latin-1 · CP1252 · UTF-16<br>
            Auto-detects double headers · missing headers · semicolon / tab / pipe separators
        </div>
    </div>
    """, unsafe_allow_html=True)
