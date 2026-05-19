import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box, PLOTLY_LAYOUT, COLORS
import plotly.express as px
import plotly.graph_objects as go
import copy

st.set_page_config(page_title="Explore · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("02", "Explore Data", "Understand your dataset before modelling.")

df = st.session_state.get("df")
if df is None:
    st.markdown('<div class="ml-warn">⚠ No dataset loaded. Go to <strong>Upload Data</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

def pl(fig):
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

tabs = st.tabs(["Preview", "Data Types", "Statistics", "Missing Values", "Distribution", "Correlations"])

# ── Tab 1: Preview ─────────────────────────────────────────────────────────────
with tabs[0]:
    n = st.slider("Rows to display", 5, min(500, len(df)), 20, label_visibility="collapsed")
    st.dataframe(df.head(n), use_container_width=True)
    st.markdown(f'<div style="font-size:0.75rem; color:#4b5563; margin-top:0.5rem;">Showing {n} of {len(df):,} rows</div>', unsafe_allow_html=True)

# ── Tab 2: Types ───────────────────────────────────────────────────────────────
with tabs[1]:
    col1, col2 = st.columns([3, 2])
    with col1:
        section("Column types")
        d = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Non-Null": df.count().values,
            "Null": df.isnull().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
            "Memory (KB)": (df.memory_usage(deep=True)[1:] / 1024).round(2).values,
        })
        st.dataframe(d, use_container_width=True, height=400)
    with col2:
        section("Type breakdown")
        type_counts = df.dtypes.astype(str).map(
            lambda x: "Float" if "float" in x else "Integer" if "int" in x else "Object/String" if x == "object" else x
        ).value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index,
                     color_discrete_sequence=COLORS["sequence"], hole=0.55)
        fig.update_traces(textposition="outside", textfont_size=11)
        pl(fig)

        section("Memory usage")
        total_kb = df.memory_usage(deep=True).sum() / 1024
        st.markdown(f'<div class="ml-metric"><div class="ml-metric-value">{total_kb:.1f} KB</div><div class="ml-metric-label">Total memory</div></div>', unsafe_allow_html=True)

# ── Tab 3: Statistics ──────────────────────────────────────────────────────────
with tabs[2]:
    section("Statistical summary")
    st.dataframe(df.describe(include="all").T.round(4), use_container_width=True, height=500)

# ── Tab 4: Missing values ──────────────────────────────────────────────────────
with tabs[3]:
    null_sum = df.isnull().sum()
    null_pct = (null_sum / len(df) * 100).round(2)
    total_null = int(null_sum.sum())

    c1,c2,c3,c4 = st.columns(4)
    for col, val, lbl, clr in zip(
        [c1,c2,c3,c4],
        [total_null, int((null_sum>0).sum()), int((null_pct>30).sum()), int((df.isnull().sum(axis=1)==0).sum())],
        ["Missing cells","Cols with nulls","Cols >30% null","Complete rows"],
        ["#ef4444","#f59e0b","#ef4444","#10b981"]
    ):
        col.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:{clr}">{val:,}</div><div class="ml-metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    null_df = pd.DataFrame({"Column": null_sum.index, "Missing": null_sum.values, "% Missing": null_pct.values})
    null_df = null_df[null_df["Missing"] > 0].sort_values("% Missing", ascending=False)

    if len(null_df) == 0:
        st.markdown('<div class="ml-success">✓ No missing values — dataset is complete.</div>', unsafe_allow_html=True)
    else:
        fig = px.bar(null_df, x="Column", y="% Missing",
                     color="% Missing", color_continuous_scale=["#6366f1","#ef4444"],
                     title="Missing Value % per Column")
        pl(fig)
        st.dataframe(null_df.reset_index(drop=True), use_container_width=True)

# ── Tab 5: Distribution ────────────────────────────────────────────────────────
with tabs[4]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    selected = st.selectbox("Select column", df.columns.tolist(), label_visibility="collapsed")

    if selected in num_cols:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=selected, nbins=40, color_discrete_sequence=[COLORS["primary"]], title=f"Distribution — {selected}")
            pl(fig)
        with c2:
            fig2 = px.box(df, y=selected, color_discrete_sequence=[COLORS["accent"]], title=f"Box plot — {selected}")
            pl(fig2)

        s = df[selected].describe()
        cols = st.columns(6)
        for col, (k,v) in zip(cols, {"Mean":s["mean"],"Median":df[selected].median(),"Std":s["std"],"Min":s["min"],"Max":s["max"],"Skew":df[selected].skew()}.items()):
            col.markdown(f'<div class="ml-card-sm" style="text-align:center"><div style="font-size:1rem;font-weight:600;color:#f9fafb">{v:.3f}</div><div style="font-size:0.7rem;color:#6b7280">{k}</div></div>', unsafe_allow_html=True)
    else:
        vc = df[selected].value_counts().reset_index()
        vc.columns = [selected, "Count"]
        fig = px.bar(vc.head(25), x=selected, y="Count",
                     color_discrete_sequence=[COLORS["primary"]], title=f"Value counts — {selected} (top 25)")
        pl(fig)
        st.markdown(f'<span class="ml-badge">Unique values: {df[selected].nunique()}</span>', unsafe_allow_html=True)

# ── Tab 6: Correlations ────────────────────────────────────────────────────────
with tabs[5]:
    if len(num_cols) < 2:
        st.markdown('<div class="ml-warn">Need at least 2 numeric columns for correlation analysis.</div>', unsafe_allow_html=True)
    else:
        corr = df[num_cols].corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Pearson Correlation Matrix", aspect="auto")
        fig.update_traces(textfont_size=9)
        pl(fig)

        section("Top correlations")
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                pairs.append({"Feature A": corr.columns[i], "Feature B": corr.columns[j], "Correlation": corr.iloc[i,j]})
        pair_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
        st.dataframe(pair_df.head(20).reset_index(drop=True), use_container_width=True)
