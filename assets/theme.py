GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0a0f !important;
    border-right: 1px solid #1e1e2e !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    padding: 0.3rem 0 0.3rem 1rem !important;
}
[data-testid="stSidebarNav"] a {
    border-radius: 8px !important;
    margin: 1px 8px !important;
    padding: 0.55rem 0.85rem !important;
    color: #9ca3af !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: #1e1e2e !important;
    color: #f9fafb !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: #1e1e2e !important;
    color: #6366f1 !important;
    border-left: 2px solid #6366f1 !important;
}

/* ── Main area ────────────────────────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1200px !important;
}
.stApp { background: #07070d !important; }

/* ── Page headers ─────────────────────────────────────────────────────── */
.ml-page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e1e2e;
}
.ml-page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f9fafb;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.3rem 0;
}
.ml-page-sub {
    font-size: 0.875rem;
    color: #6b7280;
    font-weight: 400;
    margin: 0;
}

/* ── Cards ────────────────────────────────────────────────────────────── */
.ml-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.ml-card-sm {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.ml-card:hover { border-color: #2a2a3e; }

/* ── Metric cards ─────────────────────────────────────────────────────── */
.ml-metric {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.ml-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #f9fafb;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.ml-metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ml-metric-sub {
    font-size: 0.7rem;
    color: #4b5563;
    margin-top: 0.2rem;
}

/* ── Section labels ───────────────────────────────────────────────────── */
.ml-section {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 1.75rem 0 0.75rem 0;
}

/* ── Badges / chips ───────────────────────────────────────────────────── */
.ml-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    color: #9ca3af;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.2rem;
}
.ml-badge-accent {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
    color: #818cf8;
}
.ml-badge-success {
    background: rgba(16,185,129,0.1);
    border-color: rgba(16,185,129,0.3);
    color: #34d399;
}
.ml-badge-warn {
    background: rgba(245,158,11,0.1);
    border-color: rgba(245,158,11,0.3);
    color: #fbbf24;
}
.ml-badge-error {
    background: rgba(239,68,68,0.1);
    border-color: rgba(239,68,68,0.3);
    color: #f87171;
}

/* ── Info boxes ───────────────────────────────────────────────────────── */
.ml-info {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #a5b4fc;
    margin: 0.75rem 0;
}
.ml-success {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #6ee7b7;
    margin: 0.75rem 0;
}
.ml-warn {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fcd34d;
    margin: 0.75rem 0;
}
.ml-error {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fca5a5;
    margin: 0.75rem 0;
}

/* ── Steps indicator ──────────────────────────────────────────────────── */
.ml-step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.ml-step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 600;
    color: #6b7280;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}
.ml-step-num.active {
    background: rgba(99,102,241,0.15);
    border-color: rgba(99,102,241,0.5);
    color: #818cf8;
}
.ml-step-content { flex: 1; }
.ml-step-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: #e5e7eb;
    margin-bottom: 0.1rem;
}
.ml-step-desc { font-size: 0.8rem; color: #6b7280; }

/* ── Streamlit widget overrides ───────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #f9fafb !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e1e2e !important;
    color: #9ca3af !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #1e1e2e !important;
    color: #f9fafb !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Slider ───────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Dataframe ────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Radio ────────────────────────────────────────────────────────────── */
.stRadio > div { gap: 0.5rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem !important;
    color: #e5e7eb !important;
}

/* ── Checkbox ─────────────────────────────────────────────────────────── */
.stCheckbox [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem !important;
    color: #9ca3af !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0f0f1a !important;
    border: 1px dashed #2a2a3e !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}

/* ── Plotly charts background ─────────────────────────────────────────── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── Divider ──────────────────────────────────────────────────────────── */
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }

/* ── Code blocks ──────────────────────────────────────────────────────── */
code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 6px !important;
    color: #a5b4fc !important;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #0f0f1a !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #9ca3af !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6366f1 !important; }

/* ── Progress bar ─────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border-radius: 4px !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #1e1e2e; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a2a3e; }
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f0f1a",
    font=dict(family="Inter, sans-serif", color="#9ca3af", size=12),
    title_font=dict(family="Inter, sans-serif", color="#f9fafb", size=14),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zerolinecolor="#2a2a3e"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", zerolinecolor="#2a2a3e"),
    margin=dict(t=50, l=40, r=20, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
)

COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "text": "#f9fafb",
    "muted": "#6b7280",
    "surface": "#0f0f1a",
    "border": "#1e1e2e",
    "sequence": ["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899"],
}


def page_header(icon: str, title: str, subtitle: str = ""):
    import streamlit as st
    st.markdown(f"""
    <div class="ml-page-header">
        <div class="ml-page-title">{icon}&nbsp; {title}</div>
        {"<div class='ml-page-sub'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def section(label: str):
    import streamlit as st
    st.markdown(f'<div class="ml-section">{label}</div>', unsafe_allow_html=True)


def info_box(text: str, kind: str = "info"):
    import streamlit as st
    st.markdown(f'<div class="ml-{kind}">{text}</div>', unsafe_allow_html=True)


def metric_row(metrics: dict, colors: dict = None):
    import streamlit as st
    cols = st.columns(len(metrics))
    default_colors = ["#6366f1","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ef4444"]
    for i, (col, (k, v)) in enumerate(zip(cols, metrics.items())):
        c = (colors or {}).get(k, default_colors[i % len(default_colors)])
        col.markdown(f"""
        <div class="ml-metric">
            <div class="ml-metric-value" style="color:{c}">{v}</div>
            <div class="ml-metric-label">{k}</div>
        </div>""", unsafe_allow_html=True)
