import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box
from utils.text_cleaner import STEP_REGISTRY, STEP_ORDER, clean_series, _NLTK_OK

st.set_page_config(page_title="Text Cleaner · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("3b", "Text Cleaner", "Clean and normalize raw text before NLP modelling.")

# ── NLTK status ────────────────────────────────────────────────────────────────
if _NLTK_OK:
    st.markdown('<span class="ml-badge ml-badge-success">✓ NLTK available — full stemming & lemmatization</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="ml-badge ml-badge-warn">⚠ NLTK not downloaded — using lightweight fallback for stemming/lemmatization</span>', unsafe_allow_html=True)

df = st.session_state.get("df")
task_type = st.session_state.get("task_type","")
text_col = st.session_state.get("text_col")

if df is None:
    st.markdown('<div class="ml-warn">⚠ No dataset loaded. Go to <strong>Upload Data</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

if task_type != "NLP Classification":
    st.markdown('<div class="ml-info">ℹ This page applies to <strong>NLP Classification</strong> tasks only. Set the task type in <strong>Preprocess</strong>.</div>', unsafe_allow_html=True)

# ── Column selector ────────────────────────────────────────────────────────────
section("Select text column")
text_candidates = df.select_dtypes(include="object").columns.tolist()
if not text_candidates:
    st.markdown('<div class="ml-error">❌ No text (string) columns found.</div>', unsafe_allow_html=True)
    st.stop()

selected_text_col = st.selectbox(
    "Text column", text_candidates,
    index=text_candidates.index(text_col) if text_col in text_candidates else 0,
    label_visibility="collapsed"
)

# ── Preview raw ────────────────────────────────────────────────────────────────
section("Raw text sample")
sample_raw = df[selected_text_col].dropna().astype(str).head(5).tolist()
for i, t in enumerate(sample_raw):
    st.markdown(f'<div class="ml-card-sm" style="margin-bottom:0.4rem;font-size:0.8rem;color:#9ca3af;font-family:JetBrains Mono,monospace;">[{i}] {t[:300]}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLEANING STEPS — grouped checkboxes
# ══════════════════════════════════════════════════════════════════════════════
section("Cleaning steps")
st.markdown('<div class="ml-info">Steps are always applied in the <strong>recommended order</strong> regardless of visual arrangement. Conflicting steps (e.g. lemmatize + stem) — pick one.</div>', unsafe_allow_html=True)

# Group definitions
GROUPS = {
    "🔡 Normalisation": ["lowercase", "expand_contractions", "whitespace"],
    "🌐 Web & Social": ["remove_html", "remove_urls", "remove_emails", "remove_mentions", "remove_hashtags", "remove_emojis"],
    "🔢 Characters & Tokens": ["remove_digits", "remove_punctuation", "remove_special_chars", "remove_stopwords", "remove_short_words"],
    "🌿 Morphological": ["lemmatize", "stem_porter", "stem_lancaster", "stem_snowball"],
}

# ── Presets ────────────────────────────────────────────────────────────────────
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

PRESETS = {
    "Sentiment (Social)": ["lowercase","expand_contractions","remove_html","remove_urls","remove_emails","remove_mentions","remove_hashtags","remove_emojis","remove_punctuation","remove_stopwords","whitespace","lemmatize"],
    "Topic Modelling": ["lowercase","remove_html","remove_urls","remove_digits","remove_punctuation","remove_stopwords","remove_short_words","whitespace","lemmatize"],
    "Spam Detection": ["lowercase","expand_contractions","remove_html","remove_urls","remove_emails","remove_digits","remove_punctuation","remove_stopwords","whitespace","stem_porter"],
    "Minimal": ["lowercase","whitespace"],
}

preset_selected = None
for col, (name, _) in zip([preset_col1,preset_col2,preset_col3,preset_col4], PRESETS.items()):
    if col.button(name, use_container_width=True, key=f"preset_{name}"):
        preset_selected = name

# Initialise selection state
if "cleaner_steps" not in st.session_state:
    st.session_state["cleaner_steps"] = ["lowercase", "whitespace"]

if preset_selected:
    st.session_state["cleaner_steps"] = PRESETS[preset_selected]
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
selected_steps = []

for group_name, step_keys in GROUPS.items():
    st.markdown(f'<div class="ml-section">{group_name}</div>', unsafe_allow_html=True)
    ncols = min(len(step_keys), 3)
    cols = st.columns(ncols)
    for i, key in enumerate(step_keys):
        label, _, desc = STEP_REGISTRY[key]
        with cols[i % ncols]:
            default_val = key in st.session_state["cleaner_steps"]
            checked = st.checkbox(
                label,
                value=default_val,
                key=f"chk_{key}",
                help=desc,
            )
            if checked:
                selected_steps.append(key)

# Mutually exclusive warning: stemming methods
stem_choices = [s for s in selected_steps if s.startswith("stem_")]
if len(stem_choices) > 1:
    st.markdown('<div class="ml-warn">⚠ Multiple stemming algorithms selected — only the last in order will have visible effect. Recommend picking one.</div>', unsafe_allow_html=True)

if "lemmatize" in selected_steps and stem_choices:
    st.markdown('<div class="ml-warn">⚠ Both lemmatization and stemming selected — lemmatize runs first, then stemming may reduce further. This is usually not ideal.</div>', unsafe_allow_html=True)

st.session_state["cleaner_steps"] = selected_steps

# ── Custom regex ───────────────────────────────────────────────────────────────
section("Custom regex (optional)")
c1, c2 = st.columns([2,1])
with c1:
    custom_pattern = st.text_input("Regex pattern to remove/replace", placeholder=r"e.g.  \bRT\b  or  [^\x00-\x7F]", label_visibility="collapsed")
with c2:
    custom_replacement = st.text_input("Replacement string", value=" ", label_visibility="collapsed")

# ── Live preview ───────────────────────────────────────────────────────────────
section("Live preview")
from utils.text_cleaner import clean_text

preview_text = st.text_area(
    "Try your pipeline on a custom sample",
    value=sample_raw[0] if sample_raw else "The movie was GREAT!!! Check http://review.site #film 😍 I can't believe it!",
    height=80,
    label_visibility="collapsed"
)

if selected_steps or custom_pattern.strip():
    cleaned_preview = clean_text(preview_text, selected_steps, custom_pattern, custom_replacement)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before:**")
        st.markdown(f'<div class="ml-card-sm" style="font-size:0.8rem;color:#9ca3af;font-family:JetBrains Mono,monospace;">{preview_text}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**After:**")
        st.markdown(f'<div class="ml-card-sm" style="font-size:0.8rem;color:#34d399;font-family:JetBrains Mono,monospace;">{cleaned_preview}</div>', unsafe_allow_html=True)

# ── Apply to full column ───────────────────────────────────────────────────────
section("Apply to dataset")
st.markdown(f'<div class="ml-info">Will clean column <code>{selected_text_col}</code> in the dataset ({len(df):,} rows) and save the result. The original column is preserved as <code>{selected_text_col}_raw</code>.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1,3])
with col1:
    apply_btn = st.button("Apply Cleaning", type="primary", use_container_width=True)

if apply_btn:
    if not selected_steps and not custom_pattern.strip():
        st.markdown('<div class="ml-warn">⚠ No steps selected — nothing to apply.</div>', unsafe_allow_html=True)
    else:
        with st.spinner(f"Cleaning {len(df):,} rows..."):
            df_clean = df.copy()
            # Preserve raw
            if f"{selected_text_col}_raw" not in df_clean.columns:
                df_clean[f"{selected_text_col}_raw"] = df_clean[selected_text_col]
            # Clean
            df_clean[selected_text_col] = clean_series(
                df_clean[selected_text_col], selected_steps, custom_pattern, custom_replacement
            )
            # Push back to session
            st.session_state["df"] = df_clean
            st.session_state["text_col"] = selected_text_col

        # Stats
        before = st.session_state["df"][f"{selected_text_col}_raw"].fillna("").astype(str)
        after  = st.session_state["df"][selected_text_col].fillna("").astype(str)
        avg_before = before.str.split().str.len().mean()
        avg_after  = after.str.split().str.len().mean()
        reduction  = 100 * (1 - avg_after / avg_before) if avg_before > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:#6366f1">{avg_before:.1f}</div><div class="ml-metric-label">Avg words before</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:#10b981">{avg_after:.1f}</div><div class="ml-metric-label">Avg words after</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:#f59e0b">{reduction:.1f}%</div><div class="ml-metric-label">Vocabulary reduction</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Side-by-side sample (5 rows):**")
        comp_df = pd.DataFrame({
            "Raw": before.head(5).tolist(),
            "Cleaned": after.head(5).tolist(),
        })
        st.dataframe(comp_df, use_container_width=True)
        st.markdown('<div class="ml-success">✓ Text cleaned and saved to session. Continue to <strong>Preprocess</strong> → <strong>Train Model</strong>.</div>', unsafe_allow_html=True)

# ── Applied steps summary ──────────────────────────────────────────────────────
if selected_steps:
    section(f"Active pipeline ({len(selected_steps)} steps)")
    pipeline_html = ' → '.join(
        f'<span class="ml-badge ml-badge-accent">{STEP_REGISTRY[s][0]}</span>'
        for s in STEP_ORDER if s in selected_steps
    )
    st.markdown(pipeline_html, unsafe_allow_html=True)
    if custom_pattern.strip():
        st.markdown(f'<span class="ml-badge ml-badge-warn">+ custom regex: <code>{custom_pattern}</code></span>', unsafe_allow_html=True)
