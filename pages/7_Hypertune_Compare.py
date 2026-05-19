import streamlit as st
import pandas as pd
import numpy as np
import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, PLOTLY_LAYOUT, COLORS
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Hypertune · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("4b", "Hypertune & Compare", "Cross-validation, hyperparameter search, and multi-model comparison.")

# ── Guards ─────────────────────────────────────────────────────────────────────
df_processed = st.session_state.get("df_processed")
task_type = st.session_state.get("task_type","Classification")
target = st.session_state.get("target")
features = st.session_state.get("feature_names", st.session_state.get("features",[]))
trained_model = st.session_state.get("trained_model")
trained_algo = st.session_state.get("trained_algo","")

if df_processed is None or not features or target not in df_processed.columns:
    st.markdown('<div class="ml-warn">⚠ Complete <strong>Preprocess</strong> (and optionally <strong>Train Model</strong>) before running tuning.</div>', unsafe_allow_html=True)
    st.stop()

if task_type in ("NLP Classification","Time Series"):
    st.markdown('<div class="ml-warn">⚠ Hypertune currently supports <strong>Classification</strong> and <strong>Regression</strong> tabular tasks.</div>', unsafe_allow_html=True)
    st.stop()

valid_features = [f for f in features if f in df_processed.columns]
X = df_processed[valid_features].values
y = df_processed[target].values

st.markdown(f"""
<div class="ml-card" style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center;">
    <span class="ml-badge ml-badge-accent">{task_type}</span>
    <span class="ml-badge">Target: {target}</span>
    <span class="ml-badge">{len(valid_features)} features</span>
    <span class="ml-badge">{len(X):,} samples</span>
    {"<span class='ml-badge ml-badge-success'>Current model: " + trained_algo + "</span>" if trained_algo else ""}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["📊 Cross-Validation", "🔍 Hyperparameter Search", "⚡ Model Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Cross-validation on current model")

    if trained_model is None:
        st.markdown('<div class="ml-info">Train a model first in <strong>Train Model</strong>, then come here to cross-validate it.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            cv_folds = st.slider("K-fold splits", 3, 10, 5, key="cv_folds")
        with c2:
            cv_seed = st.number_input("Random seed", 0, 9999, 42, key="cv_seed")

        st.markdown(f'<div class="ml-info">Will run <strong>{cv_folds}-fold cross-validation</strong> on <strong>{trained_algo}</strong> using all {len(X):,} samples.</div>', unsafe_allow_html=True)

        if st.button("Run Cross-Validation", type="primary", key="run_cv"):
            from utils.hypertune import run_cross_validation
            with st.spinner(f"Running {cv_folds}-fold CV on {trained_algo}..."):
                t0 = time.time()
                try:
                    cv_results = run_cross_validation(trained_model, X, y, task_type, cv_folds, int(cv_seed))
                    elapsed = time.time() - t0
                    st.session_state["cv_results"] = cv_results

                    st.markdown(f'<div class="ml-success">✓ CV complete in {elapsed:.1f}s</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="ml-error">❌ CV failed: {e}</div>', unsafe_allow_html=True)
                    st.exception(e)

        cv_results = st.session_state.get("cv_results")
        if cv_results:
            section("Results")
            # Metric cards
            cols = st.columns(len(cv_results))
            for col, (metric, data) in zip(cols, cv_results.items()):
                if data.get("mean") is not None:
                    col.markdown(f"""
                    <div class="ml-metric">
                        <div class="ml-metric-value" style="color:#6366f1">{data['mean']}</div>
                        <div class="ml-metric-label">{metric}</div>
                        <div class="ml-metric-sub">± {data['std']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Per-fold plot
            for metric, data in cv_results.items():
                if data.get("scores"):
                    fold_df = pd.DataFrame({
                        "Fold": [f"Fold {i+1}" for i in range(len(data["scores"]))],
                        metric: data["scores"]
                    })
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=fold_df["Fold"], y=fold_df[metric],
                        marker_color=COLORS["primary"], name=metric
                    ))
                    fig.add_hline(y=data["mean"], line_dash="dash", line_color=COLORS["success"],
                                  annotation_text=f"Mean: {data['mean']}", annotation_position="top right")
                    layout = copy.deepcopy(PLOTLY_LAYOUT)
                    layout["title"] = f"{metric} per Fold ({trained_algo})"
                    layout["height"] = 280
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                    break  # show only first metric chart to save space

            # Table
            rows = []
            for metric, data in cv_results.items():
                for fold_i, score in enumerate(data.get("scores", [])):
                    rows.append({"Fold": fold_i+1, "Metric": metric, "Score": score})
            if rows:
                fold_table = pd.DataFrame(rows).pivot(index="Fold", columns="Metric", values="Score")
                st.dataframe(fold_table.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Hyperparameter Search
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Algorithm & search method")
    from utils.model_trainer import REGRESSION_MODELS, CLASSIFICATION_MODELS
    from utils.hypertune import PARAM_GRIDS, run_hyperparameter_search

    registry = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
    tunable = [a for a in PARAM_GRIDS.keys() if a in registry]

    c1, c2, c3 = st.columns(3)
    with c1:
        ht_algo = st.selectbox("Algorithm to tune", tunable, key="ht_algo")
    with c2:
        ht_method = st.radio("Search method", ["random", "grid"], horizontal=True, key="ht_method",
                             help="Randomized is faster; Grid exhaustive but slow.")
    with c3:
        ht_scoring = st.selectbox(
            "Optimisation metric",
            ["accuracy","f1_weighted","roc_auc"] if task_type=="Classification" else ["r2","neg_mean_absolute_error","neg_root_mean_squared_error"],
            key="ht_scoring"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        ht_folds = st.slider("CV folds", 3, 10, 5, key="ht_folds")
    with c2:
        ht_iter = st.number_input("Random iterations (ignored for grid)", 5, 200, 20, key="ht_iter")
    with c3:
        ht_seed = st.number_input("Seed", 0, 9999, 42, key="ht_seed")

    # Show param grid
    raw_grid = PARAM_GRIDS.get(ht_algo, {})
    with st.expander("📋 Parameter grid being searched", expanded=False):
        for k, v in raw_grid.items():
            st.markdown(f'`{k}`: {v}')

    # Estimate combinations
    if ht_method == "grid":
        combos = 1
        for v in raw_grid.values():
            combos *= len(v)
        total_fits = combos * ht_folds
        warn = " ⚠ This may take a long time!" if total_fits > 500 else ""
        st.markdown(f'<div class="ml-info">Grid search: <strong>{combos} combinations × {ht_folds} folds = {total_fits} fits</strong>{warn}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ml-info">Random search: <strong>{int(ht_iter)} iterations × {ht_folds} folds = {int(ht_iter)*ht_folds} fits</strong></div>', unsafe_allow_html=True)

    if st.button("Start Search", type="primary", key="start_ht"):
        ModelClass = registry[ht_algo]()
        # Fix Decision Tree criterion for regression
        local_grid = dict(raw_grid)
        if ht_algo == "Decision Tree" and task_type == "Regression":
            local_grid["criterion"] = ["squared_error","absolute_error","friedman_mse"]

        with st.spinner(f"Running {ht_method} search on {ht_algo}..."):
            t0 = time.time()
            try:
                best_model, best_params, cv_df, best_score = run_hyperparameter_search(
                    ModelClass, local_grid, X, y,
                    task_type=task_type,
                    method=ht_method,
                    n_iter=int(ht_iter),
                    n_splits=ht_folds,
                    random_state=int(ht_seed),
                    scoring=ht_scoring,
                )
                elapsed = time.time() - t0
                st.session_state["ht_best_model"] = best_model
                st.session_state["ht_best_params"] = best_params
                st.session_state["ht_cv_df"] = cv_df
                st.session_state["ht_best_score"] = best_score
                st.session_state["ht_algo"] = ht_algo

                st.markdown(f'<div class="ml-success">✓ Search complete in {elapsed:.1f}s · Best {ht_scoring}: <strong>{best_score}</strong></div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ Search failed: {e}</div>', unsafe_allow_html=True)
                st.exception(e)

    # Results
    if st.session_state.get("ht_best_params"):
        section("Best configuration")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:#10b981">{st.session_state["ht_best_score"]}</div><div class="ml-metric-label">Best CV score ({ht_scoring})</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Best hyperparameters:**")
            for k, v in st.session_state["ht_best_params"].items():
                st.markdown(f'`{k}` = `{v}`')

        with c2:
            cv_df = st.session_state.get("ht_cv_df")
            if cv_df is not None:
                fig = px.scatter(cv_df, x=cv_df.index, y="mean_test_score",
                                 error_y="std_test_score" if "std_test_score" in cv_df.columns else None,
                                 color_discrete_sequence=[COLORS["primary"]],
                                 title="CV score per config (sorted by rank)")
                layout = copy.deepcopy(PLOTLY_LAYOUT)
                layout["height"] = 260
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        section("Full CV results")
        st.dataframe(st.session_state["ht_cv_df"].head(30), use_container_width=True, height=300)

        # Promote to trained model
        if st.button("✓ Use this as trained model", type="primary", key="promote_ht"):
            from utils.model_trainer import train_tabular
            from sklearn.model_selection import train_test_split
            clean_hp = {k: v for k, v in st.session_state["ht_best_params"].items() if v is not None}
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            best = st.session_state["ht_best_model"]
            best.fit(X_tr, y_tr)
            y_pred = best.predict(X_te)

            from utils.model_trainer import _reg_metrics, _cls_metrics
            from sklearn.metrics import confusion_matrix, classification_report
            results = {"model": best, "X_test": X_te, "y_test": y_te, "y_pred": y_pred, "task_type": task_type}
            if task_type == "Regression":
                results["metrics"] = _reg_metrics(y_te, y_pred)
            else:
                results["metrics"] = _cls_metrics(y_te, y_pred)
                results["confusion_matrix"] = confusion_matrix(y_te, y_pred)
                results["classification_report"] = classification_report(y_te, y_pred, zero_division=0)
            if hasattr(best, "feature_importances_"):
                results["feature_importances"] = best.feature_importances_
            elif hasattr(best, "coef_"):
                results["feature_importances"] = np.abs(best.coef_).flatten()

            st.session_state.update({
                "trained_model": best,
                "model_results": results,
                "trained_algo": f"{ht_algo} (tuned)",
            })
            st.markdown(f'<div class="ml-success">✓ Tuned {ht_algo} is now your active model. Check <strong>Results</strong>.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    from utils.model_trainer import REGRESSION_MODELS, CLASSIFICATION_MODELS
    from utils.hypertune import compare_models

    section("Quick model comparison (cross-validation)")
    st.markdown('<div class="ml-info">Trains each selected model with <strong>default hyperparameters</strong> and evaluates via cross-validation. Useful for baseline selection — not a replacement for proper tuning.</div>', unsafe_allow_html=True)

    registry = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
    available = list(registry.keys())

    # Remove heavy optionals if not installed
    from utils.model_trainer import XGBOOST_AVAILABLE, CATBOOST_AVAILABLE
    if not XGBOOST_AVAILABLE:
        available = [a for a in available if a != "XGBoost"]
    if not CATBOOST_AVAILABLE:
        available = [a for a in available if a != "CatBoost"]

    SAFE_DEFAULTS = {
        "Classification": ["Logistic Regression","Random Forest","Decision Tree","Gradient Boosting","KNN","Extra Trees"],
        "Regression":     ["Linear Regression","Random Forest","Decision Tree","Gradient Boosting","Ridge Regression","Extra Trees"],
    }
    default_sel = [a for a in SAFE_DEFAULTS.get(task_type,[]) if a in available]

    selected_models = st.multiselect(
        "Models to compare", available,
        default=default_sel,
        label_visibility="collapsed"
    )

    c1, c2 = st.columns(2)
    with c1:
        cmp_folds = st.slider("CV folds", 3, 10, 5, key="cmp_folds")
    with c2:
        cmp_seed = st.number_input("Seed", 0, 9999, 42, key="cmp_seed")

    if len(selected_models) == 0:
        st.markdown('<div class="ml-warn">Select at least one model.</div>', unsafe_allow_html=True)
    else:
        # Warn about slow models
        slow = [m for m in selected_models if m in ("SVM (SVC)","SVM (SVR)","Neural Network (MLP)")]
        if slow:
            st.markdown(f'<div class="ml-warn">⚠ {", ".join(slow)} can be slow on large datasets — may take minutes.</div>', unsafe_allow_html=True)

        if st.button(f"Compare {len(selected_models)} models", type="primary", key="run_cmp"):
            model_instances = {}
            for name in selected_models:
                try:
                    M = registry[name]
                    if name == "CatBoost":
                        model_instances[name] = M(verbose=0)
                    elif name == "Logistic Regression":
                        model_instances[name] = M(max_iter=500)
                    elif name == "Neural Network (MLP)":
                        model_instances[name] = M(max_iter=200)
                    else:
                        model_instances[name] = M()
                except Exception as e:
                    st.warning(f"Could not init {name}: {e}")

            progress_bar = st.progress(0)
            cmp_df_parts = []
            
            with st.spinner("Comparing models... (this may take a while)"):
                t0 = time.time()
                try:
                    cmp_df = compare_models(model_instances, X, y, task_type, cmp_folds, int(cmp_seed))
                    elapsed = time.time() - t0
                    progress_bar.progress(1.0)
                    st.session_state["cmp_df"] = cmp_df
                    st.session_state["cmp_task"] = task_type
                    st.markdown(f'<div class="ml-success">✓ Comparison done in {elapsed:.1f}s</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="ml-error">❌ Comparison failed: {e}</div>', unsafe_allow_html=True)
                    st.exception(e)

    cmp_df = st.session_state.get("cmp_df")
    if cmp_df is not None:
        section("Comparison results")
        primary_col = "Accuracy" if task_type == "Classification" else "R²"

        # Bar chart
        if primary_col in cmp_df.columns:
            cmp_sorted = cmp_df[cmp_df[primary_col].notna()].sort_values(primary_col)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cmp_sorted[primary_col],
                y=cmp_sorted["Model"],
                orientation="h",
                marker=dict(
                    color=cmp_sorted[primary_col],
                    colorscale=[[0, "#1e1e2e"], [1, "#6366f1"]],
                    showscale=False,
                ),
                error_x=dict(
                    type="data",
                    array=cmp_sorted.get(f"{primary_col} Std", pd.Series([0]*len(cmp_sorted))).tolist(),
                    color="#4b5563"
                ) if f"{primary_col} Std" in cmp_sorted.columns else None,
            ))
            layout = copy.deepcopy(PLOTLY_LAYOUT)
            layout.update({
                "title": f"Model Comparison — {primary_col} ({cmp_folds}-fold CV)",
                "xaxis_title": primary_col,
                "height": max(300, len(cmp_sorted) * 45 + 80),
            })
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        st.dataframe(cmp_df, use_container_width=True)

        # One-click promote best model
        best_row = cmp_df.dropna(subset=[primary_col]).iloc[0]
        best_name = best_row["Model"]
        best_score = best_row[primary_col]
        st.markdown(f'<div class="ml-success">🏆 Best model: <strong>{best_name}</strong> ({primary_col} = {best_score})</div>', unsafe_allow_html=True)

        if st.button(f"Train & set {best_name} as active model", type="primary", key="promote_best"):
            from utils.model_trainer import train_tabular
            registry2 = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
            M = registry2[best_name]
            hp = {}
            if best_name == "CatBoost": hp = {"verbose": 0}
            if best_name == "Logistic Regression": hp = {"max_iter": 500}
            try:
                results = train_tabular(X, y, best_name, task_type, hp, test_size=0.2, random_state=42)
                st.session_state.update({
                    "trained_model": results["model"],
                    "model_results": results,
                    "trained_algo": best_name,
                })
                st.markdown(f'<div class="ml-success">✓ {best_name} is now your active model.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ {e}</div>', unsafe_allow_html=True)
