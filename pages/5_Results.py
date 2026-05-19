import streamlit as st
import pandas as pd
import numpy as np
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, PLOTLY_LAYOUT, COLORS
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Results · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("05", "Results", "Evaluate your trained model.")

results = st.session_state.get("model_results")
task_type = st.session_state.get("task_type")
algo = st.session_state.get("trained_algo", "Model")

if results is None:
    st.markdown('<div class="ml-warn">⚠ No trained model. Complete <strong>Train Model</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

def pl(fig, height=None):
    layout = copy.deepcopy(PLOTLY_LAYOUT)
    if height:
        layout["height"] = height
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f'<span class="ml-badge ml-badge-accent">🤖 {algo}</span> <span class="ml-badge">{task_type}</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

metrics = results["metrics"]
METRIC_COLORS = {
    "Accuracy":"#10b981","F1 Score":"#6366f1","Precision":"#06b6d4","Recall":"#f59e0b",
    "R² Score":"#10b981","MAE":"#f59e0b","MSE":"#ef4444","RMSE":"#ef4444",
}
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    clr = METRIC_COLORS.get(k, "#6366f1")
    col.markdown(f'<div class="ml-metric"><div class="ml-metric-value" style="color:{clr}">{v}</div><div class="ml-metric-label">{k}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

y_test = results.get("y_test")
y_pred = results.get("y_pred")

# ══════════════════════════════════════════════════════════════════════════════
# TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
if task_type == "Time Series":
    dates_test = results.get("dates_test")
    df_full = results.get("df_full")
    date_col = results.get("date_col")
    target_col = results.get("target_col")

    tabs = st.tabs(["Forecast Plot", "Full History", "Feature Importance", "Export"])

    with tabs[0]:
        section("Test set — actual vs forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, name="Actual",
                                 line=dict(color=COLORS["accent"], width=2)))
        fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, name="Predicted",
                                 line=dict(color=COLORS["primary"], width=2, dash="dot")))
        fig.update_layout(title="Actual vs Predicted (Test Set)", xaxis_title="Index", yaxis_title=target_col)
        pl(fig, 380)

        residuals = y_test - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=list(range(len(residuals))), y=residuals,
                              marker_color=[COLORS["error"] if r < 0 else COLORS["success"] for r in residuals]))
        fig2.add_hline(y=0, line_color="#4b5563", line_dash="dash")
        fig2.update_layout(title="Residuals", xaxis_title="Index", yaxis_title="Error")
        pl(fig2, 280)

    with tabs[1]:
        section("Full time series")
        if df_full is not None and date_col in df_full.columns:
            fig3 = px.line(df_full, x=date_col, y=target_col, color_discrete_sequence=[COLORS["accent"]],
                           title=f"{target_col} over time")
            pl(fig3, 350)

    with tabs[2]:
        fi = results.get("feature_importances")
        if fi:
            fi_df = pd.DataFrame({"Feature": list(fi.keys()), "Importance": list(fi.values())})
            fi_df = fi_df.sort_values("Importance", ascending=True)
            fig4 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale=["#1e1e2e","#6366f1"],
                          title="Feature Importance")
            pl(fig4, 350)

    with tabs[3]:
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Error": y_test - y_pred})
        st.download_button("⬇ Download predictions CSV", pred_df.to_csv(index=False).encode(), "ts_predictions.csv", "text/csv")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# NLP CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
if task_type == "NLP Classification":
    tabs = st.tabs(["Confusion Matrix", "Classification Report", "Top Features", "Sample Predictions", "Export"])

    with tabs[0]:
        cm = results.get("confusion_matrix")
        le = results.get("label_encoder")
        labels = [str(c) for c in (le.classes_ if le else sorted(set(y_test)))]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        color_continuous_scale="Blues", title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"))
        pl(fig, 450)

    with tabs[1]:
        st.code(results.get("classification_report",""), language=None)

    with tabs[2]:
        if "top_features_per_class" in results:
            for cls, feats in results["top_features_per_class"].items():
                le = results.get("label_encoder")
                try:
                    cls_label = le.inverse_transform([int(cls)])[0] if le else cls
                except:
                    cls_label = cls
                st.markdown(f'<div class="ml-section">Class: {cls_label}</div>', unsafe_allow_html=True)
                feat_df = pd.DataFrame(feats, columns=["Word", "Weight"])
                fig = px.bar(feat_df.head(15), x="Weight", y="Word", orientation="h",
                             color_discrete_sequence=[COLORS["primary"]])
                pl(fig, 280)
        elif "top_features" in results:
            feat_df = pd.DataFrame(results["top_features"], columns=["Word","Importance"])
            fig = px.bar(feat_df.head(20), x="Importance", y="Word", orientation="h",
                         color_discrete_sequence=[COLORS["primary"]], title="Top Predictive Words")
            pl(fig, 380)

    with tabs[3]:
        X_test_txt = results.get("X_test_txt", [])
        le = results.get("label_encoder")
        try:
            actual_labels = le.inverse_transform(y_test) if le else y_test
            pred_labels = le.inverse_transform(y_pred) if le else y_pred
        except:
            actual_labels, pred_labels = y_test, y_pred
        sample_df = pd.DataFrame({"Text": X_test_txt[:50], "Actual": actual_labels[:50], "Predicted": pred_labels[:50]})
        sample_df["Correct"] = sample_df["Actual"] == sample_df["Predicted"]
        st.dataframe(sample_df, use_container_width=True, height=350)

    with tabs[4]:
        pred_df = pd.DataFrame({"Text": results.get("X_test_txt",[]), "Actual": y_test, "Predicted": y_pred})
        st.download_button("⬇ Download predictions CSV", pred_df.to_csv(index=False).encode(), "nlp_predictions.csv", "text/csv")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABULAR — REGRESSION / CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
tab_list = ["Predictions"]
if task_type == "Classification":
    tab_list += ["Confusion Matrix", "Report"]
else:
    tab_list += ["Residuals", "Actual vs Predicted"]
if results.get("feature_importances") is not None:
    tab_list.append("Feature Importance")
tab_list.append("Export")

tabs = st.tabs(tab_list)
tab_map = {name: tab for name, tab in zip(tab_list, tabs)}

with tab_map["Predictions"]:
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    if task_type == "Classification":
        pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]
    else:
        pred_df["Error"] = (pred_df["Actual"] - pred_df["Predicted"]).round(4)
        pred_df["Abs Error"] = pred_df["Error"].abs().round(4)
    st.dataframe(pred_df.head(100), use_container_width=True, height=350)
    st.markdown(f'<div style="font-size:0.75rem;color:#4b5563;">Showing first 100 of {len(y_test):,} test samples</div>', unsafe_allow_html=True)

if task_type == "Classification":
    with tab_map["Confusion Matrix"]:
        cm = results.get("confusion_matrix")
        labels = sorted(list(set(y_test)))
        fig = px.imshow(cm, text_auto=True, x=[str(l) for l in labels], y=[str(l) for l in labels],
                        color_continuous_scale="Blues", title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"))
        pl(fig, 450)

    with tab_map["Report"]:
        st.code(results.get("classification_report",""), language=None)
else:
    with tab_map["Residuals"]:
        residuals = y_test - y_pred
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(residuals, nbins=40, color_discrete_sequence=[COLORS["primary"]],
                               title="Residual Distribution")
            fig.add_vline(x=0, line_dash="dash", line_color=COLORS["error"])
            pl(fig, 320)
        with c2:
            fig2 = px.scatter(x=y_pred, y=residuals, color_discrete_sequence=[COLORS["accent"]],
                              title="Residuals vs Fitted",
                              labels={"x":"Fitted","y":"Residuals"})
            fig2.add_hline(y=0, line_dash="dash", line_color=COLORS["error"])
            pl(fig2, 320)

    with tab_map["Actual vs Predicted"]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                 marker=dict(color=COLORS["primary"], size=5, opacity=0.6), name="Predictions"))
        mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                 line=dict(color=COLORS["error"], dash="dash", width=1.5), name="Perfect"))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        pl(fig, 420)

if "Feature Importance" in tab_map:
    with tab_map["Feature Importance"]:
        fi = results["feature_importances"]
        feature_names = st.session_state.get("feature_names", [])
        n = min(len(fi), len(feature_names))
        fi_df = pd.DataFrame({"Feature": feature_names[:n], "Importance": fi[:n]})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(25)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#1e1e2e","#6366f1"],
                     title="Top Feature Importances")
        pl(fig, max(300, len(fi_df)*20))
        st.dataframe(fi_df.sort_values("Importance", ascending=False).reset_index(drop=True), use_container_width=True)

with tab_map["Export"]:
    pred_export = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.download_button("⬇ Download predictions CSV", pred_export.to_csv(index=False).encode(), "predictions.csv", "text/csv")
