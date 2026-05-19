import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assets.theme import GLOBAL_CSS, page_header, section, info_box
from utils.data_handler import smart_load_csv

st.set_page_config(page_title="Test · ML Studio", page_icon="⬡", layout="wide")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
page_header("06", "Test Model", "Run predictions on new data — single sample or batch upload.")

model = st.session_state.get("trained_model")
task_type = st.session_state.get("task_type","Classification")
algo = st.session_state.get("trained_algo","Model")
results = st.session_state.get("model_results")
feature_names = st.session_state.get("feature_names", [])
target = st.session_state.get("target")
target_encoder = st.session_state.get("target_encoder")

if model is None:
    st.markdown('<div class="ml-warn">⚠ No trained model found. Complete <strong>Train Model</strong> first.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(f'<span class="ml-badge ml-badge-accent">🤖 {algo}</span> <span class="ml-badge">{task_type}</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

mode = st.radio("Prediction mode", ["Single sample", "Batch CSV"], horizontal=True, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# NLP — text input
# ══════════════════════════════════════════════════════════════════════════════
if task_type == "NLP Classification":
    section("Input text")
    if mode == "Single sample":
        user_text = st.text_area("Enter text to classify", height=120, placeholder="Type or paste text here...")
        if st.button("Predict", type="primary") and user_text.strip():
            try:
                pred = model.predict([user_text])[0]
                le = target_encoder or results.get("label_encoder") or results.get("target_encoder")
                if le is not None:
                    try:
                        label = le.inverse_transform([pred])[0]
                    except:
                        label = pred
                else:
                    label = pred

                st.markdown(f"""
                <div class="ml-card" style="text-align:center; padding: 2rem;">
                    <div style="font-size:0.72rem; color:#6b7280; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.5rem;">Predicted class</div>
                    <div style="font-size:2.5rem; font-weight:700; color:#6366f1; letter-spacing:-0.02em;">{label}</div>
                </div>""", unsafe_allow_html=True)

                # Probability if available
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([user_text])[0]
                    classes = le.classes_ if le else list(range(len(probs)))
                    prob_df = pd.DataFrame({"Class": classes, "Probability": probs}).sort_values("Probability", ascending=False)
                    import plotly.express as px, copy
                    from assets.theme import PLOTLY_LAYOUT, COLORS
                    fig = px.bar(prob_df, x="Class", y="Probability",
                                 color_discrete_sequence=[COLORS["primary"]], title="Class Probabilities")
                    fig.update_layout(**copy.deepcopy(PLOTLY_LAYOUT))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ {e}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ml-info">Upload a CSV with a column containing the texts to classify.</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload batch CSV", type=["csv"], label_visibility="collapsed")
        text_col = st.session_state.get("text_col","text")
        col_name = st.text_input("Text column name in your CSV", value=text_col)
        if batch_file and st.button("Predict batch", type="primary"):
            df_batch, _ = smart_load_csv(batch_file)
            if col_name not in df_batch.columns:
                st.markdown(f'<div class="ml-error">Column <code>{col_name}</code> not found. Found: {list(df_batch.columns)}</div>', unsafe_allow_html=True)
            else:
                texts = df_batch[col_name].fillna("").astype(str).tolist()
                preds = model.predict(texts)
                le = target_encoder or font_results.get("label_encoder")
                if le is not None:
                    try:
                        labels = le.inverse_transform(preds)
                    except:
                        labels = preds
                else:
                    labels = preds
                df_batch["Predicted"] = labels
                st.dataframe(df_batch, use_container_width=True, height=350)
                st.download_button("⬇ Download predictions", df_batch.to_csv(index=False).encode(), "nlp_batch_predictions.csv")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABULAR (Regression / Classification / Time Series)
# ══════════════════════════════════════════════════════════════════════════════
df_orig = st.session_state.get("df")
label_encoders = st.session_state.get("label_encoders", {})
scaler = st.session_state.get("scaler_obj")

if not feature_names:
    st.markdown('<div class="ml-warn">⚠ Feature names not set. Re-run Preprocess.</div>', unsafe_allow_html=True)
    st.stop()

# ── RECONSTRUCT TIME SERIES LAGS TO PREVENT KEYERROR ───────────────────────────
if task_type == "Time Series" and df_orig is not None:
    df_orig = df_orig.copy()
    ts_target = st.session_state.get("ts_target") or target
    
    if ts_target in df_orig.columns:
        for feat in feature_names:
            if feat not in df_orig.columns:
                if feat.startswith("lag_"):
                    try:
                        lag_val = int(feat.split("_")[1])
                        df_orig[feat] = df_orig[ts_target].shift(lag_val)
                    except ValueError: pass
                elif feat.startswith("rolling_mean_"):
                    try:
                        window = int(feat.split("_")[2])
                        df_orig[feat] = df_orig[ts_target].rolling(window=window).mean()
                    except ValueError: pass
                elif feat.startswith("rolling_std_"):
                    try:
                        window = int(feat.split("_")[2])
                        df_orig[feat] = df_orig[ts_target].rolling(window=window).std()
                    except ValueError: pass

def preprocess_input(df_in: pd.DataFrame) -> np.ndarray:
    """Apply same preprocessing to new data."""
    from utils.preprocessor import apply_imputer
    df_in = df_in.copy()
    
    missing_cols = [c for c in feature_names if c not in df_in.columns]
    for mc in missing_cols:
        df_in[mc] = np.nan

    X = df_in[feature_names].copy()

    # Impute
    X = apply_imputer(X, strategy=st.session_state.get("imputer_strategy","mean"))

    # Encode categoricals using stored label encoders
    for col, le in label_encoders.items():
        if col in X.columns:
            try:
                X[col] = le.transform(X[col].astype(str))
            except Exception:
                X[col] = 0

    # Handle columns that became object but have no encoder (one-hot case)
    for col in X.select_dtypes(exclude="number").columns:
        X[col] = 0

    # Scale
    if scaler is not None:
        X_arr = scaler.transform(X.values)
    else:
        X_arr = X.values

    return X_arr


if mode == "Single sample":
    section(f"Enter feature values for prediction")
    st.markdown('<div class="ml-info" style="margin-bottom:1rem;">Fill in the feature values below. Leave blank to use the column median/mode.</div>', unsafe_allow_html=True)

    if df_orig is not None:
        input_vals = {}
        
        # Safely filter features that actually exist in df_orig now
        available_features = [f for f in feature_names if f in df_orig.columns]
        num_cols = df_orig[available_features].select_dtypes(include="number").columns.tolist() if feature_names else []

        # Render inputs in a 3-column grid
        n = len(feature_names)
        ncols = 3
        rows = [feature_names[i:i+ncols] for i in range(0, n, ncols)]

        for row in rows:
            cols = st.columns(ncols)
            for col, feat in zip(cols, row):
                with col:
                    if feat in df_orig.columns:
                        has_data = df_orig[feat].notna().any()
                        is_numeric = feat in num_cols
                        
                        # ── FIX: TYPE-SAFE BOUNDARY CHECKING ──────────────────────────
                        if is_numeric:
                            med = float(df_orig[feat].median()) if has_data else 0.0
                            mn = float(df_orig[feat].min()) if has_data else -10000.0
                            mx = float(df_orig[feat].max()) if has_data else 10000.0
                            
                            is_int_feature = (
                                pd.api.types.is_integer_dtype(df_orig[feat]) or 
                                (has_data and np.all(df_orig[feat].dropna() % 1 == 0))
                            )
                        else:
                            # It's string/object categorical data (like Fuel type)
                            med = str(df_orig[feat].mode()[0]) if has_data else ""
                            mn, mx = None, None
                            is_int_feature = False
                    else:
                        med, mn, mx = 0.0, -10000.0, 10000.0
                        is_numeric = True
                        is_int_feature = False

                    if is_numeric or feat not in label_encoders:
                        # Numeric entry rendering logic
                        if is_int_feature:
                            val = st.number_input(feat, value=int(med), min_value=int(mn), max_value=int(mx), step=1, key=f"inp_{feat}", format="%d")
                        else:
                            val = st.number_input(feat, value=float(med), min_value=float(mn), max_value=float(mx), key=f"inp_{feat}", format="%.4f")
                        input_vals[feat] = val
                    else:
                        # Categorical selectbox fallback parsing
                        uniq = df_orig[feat].dropna().unique().tolist()[:50] if feat in df_orig.columns else [""]
                        # Pre-select matching structural default safely
                        default_idx = uniq.index(med) if med in uniq else 0
                        sel = st.selectbox(feat, options=uniq, index=default_idx, key=f"inp_{feat}")
                        input_vals[feat] = sel

        if st.button("Predict", type="primary"):
            try:
                inp_df = pd.DataFrame([input_vals])
                X_in = preprocess_input(inp_df)
                pred = model.predict(X_in)[0]

                if task_type == "Regression" or task_type == "Time Series":
                    st.markdown(f"""
                    <div class="ml-card" style="text-align:center; padding:2rem;">
                        <div style="font-size:0.72rem;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Predicted value</div>
                        <div style="font-size:3rem;font-weight:700;color:#6366f1;letter-spacing:-0.04em;">{pred:.4f}</div>
                        <div style="font-size:0.78rem;color:#4b5563;margin-top:0.5rem;">Target: {target}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    label = pred
                    if target_encoder is not None:
                        try:
                            label = target_encoder.inverse_transform([int(pred)])[0]
                        except:
                            label = pred
                    st.markdown(f"""
                    <div class="ml-card" style="text-align:center; padding:2rem;">
                        <div style="font-size:0.72rem;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;">Predicted class</div>
                        <div style="font-size:3rem;font-weight:700;color:#6366f1;letter-spacing:-0.04em;">{label}</div>
                    </div>""", unsafe_allow_html=True)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_in)[0]
                        labels_all = list(range(len(probs)))
                        if target_encoder:
                            try:
                                labels_all = target_encoder.inverse_transform(labels_all)
                            except:
                                pass
                        prob_df = pd.DataFrame({"Class": labels_all, "Probability": probs}).sort_values("Probability", ascending=False)
                        import plotly.express as px, copy
                        from assets.theme import PLOTLY_LAYOUT, COLORS
                        fig = px.bar(prob_df, x="Class", y="Probability",
                                     color_discrete_sequence=[COLORS["primary"]])
                        fig.update_layout(**copy.deepcopy(PLOTLY_LAYOUT))
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ Prediction error: {e}</div>', unsafe_allow_html=True)
                st.exception(e)
    else:
        st.markdown('<div class="ml-error">Original dataset not found in session.</div>', unsafe_allow_html=True)

else:  # Batch CSV
    section("Batch CSV prediction")
    st.markdown('<div class="ml-info">Upload a CSV with the same feature columns used during training. Target column is optional and will be used for accuracy comparison if present.</div>', unsafe_allow_html=True)

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], label_visibility="collapsed")
    if batch_file:
        df_batch, info = smart_load_csv(batch_file)
        if info.get("issues"):
            for iss in info["issues"]:
                st.markdown(f'<div class="ml-warn">⚠ {iss}</div>', unsafe_allow_html=True)

        st.markdown(f'<span class="ml-badge">Shape: {df_batch.shape[0]} × {df_batch.shape[1]}</span>', unsafe_allow_html=True)
        st.dataframe(df_batch.head(5), use_container_width=True)

        missing = [f for f in feature_names if f not in df_batch.columns]
        if missing:
            st.markdown(f'<div class="ml-warn">⚠ Missing features in CSV: <code>{", ".join(missing)}</code><br>These will be filled automatically via model imputation rules (or defaults).</div>', unsafe_allow_html=True)

        if st.button("Run batch prediction", type="primary"):
            try:
                # Reconstruct time series attributes for uploaded batch if target is provided
                if task_type == "Time Series" and target in df_batch.columns:
                    for feat in feature_names:
                        if feat not in df_batch.columns:
                            if feat.startswith("lag_"):
                                try:
                                    lag_val = int(feat.split("_")[1])
                                    df_batch[feat] = df_batch[target].shift(lag_val)
                                except: pass
                            elif feat.startswith("rolling_mean_"):
                                try:
                                    window = int(feat.split("_")[2])
                                    df_batch[feat] = df_batch[target].rolling(window=window).mean()
                                except: pass
                            elif feat.startswith("rolling_std_"):
                                try:
                                    window = int(feat.split("_")[2])
                                    df_batch[feat] = df_batch[target].rolling(window=window).std()
                                except: pass

                X_batch = preprocess_input(df_batch)
                preds = model.predict(X_batch)

                if task_type == "Classification":
                    if target_encoder is not None:
                        try:
                            preds_labels = target_encoder.inverse_transform(preds.astype(int))
                        except:
                            preds_labels = preds
                    else:
                        preds_labels = preds
                    df_batch["Predicted"] = preds_labels
                else:
                    df_batch["Predicted"] = preds

                # If target col present, compute metrics
                if target in df_batch.columns:
                    valid_mask = df_batch[target].notna()
                    if task_type == "Regression" or task_type == "Time Series":
                        from sklearn.metrics import mean_absolute_error, r2_score
                        y_true = pd.to_numeric(df_batch.loc[valid_mask, target], errors="coerce").dropna()
                        y_hat = preds[y_true.index]
                        st.markdown(f'<div class="ml-success">MAE: {mean_absolute_error(y_true,y_hat):.4f} · R²: {r2_score(y_true,y_hat):.4f}</div>', unsafe_allow_html=True)
                    else:
                        from sklearn.metrics import accuracy_score
                        try:
                            y_true_enc = target_encoder.transform(df_batch.loc[valid_mask, target].astype(str)) if target_encoder else df_batch.loc[valid_mask, target]
                            acc = accuracy_score(y_true_enc, preds[y_true_enc.index])
                            st.markdown(f'<div class="ml-success">Accuracy on batch: {acc:.4f}</div>', unsafe_allow_html=True)
                        except:
                            pass

                st.dataframe(df_batch, use_container_width=True, height=400)
                st.download_button("⬇ Download results CSV", df_batch.to_csv(index=False).encode(), "batch_predictions.csv", "text/csv")
            except Exception as e:
                st.markdown(f'<div class="ml-error">❌ {e}</div>', unsafe_allow_html=True)
                st.exception(e)