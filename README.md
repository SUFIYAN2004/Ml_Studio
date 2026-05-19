# ⬡ ML Studio — Full Edition

End-to-end Machine Learning web app built with Streamlit.
Supports **Regression**, **Classification**, **NLP Classification**, and **Time Series** (classical + deep learning).

---

## 🚀 Setup

```bash
# 1. Unzip and enter project
cd ml_studio_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download NLTK data for full stemming/lemmatization
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# 4. Run
streamlit run app.py
```

---

## 📋 Pages

| # | Page | Purpose |
|---|------|---------|
| Home | `app.py` | Overview, task types, algorithm list |
| 01 | Upload Data | Smart CSV loading — handles double headers, encoding (UTF-8/Latin-1/CP1252), missing headers, auto-separator |
| 02 | Explore Data | head(), info(), describe(), null heatmap, distributions, correlation matrix |
| 03 | Preprocess | Task type, target, drop cols, feature select, SimpleImputer, Label/OneHot encoding, scaler |
| 3b | **Text Cleaner** | 18 NLP cleaning operations as checkboxes — lowercase, lemmatize, stemming (Porter/Lancaster/Snowball), stopwords, HTML, URLs, emojis, regex, presets |
| 04 | Train Model | 15+ algorithms for all 4 task types with hyperparameter controls |
| 4b | **Hypertune & Compare** | Cross-validation, GridSearchCV, RandomizedSearchCV, multi-model comparison with promote-best |
| 4c | **DL Time Series** | LSTM / BiLSTM / GRU with Keras — training curves, future forecast, architecture diagram |
| 05 | Results | Metrics, confusion matrix, residuals, actual vs predicted, feature importance, classification report |
| 06 | Test Model | Single-row form prediction + batch CSV upload with preprocessing pipeline |
| 07 | Save Model | Export as `.pkl` / `.joblib` bundle with encoders, scaler, metadata JSON + load code snippet |

---

## 🤖 Algorithms

### Regression
Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR, KNN, MLP, XGBoost*, CatBoost*, LightGBM*

### Classification
Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVM, KNN, MLP, Naive Bayes (Multinomial + Complement), XGBoost*, CatBoost*, LightGBM*

### NLP Classification
Logistic Regression, Naive Bayes (Multinomial + Complement), SVM (linear), Random Forest, Gradient Boosting, XGBoost*
Vectorizers: TF-IDF, Bag of Words — with n-gram control

### Time Series (Classical)
Linear, Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost*, LightGBM*
Feature engineering: lag features, rolling mean/std

### Time Series (Deep Learning)
LSTM, Bidirectional LSTM, GRU — Keras/TensorFlow
Configurable: layers, units, dropout, learning rate, early stopping, future forecast

*Optional — app works without them, shows install hint.

---

## 🧹 Text Cleaning Operations

| Category | Operations |
|----------|-----------|
| Normalisation | Lowercase, expand contractions, normalize whitespace |
| Web & Social | Remove HTML, URLs, emails, @mentions, #hashtags, emojis |
| Characters | Remove digits, punctuation, special chars |
| Tokens | Remove stopwords, remove short words |
| Morphological | Lemmatization, Porter stemming, Lancaster stemming, Snowball stemming |
| Custom | User-defined regex with replacement |

Presets: Sentiment (Social), Topic Modelling, Spam Detection, Minimal

---

## 🔧 Hypertune Features

- **Cross-validation**: K-fold / Stratified K-fold with per-fold + mean/std metrics
- **Grid Search**: Exhaustive GridSearchCV with combination count estimate
- **Random Search**: RandomizedSearchCV with configurable iterations
- **Model Comparison**: Cross-validate multiple models side-by-side, promote best
- Tuned model can be promoted as the active model with one click

---

## 📦 Loading a saved model

```python
import pickle

with open("your_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
feature_names = bundle["feature_names"]
label_encoders = bundle.get("label_encoders", {})
target_encoder = bundle.get("target_encoder", None)
scaler = bundle.get("scaler", None)

# Prepare DataFrame with same feature columns
import pandas as pd
X_new = pd.DataFrame([{feat: value for feat, value in zip(feature_names, your_values)}])

# Apply label encoding
for col, le in label_encoders.items():
    if col in X_new.columns:
        X_new[col] = le.transform(X_new[col].astype(str))

# Apply scaler
X_arr = scaler.transform(X_new.values) if scaler else X_new.values

# Predict
pred = model.predict(X_arr)
if target_encoder:
    pred = target_encoder.inverse_transform(pred.astype(int))
print(pred)
```

---

## 📝 Notes

- All state is persisted via `st.session_state` across pages
- NLTK fallback: text cleaning works without NLTK downloads using pure Python regex
- TensorFlow: DL Time Series page requires TF; all other pages work without it
- XGBoost, CatBoost, LightGBM are optional — shown as badges if missing
- Boston dataset (double header) is handled automatically by the smart CSV loader
