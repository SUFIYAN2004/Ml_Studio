import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

# ── Regression ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# ── Classification ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB

# ── NLP ─────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression,
    "Ridge Regression": Ridge,
    "Lasso Regression": Lasso,
    "ElasticNet": ElasticNet,
    "Decision Tree": DecisionTreeRegressor,
    "Random Forest": RandomForestRegressor,
    "Extra Trees": ExtraTreesRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "SVM (SVR)": SVR,
    "KNN": KNeighborsRegressor,
    "Neural Network (MLP)": MLPRegressor,
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Extra Trees": ExtraTreesClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "SVM (SVC)": SVC,
    "KNN": KNeighborsClassifier,
    "Neural Network (MLP)": MLPClassifier,
    "Naive Bayes (Multinomial)": MultinomialNB,
    "Naive Bayes (Complement)": ComplementNB,
}

NLP_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Naive Bayes (Multinomial)": MultinomialNB,
    "Naive Bayes (Complement)": ComplementNB,
    "Random Forest": RandomForestClassifier,
    "SVM (Linear)": SVC,
    "Gradient Boosting": GradientBoostingClassifier,
}

# Try optional heavy libs
try:
    from xgboost import XGBClassifier, XGBRegressor
    CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier
    REGRESSION_MODELS["XGBoost"] = XGBRegressor
    NLP_MODELS["XGBoost"] = XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CLASSIFICATION_MODELS["CatBoost"] = CatBoostClassifier
    REGRESSION_MODELS["CatBoost"] = CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    CLASSIFICATION_MODELS["LightGBM"] = LGBMClassifier
    REGRESSION_MODELS["LightGBM"] = LGBMRegressor
    NLP_MODELS["LightGBM"] = LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def _reg_metrics(y_true, y_pred):
    return {
        "R² Score": round(r2_score(y_true, y_pred), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MSE": round(mean_squared_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }


def _cls_metrics(y_true, y_pred):
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }


def train_tabular(X, y, model_name, task_type, hyperparams, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "Classification" and len(np.unique(y)) < 50 else None
    )
    registry = REGRESSION_MODELS if task_type == "Regression" else CLASSIFICATION_MODELS
    ModelClass = registry[model_name]
    hp = {k: v for k, v in hyperparams.items() if v is not None}
    if model_name in ("CatBoost", "CatBoost (Regressor)"):
        hp["verbose"] = 0
    if model_name == "SVM (SVC)":
        hp.setdefault("probability", True)

    model = ModelClass(**hp)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {"model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "task_type": task_type}
    if task_type == "Regression":
        result["metrics"] = _reg_metrics(y_test, y_pred)
    else:
        result["metrics"] = _cls_metrics(y_test, y_pred)
        result["confusion_matrix"] = confusion_matrix(y_test, y_pred)
        result["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

    if hasattr(model, "feature_importances_"):
        result["feature_importances"] = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        result["feature_importances"] = np.abs(coef).flatten()

    return result


def train_nlp(texts, labels, model_name, vectorizer_type, hyperparams, test_size=0.2, random_state=42, ngram_max=2, max_features=10000):
    """
    NLP pipeline: vectorize text → train classifier.
    Returns results dict with pipeline, metrics, top features.
    """
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state,
        stratify=labels if len(np.unique(labels)) < 50 else None
    )

    VecClass = TfidfVectorizer if vectorizer_type == "TF-IDF" else CountVectorizer
    vectorizer = VecClass(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        strip_accents="unicode",
        analyzer="word",
        stop_words="english",
    )

    ModelClass = NLP_MODELS[model_name]
    hp = {k: v for k, v in hyperparams.items() if v is not None}

    # SVM linear kernel works well for NLP
    if model_name == "SVM (Linear)":
        hp["kernel"] = "linear"
        hp.setdefault("C", 1.0)

    # Naive Bayes cannot take negative features from TF-IDF if min_df not set; fine with default
    model = ModelClass(**hp)

    # Fit
    X_train_vec = vectorizer.fit_transform(X_train_txt)
    X_test_vec = vectorizer.transform(X_test_txt)

    # NB needs non-negative features; if SVD was used with negatives, clip
    if model_name.startswith("Naive Bayes"):
        X_train_vec_arr = X_train_vec
        X_test_vec_arr = X_test_vec
    else:
        X_train_vec_arr = X_train_vec
        X_test_vec_arr = X_test_vec

    model.fit(X_train_vec_arr, y_train)
    y_pred = model.predict(X_test_vec_arr)

    pipeline = Pipeline([("vectorizer", vectorizer), ("model", model)])

    result = {
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "model": model,
        "X_test_vec": X_test_vec,
        "X_test_txt": list(X_test_txt),
        "y_test": y_test,
        "y_pred": y_pred,
        "task_type": "NLP Classification",
    }
    result["metrics"] = _cls_metrics(y_test, y_pred)
    result["confusion_matrix"] = confusion_matrix(y_test, y_pred)
    result["classification_report"] = classification_report(y_test, y_pred, zero_division=0)

    # Top features per class
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim == 1:
                top_idx = np.argsort(np.abs(coef))[-20:][::-1]
                result["top_features"] = list(zip(feature_names[top_idx], coef[top_idx]))
            else:
                result["top_features_per_class"] = {}
                for i, cls in enumerate(model.classes_):
                    top_idx = np.argsort(coef[i])[-15:][::-1]
                    result["top_features_per_class"][str(cls)] = list(zip(feature_names[top_idx], coef[i][top_idx]))
        elif hasattr(model, "feature_importances_"):
            top_idx = np.argsort(model.feature_importances_)[-20:][::-1]
            result["top_features"] = list(zip(feature_names[top_idx], model.feature_importances_[top_idx]))
    except Exception:
        pass

    return result


def train_timeseries(df_ts, date_col, target_col, model_name, hyperparams,
                     lag_features=5, test_size=0.2, random_state=42):
    """
    Time-series regression using lag features.
    Creates lag_1..lag_N features from the target column.
    """
    df = df_ts[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[target_col], inplace=True)

    # Create lag features
    for lag in range(1, lag_features + 1):
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    # Rolling stats
    df["rolling_mean_3"] = df[target_col].shift(1).rolling(3).mean()
    df["rolling_std_3"] = df[target_col].shift(1).rolling(3).std()
    df["rolling_mean_7"] = df[target_col].shift(1).rolling(7).mean()

    df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c not in [date_col, target_col]]

    X = df[feature_cols].values
    y = df[target_col].values
    dates = df[date_col].values

    # Time-aware split (no shuffle)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    registry = REGRESSION_MODELS
    ModelClass = registry[model_name]
    hp = {k: v for k, v in hyperparams.items() if v is not None}
    if model_name == "CatBoost":
        hp["verbose"] = 0

    model = ModelClass(**hp)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "dates_test": dates_test,
        "feature_cols": feature_cols,
        "task_type": "Time Series",
        "df_full": df,
        "date_col": date_col,
        "target_col": target_col,
    }
    result["metrics"] = _reg_metrics(y_test, y_pred)

    if hasattr(model, "feature_importances_"):
        result["feature_importances"] = dict(zip(feature_cols, model.feature_importances_))
    elif hasattr(model, "coef_"):
        result["feature_importances"] = dict(zip(feature_cols, np.abs(model.coef_).flatten()))

    return result