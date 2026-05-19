"""
hypertune.py
Hyperparameter tuning (GridSearchCV, RandomizedSearchCV) +
Cross-validation + Multi-model comparison utilities.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)


# ── Param grids ────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0],
        "min_samples_split": [2, 5],
    },
    "Decision Tree": {
        "max_depth": [None, 3, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],  # classification only; regressor uses mse/friedman
    },
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"],
        "max_iter": [500, 1000, 2000],
    },
    "SVM (SVC)": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "SVM (SVR)": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean", "manhattan"],
    },
    "Ridge Regression": {"alpha": [0.01, 0.1, 1, 10, 100, 500, 1000]},
    "Lasso Regression": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "Extra Trees": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 5, 10],
        "num_leaves": [31, 63, 127],
    },
}

SCORING_MAP = {
    "Classification": {
        "Accuracy": make_scorer(accuracy_score),
        "F1 (weighted)": make_scorer(f1_score, average="weighted", zero_division=0),
    },
    "Regression": {
        "R²": make_scorer(r2_score),
        "MAE (neg)": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE (neg)": make_scorer(
            lambda y, p: -np.sqrt(mean_squared_error(y, p)), greater_is_better=True
        ),
    },
}


def get_cv(task_type: str, n_splits: int = 5, random_state: int = 42):
    if task_type == "Classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def run_cross_validation(model, X, y, task_type: str, n_splits: int = 5, random_state: int = 42):
    """
    Run cross-validation and return per-fold and mean scores.
    Returns a dict: {metric_name: {"scores": [...], "mean": float, "std": float}}
    """
    cv = get_cv(task_type, n_splits, random_state)
    results = {}

    scoring = SCORING_MAP.get(task_type, SCORING_MAP["Classification"])
    primary_scorer_name = list(scoring.keys())[0]

    for metric_name, scorer in scoring.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
            results[metric_name] = {
                "scores": list(np.round(scores, 4)),
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
            }
        except Exception as e:
            results[metric_name] = {"scores": [], "mean": None, "std": None, "error": str(e)}

    return results


def run_hyperparameter_search(
    ModelClass,
    param_grid: dict,
    X, y,
    task_type: str,
    method: str = "random",   # "grid" | "random"
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = None,
):
    """
    Run GridSearchCV or RandomizedSearchCV.
    Returns (best_model, best_params, cv_results_df, best_score).
    """
    cv = get_cv(task_type, n_splits, random_state)

    if scoring is None:
        scoring = "accuracy" if task_type == "Classification" else "r2"

    common_kwargs = dict(
        estimator=ModelClass,
        param_grid=param_grid if method == "grid" else None,
        param_distributions=param_grid if method == "random" else None,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=0,
    )

    if method == "grid":
        search = GridSearchCV(
            estimator=ModelClass,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=0,
        )
    else:
        search = RandomizedSearchCV(
            estimator=ModelClass,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            random_state=random_state,
            verbose=0,
        )

    search.fit(X, y)

    cv_df = pd.DataFrame(search.cv_results_)
    # Keep useful columns
    keep = ["params","mean_test_score","std_test_score","mean_train_score","rank_test_score"]
    keep = [c for c in keep if c in cv_df.columns]
    cv_df = cv_df[keep].sort_values("rank_test_score").reset_index(drop=True)
    cv_df["mean_test_score"] = cv_df["mean_test_score"].round(4)
    cv_df["std_test_score"] = cv_df["std_test_score"].round(4)

    return search.best_estimator_, search.best_params_, cv_df, round(search.best_score_, 4)


def compare_models(
    model_dict: dict,   # {name: ModelClass(**default_params)}
    X, y,
    task_type: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Quickly cross-validate multiple models and return a comparison DataFrame.
    model_dict: {"Model Name": fitted_or_unfitted_model_instance}
    Returns DataFrame sorted by primary metric.
    """
    cv = get_cv(task_type, n_splits, random_state)
    primary = "accuracy" if task_type == "Classification" else "r2"
    secondary = "f1_weighted" if task_type == "Classification" else "neg_mean_absolute_error"

    rows = []
    for name, model in model_dict.items():
        try:
            s1 = cross_val_score(model, X, y, cv=cv, scoring=primary, n_jobs=-1)
            s2 = cross_val_score(model, X, y, cv=cv, scoring=secondary, n_jobs=-1)
            row = {
                "Model": name,
                "Primary Mean": round(np.mean(s1), 4),
                "Primary Std": round(np.std(s1), 4),
                "Secondary Mean": round(np.mean(s2), 4),
                "Secondary Std": round(np.std(s2), 4),
                "Status": "✓",
            }
        except Exception as e:
            row = {
                "Model": name,
                "Primary Mean": None,
                "Primary Std": None,
                "Secondary Mean": None,
                "Secondary Std": None,
                "Status": f"✗ {str(e)[:40]}",
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    primary_label = "Accuracy" if task_type == "Classification" else "R²"
    secondary_label = "F1 (weighted)" if task_type == "Classification" else "MAE (neg)"
    df.rename(columns={"Primary Mean": primary_label, "Primary Std": f"{primary_label} Std",
                        "Secondary Mean": secondary_label, "Secondary Std": f"{secondary_label} Std"}, inplace=True)
    df = df.sort_values(primary_label, ascending=False).reset_index(drop=True)
    return df
