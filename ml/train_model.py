from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "high_diamond_ranked_10min.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
BACKEND_MODEL_DIR = PROJECT_ROOT / "api" / "src" / "model"
TARGET_COLUMN = "blueWins"
DROP_COLUMNS = ["gameId"]
SELECTED_FEATURES = [
    "blueKills",
    "redKills",
    "blueGoldDiff",
    "blueExperienceDiff",
    "blueDragons",
    "redDragons",
    "blueHeralds",
    "redHeralds",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Kaggle League of Legends dataset."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Download high_diamond_ranked_10min.csv "
            "from Kaggle and place it in the project root."
        )
    return pd.read_csv(data_path)


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select MVP features and target, after validating the expected schema."""
    required_columns = [TARGET_COLUMN, *DROP_COLUMNS, *SELECTED_FEATURES]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data[SELECTED_FEATURES].isnull().any().any():
        raise ValueError("Missing values found in selected features.")

    # gameId is intentionally dropped because it is an identifier, not a predictive feature.
    features = data.drop(columns=[TARGET_COLUMN, *DROP_COLUMNS])[SELECTED_FEATURES]
    target = data[TARGET_COLUMN]
    return features, target


def build_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Create model pipelines and hyperparameter grids.

    StandardScaler performs feature standardization for distance- and margin-based
    models, which is the appropriate normalization strategy for KNN and SVM here.
    Tree and GaussianNB models keep the original numeric scale.
    """
    return {
        "KNeighborsClassifier": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": [5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        "DecisionTreeClassifier": (
            Pipeline(
                steps=[
                    ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 5, 7, 10, None],
                "model__min_samples_split": [2, 10, 20],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "GaussianNB": (
            Pipeline(steps=[("model", GaussianNB())]),
            {
                "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        ),
        "SVC": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", SVC(random_state=RANDOM_STATE)),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        ),
    }


def evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Run GridSearchCV, compare all models, and return the best estimator."""
    model_specs = build_models()
    comparison_rows: list[dict[str, Any]] = []
    best_estimator: BaseEstimator | None = None
    best_model_name = ""
    best_score = -1.0

    for model_name, (pipeline, param_grid) in model_specs.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)

        row = {
            "model": model_name,
            "cv_best_f1": float(search.best_score_),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
            "best_params": search.best_params_,
        }
        comparison_rows.append(row)

        if row["test_f1"] > best_score:
            best_score = row["test_f1"]
            best_model_name = model_name
            best_estimator = search.best_estimator_

    if best_estimator is None:
        raise RuntimeError("No model was trained.")

    best_predictions = best_estimator.predict(x_test)
    metrics = {
        "selected_features": SELECTED_FEATURES,
        "best_model": best_model_name,
        "accuracy": float(accuracy_score(y_test, best_predictions)),
        "precision": float(precision_score(y_test, best_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, best_predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, best_predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, best_predictions).tolist(),
        "comparison": comparison_rows,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": 5,
    }

    comparison = pd.DataFrame(comparison_rows).sort_values(
        by=["test_f1", "test_accuracy"], ascending=False
    )
    print("\nModel comparison table:")
    print(
        comparison[
            [
                "model",
                "cv_best_f1",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "best_params",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model: {best_model_name}")

    return best_estimator, metrics


def export_artifacts(model: BaseEstimator, metrics: dict[str, Any]) -> None:
    """Persist the model, selected feature list, and metrics for ML and backend use."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "lol_model.pkl"
    selected_features_path = ARTIFACTS_DIR / "selected_features.json"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(model, model_path)
    selected_features_path.write_text(json.dumps(SELECTED_FEATURES, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    shutil.copy2(model_path, BACKEND_MODEL_DIR / "lol_model.pkl")
    shutil.copy2(selected_features_path, BACKEND_MODEL_DIR / "selected_features.json")


def main() -> None:
    data = load_data()
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    best_model, metrics = evaluate_models(x_train, x_test, y_train, y_test)
    export_artifacts(best_model, metrics)


if __name__ == "__main__":
    main()
