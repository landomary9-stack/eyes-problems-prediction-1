from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


WORKSPACE_DIR = Path(__file__).resolve().parent
MODEL_PATH = WORKSPACE_DIR / "eye_problem_logistic_model.joblib"


def find_dataset_path(explicit_path: str | None = None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset not found at: {candidate}")

    for candidate in [
        WORKSPACE_DIR / "mary lando eye-clinic.xlsx",
        WORKSPACE_DIR / "mary_lando_eye_clinic.xlsx",
        WORKSPACE_DIR / "data" / "mary lando eye-clinic.xlsx",
        WORKSPACE_DIR / "data" / "mary_lando_eye_clinic.xlsx",
        WORKSPACE_DIR / "Eye_Problem_Logistic_Regression_10_Steps.ipynb",
    ]:
        if candidate.exists():
            if candidate.suffix.lower() in {".xlsx", ".xls", ".csv", ".parquet"}:
                return candidate

    raise FileNotFoundError(
        "No dataset file was found in the project folder. "
        "Place your Excel file in the same directory as this script and name it like 'mary lando eye-clinic.xlsx'."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path)
    return data


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")].copy()
    data = data.rename(columns={"Screen time ": "Screen time"})
    data = data.dropna().copy()
    data = data.drop_duplicates().copy()

    normal_labels = {"normal", "normal eye"}
    data["Eye_Problem"] = (
        ~data["Diagnosis"].astype(str).str.strip().str.lower().isin(normal_labels)
    ).astype(int)

    for column in ["Screen time", "Age", "year"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ["Visual Acuity (RE)", "Visual Acuity (LE)"]:
        data[column] = data[column].astype(str).str.strip()

    data = data.dropna(subset=["Age", "Screen time", "year", "Eye_Problem"]).copy()

    features = [
        "Age",
        "Sex",
        "year",
        "Landmark / Village / Estate",
        "Visual Acuity (RE)",
        "Visual Acuity (LE)",
        "Disability Type",
        "Screen time",
    ]

    X = data[features].copy()
    y = data["Eye_Problem"].copy()
    return X, y


def build_model() -> Pipeline:
    categorical_features = [
        "Sex",
        "Landmark / Village / Estate",
        "Visual Acuity (RE)",
        "Visual Acuity (LE)",
    ]
    numeric_features = [
        "Age",
        "year",
        "Disability Type",
        "Screen time",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "logistic_regression",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    print("Classification Report")
    print(classification_report(y_test, y_pred, target_names=["No Eye Problem", "Eye Problem"]))

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the eye problem logistic regression model.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the Excel/CSV data file.")
    args = parser.parse_args()

    dataset_path = find_dataset_path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    data = load_dataset(dataset_path)
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
