"""A reproducible employee-attrition benchmark with train-only preprocessing."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from workflow import (SEED, binary_labels, input_metadata, make_splits, numeric_features,
                      require_columns, run_experiment, write_report)

ROOT = Path(__file__).resolve().parent
NUMERIC = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
           "time_spend_company", "Work_accident", "promotion_last_5years"]
CATEGORICAL = ["department", "salary"]
ALIASES = {"sales": "department", "Departments": "department", "Departments ": "department",
           "number_projects": "number_project", "average_monthly_hours": "average_montly_hours",
           "time_spent_company": "time_spend_company", "work_accident": "Work_accident", "Salary": "salary"}


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns=lambda value: ALIASES.get(value.strip(), value.strip()))
    if result.columns.duplicated().any():
        raise ValueError("Ambiguous columns after HR column-name normalization.")
    return result


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = canonical(frame)
    require_columns(frame, [*NUMERIC, *CATEGORICAL])
    X = numeric_features(frame, NUMERIC)
    for name in ["satisfaction_level", "last_evaluation"]:
        if not X[name].dropna().between(0, 1).all():
            raise ValueError(f"{name} must be between 0 and 1.")
    for name in ["Work_accident", "promotion_last_5years"]:
        if not X[name].dropna().isin([0, 1]).all():
            raise ValueError(f"{name} must be 0 or 1.")
    for name in ["number_project", "average_montly_hours", "time_spend_company"]:
        if (X[name].dropna() < 0).any():
            raise ValueError(f"{name} must be non-negative.")
    for name in CATEGORICAL:
        X[name] = frame[name].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).replace("", np.nan)
    return X


def load_data(path: Path):
    frame = canonical(pd.read_csv(path))
    require_columns(frame, ["left", *NUMERIC, *CATEGORICAL])
    frame = frame[[*NUMERIC, *CATEGORICAL, "left"]].copy()
    binary_labels(frame.left, "left")
    initial = len(frame)
    # Exact repeated records cannot occur in both train and test.
    frame = frame.drop_duplicates().reset_index(drop=True)
    conflicts = frame.groupby([*NUMERIC, *CATEGORICAL], dropna=False).left.transform("nunique") > 1
    ambiguous = int(conflicts.sum())
    frame = frame.loc[~conflicts].reset_index(drop=True)
    return prepare_features(frame), binary_labels(frame.left, "left"), {
        "rows_read": initial, "rows_used": len(frame), "duplicates_removed": initial-len(frame)-ambiguous,
        "conflicting_rows_removed": ambiguous, "split_strategy": "stratified 60/20/20 after deduplication"}


def preprocessing():
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("category", make_pipeline(SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CATEGORICAL),
    ], remainder="drop")


def run(data: Path = ROOT / "data/hr.csv", output: Path = ROOT / "results"):
    data, output = Path(data), Path(output)
    X, y, info = load_data(data)
    splits = make_splits(y)
    models = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(preprocessing(), LogisticRegression(max_iter=600, random_state=SEED)),
        "random_forest": make_pipeline(preprocessing(), RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, max_depth=14, n_jobs=2, random_state=SEED)),
    }
    summary, model, _ = run_experiment(X, y, models, output,
        {"title": "Predictive Employee Churn", "class_names": ["Stayed", "Left"], **input_metadata(data), **info}, splits=splits)
    joblib.dump({"model": model, "threshold": summary["selected_threshold"]}, output / "model.joblib")
    importance = permutation_importance(model, X.iloc[splits["validation"]], y.iloc[splits["validation"]],
        n_repeats=3, scoring="average_precision", random_state=SEED, n_jobs=1)
    pd.DataFrame({"feature": X.columns, "validation_ap_decrease": importance.importances_mean,
                  "repeat_std": importance.importances_std}).sort_values("validation_ap_decrease", ascending=False).to_csv(
        output / "feature_importance.csv", index=False, float_format="%.8f")
    training = X.iloc[splits["train"]].copy()
    training["left"] = y.iloc[splits["train"]].to_numpy()
    training.groupby("department").left.agg(["count", "mean"]).rename(columns={"mean": "attrition_rate"}).to_csv(
        output / "department_summary.csv", float_format="%.8f")
    write_report(output, summary,
        "Predict `left=1` (employee departure) from the 14,999-row public HR Analytics benchmark. "
        "Exact duplicate records and conflicting duplicate labels are removed before splitting. "
        "The model uses the original operational features, with numeric imputation, scaling, and categorical encoding inside a pipeline.",
        "The random forest and logistic regression are compared with a constant-prevalence baseline. "
        "`feature_importance.csv` measures permutation importance on validation data; it is association, not proof of a cause. "
        "`department_summary.csv` is descriptive training-only analysis. "
        "This public benchmark has limited source provenance and no employee identifiers or observation dates, "
        "so a random holdout cannot prove future-company performance or a specific prediction horizon. "
        "It cannot establish that workload causes departures, or that intervening on a feature will retain an employee. "
        "Results support an educational analytics demonstration; external and temporal validation would be needed for workplace use.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hr.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        result = run(args.data, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Input error: {error}\n")
    print(f"Selected {result['selected_model']}; test AP={result['test']['average_precision']:.4f}; report: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
