"""
VahanMol - Used Car Price Predictor

Predicts resale price of a used car from brand, age, kilometers driven,
fuel type, transmission, seller type, ownership history, and vehicle
specs (mileage, engine size, power, seats).

Data source: "Vehicle dataset from cardekho" (Nehal Birla et al.), Kaggle
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
Licensed under DbCL-1.0. Expects data/raw/Car_details_v3.csv

Usage:
    python vahanmol.py --data data/raw/Car_details_v3.csv
    python vahanmol.py --data data/raw/Car_details_v3.csv --save-model model.joblib
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED_DEFAULT = 42
CURRENT_YEAR_DEFAULT = datetime.now().year

NUMERIC_FEATURES = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission", "owner"]
TARGET = "selling_price"

# torque is intentionally excluded: its raw format mixes Nm/kgm units and
# inconsistent "@ rpm" notation, making it too unreliable to parse cleanly.
EXPECTED_RAW_COLUMNS = {
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
}

_NUMBER_RE = re.compile(r"^([\d.]+)")


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at '{path}'.")
    df = pd.read_csv(path)

    missing = EXPECTED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected column(s): {sorted(missing)}. "
            f"Got columns: {list(df.columns)}."
        )
    return df


def clean_data(df: pd.DataFrame, current_year: int = CURRENT_YEAR_DEFAULT) -> pd.DataFrame:
    """Drop duplicates and rows with impossible price/km/year values."""
    before = len(df)
    df = df.drop_duplicates()

    df = df[
        (df["selling_price"] > 0)
        & (df["km_driven"] >= 0)
        & (df["year"] >= 1980)
        & (df["year"] <= current_year)
    ].copy()

    dropped = before - len(df)
    if dropped:
        print(f"[clean_data] Dropped {dropped} row(s) out of {before}.")
    return df.reset_index(drop=True)


def _parse_leading_number(series: pd.Series) -> pd.Series:
    """Extract the leading numeric token from strings like '23.4 kmpl',
    '1248 CC', '74 bhp'. Non-parseable or missing values become NaN."""
    return series.astype(str).str.extract(_NUMBER_RE)[0].astype(float)


def engineer_features(df: pd.DataFrame, current_year: int = CURRENT_YEAR_DEFAULT) -> pd.DataFrame:
    """car_age from year; brand from name; numeric specs parsed out of
    their unit-suffixed strings; rows with missing/impossible specs
    (0 mileage/engine/power, missing seats) are dropped."""
    df = df.copy()
    df["car_age"] = current_year - df["year"]
    df["brand"] = df["name"].astype(str).str.strip().str.split().str[0]

    df["mileage"] = _parse_leading_number(df["mileage"])
    df["engine"] = _parse_leading_number(df["engine"])
    df["max_power"] = _parse_leading_number(df["max_power"])
    df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

    before = len(df)
    df = df[
        df["mileage"].notna() & (df["mileage"] > 0)
        & df["engine"].notna() & (df["engine"] > 0)
        & df["max_power"].notna() & (df["max_power"] > 0)
        & df["seats"].notna() & (df["seats"] > 0)
    ].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[engineer_features] Dropped {dropped} row(s) with "
              f"missing/impossible spec values out of {before}.")

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Linear regression on car_age alone."""
    pipe = Pipeline(steps=[("model", LinearRegression())])
    pipe.fit(X_train[["car_age"]], y_train)
    return pipe


def train_tuned_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_SEED_DEFAULT,
    n_iter: int = 20,
) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(random_state=seed)),
        ]
    )

    param_dist = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": [2, 4, 6, 10],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=seed,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"[train_tuned_model] Best params: {search.best_params_}")
    return search.best_estimator_


@dataclass
class Metrics:
    mae: float
    rmse: float
    r2: float

    def __str__(self) -> str:
        return f"MAE={self.mae:,.0f}  RMSE={self.rmse:,.0f}  R2={self.r2:.4f}"


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
             baseline: bool = False) -> Metrics:
    X = X_test[["car_age"]] if baseline else X_test
    preds = model.predict(X)
    return Metrics(
        mae=mean_absolute_error(y_test, preds),
        rmse=np.sqrt(mean_squared_error(y_test, preds)),
        r2=r2_score(y_test, preds),
    )


def feature_importance(model: Pipeline) -> pd.Series:
    ohe: OneHotEncoder = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_names = NUMERIC_FEATURES + cat_names
    importances = model.named_steps["model"].feature_importances_
    return pd.Series(importances, index=all_names).sort_values(ascending=False)


def run(
    data_path: str | Path,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED_DEFAULT,
    search_iter: int = 20,
    save_model_path: str | Path | None = None,
) -> None:
    print(f"[run] Loading data from {data_path} ...")
    df = load_data(data_path)
    print(f"[run] Loaded {len(df)} rows.")

    df = clean_data(df)
    df = engineer_features(df)
    print(f"[run] {len(df)} rows remain after cleaning and feature engineering.")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    print(f"[run] Train/test split: {len(X_train)}/{len(X_test)} rows.")

    print("[run] Training baseline (linear regression on car_age only) ...")
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = evaluate(baseline_model, X_test, y_test, baseline=True)
    print(f"[run] Baseline  -> {baseline_metrics}")

    print(f"[run] Training tuned RandomForest (n_iter={search_iter}) ...")
    tuned_model = train_tuned_model(X_train, y_train, seed=seed, n_iter=search_iter)
    tuned_metrics = evaluate(tuned_model, X_test, y_test)
    print(f"[run] Tuned RF  -> {tuned_metrics}")

    print("\n[run] Top feature importances (tuned model):")
    print(feature_importance(tuned_model).head(10).to_string())

    if save_model_path:
        import joblib

        joblib.dump(tuned_model, save_model_path)
        print(f"\n[run] Saved trained model to {save_model_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VahanMol - Used Car Price Predictor")
    parser.add_argument(
        "--data",
        default="data/raw/Car_details_v3.csv",
        help="Path to the raw CarDekho CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT)
    parser.add_argument(
        "--search-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV iterations for the tuned model.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path to save the trained model (joblib).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            data_path=args.data,
            test_size=args.test_size,
            seed=args.seed,
            search_iter=args.search_iter,
            save_model_path=args.save_model,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
