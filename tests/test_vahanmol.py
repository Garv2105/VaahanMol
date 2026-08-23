"""Tests for VahanMol. Uses a small synthetic DataFrame, no CSV needed.

Run with: pytest tests/
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vahanmol import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    clean_data,
    engineer_features,
    train_baseline,
    train_tuned_model,
)

CURRENT_YEAR = 2026


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [
                "Maruti Swift Dzire VDI",
                "Hyundai Creta 1.6 SX",
                "Honda City i-VTEC",
                "Toyota Innova 2.5 G",
                "Ford Figo Diesel",
                "Maruti Alto 800 LXI",
                "Bad Row Example",
                "Missing Spec Example",
            ],
            "year": [2015, 2018, 2012, 2016, 2010, 2020, 2999, 2017],
            "selling_price": [450000, 950000, 300000, 800000, 200000, 350000, -1, 400000],
            "km_driven": [60000, 25000, 90000, 45000, 120000, 15000, -5, 30000],
            "fuel": ["Diesel", "Petrol", "Petrol", "Diesel", "Diesel", "Petrol", "Petrol", "Petrol"],
            "seller_type": [
                "Individual", "Dealer", "Individual", "Dealer",
                "Individual", "Individual", "Dealer", "Individual",
            ],
            "transmission": [
                "Manual", "Automatic", "Manual", "Manual",
                "Manual", "Manual", "Manual", "Manual",
            ],
            "owner": [
                "First Owner", "First Owner", "Second Owner", "First Owner",
                "Third Owner", "First Owner", "First Owner", "First Owner",
            ],
            "mileage": [
                "23.4 kmpl", "17.0 kmpl", "18.0 kmpl", "13.0 kmpl",
                "20.0 kmpl", "22.0 kmpl", "19.0 kmpl", "0.0 kmpl",
            ],
            "engine": [
                "1248 CC", "1591 CC", "1497 CC", "2494 CC",
                "1399 CC", "796 CC", "1197 CC", "1197 CC",
            ],
            "max_power": [
                "74 bhp", "121.3 bhp", "117 bhp", "100 bhp",
                "68 bhp", "47 bhp", "82 bhp", "82 bhp",
            ],
            "seats": [5, 5, 5, 7, 5, 4, 5, 5],
        }
    )


def test_clean_data_drops_impossible_rows(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    # only the year=2999/selling_price=-1/km_driven=-5 row fails raw checks
    assert len(cleaned) == len(raw_df) - 1
    assert (cleaned["selling_price"] > 0).all()
    assert (cleaned["km_driven"] >= 0).all()
    assert (cleaned["year"] <= CURRENT_YEAR).all()


def test_clean_data_drops_duplicates(raw_df):
    dup_df = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(dup_df, current_year=CURRENT_YEAR)
    assert len(cleaned) == len(clean_data(raw_df, current_year=CURRENT_YEAR))


def test_car_age_calculation(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    row = featured[featured["name"] == "Maruti Swift Dzire VDI"].iloc[0]
    assert row["car_age"] == CURRENT_YEAR - 2015


def test_brand_extraction(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    brands = dict(zip(featured["name"], featured["brand"]))
    assert brands["Maruti Swift Dzire VDI"] == "Maruti"
    assert brands["Honda City i-VTEC"] == "Honda"
    assert brands["Toyota Innova 2.5 G"] == "Toyota"


def test_spec_parsing(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    row = featured[featured["name"] == "Maruti Swift Dzire VDI"].iloc[0]
    assert row["mileage"] == pytest.approx(23.4)
    assert row["engine"] == pytest.approx(1248)
    assert row["max_power"] == pytest.approx(74)


def test_engineer_features_drops_zero_mileage(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    assert "Missing Spec Example" not in featured["name"].values
    assert (featured["mileage"] > 0).all()


def test_predictions_are_plausible(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = featured[feature_cols]
    y = featured["selling_price"]

    model = train_tuned_model(X, y, seed=0, n_iter=2)
    preds = model.predict(X)

    assert (preds > 0).all()
    assert preds.min() > 0.1 * y.min()
    assert preds.max() < 10 * y.max()


def test_baseline_uses_car_age_only(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = featured[feature_cols]
    y = featured["selling_price"]

    baseline = train_baseline(X, y)
    preds = baseline.predict(X[["car_age"]])
    assert len(preds) == len(X)


def test_preprocessor_handles_unknown_category(raw_df):
    cleaned = clean_data(raw_df, current_year=CURRENT_YEAR)
    featured = engineer_features(cleaned, current_year=CURRENT_YEAR)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = featured[feature_cols]

    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["brand"] = "TotallyNewBrand"
    preprocessor.transform(unseen)
