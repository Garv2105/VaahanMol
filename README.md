# VahanMol

**वाहन (vahan) = vehicle · मोल (mol) = price/worth**

Predicts the resale price of a used car from its listing details: brand,
age, kilometers driven, fuel type, transmission, seller type, ownership
history, and vehicle specs (mileage, engine size, power, seats).

## Data source

**Vehicle dataset from cardekho**, by Nehal Birla, Nishant Verma, and
Nikhil Kushwaha, on Kaggle:
<https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho>

Originally scraped from used-car listings on CarDekho.com. Uses the
`Car_details_v3.csv` file from that dataset (~8,100 rows, 13 columns),
included in this repo at `data/raw/Car_details_v3.csv` under the
dataset's DbCL-1.0 license (see [LICENSE](LICENSE)). The other CSVs in
`data/raw/` are additional files bundled in the same Kaggle download;
they aren't used by `vahanmol.py` but are kept here under the same
license.

## Project structure

```
VahanMol/
├── vahanmol.py           # load -> clean -> features -> model -> evaluate
├── tests/
│   └── test_vahanmol.py
├── data/
│   └── raw/
│       ├── Car_details_v3.csv          # used by vahanmol.py
│       ├── CAR_DETAILS_FROM_CAR_DEKHO.csv
│       ├── Car details v3.csv
│       ├── car data.csv
│       └── car details v4.csv
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Garv2105/VaahanMol.git
cd VaahanMol
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is already included at `data/raw/Car_details_v3.csv`, so no
download step is needed.

## Running

```bash
python vahanmol.py --data data/raw/Car_details_v3.csv
```

Trains a linear-regression baseline (car_age only) and a
RandomizedSearchCV-tuned RandomForestRegressor on the full feature set,
then prints MAE, RMSE, R², and feature importances.

Flags:

```bash
python vahanmol.py --data data/raw/Car_details_v3.csv \
    --test-size 0.2 \
    --seed 42 \
    --search-iter 30 \
    --save-model vahanmol_model.joblib
```

## Tests

```bash
pytest tests/
```

Uses a synthetic DataFrame built in-file, independent of the CSV in
`data/raw/`.

## Modeling notes

- Baseline: linear regression on `car_age` alone.
- Model: `RandomForestRegressor`, tuned via `RandomizedSearchCV` (5-fold
  CV) over tree count, depth, split/leaf sizes, and max features.
- Metrics: MAE and RMSE (in ₹), plus R².
- Split: plain random train/test split — each row is an independent
  listing.
- Numeric specs (`mileage`, `engine`, `max_power`) arrive as strings with
  units baked in (e.g. `"23.4 kmpl"`, `"1248 CC"`, `"74 bhp"`) and are
  parsed to floats during feature engineering.
- `torque` is dropped entirely — its raw format mixes Nm/kgm units and
  inconsistent `@ rpm` notation, too unreliable to parse cleanly.
- Rows with missing or zero-valued specs (a handful of listings have
  `0.0 kmpl`, which isn't physically real) are dropped rather than
  imputed.
- `max_power`, `car_age`, and `engine` dominate feature importance,
  which is the expected result — power and age are the strongest real
  drivers of resale value, not spurious correlations.

## License

Code and README in this repo are under [`LICENSE`](LICENSE): free to use,
copy, modify, and redistribute for any purpose that isn't harmful, with
attribution preserved. The dataset itself is third-party (cited above)
and not covered by this license — it's licensed separately under
DbCL-1.0 (see LICENSE).