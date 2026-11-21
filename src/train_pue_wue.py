import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = "data/processed/lbnl_ues_simulated.csv"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

FEATURE_COLS = ["cooling_system", "climate_zone", "facility_size"]


def build_quantile_pipeline(alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS)
        ]
    )
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=250,
        max_depth=6,
        learning_rate=0.06,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def train_pue_wue_models():
    df = pd.read_csv(DATA_PATH)

    # Split was done in simulate_data.py (before simulation) -> no leakage.
    train_df = df[df["split"] == "train"]
    test_df = df[(df["split"] == "test") & (df["is_simulated"] == 0)]

    X_train = train_df[FEATURE_COLS]
    y_pue_train = train_df["pue"]
    y_wue_train = train_df["wue"]

    X_real_test = test_df[FEATURE_COLS]
    y_pue_real_test = test_df["pue"]
    y_wue_real_test = test_df["wue"]

    print(f"Train rows: {len(train_df)} | Real test rows: {len(test_df)}")

    quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
    pue_models = {}
    wue_models = {}

    for label, alpha in quantiles.items():
        print(f"Training PUE & WUE quantile models ({label}, alpha={alpha})...")
        pue_pipe = build_quantile_pipeline(alpha)
        pue_pipe.fit(X_train, y_pue_train)
        pue_models[label] = pue_pipe

        wue_pipe = build_quantile_pipeline(alpha)
        wue_pipe.fit(X_train, y_wue_train)
        wue_models[label] = wue_pipe

    pue_mae = mean_absolute_error(y_pue_real_test, pue_models["p50"].predict(X_real_test))
    wue_mae = mean_absolute_error(y_wue_real_test, wue_models["p50"].predict(X_real_test))
    print(f"[EVAL] Real-test Median PUE MAE: {pue_mae:.4f} | Median WUE MAE: {wue_mae:.4f} L/kWh")

    # Valid dropdown values for the frontend (served by /categories in main.py)
    valid_categories = {
        "cooling_systems": sorted(df["cooling_system"].unique().tolist()),
        "climate_zones": sorted(df["climate_zone"].astype(str).unique().tolist()),
        "facility_sizes": sorted(df["facility_size"].unique().tolist()),
    }

    bundle = {
        "pue_models": pue_models,
        "wue_models": wue_models,
        "metrics": {"pue_mae": round(pue_mae, 4), "wue_mae": round(wue_mae, 4)},
        "valid_categories": valid_categories,
    }
    out_file = os.path.join(ARTIFACT_DIR, "pue_wue_quantile_bundle.joblib")
    joblib.dump(bundle, out_file)
    print(f"[OK] Saved PUE/WUE model bundle -> {out_file}")


if __name__ == "__main__":
    train_pue_wue_models()
