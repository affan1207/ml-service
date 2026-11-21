import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/UEs_16cases.csv"
OUT_PATH = "data/processed/lbnl_ues_simulated.csv"
os.makedirs("data/processed", exist_ok=True)


def simulate_and_augment_lbnl(samples_per_group: int = 150, random_state: int = 42):
    """
    Cleans the raw LBNL UEs_16cases dataset, splits it 80/20 FIRST, and then
    augments only the training rows with physically bounded synthetic samples.
    Test rows are never used to learn mean/std, so there is no data leakage.
    """
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Place UEs_16cases.csv there first."
        )

    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "PUE": "pue",
        "WUE": "wue",
        "Climate Zone": "climate_zone",
        "Cooling system": "cooling_system",
        "Data center size": "facility_size",
    })

    for col in ["climate_zone", "cooling_system", "facility_size"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    df = df.dropna(subset=["pue", "wue", "climate_zone", "cooling_system", "facility_size"])
    df = df[(df["pue"] >= 1.0) & (df["pue"] <= 3.5)]
    df = df[df["wue"] >= 0.0]

    keep_cols = ["cooling_system", "climate_zone", "facility_size", "pue", "wue"]
    real_clean = df[keep_cols].copy()

    # 1. Split real data FIRST (no leakage)
    train_real, test_real = train_test_split(
        real_clean, test_size=0.2, random_state=random_state
    )
    train_real = train_real.copy()
    test_real = test_real.copy()

    train_real["split"] = "train"
    train_real["is_simulated"] = 0
    test_real["split"] = "test"
    test_real["is_simulated"] = 0

    rng = np.random.default_rng(random_state)
    sim_rows = []

    # 2. Simulate ONLY from the training rows
    grouped = train_real.groupby(["cooling_system", "climate_zone", "facility_size"])

    for (cooling, zone, size), group in grouped:
        if len(group) < 2:
            continue

        pue_mean, pue_std = group["pue"].mean(), max(group["pue"].std(), 0.015)
        wue_mean, wue_std = group["wue"].mean(), max(group["wue"].std(), 0.005)
        pue_min, pue_max = max(1.02, group["pue"].min() * 0.95), min(3.2, group["pue"].max() * 1.05)
        wue_min, wue_max = max(0.0, group["wue"].min() * 0.90), group["wue"].max() * 1.10

        sim_pue = np.clip(
            rng.normal(loc=pue_mean, scale=pue_std, size=samples_per_group), pue_min, pue_max
        )
        if wue_mean < 0.15:
            # Air-cooled / dry systems: near-zero WUE
            sim_wue = np.clip(
                rng.normal(loc=wue_mean, scale=wue_std, size=samples_per_group), 0.0, wue_max
            )
        else:
            sim_wue = np.clip(
                rng.normal(loc=wue_mean, scale=wue_std, size=samples_per_group), wue_min, wue_max
            )

        for p, w in zip(sim_pue, sim_wue):
            sim_rows.append({
                "cooling_system": cooling,
                "climate_zone": zone,
                "facility_size": size,
                "pue": round(float(p), 6),
                "wue": round(float(w), 6),
                "split": "train",
                "is_simulated": 1,
            })

    sim_df = pd.DataFrame(sim_rows)
    combined = pd.concat([train_real, test_real, sim_df], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"[OK] Real Train: {len(train_real)} | Real Test: {len(test_real)} | Simulated (train only): {len(sim_df)}")
    print(f"[OK] Saved -> {OUT_PATH}")


if __name__ == "__main__":
    simulate_and_augment_lbnl(samples_per_group=150)
