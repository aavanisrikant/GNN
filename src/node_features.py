import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_PATH     = "data/processed/clean_data.csv"
NODE_MAP_PATH = "data/processed/node_mapping.json"
OUTPUT_DIR    = Path("data/processed")

WINDOW        = 24
HORIZONS      = [1, 3, 6]

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15

MAX_NAN_FRAC  = 0.8   

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

with open(NODE_MAP_PATH, "r") as f:
    node_mapping = json.load(f)

stations_order = list(node_mapping.keys())

print("Pivoting data...")

df_pivot = df.pivot(index="timestamp", columns="station", values="aqi")

df_pivot = df_pivot.reindex(columns=stations_order)
df_pivot = df_pivot.sort_index()

df_pivot = df_pivot.resample("1H").mean()

print(f"Raw time steps: {len(df_pivot)}")

print("Applying robust imputation...")

df_pivot = df_pivot.ffill(limit=6)
df_pivot = df_pivot.bfill(limit=6)

df_pivot = df_pivot.interpolate(method="linear", limit=24)

df_pivot = df_pivot.fillna(df_pivot.median())

nan_frac = df_pivot.isna().mean()

valid_cols = nan_frac[nan_frac < 0.9].index
df_pivot = df_pivot[valid_cols]

print(f"\nUsing {len(valid_cols)} stations after filtering")

timestamps = df_pivot.index
data       = df_pivot.values.astype(np.float32)
hours      = np.array(timestamps.hour)

T = data.shape[0]

train_end = int(T * TRAIN_RATIO)
val_end   = int(T * (TRAIN_RATIO + VAL_RATIO))

print(f"\nTrain cutoff: {train_end}")
print(f"Val cutoff:   {val_end}")

print("\nNormalizing...")

min_vals = np.min(data[:train_end], axis=0)
max_vals = np.max(data[:train_end], axis=0)

data_norm = (data - min_vals) / (max_vals - min_vals + 1e-8)
data_norm = np.clip(data_norm, 0, 1)

def create_sequences(data_norm, raw_data, hours, window, horizon):
    X, y, h_arr, idx_arr = [], [], [], []
    skipped = 0

    for t in range(window, len(data_norm) - horizon + 1):
        window_raw = raw_data[t - window:t]

        nan_frac = np.isnan(window_raw).mean()

        if nan_frac > MAX_NAN_FRAC:
            skipped += 1
            continue

        X.append(data_norm[t - window:t].T)
        y.append(data_norm[t + horizon - 1])
        h_arr.append(hours[t + horizon - 1])
        idx_arr.append(t)

    return (
        np.array(X),
        np.array(y),
        np.array(h_arr),
        np.array(idx_arr),
        skipped
    )

def split(idx):
    train = idx < train_end
    val   = (idx >= train_end) & (idx < val_end)
    test  = idx >= val_end
    return train, val, test

for h in HORIZONS:
    print(f"\n── Horizon {h} ──")

    X, y, h_arr, idx_arr, skipped = create_sequences(
        data_norm, data, hours, WINDOW, h
    )

    print(f"Samples: {len(X)}")
    print(f"Skipped: {skipped}")

    train, val, test = split(idx_arr)

    print(f"Train: {train.sum()} | Val: {val.sum()} | Test: {test.sum()}")

    np.save(OUTPUT_DIR / f"X_h{h}.npy", X)
    np.save(OUTPUT_DIR / f"y_h{h}.npy", y)
    np.save(OUTPUT_DIR / f"hours_h{h}.npy", h_arr)
    np.save(OUTPUT_DIR / f"time_index_h{h}.npy", idx_arr)

    np.save(OUTPUT_DIR / f"train_mask_h{h}.npy", train)
    np.save(OUTPUT_DIR / f"val_mask_h{h}.npy", val)
    np.save(OUTPUT_DIR / f"test_mask_h{h}.npy", test)

np.save(OUTPUT_DIR / "min_vals.npy", min_vals)
np.save(OUTPUT_DIR / "max_vals.npy", max_vals)

print("\n--- PHASE 2 COMPLETE ---")
