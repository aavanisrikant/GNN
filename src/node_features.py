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

MAX_NAN_FRAC  = 0.20   
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

with open(NODE_MAP_PATH, "r") as f:
    node_mapping = json.load(f)

stations_order = list(node_mapping.keys())
N = len(stations_order)

print(f"Nodes: {N}")

print("Pivoting data...")

df_pivot = df.pivot(index="timestamp", columns="station", values="aqi")

df_pivot = df_pivot.reindex(columns=stations_order)

df_pivot = df_pivot.sort_index()

T_raw = len(df_pivot)
print(f"Raw time steps: {T_raw}")

# imputation
print("Imputing missing values...")

df_pivot = df_pivot.ffill(limit=3)
df_pivot = df_pivot.interpolate(method="linear", limit=12)

nan_frac = df_pivot.isna().mean()

print("\nNaN fraction per station:")
print(nan_frac.sort_values(ascending=False).to_string())

overall_nan = df_pivot.isna().mean().mean()
print(f"\nOverall remaining NaN: {overall_nan:.3%}")

timestamps = df_pivot.index
data       = df_pivot.values.astype(np.float32)

hours = np.array(timestamps.hour, dtype=np.int32)

T = data.shape[0]

train_end = int(T * TRAIN_RATIO)
val_end   = int(T * (TRAIN_RATIO + VAL_RATIO))

print(f"\nTrain cutoff index: {train_end}")
print(f"Val cutoff index:   {val_end}")

# training split normalisation
print("\nNormalizing using TRAIN set only...")

min_vals = np.nanmin(data[:train_end], axis=0)
max_vals = np.nanmax(data[:train_end], axis=0)

data_norm = (data - min_vals) / (max_vals - min_vals + 1e-8)
data_norm = np.clip(data_norm, 0.0, 1.0)

# sequences
def create_sequences(data_norm, raw_data, hours, window, horizon, max_nan_frac):
    X_list, y_list, h_list, idx_list = [], [], [], []
    skipped = 0

    T = data_norm.shape[0]

    for t in range(window, T - horizon + 1):
        window_raw = raw_data[t - window : t]

        nan_fraction = np.isnan(window_raw).mean()
        if nan_fraction > max_nan_frac:
            skipped += 1
            continue

        x = data_norm[t - window : t]           
        y = data_norm[t + horizon - 1]           
        pred_hour = hours[t + horizon - 1]

        X_list.append(x.T)                         
        y_list.append(y)
        h_list.append(pred_hour)
        idx_list.append(t)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(h_list, dtype=np.int32),
        np.array(idx_list, dtype=np.int64),
        skipped
    )
#split function
def split_by_time(idx_arr, train_end, val_end):
    train_mask = idx_arr < train_end
    val_mask   = (idx_arr >= train_end) & (idx_arr < val_end)
    test_mask  = idx_arr >= val_end
    return train_mask, val_mask, test_mask

for h in HORIZONS:
    print(f"\n── Horizon = {h} ──")

    X, y, hour_arr, idx_arr, skipped = create_sequences(
        data_norm, data, hours, WINDOW, h, MAX_NAN_FRAC
    )

    print(f"Total samples: {len(X)}")
    print(f"Skipped (NaN-heavy): {skipped}")
    print(f"X shape: {X.shape}")   # [S, N, window]
    print(f"y shape: {y.shape}")   # [S, N]

    train_mask, val_mask, test_mask = split_by_time(idx_arr, train_end, val_end)

    print(f"Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")

    np.save(OUTPUT_DIR / f"X_h{h}.npy", X)
    np.save(OUTPUT_DIR / f"y_h{h}.npy", y)
    np.save(OUTPUT_DIR / f"hours_h{h}.npy", hour_arr)

    np.save(OUTPUT_DIR / f"time_index_h{h}.npy", idx_arr)

    np.save(OUTPUT_DIR / f"train_mask_h{h}.npy", train_mask)
    np.save(OUTPUT_DIR / f"val_mask_h{h}.npy", val_mask)
    np.save(OUTPUT_DIR / f"test_mask_h{h}.npy", test_mask)

    print(f"Saved horizon {h} files")

np.save(OUTPUT_DIR / "min_vals.npy", min_vals)
np.save(OUTPUT_DIR / "max_vals.npy", max_vals)


print("\n--- PHASE 2 COMPLETE ---")
print(f"Total timesteps: {T}")

for h in HORIZONS:
    print(f"H{h}: X_h{h}.npy, y_h{h}.npy, hours_h{h}.npy, time_index_h{h}.npy")

print("Also saved: min_vals.npy, max_vals.npy")
