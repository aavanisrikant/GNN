import pandas as pd
import numpy as np
import os

DATA_PATH = "data/processed/clean_cpcb.csv"
SAVE_DIR = "data/processed"

LOOKBACK = 24
HORIZONS = [1, 3, 6]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading data...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["timestamp", "station"])


print("Pivoting data...")

df_pivot = df.pivot(index="timestamp", columns="station", values="aqi")

df_pivot = df_pivot.resample("1h").mean()

print("Raw time steps:", len(df_pivot))

print("Applying robust imputation...")

df_pivot = df_pivot.ffill(limit=12)

df_pivot = df_pivot.bfill(limit=12)

df_pivot = df_pivot.interpolate(limit=12)

df_pivot = df_pivot.apply(lambda col: col.fillna(col.median()), axis=0)

df_pivot = df_pivot.fillna(df_pivot.mean())

print(f"\nTotal stations used: {df_pivot.shape[1]}")

data = df_pivot.values  

T = len(data)

train_end = int(T * TRAIN_RATIO)
val_end = int(T * (TRAIN_RATIO + VAL_RATIO))

print("\nTrain cutoff:", train_end)
print("Val cutoff:", val_end)

print("\nNormalizing...")

train_data = data[:train_end]

min_vals = np.min(train_data, axis=0)
max_vals = np.max(train_data, axis=0)

denom = (max_vals - min_vals)
denom[denom == 0] = 1

data_norm = (data - min_vals) / denom

np.save(f"{SAVE_DIR}/min_vals.npy", min_vals)
np.save(f"{SAVE_DIR}/max_vals.npy", max_vals)

def create_sequences(data, lookback, horizon):
    X, y, hours, time_idx = [], [], [], []

    for t in range(lookback, len(data) - horizon):
        x = data[t - lookback:t]      
        target = data[t + horizon]  

        X.append(x)
        y.append(target)

        hours.append(t % 24)
        time_idx.append(t)

    return np.array(X), np.array(y), np.array(hours), np.array(time_idx)

for h in HORIZONS:
    print(f"\n— Horizon = {h} —")

    X, y, hours, time_idx = create_sequences(data_norm, LOOKBACK, h)

    print("Samples:", len(X))

    # Split
    train_mask = time_idx < train_end
    val_mask = (time_idx >= train_end) & (time_idx < val_end)
    test_mask = time_idx >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    hours_h = hours
    time_idx_h = time_idx

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    np.save(f"{SAVE_DIR}/X_h{h}.npy", X)
    np.save(f"{SAVE_DIR}/y_h{h}.npy", y)
    np.save(f"{SAVE_DIR}/hours_h{h}.npy", hours_h)
    np.save(f"{SAVE_DIR}/time_index_h{h}.npy", time_idx_h)

    np.save(f"{SAVE_DIR}/X_train_h{h}.npy", X_train)
    np.save(f"{SAVE_DIR}/y_train_h{h}.npy", y_train)

    np.save(f"{SAVE_DIR}/X_val_h{h}.npy", X_val)
    np.save(f"{SAVE_DIR}/y_val_h{h}.npy", y_val)

    np.save(f"{SAVE_DIR}/X_test_h{h}.npy", X_test)
    np.save(f"{SAVE_DIR}/y_test_h{h}.npy", y_test)

print("\n--- PHASE 2 COMPLETE ---")
print("Total timesteps:", T)
print("Stations:", data.shape[1])
print("\nSaved files for horizons:", HORIZONS)
print("Also saved: min_vals.npy, max_vals.npy")
