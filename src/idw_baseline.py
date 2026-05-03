import numpy as np
import json
from pathlib import Path

DATA_DIR = "data/processed"
SAVE_DIR = "models/baselines"
HORIZON  = 1

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

X_test  = np.load(f"{DATA_DIR}/X_test_h{HORIZON}.npy")   # (S, N, 24)
y_test  = np.load(f"{DATA_DIR}/y_test_h{HORIZON}.npy")   # (S, N)

min_vals = np.load(f"{DATA_DIR}/min_vals.npy")           # (N,)
max_vals = np.load(f"{DATA_DIR}/max_vals.npy")           # (N,)

adj_matrix = np.load(f"{DATA_DIR}/adj_matrix.npy")       # (N, N)

S, N, T = X_test.shape
print(f"Stations: {N} | Test samples: {S} | Time window: {T}")
#idw prediction
def idw_predict(X: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """
    Predict AQI using inverse-distance weighting.

    X   : (S, N, 24)
    adj : (N, N) with weights = 1/distance

    Returns:
    preds : (S, N)
    """
    current_aqi = X[:, :, -1]   # (S, N)

    preds = np.zeros((X.shape[0], X.shape[1]))

    for i in range(N):
        weights = adj[i].copy()          
        weights[i] = 0.0                  
       
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0)

        w_sum = weights.sum()

        if w_sum == 0:
            preds[:, i] = current_aqi[:, i]
        else:
            preds[:, i] = (current_aqi * weights).sum(axis=1) / w_sum

    return preds
  
#inverse normalisation
def inverse_transform(arr_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalized values back to real AQI scale.
    arr_norm: (S, N)
    """
    return arr_norm * (max_vals - min_vals) + min_vals

def compute_metrics(preds_real: np.ndarray, targets_real: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(preds_real - targets_real)))
    rmse = float(np.sqrt(np.mean((preds_real - targets_real) ** 2)))

    mask = targets_real > 1.0
    if mask.sum() > 0:
        mape = float(np.mean(
            np.abs((preds_real[mask] - targets_real[mask]) / targets_real[mask])
        ) * 100)
    else:
        mape = float("nan")

    ss_res = np.sum((targets_real - preds_real) ** 2)
    ss_tot = np.sum((targets_real - np.mean(targets_real)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }

print("\nRunning IDW baseline (no training)...")

preds_norm   = idw_predict(X_test, adj_matrix)
targets_norm = y_test

preds_real   = inverse_transform(preds_norm)
targets_real = inverse_transform(targets_norm)

metrics = compute_metrics(preds_real, targets_real)

print("\n── IDW BASELINE — TEST RESULTS ─────────────────────────")
print(f"MAE  : {metrics['MAE']:.2f} AQI")
print(f"RMSE : {metrics['RMSE']:.2f} AQI")
print(f"MAPE : {metrics['MAPE']:.1f}%")
print(f"R²   : {metrics['R2']:.3f}")
print("─────────────────────────────────────────────────────────")

results = {
    "model":   "IDW_baseline",
    "horizon": HORIZON,
    "MAE":     round(metrics["MAE"], 3),
    "RMSE":    round(metrics["RMSE"], 3),
    "MAPE":    round(metrics["MAPE"], 3),
    "R2":      round(metrics["R2"], 3),
}

out_path = f"{SAVE_DIR}/idw_results_h{HORIZON}.json"

with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved results → {out_path}")
