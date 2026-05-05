# src/error_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR     = "models/baselines"
PROC_DIR     = "data/processed"
STATION_FILE = f"{PROC_DIR}/station_coords.csv"
PLOT_DIR     = "plots"
HORIZON      = 1

Path(PLOT_DIR).mkdir(exist_ok=True)

preds   = np.load(f"{DATA_DIR}/lstm_preds_h{HORIZON}.npy")    # (S, N)
targets = np.load(f"{DATA_DIR}/lstm_targets_h{HORIZON}.npy")  # (S, N)

station_df = pd.read_csv(STATION_FILE)
S, N = preds.shape
print(f"Samples: {S} | Stations: {N}")

time_idx_all  = np.load(f"{PROC_DIR}/time_index_h{HORIZON}.npy")
test_mask     = np.load(f"{PROC_DIR}/test_mask_h{HORIZON}.npy")
time_idx_test = time_idx_all[test_mask]

raw_ts = pd.to_datetime(
    pd.read_csv(f"{PROC_DIR}/clean_cpcb.csv", usecols=["timestamp"])["timestamp"]
).drop_duplicates().sort_values().reset_index(drop=True)

time_index = raw_ts.iloc[time_idx_test].reset_index(drop=True)
print(f"Test period: {time_index.iloc[0]} → {time_index.iloc[-1]}")

# error calculation
errors = np.abs(preds - targets)   # (S, N) — real AQI units
print(f"Overall MAE: {errors.mean():.2f} AQI units")

# error by station
station_error          = errors.mean(axis=0)   # (N,)
station_df["error"]    = station_error

print("\nTop 5 worst stations:")
print(station_df.sort_values("error", ascending=False)[
    ["station", "error"]].head().to_string(index=False))

print("\nTop 5 best stations:")
print(station_df.sort_values("error", ascending=True)[
    ["station", "error"]].head().to_string(index=False))

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    station_df["lon"],
    station_df["lat"],
    s = station_df["error"] * 5,
    c = station_df["error"],
    cmap      = "RdYlGn_r",
    alpha     = 0.85,
    edgecolors= "white",
    linewidths= 0.7,
)
plt.colorbar(sc, label="MAE (AQI units)")
for _, row in station_df.iterrows():
    short = row["station"].split("Mumbai")[0].strip()
    plt.annotate(short, (row["lon"], row["lat"]),
                 fontsize=6, ha="center", va="bottom",
                 xytext=(0, 5), textcoords="offset points")
plt.title("LSTM — Error by Station (bubble size ∝ error)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_by_station.png", dpi=150)
plt.show()
print("Saved error_by_station.png")

#error by hour
hours = time_index.dt.hour   # (S,)

df_hour    = pd.DataFrame({
    "hour":  np.repeat(hours.values, N),
    "error": errors.flatten()
})
hour_error = df_hour.groupby("hour")["error"].mean()

# Colour bars by time of day
colors = []
for h in range(24):
    if h in [7, 8, 17, 18]:
        colors.append("#e05c2a")       # rush hour
    elif h >= 23 or h <= 4:
        colors.append("#4a7ec4")       # night
    else:
        colors.append("#888888")       # normal

plt.figure(figsize=(11, 4))
plt.bar(hour_error.index, hour_error.values, color=colors, alpha=0.85, edgecolor="white")
plt.xticks(range(24), fontsize=9)
plt.xlabel("Hour of Day (IST)")
plt.ylabel("MAE (AQI units)")
plt.title("LSTM — Error by Hour of Day")
plt.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend = [Patch(color="#e05c2a", label="Rush hour (7–9am, 5–7pm)"),
          Patch(color="#4a7ec4", label="Night (11pm–5am)"),
          Patch(color="#888888", label="Other")]
plt.legend(handles=legend, fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_by_hour.png", dpi=150)
plt.show()
print("Saved error_by_hour.png")


# ── 3. ERROR BY MONTH ─────────────────────────────────────────────────────────
months = time_index.dt.month   # (S,)

df_month    = pd.DataFrame({
    "month":  np.repeat(months.values, N),
    "error":  errors.flatten()
})
month_error = df_month.groupby("month")["error"].mean()

month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
month_labels = [month_names[m] for m in month_error.index]

plt.figure(figsize=(10, 4))
plt.bar(month_labels, month_error.values, color="#5a8fd4", alpha=0.85, edgecolor="white")
plt.xlabel("Month")
plt.ylabel("MAE (AQI units)")
plt.title("LSTM — Error by Month")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_by_month.png", dpi=150)
plt.show()
print("Saved error_by_month.png")


# ── 4. PREDICTED vs ACTUAL — WORST AND BEST STATIONS ─────────────────────────
sorted_idx = np.argsort(station_error)
best_idx   = sorted_idx[0]
worst_idxs = sorted_idx[-3:][::-1]
plot_idxs  = list(worst_idxs) + [best_idx]
plot_labels= ["Worst 1", "Worst 2", "Worst 3", "Best"]

N_SHOW = min(24 * 30, S)   # last 30 days or all test data if shorter

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle("LSTM — Predicted vs Actual (last 30 days of test set)", fontsize=11)

for ax, idx, label in zip(axes, plot_idxs, plot_labels):
    ax.plot(targets[-N_SHOW:, idx], label="Actual",    color="#2a6496", lw=1.0)
    ax.plot(preds[-N_SHOW:,   idx], label="Predicted", color="#e05c2a", lw=1.0, alpha=0.8)
    short = station_df.iloc[idx]["station"].split("Mumbai")[0].strip()
    ax.set_title(f"{label}: {short}  (MAE={station_error[idx]:.1f})", fontsize=9)
    ax.set_ylabel("AQI", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pred_vs_actual.png", dpi=150)
plt.show()
print("Saved pred_vs_actual.png")


# ── 5. SUMMARY ────────────────────────────────────────────────────────────────
peak_hour  = int(hour_error.idxmax())
peak_month = int(month_error.idxmax())

print("\n── ERROR ANALYSIS SUMMARY ───────────────────────────────")
print(f"  Overall MAE    : {errors.mean():.2f} AQI units")
print(f"  Worst station  : {station_df.sort_values('error', ascending=False).iloc[0]['station']}")
print(f"                   MAE = {station_error.max():.2f}")
print(f"  Best station   : {station_df.sort_values('error').iloc[0]['station']}")
print(f"                   MAE = {station_error.min():.2f}")
print(f"  Peak error hour: {peak_hour:02d}:00")
print(f"  Peak error month: {month_names[peak_month]}")
print(f"  Plots saved to : {PLOT_DIR}/")
print("─────────────────────────────────────────────────────────")
