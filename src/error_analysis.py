# src/error_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR     = "models/baselines"
PROC_DIR     = "data/processed"
STATION_FILE = f"{PROC_DIR}/station_coords.csv"
PLOT_DIR     = "plots"
HORIZON      = 1
Path(PLOT_DIR).mkdir(exist_ok=True)

# ── LOAD ─────────────────────────────────────────────────────────────────────
preds      = np.load(f"{DATA_DIR}/lstm_preds_h{HORIZON}.npy")    # (S, N)
targets    = np.load(f"{DATA_DIR}/lstm_targets_h{HORIZON}.npy")  # (S, N)
station_df = pd.read_csv(STATION_FILE)
S, N       = preds.shape
print(f"Samples: {S} | Stations: {N}")

# ── TIMESTAMPS ────────────────────────────────────────────────────────────────
raw_ts = pd.to_datetime(
    pd.read_csv(f"{PROC_DIR}/clean_cpcb.csv", usecols=["timestamp"])["timestamp"]
).drop_duplicates().sort_values().reset_index(drop=True)

T_total    = len(raw_ts)
val_end    = int(T_total * 0.85)
time_index = raw_ts.iloc[val_end : val_end + S].reset_index(drop=True)

if len(time_index) != S:
    print(f"Warning: timestamp length {len(time_index)} != S {S}, using cyclic hours")
    hours_arr  = np.arange(S) % 24
    months_arr = None
else:
    hours_arr  = time_index.dt.hour.values
    months_arr = time_index.dt.month.values

print(f"Test period: {time_index.iloc[0]} → {time_index.iloc[-1]}")

# ── ERRORS ────────────────────────────────────────────────────────────────────
errors        = np.abs(preds - targets)   # (S, N) real AQI units
station_error = errors.mean(axis=0)       # (N,)
errors_flat   = errors.flatten()          # (S*N,)
station_df["error"] = station_error

print(f"\nOverall MAE: {errors.mean():.2f} AQI units")
print("\nTop 5 worst stations:")
print(station_df.sort_values("error", ascending=False)[
    ["station", "error"]].head().to_string(index=False))
print("\nTop 5 best stations:")
print(station_df.sort_values("error")[
    ["station", "error"]].head().to_string(index=False))


# ── SHORT NAME LOOKUP ─────────────────────────────────────────────────────────
SHORT_NAMES = {
    "Bandra Kurla Complex Mumbai IITM":                    "BKC",
    "Bandra Kurla Complex Mumbai MPCB":                    "BKC",
    "Bandra Mumbai MPCB":                                  "Bandra",
    "Borivali East Mumbai IITM":                           "Borivali E",
    "Borivali East Mumbai MPCB":                           "Borivali E",
    "Byculla Mumbai BMC":                                  "Byculla",
    "Chakala-Andheri East Mumbai IITM":                    "Chakala",
    "Chembur Mumbai MPCB":                                 "Chembur",
    "Chhatrapati Shivaji Intl. Airport (T2) Mumbai MPCB":  "Airport T2",
    "Colaba Mumbai MPCB":                                  "Colaba",
    "Deonar Mumbai IITM":                                  "Deonar",
    "Ghatkopar Mumbai BMC":                                "Ghatkopar",
    "Kandivali East Mumbai MPCB":                          "Kandivali E",
    "Kandivali West Mumbai BMC":                           "Kandivali W",
    "Kherwadi Bandra East Mumbai MPCB":                    "Kherwadi",
    "Khindipada-Bhandup West Mumbai IITM":                 "Bhandup W",
    "Kurla Mumbai MPCB":                                   "Kurla",
    "Malad West Mumbai IITM":                              "Malad W",
    "Mazgaon Mumbai IITM":                                 "Mazgaon",
    "Mindspace-Malad West Mumbai MPCB":                    "Mindspace",
    "Mulund West Mumbai MPCB":                             "Mulund W",
    "Navy Nagar-Colaba Mumbai IITM":                       "Navy Nagar",
    "Powai Mumbai MPCB":                                   "Powai",
    "Sewri Mumbai BMC":                                    "Sewri",
    "Shivaji Nagar Mumbai BMC":                            "Shivaji Ngr",
    "Siddharth Nagar-Worli Mumbai IITM":                   "Worli",
    "Sion Mumbai MPCB":                                    "Sion",
    "Vasai West Mumbai MPCB":                              "Vasai W",
    "Vile Parle West Mumbai MPCB":                         "Vile Parle W",
    "Worli Mumbai -MPCB":                                  "Worli MPCB",
}

def short(name):
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    for suffix in [" IITM", " MPCB", " BMC", "- MPCB"]:
        name = name.replace(suffix, "")
    return name.replace("Mumbai", "").replace("-", " ").strip().rstrip(",")


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATION MAP
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor("#F5F5F3")
ax.set_facecolor("#F5F5F3")

for spine in ax.spines.values():
    spine.set_visible(False)

ax.grid(True, color="white", linewidth=1.0, linestyle="-", zorder=0)
ax.set_axisbelow(True)

norm  = Normalize(vmin=station_error.min(), vmax=station_error.max())
cmap  = plt.cm.RdYlGn_r
sizes = ((station_error - station_error.min()) /
         (station_error.max() - station_error.min() + 1e-8) * 1200 + 80)

sc = ax.scatter(
    station_df["lon"], station_df["lat"],
    s          = sizes,
    c          = station_error,
    cmap       = cmap,
    norm       = norm,
    alpha      = 0.88,
    edgecolors = "white",
    linewidths = 1.5,
    zorder     = 3,
)

for _, row in station_df.iterrows():
    label = short(row["station"])
    txt   = ax.annotate(
        label,
        xy         = (row["lon"], row["lat"]),
        xytext     = (0, 9),
        textcoords = "offset points",
        fontsize   = 8,
        fontfamily = "DejaVu Sans",
        fontweight = "bold",
        color      = "#1a1a1a",
        ha         = "center",
        va         = "bottom",
        zorder     = 6,
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=3, foreground="#F5F5F3")
    ])

cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cb.set_label("Mean Absolute Error (AQI units)", fontsize=10, color="#444", labelpad=10)
cb.ax.tick_params(labelsize=9, colors="#555")
cb.outline.set_visible(False)

# Bubble size legend
ref_errors = [station_error.min(),
              (station_error.min() + station_error.max()) / 2,
              station_error.max()]
ref_labels = [f"Low  ({ref_errors[0]:.1f})",
              f"Mid  ({ref_errors[1]:.1f})",
              f"High ({ref_errors[2]:.1f})"]
ref_sizes  = [((e - station_error.min()) /
               (station_error.max() - station_error.min() + 1e-8) * 1200 + 80)
              for e in ref_errors]

for sz, lbl in zip(ref_sizes, ref_labels):
    ax.scatter([], [], s=sz, color="#aaaaaa", alpha=0.75,
               edgecolors="white", linewidths=1.2, label=lbl)

legend = ax.legend(
    title          = "MAE scale",
    title_fontsize = 8,
    fontsize       = 8,
    frameon        = True,
    facecolor      = "#F5F5F3",
    edgecolor      = "#cccccc",
    loc            = "lower left",
    labelspacing   = 1.2,
    handletextpad  = 1.0,
    borderpad      = 0.9,
    scatterpoints  = 1,
)
legend.get_title().set_color("#555")
legend.get_frame().set_linewidth(0.5)

ax.set_title("LSTM Baseline — Prediction Error by Station",
             fontsize=14, fontweight="bold", color="#1a1a1a", pad=14, loc="left")
ax.annotate("Bubble size and colour both encode MAE — larger and redder = higher error",
            xy=(0, 1.012), xycoords="axes fraction",
            fontsize=8.5, color="#888", style="italic")
ax.set_xlabel("Longitude", fontsize=10, color="#666", labelpad=8)
ax.set_ylabel("Latitude",  fontsize=10, color="#666", labelpad=8)
ax.tick_params(colors="#888", labelsize=9)
ax.set_xlim(station_df["lon"].min() - 0.02, station_df["lon"].max() + 0.02)
ax.set_ylim(station_df["lat"].min() - 0.02, station_df["lat"].max() + 0.04)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_by_station.png", dpi=180,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved error_by_station.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ERROR BY HOUR
# ══════════════════════════════════════════════════════════════════════════════
hour_repeated = np.repeat(hours_arr, N)   # (S*N,)
df_hour       = pd.DataFrame({"hour": hour_repeated, "error": errors_flat})
hour_error    = df_hour.groupby("hour")["error"].mean()

bar_colors = []
for h in range(24):
    if h in [7, 8, 17, 18]:
        bar_colors.append("#E05C2A")
    elif h >= 23 or h <= 4:
        bar_colors.append("#4A7EC4")
    else:
        bar_colors.append("#A8BAC4")

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("#F5F5F3")
ax.set_facecolor("#F5F5F3")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis="y", color="white", linewidth=1.2, zorder=0)
ax.set_axisbelow(True)

bars = ax.bar(hour_error.index, hour_error.values,
              color=bar_colors, alpha=0.9, edgecolor="white",
              linewidth=0.8, zorder=3)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#666")

ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   fontsize=8, rotation=45, color="#555")
ax.set_xlabel("Hour of Day (IST)", fontsize=10, color="#666", labelpad=8)
ax.set_ylabel("Mean MAE (AQI units)", fontsize=10, color="#666")
ax.set_title("LSTM Baseline — Prediction Error by Hour of Day",
             fontsize=14, fontweight="bold", color="#1a1a1a", pad=14, loc="left")
ax.tick_params(colors="#888", labelsize=9)

legend = [Patch(color="#E05C2A", label="Rush hour  (7–9 am, 5–7 pm)"),
          Patch(color="#4A7EC4", label="Night  (11 pm–5 am)"),
          Patch(color="#A8BAC4", label="Other hours")]
ax.legend(handles=legend, fontsize=8.5, frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/error_by_hour.png", dpi=180,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved error_by_hour.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ERROR BY MONTH
# ══════════════════════════════════════════════════════════════════════════════
month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
               7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

if months_arr is not None:
    month_repeated = np.repeat(months_arr, N)
    df_month       = pd.DataFrame({"month": month_repeated, "error": errors_flat})
    month_error    = df_month.groupby("month")["error"].mean()
    month_labels   = [month_names[m] for m in month_error.index]

    pollution_months = {10, 11, 12, 1}
    m_colors = ["#E05C2A" if m in pollution_months else "#A8BAC4"
                for m in month_error.index]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#F5F5F3")
    ax.set_facecolor("#F5F5F3")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)

    bars = ax.bar(month_labels, month_error.values,
                  color=m_colors, alpha=0.9, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8, color="#666")

    ax.set_xlabel("Month", fontsize=10, color="#666", labelpad=8)
    ax.set_ylabel("Mean MAE (AQI units)", fontsize=10, color="#666")
    ax.set_title("LSTM Baseline — Prediction Error by Month",
                 fontsize=14, fontweight="bold", color="#1a1a1a", pad=14, loc="left")
    ax.tick_params(colors="#888", labelsize=10)
    ax.annotate(
        "Oct–Jan highlighted — Mumbai's post-monsoon pollution season",
        xy=(0, 1.012), xycoords="axes fraction",
        fontsize=8.5, color="#888", style="italic"
    )

    legend = [Patch(color="#E05C2A", label="Pollution season (Oct–Jan)"),
              Patch(color="#A8BAC4", label="Other months")]
    ax.legend(handles=legend, fontsize=8.5, frameon=False)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/error_by_month.png", dpi=180,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Saved error_by_month.png")
else:
    print("Skipping month plot — timestamps unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTED vs ACTUAL — WORST AND BEST STATIONS
# ══════════════════════════════════════════════════════════════════════════════
sorted_idx  = np.argsort(station_error)
best_idx    = sorted_idx[0]
worst_idxs  = sorted_idx[-3:][::-1]
plot_idxs   = list(worst_idxs) + [best_idx]
plot_labels = ["Worst 1", "Worst 2", "Worst 3", "Best"]

N_SHOW = min(24 * 30, S)

fig, axes = plt.subplots(4, 1, figsize=(15, 11))
fig.patch.set_facecolor("#F5F5F3")
fig.suptitle(
    "LSTM Baseline — Predicted vs Actual AQI  (last 30 days of test set)",
    fontsize=13, fontweight="bold", color="#1a1a1a", y=1.01
)

for ax, idx, label in zip(axes, plot_idxs, plot_labels):
    ax.set_facecolor("#F5F5F3")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(targets[-N_SHOW:, idx], label="Actual",
            color="#2A6496", lw=1.2, alpha=0.9, zorder=3)
    ax.plot(preds[-N_SHOW:,   idx], label="Predicted",
            color="#E05C2A", lw=1.2, alpha=0.85, zorder=4)
    ax.fill_between(range(N_SHOW),
                    targets[-N_SHOW:, idx],
                    preds[-N_SHOW:,   idx],
                    alpha=0.07, color="#E05C2A")

    name  = short(station_df.iloc[idx]["station"])
    ax.set_title(
        f"{label}: {name}   —   MAE = {station_error[idx]:.2f} AQI units",
        fontsize=9, color="#333", loc="left", pad=6
    )
    ax.set_ylabel("AQI", fontsize=8, color="#666")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.tick_params(labelsize=7, colors="#888")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pred_vs_actual.png", dpi=180,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved pred_vs_actual.png")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
peak_hour  = int(hour_error.idxmax())
peak_month = int(month_error.idxmax()) if months_arr is not None else None

print("\n── ERROR ANALYSIS SUMMARY ───────────────────────────────")
print(f"  Overall MAE     : {errors.mean():.2f} AQI units")
print(f"  Worst station   : {short(station_df.sort_values('error', ascending=False).iloc[0]['station'])}")
print(f"                    MAE = {station_error.max():.2f}")
print(f"  Best station    : {short(station_df.sort_values('error').iloc[0]['station'])}")
print(f"                    MAE = {station_error.min():.2f}")
print(f"  Peak error hour : {peak_hour:02d}:00")
if peak_month:
    print(f"  Peak error month: {month_names[peak_month]}")
print(f"  Plots saved to  : {PLOT_DIR}/")
print("─────────────────────────────────────────────────────────")
