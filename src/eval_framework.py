import json
from pathlib import Path

SAVE_DIR = Path("models/baselines")
HORIZON  = 1

print(f"\n── Model Comparison | Horizon = {HORIZON}h ─────────────────────")

files = sorted(SAVE_DIR.glob(f"*results_h{HORIZON}.json"))

if not files:
    print("No result files found.")
    exit()

def fmt(val, fmt_str="{:.2f}", suffix=""):
    if val is None:
        return "—"
    if isinstance(val, str):
        return val
    try:
        return fmt_str.format(val) + suffix
    except:
        return "—"
      
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>8} {'Time(s)':>10} {'Params':>10}")
print("─" * 85)

for fpath in files:
    with open(fpath) as f:
        r = json.load(f)

    model = r.get("model", fpath.stem)

    mae   = r.get("MAE")
    rmse  = r.get("RMSE")
    mape  = r.get("MAPE")
    r2    = r.get("R2")

    time   = r.get("training_time", "N/A")
    params = r.get("n_parameters", "N/A")

    print(
        f"{model:<25} "
        f"{fmt(mae):>8} "
        f"{fmt(rmse):>8} "
        f"{fmt(mape, '{:.1f}', '%'):>8} "
        f"{fmt(r2, '{:.3f}'):>8} "
        f"{fmt(time, '{:.1f}'):>10} "
        f"{fmt(params, '{:,.0f}'):>10}"
    )

print("─" * 85)
print(f"Total models compared: {len(files)}")
