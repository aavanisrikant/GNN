"""
Run the whole pipeline from the repository root.

    python main.py                       # everything that does not need internet
    python main.py --stages all          # also re-download coords + OSM morphology
    python main.py --stages train compare --horizons 1
    python main.py --quick               # small, fast CPU run to check everything works

Stages (in order)
  collect     raw CPCB .xlsx → data/processed/clean_cpcb.csv        (needs data/raw/CPCB)
  features    sliding windows, masks, normalisation
  coords      geocode stations                                     (internet)
  graph       distance graph between stations
  morphology  OSM road / land-use / building features              (internet)
  baselines   persistence, IDW, LSTM
  train       T-GCN, T-GCN (AQI only), temporal T-GCN, T-GAT  → test metrics
  compare     results/model_comparison.md
  analyze     error-analysis plots for the main model
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

ALL_STAGES = ["collect", "features", "coords", "graph", "morphology",
              "baselines", "train", "compare", "analyze"]
DEFAULT_STAGES = ["features", "graph", "baselines", "train", "compare", "analyze"]


def run(script, *args):
    cmd = [PY, f"src/{script}", *map(str, args)]
    print("\n" + "▶ " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"✓ {script} ({time.time() - t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", default=DEFAULT_STAGES, choices=ALL_STAGES + ["all"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 6])
    p.add_argument("--models", nargs="+", default=None, help="subset of graph models for 'train'")
    p.add_argument("--max_train", type=int, default=None, help="most recent N training samples")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--quick", action="store_true", help="max_train 4000, 8 epochs, horizon 1")
    args = p.parse_args()

    stages = ALL_STAGES if "all" in args.stages else [s for s in ALL_STAGES if s in args.stages]

    if args.quick:
        args.max_train = args.max_train or 4000
        args.epochs = 8
        args.horizons = [1]

    train_args = ["--epochs", args.epochs]
    if args.max_train:
        train_args += ["--max_train", args.max_train]

    for stage in stages:
        if stage == "collect":
            run("data_collection.py")
        elif stage == "features":
            run("node_features.py")
        elif stage == "coords":
            run("station_coordinates.py")
        elif stage == "graph":
            run("graph_construction.py")
        elif stage == "morphology":
            run("morphology.py")
        elif stage == "baselines":
            for h in args.horizons:
                run("idw_baseline.py", "--horizon", h)
                run("lstm_baseline.py", "--horizon", h, *train_args)
        elif stage == "train":
            extra = ["--models", *args.models] if args.models else []
            run("final_evaluation.py", "--horizons", *args.horizons, *train_args, *extra)
        elif stage == "compare":
            run("eval_framework.py")
        elif stage == "analyze":
            for h in args.horizons:
                run("error_analysis.py", "--model", "tgcn", "--horizon", h)

    print("\nAll done. Results: results/model_comparison.md  ·  Dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
