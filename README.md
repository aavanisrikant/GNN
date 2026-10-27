# Mumbai AQI forecasting with graph neural networks

Forecasting hourly air quality (AQI) at **30 CPCB monitoring stations across Mumbai** 1, 3 and 6 hours ahead. Each station is a node in a graph, connected to nearby stations by inverse distance. The graph models combine this spatial structure with each station's recent AQI history and the urban form around it (roads, land use, buildings, distance to the coast).

| | |
|---|---|
| **Data** | Hourly AQI, Jan 2019 – Dec 2025, Central Pollution Control Board (CPCB) |
| **Graph** | 30 stations, edges between stations ≤ 30 km apart, weight = 1 / distance |
| **Node features** | AQI (last 24 h) + 7 OpenStreetMap morphology features (2 km buffer) |
| **Models** | T-GCN, T-GCN with rush-hour/night edge regimes, T-GAT (graph attention) |
| **Baselines** | Persistence, inverse-distance weighting, per-station LSTM |

## Pipeline

```
CPCB .xlsx ─► data_collection ─► clean_cpcb.csv ─► node_features ─► X / y / mask windows
                                                         │
station_coordinates ─► graph_construction ─► edge_index, edge_weight
                    └► morphology (OSM) ───► node_static_features (N × 7)
                                                         │
        baselines (persistence · IDW · LSTM)   graph models (T-GCN · temporal · T-GAT)
                                    └──────────┬──────────┘
                     final_evaluation ─► eval_framework ─► error_analysis ─► dashboard
```

## Quick start

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate gnn-env

# put clean_cpcb.csv in data/processed/   (or the raw CPCB .xlsx files in data/raw/CPCB/)
python main.py --stages coords morphology   # once, needs internet (OpenStreetMap)
python main.py                              # features → graph → baselines → train → compare → plots

streamlit run app/dashboard.py              # interactive results
```

`python main.py --quick` does a small CPU run (most recent 4 000 training samples, 8 epochs, 1 h horizon) to check everything works before a full run. Training uses a GPU automatically if one is available.

Every script can also be run on its own from the repository root, e.g. `python src/final_evaluation.py --models tgcn tgat --horizons 1 3`.

## Repository layout

```
main.py                     pipeline runner (stages, horizons, quick mode)
app/dashboard.py            Streamlit + Plotly results dashboard
src/
  config.py                 paths, constants, station registry + name matching
  metrics.py                masked metrics / losses, result writer
  ── data ──
  data_collection.py        raw CPCB Excel → clean_cpcb.csv
  node_features.py          sliding windows, causal gap filling, masks, splits
  station_coordinates.py    geocode stations (osmnx)
  graph_construction.py     distance graph, node order aligned with AQI columns
  morphology.py             OSM road / land-use / building / coast features
  data_loader.py            load_real_data(): tensors shaped (B, T, N, F)
  ── models ──
  gcn.py, tgcn_graph.py     GCN layer built from scratch (MessagePassing / sparse)
  gru_component.py          T-GCN GRU cell (graph conv inside the gates)
  tgcn_model.py             stacked T-GCN
  temporal_edge_weights.py  T-GCN with rush-hour / normal / night adjacency
  attention_mechanism.py    T-GAT (GAT inside the GRU) + attention maps
  training_loop.py          training with early stopping, LR schedule, masked loss
  ── evaluation ──
  idw_baseline.py           persistence + IDW baselines
  lstm_baseline.py          per-station LSTM baseline
  final_evaluation.py       train all graph models, score once on the test set
  eval_framework.py         comparison table across models and horizons
  error_analysis.py         error by station / hour / month, predicted vs actual
  ── development experiments (validation set) ──
  tgcn_train.py             Day 27 – first T-GCN training run + diagnostics
  morphology_ablation.py    Day 28 – AQI only vs + roads vs + all morphology
  (temporal_edge_weights.py Day 29 and attention_mechanism.py Day 30 also run as experiments)
  shapes_check.py           shape / NaN report for every processed array
tests/                      pytest: model shapes, batching equivalence, masks, leakage
```

## Evaluation protocol

* **Chronological split** 70 / 15 / 15 % (train to Nov 2023, validation to Dec 2024, test to Dec 2025). Samples are assigned by the time of their *target*, so no target leaks across splits.
* **Only real measurements are scored.** About half of the station-hours are missing (many stations only came online in 2022–23). Gaps are filled for the *inputs* so models can run, but a mask records which targets were really measured. Losses and metrics ignore the filled ones.
* **Causal gap filling.** Short gaps are forward-filled (≤ 12 h). Longer ones get the station's training-set median. No back-filling or interpolation, which would copy future values into the input window.
* **Same test set, same metric code, real AQI units** for every model.
* **Residual forecasting** (default in `final_evaluation.py`). The graph models predict the *change* from the last observed AQI and start training at exactly the persistence forecast. CPCB AQI is built from 24 h rolling averages, so it moves slowly and "same as last hour" is a strong baseline. Use `--no-residual` to predict absolute values.
* **MAE training loss** (default, `--loss mse|mae|huber`). About 1% of hours, mostly right after a data gap, contain sudden jumps that make up 85–90% of the squared error. MSE training is dominated by them; MAE matches the reported metric.

## Results

`python main.py` writes the full table to `results/model_comparison.md` and per-model plots to `results/plots/`.

**Verification run** (test set Dec 2024 – Dec 2025, MAE in AQI points, observed hours only):

| Model | 1 h | 3 h | 6 h |
|---|---:|---:|---:|
| Persistence (last value) | **1.74** | **4.35** | **7.38** |
| IDW (spatial) | 23.53 | 23.60 | 23.80 |
| LSTM (per station) | 1.92 | 5.52 | 8.57 |
| T-GCN (AQI only) | 1.76 | | |
| T-GCN (AQI + morphology) | 1.75 | 4.35 | 7.38 |
| T-GCN + temporal edges | 1.75 | | |
| T-GAT (attention) | 1.74 | | |

<sub>Trained on the most recent 4 000–8 000 training hours for 8–15 epochs on a CPU. Graph models used approximate station coordinates and no morphology (OpenStreetMap was not reachable in that environment), so rerun `python main.py` with your real `station_coords.csv` and `node_static_features.npy` for final numbers. Persistence and LSTM do not depend on either.</sub>

**What this shows.** Because CPCB AQI is a smoothed 24-hour index, *the last observed value* is already very accurate. No model beats it meaningfully. The graph models learn to stay at persistence, and the LSTM, which has to rebuild the level from scratch, does worse. Even a linear model on the last 24 hourly changes plus hour of day only reaches 1.73 / 7.31 (1 h / 6 h). The AQI history itself carries almost no information about the next change. A 24 h rolling index moves according to the hour that leaves the window and the hour that enters it, and neither is visible in the AQI series.

**Where the gains are likely to be**

1. **Raw pollutant concentrations** (PM2.5, PM10, NO₂ … hourly) as inputs or as the target. `data_collection.py` already parses every CPCB parameter. Knowing the concentration from 24 h ago tells the model exactly what leaves the window.
2. **Weather** (wind speed / direction, humidity, boundary-layer height). Wind direction is also the natural thing to put on the graph edges, in place of fixed distance or hour-of-day regimes.
3. **Time features** (hour of day, day of week) as node inputs.
4. **Gap handling.** Most of the largest errors occur in the first hours after a data gap, where the inputs are filled-in values. Adding "is this input observed?" as a feature lets the model tell real readings from filled ones.

`eval_framework.py` reports every model as **MAE vs persistence**, so any improvement from these changes will be visible right away.

## Development experiments (validation set)

The Day 27–30 scripts compare design choices on the validation split. Add `--real` to run them on the processed data instead of synthetic tensors:

```bash
python src/tgcn_train.py           --real
python src/morphology_ablation.py  --real --max_epochs 30
python src/temporal_edge_weights.py --real
python src/attention_mechanism.py  --real          # also writes an attention map
```

Outputs go to `results/experiments/day2X_*/`. These numbers are in normalised units on the validation set; the headline numbers come from `final_evaluation.py`.

## Tests

```bash
pytest -q
```

Checks that the batched models give exactly the same output as the original per-sample loops, that the dense attention matches PyG's `GATConv`, horizon alignment, causal imputation and masked metrics.

## Data

Air-quality data: [Central Pollution Control Board, India](https://airquality.cpcb.gov.in/ccr/) (hourly AQI per station). Map and morphology data: © OpenStreetMap contributors. `data/` is git-ignored apart from `station_coords.csv`; put `clean_cpcb.csv` in `data/processed/` or the raw Excel files in `data/raw/CPCB/`.
