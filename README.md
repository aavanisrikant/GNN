# Mumbai AQI forecasting with graph neural networks

This project forecasts hourly air quality (AQI) at **30 CPCB monitoring stations across Mumbai**. Each station is a node in a graph, and nearby stations are connected by edges weighted by inverse distance. The graph models combine this spatial structure with each station's last 24 hours of AQI and with features describing the urban area around the station (roads, land use, buildings, distance to the coast).

| | |
|---|---|
| **Data** | Hourly AQI per station, Central Pollution Control Board (CPCB), 2019 onwards |
| **Graph** | 30 stations, edges between stations ≤ 30 km apart, weight = 1 / distance (km) |
| **Node features** | AQI over the last 24 h + 7 OpenStreetMap morphology features (2 km buffer) |
| **Models** | T-GCN, T-GCN with rush-hour / normal / night adjacency, T-GAT (graph attention) |
| **Baselines** | Inverse-distance weighting (IDW), per-station LSTM |
| **Horizons** | 1, 3 and 6 steps ahead |

## Pipeline

```
CPCB .xlsx ─► data_collection ─► clean_cpcb.csv ─► node_features ─► X / y windows (train / val / test)
                                                         │
station_coordinates ─► graph_construction ─► adj_matrix, edge_index, edge_weight
                    └► morphology (OSM) ───► node_static_features (N × 7)
                                                         │
                     data_loader ─► tensors shaped (B, T, N, F), F = 1 AQI + 7 morphology
                                                         │
        baselines (IDW · LSTM)        graph models (T-GCN · temporal T-GCN · T-GAT)
                    └──────────────┬───────────────┘
                     eval_framework ─► error_analysis ─► plots
```

## Setup

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate gnn-env
```

Put the raw CPCB Excel downloads in `data/raw/CPCB/`. The parser reads the parameter from the first underscore-separated part of each filename (for example `AQI_...`) and the month name and four-digit year from the rest of the name. Each file holds one station-month, with days as rows and hours as columns.

## Running

All scripts run from the repository root.

**Data, graph and baselines** (the `coords` and `morphology` stages need internet access for OpenStreetMap):

```bash
python main.py --stages collect features coords graph morphology baselines compare --horizons 1
```

or one step at a time:

```bash
python src/data_collection.py        # raw .xlsx → data/processed/clean_cpcb.csv
python src/node_features.py          # 24 h windows, train-set min–max scaling, 70/15/15 split
python src/station_coordinates.py    # geocode stations → data/processed/station_coords.csv
python src/graph_construction.py     # distance graph + connectivity check
python src/morphology.py             # OSM features → data/processed/node_static_features.npy
python src/shapes_check.py           # shape / NaN report for the processed arrays
python src/idw_baseline.py           # IDW baseline, test metrics in AQI units
python src/lstm_baseline.py          # LSTM baseline, test metrics in AQI units
python src/eval_framework.py         # comparison table of saved baseline results
python src/error_analysis.py         # LSTM error by station, hour and month → plots/
```

**T-GCN training on the processed data:**

```bash
python src/training_loop.py          # T-GCN on AQI + morphology, early stopping, LR schedule
```

**Model checks** (run on small random graphs):

```bash
python src/gcn.py
python src/tgcn_graph.py
python src/gru_component.py
python src/tgcn_model.py
```

**Experiment scripts.** Each one takes `--X_train --y_train --X_val --y_val` (`.npy` arrays shaped `(samples, 24, N, F)` and `(samples, N, 1)`) plus `--edge_index --edge_weight`. Without them, it runs on synthetic data as a smoke test.

```bash
python src/result1.py                 # first T-GCN training run with diagnostics (gradient flow, output range)
python src/morphology_ablation1.py    # AQI only vs AQI + roads vs AQI + all morphology
python src/temporal_edge_weights.py   # T-GCN with rush-hour / normal / night adjacency (--hours_train, --hours_val)
python src/attention_mechanism.py     # T-GAT vs T-GCN, plus a map of the strongest attention edges
```

## Methods

### Data preparation (`data_collection.py`, `node_features.py`)
- Each CPCB Excel sheet (days × hours) is reshaped to long format with Asia/Kolkata timestamps, then pivoted to one column per station on an hourly grid.
- Gaps are filled with forward-fill, back-fill and linear interpolation (each up to 12 h), then with the station median.
- Inputs are 24-hour windows. Values are min–max scaled per station using training-period statistics only.
- The split is chronological, 70 / 15 / 15 %.

### Graph (`station_coordinates.py`, `graph_construction.py`)
- Stations are geocoded with OSMnx. Four stations use manually set coordinates.
- Edges connect stations within 30 km (haversine distance), weighted 1 / distance.
- A breadth-first search checks that the graph is connected and reports any isolated stations.

### Urban morphology (`morphology.py`)
For each station, the script uses a 2 km buffer projected to UTM:

| Feature | Definition |
|---|---|
| `road_density` | drivable road length / buffer area |
| `arterial_fraction` | share of road length that is motorway, trunk, primary or secondary |
| `industrial_fraction` | industrial / port / quarry land-use area / buffer area |
| `residential_fraction` | residential land-use area / buffer area |
| `green_fraction` | parks, grass, forest and similar / buffer area |
| `building_density` | building footprint area / buffer area |
| `coastline_distance` | km to a simplified western-coastline line |

OSM responses are cached per station. Features are min–max scaled across stations.

### Models
- **Graph convolution (`gcn.py`, `tgcn_graph.py`)**, written from scratch: `H = D^{-1/2}(A + I)D^{-1/2} X W + b` with weighted degrees. There is a PyG `MessagePassing` version and a sparse-matrix version.
- **T-GCN cell (`gru_component.py`)**: a GRU whose update, reset and candidate transforms are graph convolutions over `[x_t, h_{t-1}]`.
- **T-GCN (`tgcn_model.py`)**: two stacked T-GCN cells unrolled over the 24-hour window, with dropout and a linear output head for each node.
- **Temporal T-GCN (`temporal_edge_weights.py`)**: the normalised adjacency is scaled by hour of day (×1.5 at rush hour, 07–09 h and 17–19 h; ×0.7 at night, 23–05 h; ×1.0 otherwise).
- **T-GAT (`attention_mechanism.py`)**: the graph convolutions in the GRU are replaced by 4-head `GATConv` layers. Attention weights from the last step can be exported and drawn on a station map.

### Training (`training_loop.py`)
Adam (lr 1e-3), MSE loss, gradient clipping at 5.0, `ReduceLROnPlateau` (factor 0.5), early stopping (patience 10), and checkpointing of the best validation model.

### Baselines
- **IDW (`idw_baseline.py`)**: each station is predicted as the inverse-distance-weighted average of the other stations' latest values.
- **LSTM (`lstm_baseline.py`)**: a single-layer LSTM (64 hidden units) applied to each station's 24-hour series separately, with no spatial information.

### Evaluation
MAE, RMSE, MAPE (on targets > 1) and R² on the test split. Baselines report these in real AQI units. `eval_framework.py` collects them into one table. `error_analysis.py` breaks the error down by station (with a map), by hour of day (rush hours highlighted) and by month (post-monsoon/winter highlighted), and plots predicted against actual values.

## Limitations and next steps

- **Persistence is a strong baseline.** CPCB AQI is computed from rolling 24-hour averages, so it changes slowly from hour to hour. Any model should be compared with simply repeating the last observed value.
- **Imputed values.** Many stations started reporting only in 2022–23, so a large share of station-hours are filled in. Back-fill and interpolation use values from after the forecast time. Training and scoring currently include imputed targets. Next: forward-fill only, plus a mask so that only real measurements are scored.
- **Target indexing.** In `node_features.py` the target is `data[t + h]` for a window ending at `t − 1`, which is h + 1 hours ahead. Next: use `data[t + h − 1]`.
- **Station alignment.** AQI columns and graph / morphology rows are matched by position. Next: join them explicitly by station name.
- **Temporal edges** apply the same multiplier to every edge, so they change the overall scale but not which neighbours matter. Next: learn separate weights for each regime, or use wind direction to weight edges.
- **More inputs.** Raw pollutant concentrations (PM2.5, PM10, NO₂, already parsed by `data_collection.py`), weather (wind, humidity) and time features are the most promising additions.

## Repository layout

```
main.py                       pipeline runner (stages, horizons)
environment.yml, requirements.txt
src/
  data_collection.py          raw CPCB Excel → clean_cpcb.csv
  node_features.py            windows, imputation, scaling, splits
  station_coordinates.py      geocoding (OSMnx)
  graph_construction.py       distance graph, adjacency, connectivity check
  morphology.py               OSM road / land-use / building / coast features
  data_loader.py              builds (B, T, N, F) tensors with morphology
  shapes_check.py             shape / NaN report
  gcn.py, tgcn_graph.py       graph convolution layers
  gru_component.py            T-GCN GRU cell
  tgcn_model.py               stacked T-GCN
  training_loop.py            training loop
  temporal_edge_weights.py    T-GCN with time-of-day adjacency
  attention_mechanism.py      T-GAT and attention maps
  result1.py                  first T-GCN run with diagnostics
  morphology_ablation1.py     morphology feature ablation
  idw_baseline.py             IDW baseline
  lstm_baseline.py            LSTM baseline
  eval_framework.py           results table
  error_analysis.py           error breakdown plots
```

## Data sources

Air-quality data: [Central Pollution Control Board, India](https://airquality.cpcb.gov.in/ccr/). Map and morphology data: © OpenStreetMap contributors.
