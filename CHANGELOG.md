# Changelog

## October 2026 — project completion and cleanup

### Correctness fixes (these change results)

- **Horizon off-by-one.** `node_features.py` used `data[t + h]` as the target for a window ending at `t - 1`, so "1 h ahead" was really 2 h ahead (3 → 4 h, 6 → 7 h). Targets are now `data[t + h - 1]`.
- **Future data leaking into inputs.** Gaps were back-filled and interpolated, which copied values from after the forecast time into the input window. Gap filling is now causal: forward-fill up to 12 h, then the station's *training-set* median.
- **Scoring on imputed values.** About 47% of station-hours are missing (many stations started in 2022–23), yet models were trained and scored on median-filled values. A per-target mask now records real measurements; losses and metrics use only those.
- **Split leakage.** Samples are assigned to train / val / test by the time of their target, not their input start.
- **LSTM predictions saved wrongly.** `(S·N, 1)` predictions were broadcast against `(N,)` min / max, giving an `(S·N, N)` array. Now reshaped to `(S, N)` first. The LSTM is also evaluated on the full test set (previously the first 2 000 samples).
- **IDW averaged normalised values** from stations with different scalings. It now averages in real AQI units (test MAE 39.8 → 23.5).
- **Station order.** `station_coords.csv` (geocoder names, e.g. `Bandra, Mumbai - MPCB`) and `clean_cpcb.csv` (`Bandra Mumbai MPCB`) used different names, so graph nodes, morphology rows and AQI columns could silently fall out of line. All names are now matched through `config.canonical_station()` and every array follows `station_order.json`.
- **T-GAT attention export.** Attention weights include self-loops, but they were indexed against the edge list without them (wrong pairs or an `IndexError`). They are now mapped back to (source, target) pairs, and the map / report show attention *relative to uniform*, since raw α is inflated for stations with few neighbours.
- **Hour of day for temporal edges** came from `sample_index % 24`; it now comes from the real timestamp.
- **UTM zone** for morphology: Mumbai is in zone 43N (EPSG:32643), not 44N.

### New

- `final_evaluation.py` trains T-GCN, T-GCN (AQI only), temporal T-GCN and T-GAT for 1 / 3 / 6 h and scores everything once on the test set in AQI units.
- Persistence baseline (`idw_baseline.py`). It is the bar to beat: test MAE 1.74 at 1 h.
- Residual forecasting (start from persistence, learn the change) and masked MAE / Huber losses.
- `eval_framework.py`: multi-horizon table with "MAE vs persistence" → `results/model_comparison.md`.
- `error_analysis.py --model X --horizon h` works for any model; uses real timestamps and only observed hours.
- Streamlit dashboard (`app/dashboard.py`).
- `main.py` pipeline runner with stages and `--quick` mode.
- `--real` flag on the Day 27–30 experiment scripts (they previously needed hand-made 4-D `.npy` files or used synthetic data).
- `config.py` (paths, station registry), `metrics.py` (shared metrics), tests, README, `.gitignore`, `requirements.txt`.

### Performance

- T-GCN, temporal T-GCN and T-GAT process the whole batch at once instead of looping over samples (identical outputs, verified in `tests/`). T-GCN training ≈ 5–10× faster.
- T-GAT computes attention densely (30 × 30) from the same `GATConv` parameters (checkpoint-compatible, verified against PyG). This uses about 5× less memory per sample; the sparse batched version needed > 6 GB at batch size 64.
- `data_loader.py` no longer loads every array onto the GPU at import time; `training_loop.py` imports it lazily.

### Removed / renamed

- `result1.py` → `tgcn_train.py`, `morphology_ablation1.py` → `morphology_ablation.py`.
- Unused helpers in `data_collection.py` (`handle_missing_values`, `train_test_split_time`, `normalize_train_test`, `create_sequences`), superseded by `node_features.py`.
- `torchvision`, `torchaudio` and `earthengine-api` dropped from the environment (not used). Add `earthengine-api` back if you start on satellite features.
