import pandas as pd
import numpy as np
import json
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from collections import deque

# Configuration
INPUT_PATH      = "data/processed/station_coords.csv"
OUTPUT_DIR      = Path("data/processed")
DIST_THRESHOLD_KM = 30.0
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Haversine distance function 
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

#BFS
def check_connectivity(edge_index: np.ndarray, n_nodes: int, station_names: list):
    adj = {i: [] for i in range(n_nodes)}
    if edge_index.shape[1] > 0:
        for src, dst in zip(edge_index[0], edge_index[1]):
            adj[src].append(dst)

    visited = set()
    queue = deque([0])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj[node])

    isolated = set(range(n_nodes)) - visited
    is_connected = len(isolated) == 0

    print(f"  Graph fully connected: {is_connected}")
    if not is_connected:
        print(f"  Isolated nodes ({len(isolated)}):")
        for idx in isolated:
            print(f"      [{idx}] {station_names[idx]}")
        print("  → Consider raising DIST_THRESHOLD_KM or dropping these nodes.")

    return is_connected, isolated

df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

stations = df["station"].tolist()
coords   = df[["lat", "lon"]].values
N        = len(stations)
print(f"Total stations (nodes): {N}")

df["node_idx"] = df.index
df.to_csv(OUTPUT_DIR / "station_inventory.csv", index=False)
print("Saved station_inventory.csv")

# Node mapping
node_mapping = {station: i for i, station in enumerate(stations)}
with open(OUTPUT_DIR / "node_mapping.json", "w") as f:
    json.dump(node_mapping, f, indent=4)
print("Saved node_mapping.json")

# edges + adjacency matrix
adj_matrix  = np.zeros((N, N), dtype=np.float32)
edge_list   = []   
weight_list = []  

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        if dist <= DIST_THRESHOLD_KM and dist > 0:
            weight = 1.0 / dist
            adj_matrix[i, j] = weight
            edge_list.append([i, j])
            weight_list.append(weight)

if edge_list:
    edge_index  = np.array(edge_list, dtype=np.int64).T   
    edge_weight = np.array(weight_list, dtype=np.float32) 
else:
    edge_index  = np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.zeros((0,),   dtype=np.float32)
    print(" WARNING: No edges found")


np.save(OUTPUT_DIR / "adj_matrix.npy",  adj_matrix)
np.save(OUTPUT_DIR / "edge_index.npy",  edge_index)
np.save(OUTPUT_DIR / "edge_weight.npy", edge_weight)
print("Saved adj_matrix.npy")
print("Saved edge_index.npy")
print("Saved edge_weight.npy")

edge_df = pd.DataFrame({
    "src_idx":      edge_index[0],
    "dst_idx":      edge_index[1],
    "src_station":  [stations[i] for i in edge_index[0]],
    "dst_station":  [stations[i] for i in edge_index[1]],
    "distance_km":  [1.0 / w for w in edge_weight],
    "weight":       edge_weight,
})
edge_df.to_csv(OUTPUT_DIR / "edge_list.csv", index=False)
print("Saved edge_list.csv")

print("\n--- GRAPH STATS ---")
print(f"  Nodes:             {N}")
print(f"  Edges (directed):  {edge_index.shape[1]}")
print(f"  Edges (undirected): {edge_index.shape[1] // 2}")
print(f"  Adjacency shape:   {adj_matrix.shape}")

degrees = (adj_matrix > 0).sum(axis=1)
print(f"  Min degree:        {int(degrees.min())}")
print(f"  Max degree:        {int(degrees.max())}")
print(f"  Avg degree:        {degrees.mean():.2f}")

print(f"\n  Per-station degrees:")
for i, station in enumerate(stations):
    print(f"    [{i:02d}] {station:<50s} degree={int(degrees[i])}")

print("\n--- CONNECTIVITY ---")
check_connectivity(edge_index, N, stations)

print("\n Phase 1 complete. Outputs in:", OUTPUT_DIR)
