import numpy as np

files = [
    "X_train_h1.npy",
    "y_train_h1.npy",
    "X_val_h1.npy",
    "y_val_h1.npy",
    "X_test_h1.npy",
    "y_test_h1.npy",
    "edge_index.npy",
    "edge_weight.npy",
    "node_static_features.npy",
]

BASE = "data/processed"

for f in files:

    arr = np.load(f"{BASE}/{f}")

    print(f"{f}")
    print(f"shape = {arr.shape}")
    print(f"dtype = {arr.dtype}")

    if np.isnan(arr).any():
        print("NaNs detected")

    print("-" * 50)
