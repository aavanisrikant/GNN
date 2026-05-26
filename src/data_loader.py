import numpy as np
import torch
# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/processed"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD NUMPY FILES
# ============================================================

print("=" * 70)
print("REAL DATA LOADER")
print("=" * 70)

print("\nLoading processed tensors...")


X_train = np.load(f"{DATA_DIR}/X_train_h1.npy")
y_train = np.load(f"{DATA_DIR}/y_train_h1.npy")

X_val = np.load(f"{DATA_DIR}/X_val_h1.npy")
y_val = np.load(f"{DATA_DIR}/y_val_h1.npy")

X_test = np.load(f"{DATA_DIR}/X_test_h1.npy")
y_test = np.load(f"{DATA_DIR}/y_test_h1.npy")

edge_index = np.load(f"{DATA_DIR}/edge_index.npy")
edge_weight = np.load(f"{DATA_DIR}/edge_weight.npy")

node_static = np.load(
    f"{DATA_DIR}/node_static_features.npy"
)


# ============================================================
# PRINT ORIGINAL SHAPES
# ============================================================

print("\nOriginal Shapes:")

print(f"X_train : {X_train.shape}")
print(f"y_train : {y_train.shape}")

print(f"X_val   : {X_val.shape}")
print(f"y_val   : {y_val.shape}")

print(f"X_test  : {X_test.shape}")
print(f"y_test  : {y_test.shape}")

print(f"edge_index : {edge_index.shape}")
print(f"edge_weight: {edge_weight.shape}")

print(f"node_static: {node_static.shape}")


# ============================================================
# CURRENT FORMAT
# (samples, nodes, timesteps)
#
# T-GCN NEEDS:
# (samples, timesteps, nodes, features)
# ============================================================

print("\nTransposing dimensions...")

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

print(f"X_train transposed: {X_train.shape}")


# ============================================================
# ADD FEATURE DIMENSION
#
# FROM:
# (B,T,N)
#
# TO:
# (B,T,N,1)
# ============================================================

print("\nAdding AQI feature dimension...")

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

print(f"X_train expanded: {X_train.shape}")
print(f"y_train expanded: {y_train.shape}")


# ============================================================
# BUILD MORPHOLOGY FEATURES
#
# node_static shape:
# (N,7)
#
# Need:
# (B,T,N,7)
# ============================================================

print("\nBroadcasting morphology features...")

B_train, T, N, _ = X_train.shape
B_val = X_val.shape[0]
B_test = X_test.shape[0]

# (1,1,N,7)
node_static_expanded = np.expand_dims(
    node_static,
    axis=(0, 1),
)

print(
    f"node_static_expanded: "
    f"{node_static_expanded.shape}"
)

# Repeat across batch + time
static_train = np.repeat(
    node_static_expanded,
    repeats=B_train,
    axis=0,
)

static_train = np.repeat(
    static_train,
    repeats=T,
    axis=1,
)

static_val = np.repeat(
    node_static_expanded,
    repeats=B_val,
    axis=0,
)

static_val = np.repeat(
    static_val,
    repeats=T,
    axis=1,
)

static_test = np.repeat(
    node_static_expanded,
    repeats=B_test,
    axis=0,
)

static_test = np.repeat(
    static_test,
    repeats=T,
    axis=1,
)

print(f"static_train: {static_train.shape}")


# ============================================================
# CONCAT AQI + MORPHOLOGY
#
# AQI:
# (B,T,N,1)
#
# morphology:
# (B,T,N,7)
#
# FINAL:
# (B,T,N,8)
# ============================================================

print("\nConcatenating AQI + morphology...")

X_train_full = np.concatenate(
    [X_train, static_train],
    axis=-1,
)

X_val_full = np.concatenate(
    [X_val, static_val],
    axis=-1,
)

X_test_full = np.concatenate(
    [X_test, static_test],
    axis=-1,
)

print(f"X_train_full: {X_train_full.shape}")
print(f"X_val_full  : {X_val_full.shape}")
print(f"X_test_full : {X_test_full.shape}")


# ============================================================
# NAN / INF CHECK
# ============================================================

print("\nChecking NaNs / Infs...")


def check_tensor(name, arr):

    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:

        print(f"[FAIL] {name}")

        print(f"NaN: {has_nan}")
        print(f"Inf: {has_inf}")

    else:

        print(f"[PASS] {name}")


check_tensor("X_train_full", X_train_full)
check_tensor("X_val_full", X_val_full)
check_tensor("X_test_full", X_test_full)

check_tensor("y_train", y_train)
check_tensor("y_val", y_val)
check_tensor("y_test", y_test)


# ============================================================
# CONVERT TO TORCH
# ============================================================

print("\nConverting to torch tensors...")

X_train_full = torch.tensor(
    X_train_full,
    dtype=torch.float32,
).to(DEVICE)

X_val_full = torch.tensor(
    X_val_full,
    dtype=torch.float32,
).to(DEVICE)

X_test_full = torch.tensor(
    X_test_full,
    dtype=torch.float32,
).to(DEVICE)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32,
).to(DEVICE)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32,
).to(DEVICE)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32,
).to(DEVICE)

edge_index = torch.tensor(
    edge_index,
    dtype=torch.long,
).to(DEVICE)

edge_weight = torch.tensor(
    edge_weight,
    dtype=torch.float32,
).to(DEVICE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL REAL DATA SUMMARY")
print("=" * 70)

print("\nAQI-only tensor:")
print(f"Shape: {X_train.shape}")

print("\nAQI + morphology tensor:")
print(f"Shape: {X_train_full.shape}")

print("\nTargets:")
print(f"Shape: {y_train.shape}")

print("\nGraph:")
print(f"edge_index : {edge_index.shape}")
print(f"edge_weight: {edge_weight.shape}")

print("\nFeature meaning:")
print("  Feature 0 = AQI")
print("  Features 1-7 = morphology")

print("\nDevice:")
print(DEVICE)

print("\nREADY FOR REAL T-GCN TRAINING")
print("=" * 70)


# ============================================================
# OPTIONAL EXPORTS
# ============================================================

REAL_DATA = {

    "X_train_aqi": X_train,
    "X_val_aqi": X_val,
    "X_test_aqi": X_test,

    "X_train_full": X_train_full,
    "X_val_full": X_val_full,
    "X_test_full": X_test_full,

    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,

    "edge_index": edge_index,
    "edge_weight": edge_weight,
}


if __name__ == "__main__":

    print("\nLoader test complete.")
