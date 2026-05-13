import pandas as pd
import numpy as np
import osmnx as ox
import geopandas as gpd

from shapely.geometry import (
    Point,
    MultiLineString,
    LineString
)

from shapely.ops import (
    nearest_points,
    unary_union
)

from pathlib import Path

import json
import warnings
import os
import time

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────
RADIUS_M = 2000

COORDS_PATH = "data/processed/station_coords.csv"

CACHE_DIR = "data/morphology_cache"

SAVE_CSV = "data/processed/node_static_features.csv"
SAVE_NPY = "data/processed/node_static_features.npy"

# Better CRS for Mumbai region
UTM_CRS = "EPSG:32644"

Path(CACHE_DIR).mkdir(
    parents=True,
    exist_ok=True
)

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

# ── OSMNX SETTINGS ────────────────────────────────────────────────────────
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = "osm_cache"
ox.settings.timeout = 180

# ── LOAD MUMBAI BOUNDARY SAFELY ──────────────────────────────────────────
print("Loading Mumbai boundary...")

# IMPORTANT:
# which_result=2 usually returns the polygon
# instead of the city-center point
MUMBAI_GDF = ox.geocode_to_gdf(
    "Mumbai, Maharashtra, India",
    which_result=2
)

print(
    "Returned geometry types:"
)

print(MUMBAI_GDF.geom_type)

# Keep polygon geometries only
MUMBAI_GDF = MUMBAI_GDF[
    MUMBAI_GDF.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
]

if MUMBAI_GDF.empty:

    raise ValueError(
        "No polygon geometry returned "
        "for Mumbai."
    )

# Merge polygons safely
MUMBAI_GEOM = unary_union(
    MUMBAI_GDF.geometry
)

# SAFER coastline extraction
if MUMBAI_GEOM.geom_type == "Polygon":

    MUMBAI_COASTLINE = (
        MUMBAI_GEOM.exterior
    )

elif MUMBAI_GEOM.geom_type == "MultiPolygon":

    MUMBAI_COASTLINE = unary_union(
        [
            poly.exterior
            for poly in MUMBAI_GEOM.geoms
        ]
    )

else:

    raise ValueError(
        f"Unexpected geometry type: "
        f"{MUMBAI_GEOM.geom_type}"
    )

print(
    f"Coastline geometry type: "
    f"{MUMBAI_COASTLINE.geom_type}"
)

print(
    "Mumbai boundary loaded successfully."
)

# ── ROAD TYPE DEFINITIONS ────────────────────────────────────────────────
ARTERIAL_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
}

# ── LANDUSE GROUPINGS ─────────────────────────────────────────────────────
LANDUSE_CATEGORIES = {

    "industrial": {
        "industrial",
        "port",
        "quarry",
    },

    "residential": {
        "residential",
        "housing",
        "apartments",
    },

    "green": {
        "grass",
        "forest",
        "park",
        "recreation_ground",
        "village_green",
        "meadow",
        "orchard",
        "allotments",
    },
}

# ── STATIC FEATURE COLUMNS ────────────────────────────────────────────────
STATIC_COLS = [
    "road_density",
    "arterial_fraction",
    "industrial_fraction",
    "residential_fraction",
    "green_fraction",
    "building_density",
    "coastline_distance",
]

# ── HELPERS ───────────────────────────────────────────────────────────────
def is_arterial(hw):

    if hw is None:
        return False

    if isinstance(hw, float) and np.isnan(hw):
        return False

    if isinstance(hw, list):

        return any(
            h in ARTERIAL_TYPES
            for h in hw
            if isinstance(h, str)
        )

    return hw in ARTERIAL_TYPES


def safe_buffer_fix(gdf):

    try:

        gdf = gdf.copy()

        gdf["geometry"] = (
            gdf.geometry.buffer(0)
        )

    except Exception:
        pass

    return gdf


def format_float(val, precision=5):

    if val is None:
        return "nan"

    if isinstance(val, float) and np.isnan(val):
        return "nan"

    return f"{val:.{precision}f}"


# ── ROAD FEATURES ─────────────────────────────────────────────────────────
def get_road_features(lat, lon):

    for attempt in range(3):

        try:

            G = ox.graph_from_point(
                (lat, lon),
                dist=RADIUS_M,
                network_type="drive"
            )

            edges = ox.graph_to_gdfs(
                G,
                nodes=False
            )

            edges_proj = edges.to_crs(
                UTM_CRS
            )

            total_length = (
                edges_proj.geometry
                .length
                .sum()
            )

            buffer_area = (
                np.pi * (RADIUS_M ** 2)
            )

            road_density = (
                total_length / buffer_area
            )

            art_mask = (
                edges_proj["highway"]
                .apply(is_arterial)
            )

            arterial_length = (
                edges_proj.loc[art_mask]
                .geometry.length
                .sum()
            )

            arterial_fraction = (
                arterial_length / total_length
                if total_length > 0 else 0.0
            )

            return (
                float(road_density),
                float(arterial_fraction)
            )

        except Exception as e:

            print(
                f"    Road retry "
                f"{attempt+1}/3 failed: {e}"
            )

            time.sleep(5)

    return np.nan, np.nan


# ── LANDUSE FEATURES ──────────────────────────────────────────────────────
def get_landuse_features(lat, lon):

    fallback = {
        k + "_fraction": 0.0
        for k in LANDUSE_CATEGORIES
    }

    gdf = None

    for attempt in range(3):

        try:

            gdf = ox.features_from_point(
                (lat, lon),
                tags={"landuse": True},
                dist=RADIUS_M
            )

            break

        except Exception as e:

            print(
                f"    Landuse retry "
                f"{attempt+1}/3 failed: {e}"
            )

            time.sleep(5)

    if gdf is None or gdf.empty:
        return fallback

    try:

        point_proj = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326"
        ).to_crs(UTM_CRS).iloc[0]

        buffer = point_proj.buffer(
            RADIUS_M
        )

        gdf_proj = gdf.to_crs(
            UTM_CRS
        )

        gdf_proj = gdf_proj[
            gdf_proj.geometry.notna()
        ]

        gdf_proj = safe_buffer_fix(
            gdf_proj
        )

        gdf_proj = gdf_proj[
            gdf_proj.geometry.intersects(
                buffer
            )
        ].copy()

        if gdf_proj.empty:
            return fallback

        gdf_proj["clipped_area"] = (
            gdf_proj.geometry
            .intersection(buffer)
            .area
        )

        total_area = buffer.area

        results = {}

        for category, tags in LANDUSE_CATEGORIES.items():

            mask = (
                gdf_proj["landuse"]
                .isin(tags)
            )

            area = (
                gdf_proj.loc[
                    mask,
                    "clipped_area"
                ].sum()
            )

            results[
                f"{category}_fraction"
            ] = float(area / total_area)

        return results

    except Exception as e:

        print(
            f"    Landuse processing failed: {e}"
        )

        return fallback


# ── BUILDING DENSITY ──────────────────────────────────────────────────────
def get_building_density(lat, lon):

    gdf = None

    for attempt in range(3):

        try:

            gdf = ox.features_from_point(
                (lat, lon),
                tags={"building": True},
                dist=RADIUS_M
            )

            break

        except Exception as e:

            print(
                f"    Building retry "
                f"{attempt+1}/3 failed: {e}"
            )

            time.sleep(5)

    if gdf is None or gdf.empty:
        return 0.0

    try:

        point_proj = gpd.GeoSeries(
            [Point(lon, lat)],
            crs="EPSG:4326"
        ).to_crs(UTM_CRS).iloc[0]

        buffer = point_proj.buffer(
            RADIUS_M
        )

        gdf_proj = gdf.to_crs(
            UTM_CRS
        )

        gdf_proj = gdf_proj[
            gdf_proj.geometry.notna()
        ]

        gdf_proj = gdf_proj[
            gdf_proj.geom_type.isin(
                ["Polygon", "MultiPolygon"]
            )
        ]

        gdf_proj = safe_buffer_fix(
            gdf_proj
        )

        gdf_proj = gdf_proj[
            gdf_proj.geometry.intersects(
                buffer
            )
        ].copy()

        if gdf_proj.empty:
            return 0.0

        footprint_area = (
            gdf_proj.geometry
            .intersection(buffer)
            .area
            .sum()
        )

        return float(
            footprint_area / buffer.area
        )

    except Exception as e:

        print(
            f"    Building processing failed: {e}"
        )

        return 0.0


# ── COASTLINE DISTANCE ────────────────────────────────────────────────────
def get_coastline_distance(lat, lon):

    station_pt = Point(
        lon,
        lat
    )

    nearest_pt = nearest_points(
        station_pt,
        MUMBAI_COASTLINE
    )[1]

    s_proj = gpd.GeoSeries(
        [station_pt],
        crs="EPSG:4326"
    ).to_crs(UTM_CRS)

    n_proj = gpd.GeoSeries(
        [nearest_pt],
        crs="EPSG:4326"
    ).to_crs(UTM_CRS)

    return float(
        s_proj.iloc[0].distance(
            n_proj.iloc[0]
        ) / 1000.0
    )


# ── LOAD STATIONS ─────────────────────────────────────────────────────────
df = pd.read_csv(
    COORDS_PATH
)

features = []

print(
    f"\nProcessing "
    f"{len(df)} stations...\n"
)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
for idx, row in df.iterrows():

    station = row["station"]
    lat = row["lat"]
    lon = row["lon"]

    safe_id = (
        station
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "")
    )

    cache_file = (
        f"{CACHE_DIR}/{safe_id}.json"
    )

    print(
        f"[{idx+1}/{len(df)}] "
        f"{station}"
    )

    # ── CACHE HIT ─────────────────────────────────────────────────────────
    if os.path.exists(cache_file):

        print("  [CACHE HIT]")

        with open(cache_file) as f:
            feats = json.load(f)

        features.append(feats)

        continue

    feats = {
        "station": station
    }

    # ── ROAD FEATURES ─────────────────────────────────────────────────────
    try:

        road_density, arterial_fraction = (
            get_road_features(lat, lon)
        )

        feats["road_density"] = (
            road_density
        )

        feats["arterial_fraction"] = (
            arterial_fraction
        )

        print(
            f"  road_density="
            f"{format_float(road_density, 5)} "
            f"arterial_fraction="
            f"{format_float(arterial_fraction, 3)}"
        )

    except Exception as e:

        print(f"  ROAD FAILED: {e}")

        feats["road_density"] = np.nan
        feats["arterial_fraction"] = np.nan

    # ── LANDUSE FEATURES ──────────────────────────────────────────────────
    try:

        lu = get_landuse_features(
            lat,
            lon
        )

        feats.update(lu)

        print(
            f"  industrial="
            f"{lu['industrial_fraction']:.3f} "
            f"residential="
            f"{lu['residential_fraction']:.3f} "
            f"green="
            f"{lu['green_fraction']:.3f}"
        )

    except Exception as e:

        print(f"  LANDUSE FAILED: {e}")

        feats["industrial_fraction"] = np.nan
        feats["residential_fraction"] = np.nan
        feats["green_fraction"] = np.nan

    # ── BUILDING DENSITY ──────────────────────────────────────────────────
    try:

        bd = get_building_density(
            lat,
            lon
        )

        feats["building_density"] = bd

        print(
            f"  building_density="
            f"{bd:.4f}"
        )

    except Exception as e:

        print(
            f"  BUILDINGS FAILED: {e}"
        )

        feats["building_density"] = np.nan

    # ── COASTLINE DISTANCE ────────────────────────────────────────────────
    try:

        cd = get_coastline_distance(
            lat,
            lon
        )

        feats["coastline_distance"] = cd

        print(
            f"  coastline_distance="
            f"{cd:.2f} km"
        )

    except Exception as e:

        print(
            f"  COASTLINE FAILED: {e}"
        )

        feats["coastline_distance"] = np.nan

    # ── SAVE CACHE ────────────────────────────────────────────────────────
    with open(cache_file, "w") as f:

        json.dump(
            feats,
            f,
            indent=2
        )

    features.append(feats)

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────
feat_df = pd.DataFrame(
    features
)

# ── FILL MISSING VALUES ──────────────────────────────────────────────────
for col in STATIC_COLS:

    if col in feat_df.columns:

        median_val = (
            feat_df[col]
            .median()
        )

        feat_df[col] = (
            feat_df[col]
            .fillna(median_val)
        )

    else:

        feat_df[col] = 0.0

# ── SAVE RAW CSV ─────────────────────────────────────────────────────────
feat_df.to_csv(
    SAVE_CSV,
    index=False
)

print(
    f"\nSaved CSV → {SAVE_CSV}"
)

# ── NORMALISE FEATURES ───────────────────────────────────────────────────
# NOTE:
# For final train/test experiments,
# fit scaler only on training set.

X_static = feat_df[
    STATIC_COLS
].values.astype(np.float32)

X_min = X_static.min(axis=0)
X_max = X_static.max(axis=0)

X_norm = (
    (X_static - X_min)
    / (X_max - X_min + 1e-8)
)

# ── SAVE NUMPY ───────────────────────────────────────────────────────────
np.save(
    SAVE_NPY,
    X_norm
)

np.save(
    "data/processed/static_min.npy",
    X_min
)

np.save(
    "data/processed/static_max.npy",
    X_max
)

print(
    f"Saved NPY → {SAVE_NPY}"
)

print(
    f"Shape     → {X_norm.shape}"
)

# ── FEATURE SUMMARY ──────────────────────────────────────────────────────
print("\nFeature Summary:")

for i, col in enumerate(STATIC_COLS):

    print(
        f"  {col:<25s} "
        f"min={X_norm[:, i].min():.3f} "
        f"max={X_norm[:, i].max():.3f}"
    )

print(
    "\nMorphology extraction complete."
)
