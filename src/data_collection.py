import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parsing single files
def parse_cpcb_file(filepath: str, station: str, parameter: str) -> pd.DataFrame:

    df = pd.read_excel(
        filepath,
        header=0,
        na_values=["NA", "N/A", "none", "None", "--", ""],
    )

    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: "day"})

    fname = Path(filepath).stem
    parts = fname.split("_")

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }

    year = None
    month_str = None

    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = int(part)
        if part in month_map:
            month_str = part

    if year is None or month_str is None:
        raise ValueError(f"Could not extract year/month from {fname}")

    month = month_map[month_str]

    hour_cols = [c for c in df.columns if c != "day"]

    df_long = df.melt(
        id_vars=["day"],
        value_vars=hour_cols,
        var_name="hour_str",
        value_name="value"
    )

    df_long["hour"] = pd.to_numeric(
        df_long["hour_str"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    df_long = df_long[pd.to_numeric(df_long["day"], errors="coerce").notna()]
    df_long["day"] = df_long["day"].astype(int)

    df_long["timestamp"] = pd.to_datetime(
        dict(year=year, month=month, day=df_long["day"], hour=df_long["hour"]),
        errors="coerce"
    )

    df_long["timestamp"] = df_long["timestamp"].dt.tz_localize(
        "Asia/Kolkata", nonexistent="shift_forward", ambiguous="NaT"
    )

    df_long = df_long.dropna(subset=["timestamp"])

    df_long["station"] = station
    df_long["parameter"] = parameter.lower()

    result = df_long[["timestamp", "station", "parameter", "value"]].copy()
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result
#all files
def load_all_cpcb_files(folder_path: str):

    folder = Path(folder_path)
    files = sorted(folder.glob("*.xlsx"))

    all_dfs = []

    for filepath in files:
        try:
            parts = filepath.stem.split("_")

            parameter = parts[0].lower()
            station = " ".join(parts[4:-2])

            df = parse_cpcb_file(str(filepath), station, parameter)

            if not df.empty:
                all_dfs.append(df)

        except Exception as e:
            logger.error(f"{filepath.name} failed: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined

def pivot_to_wide(df):

    df_wide = df.pivot_table(
        index=["timestamp", "station"],
        columns="parameter",
        values="value"
    ).reset_index()

    df_wide.columns.name = None
    return df_wide

def handle_missing_values(df):

    df = df.sort_values(["station", "timestamp"])

    def process(group):
        group = group.set_index("timestamp")
        for col in group.columns:
            try:
                group[col] = pd.to_numeric(group[col])
            except:
                pass
        group = group.ffill(limit=3)
        group = group.interpolate(limit=12)
        return group.reset_index()

    df = df.groupby("station", group_keys=False).apply(process)
    return df

def train_test_split_time(df, split_ratio=0.8):

    df = df.sort_values("timestamp")

    split_index = int(len(df) * split_ratio)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    return train, test


def normalize_train_test(train, test):

    features = [c for c in train.columns if c not in ["timestamp", "station"]]

    min_vals = train[features].min()
    max_vals = train[features].max()

    train[features] = (train[features] - min_vals) / (max_vals - min_vals)
    test[features] = (test[features] - min_vals) / (max_vals - min_vals)

    return train, test


def create_sequences(df, window=24, horizon=1):

    sequences = []

    for station in df["station"].unique():
        station_df = df[df["station"] == station].sort_values("timestamp")

        values = station_df.drop(columns=["timestamp", "station"]).values

        for i in range(len(values) - window - horizon):
            X = values[i:i+window]
            y = values[i+window+horizon-1]

            sequences.append((X, y))

    return sequences

if __name__ == "__main__":

    print("Running pipeline...\n")

    df = load_all_cpcb_files("data/raw/CPCB")

    df_wide = pivot_to_wide(df)

    df_clean = handle_missing_values(df_wide)

    train, test = train_test_split_time(df_clean)

    train, test = normalize_train_test(train, test)

    seq_1 = create_sequences(train, horizon=1)
    seq_3 = create_sequences(train, horizon=3)
    seq_6 = create_sequences(train, horizon=6)

    print("Sequences (1 step):", len(seq_1))
    print("Sequences (3 step):", len(seq_3))
    print("Sequences (6 step):", len(seq_6))

    print("Input shape:", seq_1[0][0].shape)
