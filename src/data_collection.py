import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# parse single file
def parse_cpcb_file(filepath: str, station: str, parameter: str) -> pd.DataFrame:
    """
    Parse one CPCB Excel file (wide → long format).
    """

    df = pd.read_excel(
        filepath,
        header=0,
        na_values=["NA", "N/A", "none", "None", "--", ""],
    )

    # Column names
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
        raise ValueError(f"Could not extract year/month from filename: {fname}")

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

    # IST timezone
    df_long["timestamp"] = df_long["timestamp"].dt.tz_localize(
        "Asia/Kolkata",
        nonexistent="shift_forward",
        ambiguous="NaT"
    )

    df_long = df_long.dropna(subset=["timestamp"])
    df_long["station"] = station
    df_long["parameter"] = parameter.lower()

    result = df_long[["timestamp", "station", "parameter", "value"]].copy()
    result = result.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Parsed {Path(filepath).name}: "
        f"{len(result)} rows, "
        f"{result['value'].notna().sum()} non-null values"
    )

    return result

# all files
def load_all_cpcb_files(data_folder: str) -> pd.DataFrame:
    """
    Load all CPCB Excel files into one DataFrame.
    """

    folder = Path(data_folder)
    files = sorted(folder.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {data_folder}")

    all_dfs = []

    for filepath in files:
        try:
            fname = filepath.stem
            parts = fname.split("_")

            parameter = parts[0].lower()
            station = " ".join(parts[4:-2])

            df = parse_cpcb_file(
                str(filepath),
                station=station,
                parameter=parameter
            )

            if df.empty:
                continue

            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Failed: {filepath.name} — {e}")
            continue

    if not all_dfs:
        raise ValueError("No files loaded successfully")

    combined = pd.concat(all_dfs, ignore_index=True)

    logger.info(
        f"Total: {len(combined)} rows from {len(all_dfs)} files, "
        f"{combined['station'].nunique()} stations"
    )

    return combined

def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long format → wide format
    """

    df_wide = df.pivot_table(
        index=["timestamp", "station"],
        columns="parameter",
        values="value"
    ).reset_index()

    df_wide.columns.name = None

    return df_wide

# testing
if __name__ == "__main__":
    print("🚀 Running CPCB data loader...\n")

    df = load_all_cpcb_files("data/raw/CPCB")

    print("\n=== LONG DATA PREVIEW ===")
    print(df.head())

    print("\n=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("Stations:", df["station"].unique())

    print("\n=== TIME RANGE ===")
    print(df["timestamp"].min(), "→", df["timestamp"].max())

    print("\n=== MISSING VALUES ===")
    print(df["value"].isna().sum(), "/", len(df))
    df_wide = pivot_to_wide(df)

    print("\n=== WIDE DATA PREVIEW ===")
    print(df_wide.head())

    print("\n=== WIDE SHAPE ===")
    print(df_wide.shape)

    print("\n=== COLUMNS ===")
    print(df_wide.columns)
