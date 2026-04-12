import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_stations(data_folder: str) -> pd.DataFrame:
    """
    Load all CPCB/SAFAR station CSVs from a folder into a single DataFrame.
    
    Args:
        data_folder: Path to folder containing one CSV per station.
    
    Returns:
        Combined DataFrame with columns: timestamp, station,
        pm25, pm10, no2, aqi (and others if present).
    """
    dfs = []

    csv_files = sorted([
        f for f in os.listdir(data_folder)
        if f.endswith(".csv") and not f.startswith(".")
    ])

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    for file in csv_files:
        filepath = os.path.join(data_folder, file)

        try:
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                skipinitialspace=True,
                na_values=["NA", "N/A", "none", "None", "--", ""],
            )

            # standardised column names
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            timestamp_col = next(
                (c for c in df.columns if "date" in c or "time" in c),
                None
            )

            if timestamp_col:
                df["timestamp"] = pd.to_datetime(
                    df[timestamp_col],
                    dayfirst=True,
                    errors="coerce"
                )

                # setting timezone to IST
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize(
                        "Asia/Kolkata",
                        nonexistent="shift_forward",
                        ambiguous="NaT"
                    )
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")

                df.drop(columns=[timestamp_col], inplace=True)

            else:
                logger.warning(f"No timestamp column found in {file}")

            # station names
            station_name = os.path.splitext(file)[0].replace("_", " ").title()
            df["station"] = station_name

            dfs.append(df)
            logger.info(f"Loaded {file}: {len(df)} rows")

        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No CSVs loaded successfully — check your data folder")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined)} from {len(dfs)} stations")

    return combined


# cleaning data
def clean_aqi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise and clean the combined station DataFrame.
    """

    # Sorting
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
  
    df = df.dropna(subset=["timestamp"])

    df = df.drop_duplicates(subset=["station", "timestamp"], keep="first")

    df = df.set_index(["station", "timestamp"])

    return df
