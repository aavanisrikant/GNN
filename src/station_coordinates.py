import osmnx as ox
import pandas as pd
import time
import os

stations = [
    "Bandra Kurla Complex, Mumbai - IITM",
    "Bandra Kurla Complex, Mumbai - MPCB",
    "Bandra, Mumbai - MPCB",
    "Borivali East, Mumbai - IITM",
    "Borivali East, Mumbai - MPCB",
    "Byculla, Mumbai - BMC",
    "Chakala-Andheri East, Mumbai - IITM",
    "Chembur, Mumbai - MPCB",
    "Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB",
    "Colaba, Mumbai - MPCB",
    "Deonar, Mumbai - IITM",
    "Ghatkopar, Mumbai - BMC",
    "Kandivali East, Mumbai - MPCB",
    "Kandivali West, Mumbai - BMC",
    "Kherwadi_Bandra East, Mumbai - MPCB",
    "Khindipada-Bhandup West, Mumbai - IITM",
    "Kurla, Mumbai - MPCB",
    "Malad West, Mumbai - IITM",
    "Mazgaon, Mumbai - IITM",
    "Mindspace-Malad West, Mumbai - MPCB",
    "Mulund West, Mumbai - MPCB",
    "Navy Nagar-Colaba, Mumbai - IITM",
    "Powai, Mumbai - MPCB",
    "Sewri, Mumbai - BMC",
    "Shivaji Nagar, Mumbai - BMC",
    "Siddharth Nagar-Worli, Mumbai - IITM",
    "Sion, Mumbai - MPCB",
    "Vasai West, Mumbai - MPCB",
    "Vile Parle West, Mumbai - MPCB",
    "Worli, Mumbai - MPCB",
]


def clean_query(name):
    query = name
    query = query.replace(" - IITM", "")
    query = query.replace(" - MPCB", "")
    query = query.replace(" - BMC", "")
    query = query.replace("_", " ")
    return query


if os.path.exists("data/processed/station_coords.csv"):
    df_existing = pd.read_csv("data/processed/station_coords.csv", index_col="station")
else:
    df_existing = pd.DataFrame(columns=["lat", "lon"])


results = {}

for name in stations:
    try:
        if name in df_existing.index and pd.notna(df_existing.loc[name, "lat"]):
            print(f"Skipping (cached): {name}")
            results[name] = (
                df_existing.loc[name, "lat"],
                df_existing.loc[name, "lon"]
            )
            continue

        query = clean_query(name)
        full_query = f"{query}, Mumbai, India"

        print(f"Fetching: {full_query}")

        lat, lon = ox.geocode(full_query)
        results[name] = (lat, lon)

        time.sleep(1)

    except Exception:
        print(f"Failed: {name}")
        results[name] = (None, None)

# Combine old + new
df_new = pd.DataFrame.from_dict(results, orient="index", columns=["lat", "lon"])
df_new.index.name = "station"

df_final = df_existing.combine_first(df_new)

df_final.to_csv("data/processed/station_coords.csv")

print("\nSaved station_coords.csv")
