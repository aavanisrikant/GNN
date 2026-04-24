import os
import osmnx as ox
import pandas as pd
import time

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

manual_coords = {
    "Chhatrapati Shivaji Intl. Airport (T2), Mumbai - MPCB": (19.0886300, 72.8681400),
    "Shivaji Nagar, Mumbai - BMC": (19.0667800, 72.9287000),
    "Vasai West, Mumbai - MPCB": (19.3919000, 72.8397000),
    "Siddharth Nagar-Worli, Mumbai - IITM": (19.0100000, 72.8150000),
}

def clean_query(name):
    query = name
    query = query.replace(" - IITM", "")
    query = query.replace(" - MPCB", "")
    query = query.replace(" - BMC", "")
    query = query.replace("_", " ")
    query = query.replace("-", " ")
    query = query.replace("(", "")
    query = query.replace(")", "")
    return query + ", Mumbai, India"

cache_path = "data/processed/station_coords.csv"

if os.path.exists(cache_path):
    df_existing = pd.read_csv(cache_path)
    existing_dict = dict(
        zip(df_existing["station"], zip(df_existing["lat"], df_existing["lon"]))
    )
else:
    existing_dict = {}

results = {}

for name in stations:
    if name in existing_dict and pd.notna(existing_dict[name][0]):
        print(f"Skipping (cached): {name}")
        results[name] = existing_dict[name]
        continue

    if name in manual_coords:
        print(f"Using manual coords: {name}")
        results[name] = manual_coords[name]
        continue

    try:
        query = clean_query(name)
        print(f"Fetching: {query}")

        lat, lon = ox.geocode(query)
        results[name] = (lat, lon)

        time.sleep(1)  

    except Exception:
        print(f"Failed: {name}")
        results[name] = (None, None)

df_new = pd.DataFrame.from_dict(results, orient="index", columns=["lat", "lon"])
df_new.index.name = "station"
df_new.reset_index(inplace=True)

if os.path.exists(cache_path):
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new.copy()


df_final = df_final.drop_duplicates(subset="station", keep="last")
df_final.to_csv(cache_path, index=False)

print("\n Saved station_coords.csv")
print("\n--- VALIDATION ---")
print("Total rows:", len(df_final))
print("Unique stations:", df_final["station"].nunique())
print("Missing coords:", df_final[["lat", "lon"]].isna().sum().sum())
