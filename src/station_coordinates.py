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

def clean_query(name):
    query = name
    query = query.replace(" - IITM", "")
    query = query.replace(" - MPCB", "")
    query = query.replace(" - BMC", "")
  
    query = query.replace("_", " ")
    query = query.replace("-", " ")
    query = query.replace("(", "")
    query = query.replace(")", "")

    return query

#coordinates
results = {}

for name in stations:
    try:
        query = clean_query(name)

        print(f"Fetching: {query}")

        lat, lon = ox.geocode(query)
        results[name] = (lat, lon)

        time.sleep(1)

    except Exception:
        print(f"Failed: {name}")
        results[name] = (None, None)


df = pd.DataFrame.from_dict(results, orient="index", columns=["lat", "lon"])
df.index.name = "station"

df.to_csv("data/processed/station_coords.csv")

print("\n Saved station_coords.csv")
