from src.data_collection import load_all_stations, clean_aqi_data


df = load_all_stations("data/raw")
df = clean_aqi_data(df)

print(df.head(20))
print(df.shape)
