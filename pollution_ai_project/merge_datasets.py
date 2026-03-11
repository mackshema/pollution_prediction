import pandas as pd

# Load datasets
wind = pd.read_csv("wind_trajectory.csv")
sat = pd.read_csv("sentinel_pollution.csv")
iot = pd.read_csv("iot_pollution_data.csv")

# Convert timestamps
wind["timestamp"] = pd.to_datetime(wind["timestamp"], utc=True)
iot["timestamp_UTC"] = pd.to_datetime(iot["timestamp_UTC"], utc=True)

# Round timestamps to nearest hour
wind["hour"] = wind["timestamp"].dt.floor("h")
iot["hour"] = iot["timestamp_UTC"].dt.floor("h")

# Merge IoT pollution with wind data
merged = pd.merge(iot, wind, left_on="hour", right_on="hour", how="inner")

# Add satellite pollution data
sat["date"] = pd.to_datetime(sat["date"])
# Make date column compatible for merge
merged["date"] = pd.to_datetime(merged["hour"].dt.date)

merged = pd.merge(merged, sat, on="date", how="left")

# Save dataset
merged.to_csv("training_dataset.csv", index=False)

print("Training dataset created")
print("Rows:", len(merged))
