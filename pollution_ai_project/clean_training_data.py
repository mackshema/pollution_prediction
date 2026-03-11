import pandas as pd
import numpy as np

INPUT_FILE = "training_dataset.csv"
OUTPUT_FILE = "training_dataset_clean.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

print("Initial rows:", len(df))

# -------------------------------------------------
# 1. Remove duplicates
# -------------------------------------------------
df = df.drop_duplicates()

# -------------------------------------------------
# 2. Handle missing values
# -------------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["pm25","wind_speed","humidity"])

# -------------------------------------------------
# 3. Sensor sanity checks
# -------------------------------------------------

df = df[(df["pm25"] >= 0) & (df["pm25"] <= 1000)]
df = df[(df["pm10"] >= 0) & (df["pm10"] <= 1000)]
df = df[(df["co"] >= 0) & (df["co"] <= 50)]
df = df[(df["no2"] >= 0) & (df["no2"] <= 500)]
df = df[(df["so2"] >= 0) & (df["so2"] <= 500)]
df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
df = df[(df["temperature"] >= -20) & (df["temperature"] <= 60)]

# -------------------------------------------------
# 4. Wind sanity checks
# -------------------------------------------------

df = df[(df["wind_speed"] >= 0) & (df["wind_speed"] <= 60)]
df = df[(df["wind_dir"] >= 0) & (df["wind_dir"] <= 360)]

# -------------------------------------------------
# 5. Create additional ML features
# -------------------------------------------------

df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
df["day"] = pd.to_datetime(df["timestamp"]).dt.day
df["month"] = pd.to_datetime(df["timestamp"]).dt.month

# cyclic encoding (important for ML)
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

# -------------------------------------------------
# 6. Save clean dataset
# -------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("Clean dataset saved:", OUTPUT_FILE)
print("Final rows:", len(df))
