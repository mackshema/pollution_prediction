import xarray as xr
import pandas as pd
import numpy as np

# Input GRIB file
grib_file = r"C:\Users\sivas\OneDrive\ECE_climate_forcasting_hackthon\pollution_ai_project\wind_data.nc"

# Output CSV file
output_csv = r"C:\Users\sivas\OneDrive\ECE_climate_forcasting_hackthon\pollution_ai_project\wind_trajectory.csv"

print("Loading dataset...")

# Open dataset
ds = xr.open_dataset(grib_file)

print(ds)

# Extract variables
u = ds["u10"].values
v = ds["v10"].values
lats = ds.latitude.values
lons = ds.longitude.values
times = ds.valid_time.values

rows = []

print("Processing wind vectors...")

for t in range(len(times)):

    for i, lat in enumerate(lats):

        for j, lon in enumerate(lons):

            u_val = u[t, i, j]
            v_val = v[t, i, j]

            # wind speed
            speed = np.sqrt(u_val**2 + v_val**2)

            # wind direction
            direction = (np.degrees(np.arctan2(u_val, v_val)) + 360) % 360

            # estimate next location after 1 hour
            lat_b = lat + (v_val * 3600) / 111320
            lon_b = lon + (u_val * 3600) / (111320 * np.cos(np.radians(lat)))

            rows.append({
                "timestamp": str(times[t]),
                "lat_A": float(lat),
                "lon_A": float(lon),
                "wind_speed": float(speed),
                "wind_dir": float(direction),
                "u_comp": float(u_val),
                "v_comp": float(v_val),
                "lat_B": float(lat_b),
                "lon_B": float(lon_b)
            })

df = pd.DataFrame(rows)

print("Saving CSV...")

df.to_csv(output_csv, index=False)

print("Dataset created successfully!")
print("Saved to:", output_csv)
