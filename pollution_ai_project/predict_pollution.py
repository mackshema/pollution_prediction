"""
predict_pollution.py
────────────────────────────────────────────────────────────
Uses the saved Scikit-Learn Random Forest model to predict
PM2.5 / AQI and writes prediction_output.csv continuously.

Runs every 60 seconds so the dashboard always has fresh data.
"""

import os, math, time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

# ── Paths ──────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE   = os.path.join(BASE, "pollution_prediction_model.pkl")
SCALER_X     = os.path.join(BASE, "scaler_X.pkl")
SCALER_Y     = os.path.join(BASE, "scaler_y.pkl")
WIND_FILE    = os.path.join(BASE, "live_wind_buffer.csv")
IOT_FILE     = os.path.join(BASE, "iot_pollution_data.csv")
OUTPUT_FILE  = os.path.join(BASE, "prediction_output.csv")

INTERVAL = 60   # seconds between prediction cycles

# ── Load model once ────────────────────────────────────────
print("=" * 52)
print("  Pollution Prediction Engine  (sklearn RF)")
print("=" * 52)

try:
    model = joblib.load(MODEL_FILE)
    print(f"  ✓ Model loaded   : {MODEL_FILE}")
except Exception as e:
    print(f"  ✗ Model load failed: {e}")
    model = None

try:
    sx = joblib.load(SCALER_X)
    sy = joblib.load(SCALER_Y)
    print(f"  ✓ Scalers loaded")
except Exception:
    sx = sy = None
    print("  ⚠  Scalers not found — raw values used")


# ── Helper: AQI from PM2.5 ─────────────────────────────────
def pm25_to_aqi(pm):
    """
    US EPA linear interpolation breakpoints for PM2.5.
    """
    bp = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,   51,  100),
        (35.5,  55.4,  101,  150),
        (55.5, 150.4,  151,  200),
        (150.5,250.4,  201,  300),
        (250.5,500.4,  301,  500),
    ]
    for lo_c, hi_c, lo_a, hi_a in bp:
        if lo_c <= pm <= hi_c:
            return round((hi_a - lo_a) / (hi_c - lo_c) * (pm - lo_c) + lo_a)
    return 500


# ── Helper: destination from bearing + distance ─────────────
def move_point(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    b = math.radians(bearing_deg)
    d = dist_km / R
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(d) +
                     math.cos(lat1)*math.sin(d)*math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b)*math.sin(d)*math.cos(lat1),
                              math.cos(d) - math.sin(lat1)*math.sin(lat2))
    return round(math.degrees(lat2), 6), round(math.degrees(lon2), 6)


# ── Predict loop ───────────────────────────────────────────
def predict_once():
    # 1) Load latest IoT row
    try:
        iot = pd.read_csv(IOT_FILE, on_bad_lines="skip")
        iot.columns = [c.strip().lower().replace("timestamp_utc", "timestamp")
                       for c in iot.columns]
        iot_row = iot.dropna(subset=["pm25"]).tail(1).iloc[0]
    except Exception as e:
        print(f"  ✗ IoT read error: {e}")
        return

    # 2) Load latest wind row (flexible column names)
    try:
        wdf = pd.read_csv(WIND_FILE, on_bad_lines="skip")
        wdf.columns = [c.strip().lower() for c in wdf.columns]

        # Map flexible column names
        speed_col = next((c for c in wdf.columns if "speed" in c and "kmh" not in c), None)
        dir_col   = next((c for c in wdf.columns if "dir" in c), None)
        u_col     = next((c for c in wdf.columns if "u_comp" in c or c == "u"), None)
        v_col     = next((c for c in wdf.columns if "v_comp" in c or c == "v"), None)

        wdf_last = wdf.dropna(subset=[speed_col] if speed_col else []).tail(3)
        w_speed = float(wdf_last[speed_col].mean())  if speed_col else 3.5
        w_dir   = float(wdf_last[dir_col].mean())    if dir_col   else 180.0
        w_u     = float(wdf_last[u_col].mean())      if u_col     else 0.0
        w_v     = float(wdf_last[v_col].mean())      if v_col     else 0.0
    except Exception as e:
        print(f"  ⚠  Wind read error: {e} — using defaults")
        w_speed, w_dir, w_u, w_v = 3.5, 180.0, 0.0, 0.0

    # 3) Build feature vector
    pm25 = float(iot_row.get("pm25", 50))
    pm10 = float(iot_row.get("pm10", 80))
    co   = float(iot_row.get("co",   1.0))
    no2  = float(iot_row.get("no2",  30))
    so2  = float(iot_row.get("so2",  10))
    temp = float(iot_row.get("temperature", 30))
    hum  = float(iot_row.get("humidity",    60))
    ref_lat = float(iot_row.get("lat", 10.90))
    ref_lon = float(iot_row.get("lon", 78.70))

    features = np.array([[pm25, pm10, co, no2, so2, temp, hum,
                          w_speed, w_dir, w_u, w_v]])

    # 4) Run model if available, else physics-based fallback
    if model is not None:
        try:
            X_in = sx.transform(features) if sx else features
            y_raw = model.predict(X_in)
            if sy:
                y_out = sy.inverse_transform(y_raw.reshape(1, -1))[0]
            else:
                y_out = y_raw[0]
            # Model outputs: expect at least [pm25, aqi] or just [pm25]
            pred_pm25 = max(0.0, float(y_out[0]))
            pred_aqi  = float(y_out[1]) if len(y_out) > 1 else float(pm25_to_aqi(pred_pm25))
        except Exception as e:
            print(f"  ⚠  Model inference error: {e} — using physics fallback")
            pred_pm25 = pm25 * 1.08
            pred_aqi  = pm25_to_aqi(pred_pm25)
    else:
        # Pure physics-based prediction
        pred_pm25 = pm25 * 1.08
        pred_aqi  = pm25_to_aqi(pred_pm25)

    # 5) Compute spatial spread (1-hour forecast)
    TIME_HOURS   = 1.0
    dist_km      = w_speed * 3.6 * TIME_HOURS
    arrival_min  = (dist_km / (w_speed * 3.6) * 60) if w_speed > 0.01 else 999
    pred_lat, pred_lon = move_point(ref_lat, ref_lon, w_dir, dist_km)

    # 6) Write output
    row = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "predicted_pm25":    round(pred_pm25, 2),
        "predicted_aqi":     round(pred_aqi,  1),
        "predicted_lat":     pred_lat,
        "predicted_lon":     pred_lon,
        "distance_km":       round(dist_km, 3),
        "arrival_time_min":  round(arrival_min, 1),
        "wind_speed_ms":     round(w_speed, 3),
        "wind_dir_deg":      round(w_dir, 1),
        "ref_lat":           ref_lat,
        "ref_lon":           ref_lon,
        "ref_pm25":          round(pm25, 2),
        "ref_aqi":           pm25_to_aqi(pm25),
    }
    df_out = pd.DataFrame([row])
    write_header = not os.path.exists(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    df_out.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] ──────────────────────────────")
    print(f"  Ref PM2.5    : {pm25:.1f}  →  Predicted: {pred_pm25:.1f} µg/m³")
    print(f"  Ref AQI      : {pm25_to_aqi(pm25)}    →  Predicted: {pred_aqi:.0f}")
    print(f"  Wind         : {w_speed:.2f} m/s  @  {w_dir:.0f}°")
    print(f"  Travel dist  : {dist_km:.2f} km  in  {arrival_min:.0f} min")
    print(f"  Pred location: ({pred_lat}, {pred_lon})")
    print(f"  ✓ Written to  : prediction_output.csv")


if __name__ == "__main__":
    print("\n  Running every 60 seconds. Press Ctrl+C to stop.\n")
    while True:
        try:
            predict_once()
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
        time.sleep(INTERVAL)
