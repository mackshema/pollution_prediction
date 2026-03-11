import os
import requests
import pandas as pd
import time
import math
import logging
from datetime import datetime, timezone

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Open-Meteo: FREE, no API key, 1–2km resolution, updates every 15 min
BASE_URL = "https://api.open-meteo.com/v1/forecast"

LOCATIONS = [
    {"id": "sensor_1", "lat": 8.7139, "lon": 77.7567},  # Tirunelveli city
    {"id": "sensor_2", "lat": 8.9500, "lon": 77.4900},  # ~35km NW
    {"id": "sensor_3", "lat": 8.5000, "lon": 78.0500},  # ~40km SE (coastal)
]

OUTPUT_FILE = r"C:\Users\sivas\OneDrive\ECE_climate_forcasting_hackthon\pollution_ai_project\live_wind_buffer.csv"

INTERVAL_SECONDS = 60    # 60 s — fast refresh for live demo (change to 900 for production)

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wind_collector.log")
    ]
)
log = logging.getLogger(__name__)

# ─── FETCH ────────────────────────────────────────────────────────────────────

def get_wind(sensor_id, lat, lon, retries=3, backoff=5):
    """
    Fetch current wind from Open-Meteo.
    Returns a dict or None on failure.

    Open-Meteo current fields used:
      wind_speed_10m        → m/s at 10m height
      wind_direction_10m    → degrees (meteorological)
      wind_gusts_10m        → m/s gust
      temperature_2m        → °C (bonus context)
      relative_humidity_2m  → % (useful for pollution model)
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "temperature_2m",
            "relative_humidity_2m"
        ],
        "wind_speed_unit": "ms",      # m/s (not default kmh)
        "timezone": "Asia/Kolkata",   # IST timestamps
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            cur = data.get("current", {})

            speed    = cur.get("wind_speed_10m")
            direction= cur.get("wind_direction_10m")
            gust     = cur.get("wind_gusts_10m")
            temp     = cur.get("temperature_2m")
            humidity = cur.get("relative_humidity_2m")
            api_time = cur.get("time")   # ISO string in IST

            # ── Validate ──────────────────────────────────────────────────────
            if speed is None or direction is None:
                log.warning(f"[{sensor_id}] Missing wind fields in response")
                return None
            if not (0 <= speed <= 100):
                log.warning(f"[{sensor_id}] Implausible speed: {speed} m/s — skipped")
                return None
            if not (0 <= direction <= 360):
                log.warning(f"[{sensor_id}] Implausible direction: {direction}° — skipped")
                return None
            if gust is not None and gust < speed:
                log.warning(f"[{sensor_id}] Gust ({gust}) < speed ({speed}) — discarding gust")
                gust = None

            # ── U/V components ────────────────────────────────────────────────
            u = round(-speed * math.sin(math.radians(direction)), 6)
            v = round(-speed * math.cos(math.radians(direction)), 6)

            return {
                "api_timestamp_IST": api_time,
                "wind_speed_ms":     round(speed, 4),
                "wind_speed_kmh":    round(speed * 3.6, 3),
                "wind_dir_deg":      direction,
                "wind_gust_ms":      round(gust, 4) if gust else None,
                "u_comp":            u,
                "v_comp":            v,
                "temp_c":            temp,
                "humidity_pct":      humidity,
            }

        except requests.exceptions.Timeout:
            log.warning(f"[{sensor_id}] Timeout — attempt {attempt}/{retries}")
        except requests.exceptions.RequestException as e:
            log.warning(f"[{sensor_id}] Request error attempt {attempt}: {e}")

        time.sleep(backoff * attempt)

    log.error(f"[{sensor_id}] All {retries} attempts failed")
    return None


# ─── COLLECT ──────────────────────────────────────────────────────────────────

def collect_data():
    collection_time = datetime.now(timezone.utc).isoformat()
    rows = []

    for loc in LOCATIONS:
        result = get_wind(loc["id"], loc["lat"], loc["lon"])

        if result is None:
            log.warning(f"Skipping {loc['id']} — no valid data")
            continue

        rows.append({
            "collection_timestamp_UTC": collection_time,
            "api_timestamp_IST":        result["api_timestamp_IST"],
            "sensor_id":                loc["id"],
            "lat":                      loc["lat"],
            "lon":                      loc["lon"],
            "wind_speed_ms":            result["wind_speed_ms"],
            "wind_speed_kmh":           result["wind_speed_kmh"],
            "wind_dir_deg":             result["wind_dir_deg"],
            "wind_gust_ms":             result["wind_gust_ms"],
            "u_comp":                   result["u_comp"],
            "v_comp":                   result["v_comp"],
            "temp_c":                   result["temp_c"],
            "humidity_pct":             result["humidity_pct"],
        })

    if not rows:
        log.warning("No data collected this cycle — file not updated")
        return

    df = pd.DataFrame(rows)

    # Safe append — write header only if file is new or empty
    write_header = (
        not os.path.exists(OUTPUT_FILE)
        or os.path.getsize(OUTPUT_FILE) == 0
    )
    df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)

    log.info(
        f"✓ Saved {len(rows)} rows | "
        f"speeds: {[r['wind_speed_ms'] for r in rows]} m/s | "
        f"dirs: {[r['wind_dir_deg'] for r in rows]}°"
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 55)
    log.info("  Wind collector started — Open-Meteo (no API key)")
    log.info(f"  Polling every {INTERVAL_SECONDS}s | {len(LOCATIONS)} sensors")
    log.info("=" * 55)

    while True:
        try:
            collect_data()
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
        time.sleep(INTERVAL_SECONDS)