
# ============================================================
#  Smart Tamil Nadu Air Quality Monitoring & Prediction System
#  dashboard.py  –  Real-Time Streamlit Dashboard
# ============================================================

import os, math, time
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Tamil Nadu – AQI Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────
# CUSTOM CSS  (dark glassmorphism theme)
# ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #050d1a;
    color: #e0eaff;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem; }

/* ── HERO BANNER ── */
.hero-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0e2040 40%, #0a2a1a 100%);
    border: 1px solid rgba(0, 210, 180, 0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(0,210,180,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(0,120,255,0.06) 0%, transparent 60%);
}
.hero-title {
    font-size: 2.2rem; font-weight: 900; margin: 0;
    background: linear-gradient(90deg, #00d2b4, #00aaff, #7c63ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-shadow: none; letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.95rem; color: #7aa3cc; margin-top: 6px; font-weight: 300;
}
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,210,100,0.15); border: 1px solid rgba(0,210,100,0.4);
    color: #00d264; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 1px;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #00d264;
    animation: pulse-dot 1.4s infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.7); }
}

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #00d2b4; margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
}

/* ── STATUS CHIPS ── */
.status-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.status-chip {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.05); border-radius: 10px;
    padding: 10px 18px; border: 1px solid rgba(255,255,255,0.07);
    flex: 1; min-width: 160px;
}
.chip-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.chip-label { font-size: 0.75rem; color: #7aa3cc; }
.chip-value { font-size: 0.88rem; font-weight: 600; }
.dot-green  { background: #00d264; box-shadow: 0 0 8px #00d264; }
.dot-blue   { background: #00aaff; box-shadow: 0 0 8px #00aaff; }
.dot-purple { background: #a855f7; box-shadow: 0 0 8px #a855f7; }
.dot-orange { background: #f97316; box-shadow: 0 0 8px #f97316; }
.dot-red    { background: #ef4444; box-shadow: 0 0 8px #ef4444; }
.dot-yellow { background: #eab308; box-shadow: 0 0 8px #eab308; }

/* ── METRIC TILES ── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: 12px; }
.metric-tile {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px; padding: 16px 18px; text-align: center;
}
.metric-val { font-size: 1.9rem; font-weight: 800; line-height: 1.1; }
.metric-lbl { font-size: 0.72rem; color: #7aa3cc; margin-top: 4px; letter-spacing: 0.5px; }
.metric-unit { font-size: 0.75rem; color: #4a6a8a; margin-top: 2px; }

/* ── WIND COMPASS ── */
.compass-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 10px;
}
.compass-ring {
    width: 120px; height: 120px; border-radius: 50%;
    border: 2px solid rgba(0,210,180,0.3);
    position: relative; background: radial-gradient(circle, rgba(0,30,60,0.8), rgba(0,10,30,0.9));
    box-shadow: 0 0 30px rgba(0,210,180,0.15);
}
.compass-label {
    font-size: 0.65rem; font-weight: 700; color: #00d2b4;
    position: absolute; text-align: center;
}
.compass-n { top: 3px; left: 50%; transform: translateX(-50%); }
.compass-s { bottom: 3px; left: 50%; transform: translateX(-50%); }
.compass-e { right: 3px; top: 50%; transform: translateY(-50%); }
.compass-w { left: 3px; top: 50%; transform: translateY(-50%); }
.compass-arrow-svg {
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
}
.wind-speed-label {
    font-size: 1.3rem; font-weight: 800;
    color: #00d2b4; margin-top: 10px; text-align: center;
}
.wind-dir-label {
    font-size: 0.8rem; color: #7aa3cc; text-align: center;
}

/* ── FORECAST TABLE ── */
.forecast-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
}
.forecast-key { font-size: 0.82rem; color: #7aa3cc; }
.forecast-val { font-size: 0.88rem; font-weight: 600; color: #e0eaff; }

/* ── AQI COLOR LEGEND ── */
.legend-bar {
    display: flex; border-radius: 8px; overflow: hidden; height: 12px; margin: 8px 0 4px;
}
.legend-seg { flex: 1; }
.legend-labels { display: flex; justify-content: space-between; font-size: 0.65rem; color: #7aa3cc; }

/* ── SCROLLABLE TABLE ── */
.stDataFrame { border-radius: 10px !important; }

/* ── DIVIDER ── */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 20px 0; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
IOT_FILE        = os.path.join(BASE, "iot_pollution_data.csv")
WIND_FILE       = os.path.join(BASE, "live_wind_buffer.csv")
PREDICT_FILE    = os.path.join(BASE, "prediction_output.csv")

# ────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────

def aqi_color(aqi_val):
    if aqi_val is None: return "#888"
    if aqi_val <= 50:   return "#22c55e"
    if aqi_val <= 100:  return "#eab308"
    if aqi_val <= 150:  return "#f97316"
    if aqi_val <= 200:  return "#ef4444"
    return "#a855f7"

def aqi_dot_class(aqi_val):
    if aqi_val is None: return "dot-orange"
    if aqi_val <= 50:   return "dot-green"
    if aqi_val <= 100:  return "dot-yellow"
    if aqi_val <= 150:  return "dot-orange"
    if aqi_val <= 200:  return "dot-red"
    return "dot-purple"

def wind_dir_label(deg):
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = int((deg + 11.25) / 22.5) % 16
    return dirs[idx]

def safe_read(path, **kwargs):
    try:
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception:
        return pd.DataFrame()

def calc_destination(lat, lon, bearing_deg, dist_km):
    R = 6371.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dist  = dist_km / R
    lat2  = math.asin(math.sin(lat1)*math.cos(dist) +
                      math.cos(lat1)*math.sin(dist)*math.cos(bearing))
    lon2  = lon1 + math.atan2(math.sin(bearing)*math.sin(dist)*math.cos(lat1),
                               math.cos(dist) - math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def wind_arrow_svg(direction_deg, speed_ms, size=80):
    """Returns HTML for a wind arrow SVG that rotates to show wind direction."""
    # Wind direction: where wind is coming FROM. Arrow points TO where it goes.
    angle = direction_deg  # rotate so 0°=N
    color_intensity = min(255, int(speed_ms * 30))
    r = 0; g = color_intensity; b = min(255, 255 - color_intensity//2)
    stroke = f"rgb({r},{g},{b})"
    half = size // 2
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"
         style="transform:rotate({angle}deg); transition:transform 0.8s ease;">
      <defs>
        <marker id="arrowhead" markerWidth="6" markerHeight="6"
                refX="3" refY="3" orient="auto">
          <polygon points="0 0, 6 3, 0 6" fill="{stroke}" opacity="0.9"/>
        </marker>
      </defs>
      <!-- Tail feathers (wind barb style) -->
      <line x1="{half}" y1="{size-8}" x2="{half}" y2="8"
            stroke="{stroke}" stroke-width="2.5" opacity="0.85"
            marker-end="url(#arrowhead)"/>
      <line x1="{half}" y1="{size-8}" x2="{half-12}" y2="{size-22}"
            stroke="{stroke}" stroke-width="1.8" opacity="0.7"/>
      <line x1="{half}" y1="{size-18}" x2="{half-9}" y2="{size-28}"
            stroke="{stroke}" stroke-width="1.4" opacity="0.55"/>
      <!-- Glow circle at root -->
      <circle cx="{half}" cy="{size-8}" r="3"
              fill="{stroke}" opacity="0.6"/>
    </svg>"""

# ────────────────────────────────────────────────
# DATA LOADERS
# ────────────────────────────────────────────────

@st.cache_data(ttl=12)
def load_iot():
    df = safe_read(IOT_FILE)
    if df.empty:
        return df
    # normalise column names
    df.columns = [c.strip().lower().replace("timestamp_utc","timestamp") for c in df.columns]
    if "timestamp" not in df.columns and "timestamp_utc" in df.columns:
        df.rename(columns={"timestamp_utc":"timestamp"}, inplace=True)
    for col in ["lat","lon","pm25","pm10","co","no2","so2","temperature","humidity","aqi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=12)
def load_wind():
    df = safe_read(WIND_FILE, on_bad_lines="skip")
    if df.empty:
        return df
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Explicit priority mapping (new live_wind_feed.py names first) ──
    # Speed: prefer wind_speed_ms > wind_speed > any col with 'speed' (but NOT kmh)
    if "wind_speed_ms" in df.columns:
        spd_col = "wind_speed_ms"
    elif "wind_speed" in df.columns:
        spd_col = "wind_speed"
    else:
        spd_col = next((c for c in df.columns if "speed" in c and "kmh" not in c), None)

    # Direction: prefer wind_dir_deg > wind_dir > any col with 'dir'
    if "wind_dir_deg" in df.columns:
        dir_col = "wind_dir_deg"
    elif "wind_dir" in df.columns:
        dir_col = "wind_dir"
    else:
        dir_col = next((c for c in df.columns if "dir" in c), None)

    # U/V components
    u_col = next((c for c in df.columns if "u_comp" in c or c == "u"), None)
    v_col = next((c for c in df.columns if "v_comp" in c or c == "v"), None)

    # Rename to standard names
    rename_map = {}
    if spd_col and spd_col != "wind_speed": rename_map[spd_col] = "wind_speed"
    if dir_col and dir_col != "wind_dir":   rename_map[dir_col] = "wind_dir"
    if u_col   and u_col   != "u_comp":     rename_map[u_col]   = "u_comp"
    if v_col   and v_col   != "v_comp":     rename_map[v_col]   = "v_comp"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Coerce numerics
    for col in ["lat", "lon", "wind_speed", "wind_dir", "u_comp", "v_comp",
                "wind_gust_ms", "temp_c", "humidity_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where both speed and dir are NaN
    df = df.dropna(subset=["wind_speed", "wind_dir"], how="all")
    return df

@st.cache_data(ttl=12)
def load_predictions():
    return safe_read(PREDICT_FILE) if os.path.exists(PREDICT_FILE) else pd.DataFrame()

# ────────────────────────────────────────────────
# AUTO-REFRESH  (JavaScript ping every 12 s)
# ────────────────────────────────────────────────
st.markdown("""
<script>
setTimeout(function(){ window.location.reload(); }, 12000);
</script>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────
iot_df   = load_iot()
wind_df  = load_wind()
pred_df  = load_predictions()

iot_ok   = not iot_df.empty
wind_ok  = not wind_df.empty
pred_ok  = not pred_df.empty

# Latest rows
latest_iot  = iot_df.sort_values("timestamp").groupby("device_id").last().reset_index() if iot_ok else pd.DataFrame()
latest_wind = wind_df.dropna(subset=["wind_speed","wind_dir"]).tail(3) if wind_ok else pd.DataFrame()

# Dominant wind values (mean of latest)
if not latest_wind.empty:
    w_speed = float(latest_wind["wind_speed"].mean())
    w_dir   = float(latest_wind["wind_dir"].mean())
    w_u     = float(latest_wind["u_comp"].mean()) if "u_comp" in latest_wind.columns else 0.0
    w_v     = float(latest_wind["v_comp"].mean()) if "v_comp" in latest_wind.columns else 0.0
else:
    w_speed, w_dir, w_u, w_v = 0.0, 0.0, 0.0, 0.0

# Reference sensor location (worst AQI sensor or first)
if iot_ok and not latest_iot.empty:
    worst = latest_iot.sort_values("aqi", ascending=False).iloc[0]
    ref_lat, ref_lon = float(worst["lat"]), float(worst["lon"])
    ref_pm25 = float(worst["pm25"]) if "pm25" in worst.index else 0.0
    ref_aqi  = float(worst["aqi"])  if "aqi"  in worst.index else 0.0
else:
    ref_lat, ref_lon, ref_pm25, ref_aqi = 10.90, 78.70, 0.0, 0.0

# ── Pollution forecast maths ──
TIME_HOURS = 1.0
dist_km = w_speed * 3.6 * TIME_HOURS
lat_change = dist_km / 111.0
lon_change = dist_km / (111.0 * math.cos(math.radians(ref_lat))) if ref_lat != 0 else 0
pred_lat = ref_lat + lat_change * math.sin(math.radians(w_dir))
pred_lon = ref_lon + lon_change * math.cos(math.radians(w_dir))
arrival_min = (dist_km / (w_speed * 3.6) * 60) if w_speed > 0 else 0
pred_pm25   = ref_pm25 * 1.08
pred_aqi    = ref_aqi  * 1.05

# Override with prediction file if available
if pred_ok:
    pr = pred_df.iloc[-1]
    for col in ["predicted_pm25","predicted_aqi","predicted_lat",
                "predicted_lon","distance_km","arrival_time_min"]:
        if col in pred_df.columns:
            pr[col] = pd.to_numeric(pr[col], errors="coerce")
    if "predicted_pm25"    in pred_df.columns: pred_pm25   = float(pr["predicted_pm25"])
    if "predicted_aqi"     in pred_df.columns: pred_aqi    = float(pr["predicted_aqi"])
    if "predicted_lat"     in pred_df.columns: pred_lat    = float(pr["predicted_lat"])
    if "predicted_lon"     in pred_df.columns: pred_lon    = float(pr["predicted_lon"])
    if "distance_km"       in pred_df.columns: dist_km     = float(pr["distance_km"])
    if "arrival_time_min"  in pred_df.columns: arrival_min = float(pr["arrival_time_min"])

now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")

# ════════════════════════════════════════════════
#  HERO BANNER
# ════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
    <div>
      <p class="hero-title">🌬️ Smart Tamil Nadu AQI</p>
      <p class="hero-subtitle">
        Real-Time Air Quality Monitoring · Wind Trajectory · Pollution Forecast
      </p>
    </div>
    <div>
      <div class="live-badge">
        <div class="live-dot"></div> LIVE FEED
      </div>
      <div style="font-size:0.75rem;color:#4a6a8a;margin-top:6px;text-align:right;">
        🕐 {now_str}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  SECTION 1 – SYSTEM STATUS
# ════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⬡ Section 1 — System Status</div>', unsafe_allow_html=True)

chips_html = '<div class="status-grid">'
chips = [
    ("dot-green"  if iot_ok  else "dot-red",   "IoT Sensors",    "ONLINE"  if iot_ok  else "OFFLINE"),
    ("dot-blue"   if wind_ok else "dot-red",   "Wind Feed",      "ACTIVE"  if wind_ok else "INACTIVE"),
    ("dot-purple" if pred_ok else "dot-orange","Prediction AI",  "RUNNING" if pred_ok else "WIND CALC"),
    ("dot-yellow", "Auto-Refresh", "12 s"),
    ("dot-green",  "Last Update",  now_str[:20]),
]
for dot_cls, label, value in chips:
    chips_html += f"""
    <div class="status-chip">
      <div class="chip-dot {dot_cls}"></div>
      <div>
        <div class="chip-label">{label}</div>
        <div class="chip-value">{value}</div>
      </div>
    </div>"""
chips_html += '</div>'
st.markdown(chips_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  SECTION 2 – SENSOR DATA TABLE
# ════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📡 Section 2 — Live Sensor Readings</div>', unsafe_allow_html=True)

if iot_ok and not latest_iot.empty:
    display_cols = [c for c in [
        "device_id","lat","lon","pm25","pm10","co","no2","so2",
        "temperature","humidity","aqi","category","timestamp"
    ] if c in latest_iot.columns]

    styled = latest_iot[display_cols].copy()
    # Round floats
    for col in ["pm25","pm10","co","no2","so2","temperature","humidity","aqi"]:
        if col in styled.columns:
            styled[col] = styled[col].round(1)

    def highlight_aqi(row):
        val = row.get("aqi", None)
        if pd.isna(val): return [""] * len(row)
        c = aqi_color(val)
        return [f"color:{c};font-weight:700" if col == "aqi" else "" for col in row.index]

    st.dataframe(
        styled,
        use_container_width=True,
        height=200,
        column_config={
            "device_id":   st.column_config.TextColumn("Device"),
            "lat":         st.column_config.NumberColumn("Lat", format="%.4f"),
            "lon":         st.column_config.NumberColumn("Lon", format="%.4f"),
            "pm25":        st.column_config.NumberColumn("PM2.5 (µg/m³)"),
            "pm10":        st.column_config.NumberColumn("PM10 (µg/m³)"),
            "co":          st.column_config.NumberColumn("CO (ppm)"),
            "no2":         st.column_config.NumberColumn("NO₂ (ppb)"),
            "so2":         st.column_config.NumberColumn("SO₂ (ppb)"),
            "temperature": st.column_config.NumberColumn("Temp (°C)"),
            "humidity":    st.column_config.NumberColumn("Humidity (%)"),
            "aqi":         st.column_config.NumberColumn("AQI"),
            "category":    st.column_config.TextColumn("Category"),
        },
        hide_index=True,
    )
else:
    st.info("⚠️  No IoT sensor data found. Ensure `iot_pollution_data.csv` is populated.")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  SECTION 3 – INTERACTIVE MAP + WIND PANEL
# ════════════════════════════════════════════════
st.markdown('<div class="section-title" style="padding:0 4px 8px;">🗺️ Section 3 — Smart Tamil Nadu Pollution Map</div>', unsafe_allow_html=True)

map_col, wind_col = st.columns([3, 1])

with map_col:
    # Build Folium map
    m = folium.Map(
        location=[10.90, 78.70],
        zoom_start=7,
        tiles=None,
    )
    # Dark tile layer
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Dark Map",
        max_zoom=19,
    ).add_to(m)

    # ── Heatmap ──
    if iot_ok and not latest_iot.empty:
        heat_data = []
        for _, row in latest_iot.iterrows():
            if not (pd.isna(row.get("lat")) or pd.isna(row.get("lon")) or pd.isna(row.get("pm25"))):
                heat_data.append([float(row["lat"]), float(row["lon"]),
                                   min(float(row["pm25"]) / 200.0, 1.0)])
        if heat_data:
            HeatMap(
                heat_data,
                radius=45, blur=30, max_zoom=9,
                gradient={0.3:"#22c55e", 0.55:"#eab308",
                           0.70:"#f97316", 0.85:"#ef4444", 1.0:"#a855f7"},
            ).add_to(m)

    # ── IoT sensor markers ──
    if iot_ok and not latest_iot.empty:
        for _, row in latest_iot.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            clr   = aqi_color(row.get("aqi"))
            lat_r = float(row["lat"]); lon_r = float(row["lon"])
            aqi_v = row.get("aqi","–"); pm25_v = row.get("pm25","–")
            temp_v = row.get("temperature","–"); hum_v = row.get("humidity","–")
            cat_v = row.get("category","–"); dev = row.get("device_id","–")
            popup_html = f"""
            <div style="font-family:Inter,sans-serif;width:180px;background:#0a1628;
                        color:#e0eaff;border-radius:10px;padding:12px;">
              <div style="font-weight:700;font-size:0.95rem;color:{clr};
                          border-bottom:1px solid rgba(255,255,255,0.1);
                          padding-bottom:6px;margin-bottom:8px;">
                📡 {dev}
              </div>
              <table style="width:100%;font-size:0.8rem;">
                <tr><td style="color:#7aa3cc;">PM2.5</td>
                    <td style="text-align:right;font-weight:600;">{pm25_v} µg/m³</td></tr>
                <tr><td style="color:#7aa3cc;">AQI</td>
                    <td style="text-align:right;font-weight:700;color:{clr};">{aqi_v}</td></tr>
                <tr><td style="color:#7aa3cc;">Category</td>
                    <td style="text-align:right;">{cat_v}</td></tr>
                <tr><td style="color:#7aa3cc;">Temp</td>
                    <td style="text-align:right;">{temp_v} °C</td></tr>
                <tr><td style="color:#7aa3cc;">Humidity</td>
                    <td style="text-align:right;">{hum_v} %</td></tr>
              </table>
            </div>"""
            folium.CircleMarker(
                location=[lat_r, lon_r],
                radius=14,
                color=clr, fill=True, fill_color=clr,
                fill_opacity=0.75, weight=2.5,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{dev} | AQI {aqi_v}",
            ).add_to(m)
            # Pulsing outer ring via DivIcon
            folium.Marker(
                location=[lat_r, lon_r],
                icon=folium.DivIcon(html=f"""
                <div style="
                    width:30px;height:30px;border-radius:50%;
                    border:2px solid {clr};opacity:0.5;
                    animation:ripple 2s infinite;
                    transform:translate(-15px,-15px);
                    background:transparent;">
                </div>
                <style>
                @keyframes ripple {{
                  0%  {{transform:translate(-15px,-15px) scale(1);opacity:0.5;}}
                  100%{{transform:translate(-15px,-15px) scale(2.5);opacity:0;}}
                }}
                </style>""", icon_size=(0,0)),
            ).add_to(m)

    # ── Wind arrows for each sensor ──
    if iot_ok and not latest_iot.empty and w_speed > 0:
        for _, row in latest_iot.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                continue
            lat_s = float(row["lat"]); lon_s = float(row["lon"])
            # Arrow endpoint: proportional to wind speed (max 0.12° at 10 m/s)
            arrow_scale = min(w_speed / 10.0, 1.0) * 0.10
            d_lat = arrow_scale * math.cos(math.radians(w_dir))
            d_lon = arrow_scale * math.sin(math.radians(w_dir)) / math.cos(math.radians(lat_s))
            end_lat = lat_s + d_lat
            end_lon = lon_s + d_lon

            # Shaft
            folium.PolyLine(
                locations=[[lat_s, lon_s],[end_lat, end_lon]],
                color="#00d2b4", weight=2.5, opacity=0.85,
                tooltip=f"Wind {w_speed:.1f} m/s → {wind_dir_label(w_dir)}"
            ).add_to(m)
            # Arrowhead using a small rotated triangle marker
            bearing_rad = math.radians(w_dir)
            # barb 1
            b1_lat = end_lat - 0.012*math.cos(math.radians(w_dir-20))
            b1_lon = end_lon - 0.012*math.sin(math.radians(w_dir-20))/math.cos(math.radians(end_lat))
            # barb 2
            b2_lat = end_lat - 0.012*math.cos(math.radians(w_dir+20))
            b2_lon = end_lon - 0.012*math.sin(math.radians(w_dir+20))/math.cos(math.radians(end_lat))
            folium.PolyLine([[end_lat,end_lon],[b1_lat,b1_lon]],
                            color="#00d2b4",weight=2,opacity=0.85).add_to(m)
            folium.PolyLine([[end_lat,end_lon],[b2_lat,b2_lon]],
                            color="#00d2b4",weight=2,opacity=0.85).add_to(m)

            # Wind barbs (feathers) along the shaft for speed indication
            num_barbs = max(1, int(w_speed // 2))
            for bi in range(num_barbs):
                frac = 0.3 + bi * 0.15
                if frac > 0.85: break
                tick_lat = lat_s + frac * d_lat
                tick_lon = lon_s + frac * d_lon
                perp = math.radians(w_dir + 90)
                tk_len = 0.018
                tk1_lat = tick_lat + tk_len * math.cos(perp)
                tk1_lon = tick_lon + tk_len * math.sin(perp)/math.cos(math.radians(tick_lat))
                folium.PolyLine(
                    [[tick_lat,tick_lon],[tk1_lat,tk1_lon]],
                    color="#00d2b4", weight=1.5, opacity=0.65
                ).add_to(m)

    # ── Predicted pollution location marker ──
    if pred_lat and pred_lon:
        pred_popup = f"""
        <div style="font-family:Inter,sans-serif;background:#0a1628;
                    color:#e0eaff;border-radius:10px;padding:12px;width:180px;">
          <div style="font-weight:700;color:#00aaff;
                      border-bottom:1px solid rgba(255,255,255,0.1);
                      padding-bottom:6px;margin-bottom:8px;">
            🔵 Predicted Location
          </div>
          <table style="width:100%;font-size:0.8rem;">
            <tr><td style="color:#7aa3cc;">PM2.5 Forecast</td>
                <td style="text-align:right;font-weight:600;">{pred_pm25:.1f} µg/m³</td></tr>
            <tr><td style="color:#7aa3cc;">AQI Forecast</td>
                <td style="text-align:right;font-weight:700;color:#00aaff;">{pred_aqi:.0f}</td></tr>
            <tr><td style="color:#7aa3cc;">Distance</td>
                <td style="text-align:right;">{dist_km:.1f} km</td></tr>
            <tr><td style="color:#7aa3cc;">Arrival In</td>
                <td style="text-align:right;">{arrival_min:.0f} min</td></tr>
          </table>
        </div>"""
        folium.Marker(
            location=[pred_lat, pred_lon],
            icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
            popup=folium.Popup(pred_popup, max_width=200),
            tooltip=f"Predicted location in {arrival_min:.0f} min",
        ).add_to(m)

        # ── Movement path line ──
        if iot_ok and not latest_iot.empty:
            folium.PolyLine(
                locations=[[ref_lat, ref_lon],[pred_lat, pred_lon]],
                color="#f97316", weight=2, opacity=0.75,
                dash_array="8 4",
                tooltip="Pollution travel path",
            ).add_to(m)
            # Arrow at midpoint
            mid_lat = (ref_lat + pred_lat) / 2
            mid_lon = (ref_lon + pred_lon) / 2
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(html=f"""
                <div style="font-size:18px;transform:translate(-9px,-9px)
                            rotate({w_dir:.0f}deg);color:#f97316;">➤</div>""",
                    icon_size=(0, 0)),
                tooltip=f"Moving {wind_dir_label(w_dir)} at {w_speed:.1f} m/s",
            ).add_to(m)

    # AQI legend overlay
    legend_html = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:rgba(5,13,26,0.88);border:1px solid rgba(255,255,255,0.12);
                border-radius:10px;padding:10px 14px;font-family:Inter,sans-serif;
                font-size:11px;color:#e0eaff;min-width:170px;">
      <b style="color:#00d2b4;">AQI Legend</b>
      <div style="margin-top:6px;display:flex;flex-direction:column;gap:4px;">
        <span><span style="color:#22c55e;">■</span>  0–50 &nbsp;&nbsp;Good</span>
        <span><span style="color:#eab308;">■</span>  51–100  Moderate</span>
        <span><span style="color:#f97316;">■</span>  101–150 Unhealthy (Sensitive)</span>
        <span><span style="color:#ef4444;">■</span>  151–200 Unhealthy</span>
        <span><span style="color:#a855f7;">■</span>  200+  &nbsp; Very Unhealthy / Hazardous</span>
      </div>
      <div style="margin-top:8px;border-top:1px solid rgba(255,255,255,0.1);
                  padding-top:6px;color:#7aa3cc;">
        <span style="color:#00d2b4;">→</span> Wind arrow (barb = speed)<br>
        <span style="color:#f97316;">---</span> Pollution travel path<br>
        <span style="color:#00aaff;">●</span> Predicted location
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width="100%", height=520, returned_objects=[])

# ── Wind Panel ──
with wind_col:
    st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💨 Wind Status</div>', unsafe_allow_html=True)

    # Compass
    compass_html = f"""
    <div class="compass-wrap">
      <div class="compass-ring">
        <span class="compass-label compass-n">N</span>
        <span class="compass-label compass-s">S</span>
        <span class="compass-label compass-e">E</span>
        <span class="compass-label compass-w">W</span>
        <div class="compass-arrow-svg">
          {wind_arrow_svg(w_dir, w_speed, size=76)}
        </div>
      </div>
      <div class="wind-speed-label">{w_speed:.1f} m/s</div>
      <div class="wind-dir-label">{w_dir:.0f}° · {wind_dir_label(w_dir)}</div>
    </div>"""
    st.markdown(compass_html, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Wind metrics
    wind_metrics = [
        ("Speed",     f"{w_speed:.2f} m/s",              f"{w_speed*3.6:.1f} km/h"),
        ("Direction", f"{w_dir:.1f}°",                    wind_dir_label(w_dir)),
        ("U-comp",    f"{w_u:.2f} m/s",                  "East →"),
        ("V-comp",    f"{w_v:.2f} m/s",                  "North ↑"),
        ("Gust est.", f"{w_speed*1.4:.2f} m/s",          "approx."),
    ]
    for lbl, val, sub in wind_metrics:
        st.markdown(f"""
        <div class="forecast-row">
          <span class="forecast-key">{lbl}</span>
          <div style="text-align:right;">
            <div class="forecast-val">{val}</div>
            <div style="font-size:0.68rem;color:#4a6a8a;">{sub}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  SECTION 4 – POLLUTION FORECAST PANEL
# ════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔮 Section 4 — Pollution Forecast</div>', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

def metric_tile(col_obj, val, label, unit="", color="#00d2b4"):
    col_obj.markdown(f"""
    <div class="metric-tile">
      <div class="metric-val" style="color:{color};">{val}</div>
      <div class="metric-lbl">{label}</div>
      <div class="metric-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

metric_tile(f1, f"{ref_pm25:.1f}", "Current PM2.5", "µg/m³", "#eab308")
metric_tile(f2, f"{pred_pm25:.1f}", "Predicted PM2.5", "µg/m³", "#f97316")
metric_tile(f3, f"{ref_aqi:.0f}",  "Current AQI",  "", aqi_color(ref_aqi))
metric_tile(f4, f"{pred_aqi:.0f}", "Predicted AQI", "", aqi_color(pred_aqi))

st.markdown("<br>", unsafe_allow_html=True)

r1, r2 = st.columns(2)
with r1:
    rows = [
        ("🌬️ Wind Speed",              f"{w_speed:.2f} m/s   ({w_speed*3.6:.1f} km/h)"),
        ("🧭 Wind Direction",           f"{w_dir:.1f}°  →  {wind_dir_label(w_dir)}"),
        ("📍 Ref Sensor",               f"({ref_lat:.4f}°N, {ref_lon:.4f}°E)"),
        ("🎯 Predicted Location",       f"({pred_lat:.4f}°N, {pred_lon:.4f}°E)"),
    ]
    for k, v in rows:
        st.markdown(f"""
        <div class="forecast-row">
          <span class="forecast-key">{k}</span>
          <span class="forecast-val">{v}</span>
        </div>""", unsafe_allow_html=True)

with r2:
    rows2 = [
        ("📏 Pollution Travel Distance", f"{dist_km:.2f} km"),
        ("⏱️ Estimated Arrival Time",    f"{arrival_min:.0f} minutes"),
        ("🌡️ Time Horizon",              "1 hour"),
        ("📐 Formula",                   "d = v × 3.6 × t"),
    ]
    for k, v in rows2:
        st.markdown(f"""
        <div class="forecast-row">
          <span class="forecast-key">{k}</span>
          <span class="forecast-val">{v}</span>
        </div>""", unsafe_allow_html=True)

# AQI color legend bar
st.markdown("""
<br>
<div style="padding:0 4px;">
  <div style="font-size:0.7rem;color:#7aa3cc;margin-bottom:4px;letter-spacing:1px;">
    AQI SCALE
  </div>
  <div class="legend-bar">
    <div class="legend-seg" style="background:#22c55e;"></div>
    <div class="legend-seg" style="background:#eab308;"></div>
    <div class="legend-seg" style="background:#f97316;"></div>
    <div class="legend-seg" style="background:#ef4444;"></div>
    <div class="legend-seg" style="background:#a855f7;"></div>
  </div>
  <div class="legend-labels">
    <span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300+</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  SECTION 5 – HISTORICAL CHARTS
# ════════════════════════════════════════════════
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📈 Section 5 — Historical Trends</div>', unsafe_allow_html=True)

CHART_BG = "rgba(0,0,0,0)"
GRID_CLR  = "rgba(255,255,255,0.05)"
FONT_CLR  = "#7aa3cc"

def base_layout(title, yaxis_title):
    return dict(
        title=dict(text=title, font=dict(color="#e0eaff", size=13)),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_CLR, family="Inter"),
        xaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                   tickfont=dict(size=10), title="Time"),
        yaxis=dict(gridcolor=GRID_CLR, linecolor=GRID_CLR,
                   tickfont=dict(size=10), title=yaxis_title),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )

if iot_ok and len(iot_df) > 0:
    # Sort by timestamp for clean charts
    plot_df = iot_df.copy()
    if "timestamp" in plot_df.columns:
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], errors="coerce", utc=True)
        plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    chart1, chart2, chart3 = st.columns(3)

    # ── PM2.5 Trend ──
    with chart1:
        fig1 = go.Figure()
        for dev in plot_df["device_id"].unique() if "device_id" in plot_df.columns else []:
            sub = plot_df[plot_df["device_id"] == dev]
            if "pm25" not in sub.columns: continue
            fig1.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["pm25"],
                name=dev, mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=5),
            ))
        fig1.add_hline(y=35,  line_dash="dot", line_color="#22c55e",
                       annotation_text="Good 35",
                       annotation_font=dict(color="#22c55e", size=9))
        fig1.add_hline(y=75,  line_dash="dot", line_color="#f97316",
                       annotation_text="Moderate 75",
                       annotation_font=dict(color="#f97316", size=9))
        fig1.add_hrect(y0=0,  y1=35,  fillcolor="rgba(34,197,94,0.04)",  line_width=0)
        fig1.add_hrect(y0=35, y1=75,  fillcolor="rgba(234,179,8,0.04)",  line_width=0)
        fig1.add_hrect(y0=75, y1=500, fillcolor="rgba(239,68,68,0.04)",  line_width=0)
        fig1.update_layout(**base_layout("PM2.5 Concentration Trend", "PM2.5 (µg/m³)"))
        st.plotly_chart(fig1, use_container_width=True)

    # ── AQI Trend ──
    with chart2:
        fig2 = go.Figure()
        aqi_colors_trace = ["#00d2b4","#eab308","#f97316"]
        for i, dev in enumerate(plot_df["device_id"].unique() if "device_id" in plot_df.columns else []):
            sub = plot_df[plot_df["device_id"] == dev]
            if "aqi" not in sub.columns: continue
            c = aqi_colors_trace[i % len(aqi_colors_trace)]
            fig2.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["aqi"],
                name=dev, mode="lines+markers",
                line=dict(color=c, width=2),
                marker=dict(size=5, color=sub["aqi"].apply(aqi_color)),
                fill="tozeroy", fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:],16)},0.06)"
                    if c.startswith("#") and len(c)==7 else "rgba(0,210,180,0.06)",
            ))
        fig2.add_hline(y=100, line_dash="dot", line_color="#eab308",
                       annotation_text="Moderate 100",
                       annotation_font=dict(color="#eab308", size=9))
        fig2.add_hline(y=150, line_dash="dot", line_color="#ef4444",
                       annotation_text="Unhealthy 150",
                       annotation_font=dict(color="#ef4444", size=9))
        fig2.update_layout(**base_layout("AQI Trend Over Time", "AQI"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Wind Speed Trend ──
    with chart3:
        fig3 = go.Figure()
        if wind_ok and "wind_speed" in wind_df.columns:
            ts_col = next((c for c in wind_df.columns if "timestamp" in c), None)
            wdf_plot = wind_df.copy()
            if ts_col:
                wdf_plot[ts_col] = pd.to_datetime(wdf_plot[ts_col], errors="coerce", utc=True)
                wdf_plot = wdf_plot.dropna(subset=[ts_col]).sort_values(ts_col)
                for sid in wdf_plot["sensor_id"].unique() if "sensor_id" in wdf_plot.columns else []:
                    sub = wdf_plot[wdf_plot["sensor_id"] == sid]
                    fig3.add_trace(go.Scatter(
                        x=sub[ts_col], y=sub["wind_speed"],
                        name=sid, mode="lines+markers",
                        line=dict(width=2),
                        marker=dict(size=5),
                    ))
            # Wind direction as secondary axis if available (small scatter)
            if "wind_dir" in wdf_plot.columns and ts_col:
                fig3.add_trace(go.Scatter(
                    x=wdf_plot[ts_col], y=wdf_plot["wind_dir"] / 36.0,  # scale 0-10
                    name="Dir (÷36°)", mode="lines",
                    line=dict(color="#a855f7", width=1.5, dash="dot"),
                    opacity=0.6,
                ))
        else:
            fig3.add_annotation(text="No wind data", x=0.5, y=0.5,
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(color="#7aa3cc"))

        fig3.update_layout(**base_layout("Wind Speed & Direction Trend", "Speed (m/s)"))
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("📊 Insufficient data for charts. Run IoT gateway and wind collector first.")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center;padding:20px;color:#2a4a6a;font-size:0.75rem;">
  Smart Tamil Nadu AQI · IoT + Wind + AI Pollution Forecasting System<br>
  Auto-refreshes every 12 seconds · Built with Streamlit + Folium + Plotly<br>
  <span style="color:#00d2b4;font-weight:600;">🌍 Protecting Tamil Nadu's Air Quality</span>
</div>
""", unsafe_allow_html=True)
