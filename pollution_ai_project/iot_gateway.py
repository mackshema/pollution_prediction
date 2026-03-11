import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime, timezone

BROKER = "broker.hivemq.com"
PORT = 1883

TOPIC = "hackathon/pollution/sensors"

OUTPUT_FILE = r"C:\Users\sivas\OneDrive\ECE_climate_forcasting_hackthon\pollution_ai_project\iot_pollution_data.csv"


def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected to MQTT Broker")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):

    try:
        data = json.loads(msg.payload.decode())

        row = {
            "timestamp_UTC": datetime.now(timezone.utc).isoformat(),
            "device_id": data.get("device_id"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),

            "pm25": data.get("PM2.5"),
            "pm10": data.get("PM10"),
            "co": data.get("CO"),
            "no2": data.get("NO2"),
            "so2": data.get("SO2"),

            "temperature": data.get("Temperature"),
            "humidity": data.get("Humidity"),

            "aqi": data.get("AQI"),
            "category": data.get("Category")
        }

        df = pd.DataFrame([row])

        try:
            with open(OUTPUT_FILE, 'r'):
                df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
        except FileNotFoundError:
            df.to_csv(OUTPUT_FILE, index=False)

        print("Data saved:", row)

    except Exception as e:
        print("Error:", e)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

print("Waiting for IoT sensor data...")

client.loop_forever() 
