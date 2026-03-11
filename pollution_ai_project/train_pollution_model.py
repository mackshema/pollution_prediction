import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("training_dataset_clean.csv")

features = [
    "pm25",
    "pm10",
    "co",
    "no2",
    "so2",
    "temperature",
    "humidity",
    "wind_speed",
    "wind_dir",
    "u_comp",
    "v_comp"
]

targets = [
    "pm25",
    "aqi",
    "lat_B",
    "lon_B"
]

X = df[features].values
y = df[targets].values

# -----------------------------
# Normalize data
# -----------------------------

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# -----------------------------
# Convert to LSTM sequences
# -----------------------------

TIME_STEPS = 24

X_seq = []
y_seq = []

for i in range(TIME_STEPS, len(X_scaled)):
    X_seq.append(X_scaled[i-TIME_STEPS:i])
    y_seq.append(y_scaled[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence shape:", X_seq.shape)

# -----------------------------
# Train / validation split
# -----------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

# -----------------------------
# Build Bi-LSTM + Attention
# -----------------------------

inputs = tf.keras.layers.Input(shape=(X_seq.shape[1], X_seq.shape[2]))

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(128, return_sequences=True)
)(inputs)

x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True)
)(x)

# attention layer
attention = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=32
)(x, x)

x = tf.keras.layers.Add()([x, attention])

x = tf.keras.layers.GlobalAveragePooling1D()(x)

x = tf.keras.layers.Dense(64, activation="relu")(x)

outputs = tf.keras.layers.Dense(4)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="huber",
    metrics=["mae"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32
)

# -----------------------------
# Save Model
# -----------------------------

model.save("pollution_prediction_model.h5")

print("Model saved!")

# -----------------------------
# Plot Training
# -----------------------------

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train", "validation"])
plt.title("Training Loss")
plt.savefig("training_loss.png")
