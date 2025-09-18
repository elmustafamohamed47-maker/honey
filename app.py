import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ===============================
# Load the trained model
# ===============================
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ===============================
# Streamlit UI
# ===============================
st.title("🐝 Hive Weight Prediction App")

st.write("Enter the environmental parameters below to predict hive weight:")

# User inputs
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
hour = st.slider("⏱️ Hour (0–23)", 0, 23, 12)
day_of_week = st.selectbox("📅 Day of week (0=Mon ... 6=Sun)", list(range(7)), index=0)
month = st.selectbox("📆 Month (1–12)", list(range(1, 13)), index=datetime.now().month - 1)

# Derived features
thi = 0.8 * temperature + (temperature * humidity) / 500
light = 100 if (6 < hour < 18) else 0
heat_index = temperature + 0.5 * humidity
comfort = 70 - abs(temperature - 22) - 0.2 * abs(humidity - 50)
it = 25.0
ratio = temperature / (humidity + 1)

# Cyclical encodings
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

day_of_year = datetime.now().timetuple().tm_yday

# Base features
input_data = {
    'humidity': [humidity],
    'temperature': [temperature],
    'thi': [thi],
    'light': [light],
    'heat_index': [heat_index],
    'comfort': [comfort],
    'it': [it],
    'ratio': [ratio],
    'hour': [hour],
    'day_of_week': [day_of_week],
    'month': [month],
    'day_of_year': [day_of_year],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'month_sin': [month_sin],
    'month_cos': [month_cos]
}

# Add missing lag/rolling/diff/interaction features with 0
expected_features = model.get_booster().feature_names
for feat in expected_features:
    if feat not in input_data:
        input_data[feat] = [0]

# Build DataFrame in correct order
input_df = pd.DataFrame(input_data)[expected_features]

# ===============================
# Prediction
# ===============================
if st.button("🔮 Predict Hive Weight"):
    prediction = model.predict(input_df)[0]
    st.success(f"🤖 Predicted Hive Weight: {prediction:.2f} kg")

