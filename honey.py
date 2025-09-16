import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# -------------------------------
# 1. Preprocessing Function
# -------------------------------
def preprocess_data(df):
    df['recordedAt'] = pd.to_datetime(df['recordedAt'])
    df = df.sort_values('recordedAt').reset_index(drop=True)

    # Time features
    df['hour'] = df['recordedAt'].dt.hour
    df['day_of_week'] = df['recordedAt'].dt.dayofweek
    df['month'] = df['recordedAt'].dt.month
    df['day_of_year'] = df['recordedAt'].dt.dayofyear

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
        df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)

    # Rolling stats
    for window in [3, 6, 12, 24]:
        df[f'temperature_rolling_mean_{window}'] = df['temperature'].rolling(window=window).mean()
        df[f'humidity_rolling_mean_{window}'] = df['humidity'].rolling(window=window).mean()
        df[f'temperature_rolling_std_{window}'] = df['temperature'].rolling(window=window).std()
        df[f'humidity_rolling_std_{window}'] = df['humidity'].rolling(window=window).std()

    # Diff + interaction
    df['temperature_diff'] = df['temperature'].diff()
    df['humidity_diff'] = df['humidity'].diff()
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

    return df.dropna()

# -------------------------------
# 2. Model Loader (cached)
# -------------------------------
@st.cache_resource
def load_or_train_model(df):
    processed_df = preprocess_data(df)

    feature_columns = [
        'humidity', 'temperature', 'thi', 'light', 'heat_index',
        'comfort', 'it', 'ratio', 'hour', 'day_of_week', 'month',
        'day_of_year', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    lag_features = [col for col in processed_df.columns if 'lag' in col or 'rolling' in col or 'diff' in col or 'interaction' in col]
    feature_columns.extend(lag_features)

    X = processed_df[feature_columns]
    y = processed_df['weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Save model
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, feature_columns

# -------------------------------
# 3. Streamlit App
# -------------------------------
st.title("🐝 Honey Weight Prediction")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Train or load model only once
    model, feature_columns = load_or_train_model(df)

    processed_df = preprocess_data(df)
    X = processed_df[feature_columns]
    y = processed_df['weight']

    preds = model.predict(X)

    st.success("✅ Predictions ready!")
    st.line_chart(pd.DataFrame({"True": y.values, "Predicted": preds}, index=processed_df['recordedAt']))

    st.write("📊 Latest Prediction:", preds[-1])
else:
    st.info("Upload an Excel file to start.")

import numpy as np
import pickle

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Live prediction function
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

def predict_weight_streamlit(model, X):
    """
    Streamlit function to predict hive weight based on user input
    """
    st.subheader("🐝 Enter Hive Parameters")

    # User inputs
    temperature = st.number_input("🌡️ Temperature (°C):", value=25.0, min_value=-20.0, max_value=50.0)
    humidity = st.number_input("💧 Humidity (%):", value=50.0, min_value=0.0, max_value=100.0)
    hour = st.slider("⏱️ Hour of day:", 0, 23, 12)
    day_of_week = st.slider("📅 Day of week (0=Mon, 6=Sun):", 0, 6, 2)
    month = st.slider("📆 Month (1-12):", 1, 12, 6)

    # Derived features
    thi = 0.8 * temperature + (temperature * humidity) / 500
    light = 100 if (hour > 6) and (hour < 18) else 0
    heat_index = temperature + 0.5 * humidity
    comfort = 70 - abs(temperature - 22) - 0.2 * abs(humidity - 50)
    it = 25.0
    ratio = temperature / (humidity + 1)

    # Cyclical features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Day of year
    day_of_year = datetime.now().timetuple().tm_yday

    # Build input data
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

    # Add missing lag/rolling features as 0
    for col in X.columns:
        if col not in input_data:
            input_data[col] = [0]

    # Create DataFrame with correct column order
    input_df = pd.DataFrame(input_data)[X.columns]

    # Predict when button clicked
    if st.button("🔮 Predict Hive Weight"):
        prediction = model.predict(input_df)[0]
        st.success(f"🤖 Predicted Hive Weight: {prediction:.2f} kg")
        
predict_weight_streamlit(model, X)





