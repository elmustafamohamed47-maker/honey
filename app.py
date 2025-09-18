# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

# --------------------------------------------------
# 1. LOAD MODEL + SCALER
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()
# if you also need a separate scaler, load it the same way or re-save together # list of derived feature names

# --------------------------------------------------
# 2. FEATURE ENGINEERING (exactly like training)
# --------------------------------------------------
def engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # ensure datetime
    df["recordedAt"] = pd.to_datetime(df["recordedAt"])

    # ---- time-based ----
    df["hour"] = df["recordedAt"].dt.hour
    df["day_of_week"] = df["recordedAt"].dt.dayofweek
    df["month"] = df["recordedAt"].dt.month
    df["day_of_year"] = df["recordedAt"].dt.dayofyear

    # ---- cyclical ----
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- lags ----
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"temperature_lag_{lag}"] = df["temperature"].shift(lag)
        df[f"humidity_lag_{lag}"] = df["humidity"].shift(lag)

    # ---- rolling stats ----
    for window in [3, 6, 12, 24]:
        df[f"temperature_rolling_mean_{window}"] = (
            df["temperature"].rolling(window=window).mean()
        )
        df[f"humidity_rolling_mean_{window}"] = (
            df["humidity"].rolling(window=window).mean()
        )
        df[f"temperature_rolling_std_{window}"] = (
            df["temperature"].rolling(window=window).std()
        )
        df[f"humidity_rolling_std_{window}"] = (
            df["humidity"].rolling(window=window).std()
        )

    # ---- diffs ----
    df["temperature_diff"] = df["temperature"].diff()
    df["humidity_diff"] = df["humidity"].diff()

    # ---- interaction ----
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]

    # ---- keep only what the model saw ----
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing derived columns: {missing}")
    return df[EXPECTED]
