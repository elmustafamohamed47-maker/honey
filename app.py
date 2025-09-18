# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")   # single XGBClassifier object

model = load_model()
expected_cols = model.get_booster().feature_names   # XGBoost native list

st.title("🍯 Honey Authenticity Predictor")
uploaded = st.file_uploader("Choose Excel file", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded)

    missing = set(expected_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    X = df[expected_cols]
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["prediction"] = preds
    out["prob_real"]  = proba.round(3)
    st.dataframe(out)

    csv = out.to_csv(index=False).encode()
    st.download_button("Download results", csv, "predictions.csv", "text/csv")
