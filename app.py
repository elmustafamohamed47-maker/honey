# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🐝 Honey Authenticity Checker", layout="centered")

# ------------- LOAD PRE-TRAINED PIPELINE -------------
@st.cache_resource
def load_model():
    # xgb_model.pkl contains ONLY the trained model
    return joblib.load("xgb_model.pkl")

model = load_model()
expected_cols = scaler.feature_names_in_

# ------------- UI -------------
st.title("🍯 Honey Authenticity Predictor")
st.markdown("Upload an **Excel file (.xlsx)** with the same feature columns used during training.")

uploaded = st.file_uploader("Choose Excel file", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded)          # <-- Excel read

    missing = set(expected_cols) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    X = df[expected_cols]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1]

    out = df.copy()
    out["prediction"] = preds
    out["prob_real"]  = proba.round(3)
    st.success("Done!")
    st.dataframe(out)

    csv = out.to_csv(index=False).encode()
    st.download_button("Download results", csv, "predictions.csv", "text/csv")
