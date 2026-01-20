import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="IDS", layout="centered")

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛡️ Intrusion Detection System")

file = st.file_uploader("Upload Test_data.csv", type="csv")

if file:
    df = pd.read_csv(file)

    # نحيد label كان موجود
    if "class" in df.columns:
        X = df.drop("class", axis=1)
    else:
        X = df.copy()

    # نحولو أرقام
    X = X.apply(pd.to_numeric, errors="coerce")

    # نعوّض NaN
    X = X.fillna(0)

    # 🔥 نضبط عدد الأعمدة بالقوة
    n_features = scaler.mean_.shape[0]
    X = X.iloc[:, :n_features]

    # نحولو numpy
    X = X.to_numpy()

    # scaling
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    df["Label"] = ["🚨 Anomaly" if p == 1 else "✅ Normal" for p in preds]

    st.success("Prediction OK ✅")


