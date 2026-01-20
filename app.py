import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page config
st.set_page_config(page_title="IDS", layout="centered")

# Load model & scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛡️ Intrusion Detection System")

# Upload file
file = st.file_uploader("Upload Test_data.csv", type="csv")

if file:
    # Read CSV
    df = pd.read_csv(file)

    # Remove label if exists
    X = df.drop("class", axis=1) if "class" in df.columns else df.copy()

    # Keep same columns as training
    X = X[scaler.feature_names_in_]

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace NaN
    X = X.fillna(0)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)

    # Add result column
    df["Label"] = ["🚨 Anomaly" if p == 1 else "✅ Normal" for p in preds]

    # Display results
    st.write("### Résultat")
    st.metric("Total", len(df))
    st.metric("Anomalies", int(sum(preds)))

    fig = px.pie(df, names="Label", hole=0.4)
    st.plotly_chart(fig)

    st.dataframe(df[["Label"]].head(20))
