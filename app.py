import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# إعدادات الصفحة
st.set_page_config(page_title="IDS Dashboard", layout="wide")

# تحميل الموديلات
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🛡️ Intrusion Detection System Dashboard")

file = st.file_uploader("📥 Upload CSV file for Analysis", type="csv")

if file:
    df = pd.read_csv(file)
    
    # 1. تنظيف البيانات: نأخذ فقط الأعمدة الرقمية ونحولها لـ Numpy لنتفادى الـ ValueError
    X = df.drop("class", axis=1) if "class" in df.columns else df.copy()
    X = X.select_dtypes(include=[np.number]) # نضمن أننا نرسل أرقاماً فقط
    
    # 2. ضبط عدد الأعمدة ليتوافق مع ما تعلمه الـ Scaler
    n_features = scaler.n_features_in_
    X_input = X.iloc[:, :n_features].values  # تحويل لـ Numpy Array (هنا يكمن الحل)

    try:
        # 3. التحويل والتوقع
        X_scaled = scaler.transform(X_input)
        preds = model.predict(X_scaled)
        
        # 4. إضافة النتائج للعرض
        df["Result"] = ["🚨 Anomaly" if p == 1 else "✅ Normal" for p in preds]

        # --- Dashboard (الأرقام والرسوم) ---
        st.markdown("### 📊 Statistiques de Détection")
        c1, c2, c3 = st.columns(3)
        total = len(df)
        anomalies = int(sum(preds))
        
        c1.metric("Total Trafic", total)
        c2.metric("Anomalies", anomalies, delta=f"{(anomalies/total)*100:.1f}%", delta_color="inverse")
        c3.metric("Normal", total - anomalies)

        st.divider()

        # رسم الـ Pie Chart
        fig = px.pie(df, names="Result", hole=0.5, 
                     color="Result", 
                     color_discrete_map={"🚨 Anomaly": "#FF4B4B", "✅ Normal": "#00CC96"},
                     title="Répartition des Prédictions")
        st.plotly_chart(fig, use_container_width=True)

        # عرض الجدول
        st.write("### 📋 Liste des Détections (Top 10)")
        st.dataframe(df[["Result"]].head(10), use_container_width=True)
        
        st.success("Analyse terminée مع نجاح العملية ✅")

    except Exception as e:
        st.error(f"⚠️ خطأ في معالجة البيانات: {e}")
        st.info
