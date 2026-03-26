import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess, create_features
from src.failure_prediction import train_model, predict
from src.decision_engine import compute_risk
from src.anomaly_detection import detect_anomalies, evaluate_anomaly


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("Space AI Monitoring Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_pipeline():
    df_train, df_test, _ = load_data(
        "data/raw/train_FD001.txt",
        "data/raw/test_FD001.txt",
        "data/raw/RUL_FD001.txt"
    )

    df_train, df_test = preprocess(df_train, df_test)

    df_train = create_features(df_train)
    df_test = create_features(df_test)

    feature_cols = [col for col in df_train.columns if 'sensor_' in col]

    model = train_model(df_train, feature_cols)

    df_train['RUL_pred'] = predict(model, df_train, feature_cols)

    df_train = detect_anomalies(df_train, feature_cols)
    df_train = compute_risk(df_train)
    cm, report, acc = evaluate_anomaly(df_train)

    return df_train, model, feature_cols, cm, report, acc


df, model, feature_cols, cm, report, acc = load_pipeline()

# =========================
# SIDEBAR
# =========================
engine_id = st.sidebar.selectbox("Select Engine", df['engine_id'].unique())

engine_data = df[df['engine_id'] == engine_id].reset_index(drop=True)

cycle = st.sidebar.slider("Cycle", 0, len(engine_data)-1, 0)

current = engine_data.iloc[cycle]

# =========================
# METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Actual RUL", int(current['RUL']))
col2.metric("Predicted RUL", int(current['RUL_pred']))

status = "ANOMALY" if current['anomaly'] == 1 else "NORMAL"
col3.metric("Status", status)

# =========================
# ALERT
# =========================
if current['risk_score'] > 0.8:
    st.error("CRITICAL CONDITION")
elif current['risk_score'] > 0.6:
    st.warning("WARNING: CHECK SYSTEM")
else:
    st.success("SYSTEM NORMAL")

# =========================
# RUL PLOT
# =========================
fig, ax = plt.subplots()

ax.plot(engine_data['cycle'], engine_data['RUL'], label='Actual RUL')
ax.plot(engine_data['cycle'], engine_data['RUL_pred'].rolling(5).mean(), label='Predicted RUL')

ax.set_xlabel("Cycle")
ax.set_ylabel("Remaining Useful Life (RUL)")
ax.set_title("RUL Prediction vs Actual")

ax.legend()

st.pyplot(fig)

# =========================
# SENSOR PLOT
# =========================
st.subheader("Sensor Analysis")

sensor_cols = [c for c in df.columns if 'sensor_' in c]
sensor = st.selectbox("Select Sensor", sensor_cols)

fig, ax = plt.subplots()

ax.plot(engine_data['cycle'], engine_data[sensor])

ax.set_xlabel("Cycle")
ax.set_ylabel(sensor)
ax.set_title(f"{sensor} vs Cycle")

st.pyplot(fig)

# =========================
# RISK SCORE
# =========================
st.subheader("Risk Score")
fig, ax = plt.subplots()

ax.plot(engine_data['cycle'], engine_data['risk_score'])

ax.set_xlabel("Cycle")
ax.set_ylabel("Risk Score")
ax.set_title("Risk Score Over Time")

st.pyplot(fig)

# =========================
# ANOMALY TABLE
# =========================
st.subheader("Anomaly Log")

anomalies = engine_data[engine_data['anomaly'] == 1]

st.dataframe(anomalies[['cycle', 'RUL_pred', 'risk_score']].tail(10))

import seaborn as sns

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.subheader("Model Performance Metrics")

precision = report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{acc:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Top Features")

importance = pd.Series(model.feature_importances_, index=feature_cols)
top_features = importance.sort_values(ascending=False).head(10)

st.bar_chart(top_features)