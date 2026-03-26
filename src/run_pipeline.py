from src.data_preprocessing import load_data, preprocess, create_features
from src.failure_prediction import train_model, predict
from src.anomaly_detection import detect_anomalies
from src.decision_engine import compute_risk

# Load
df_train, df_test, y_true = load_data(
    "data/raw/train_FD001.txt",
    "data/raw/test_FD001.txt",
    "data/raw/RUL_FD001.txt"
)

# Preprocess
df_train, df_test = preprocess(df_train, df_test)

# Features
df_train = create_features(df_train)
df_test = create_features(df_test)

feature_cols = [col for col in df_train.columns if 'sensor_' in col]

# Train model
model = train_model(df_train, feature_cols)

# Predict
df_test['RUL_pred'] = predict(model, df_test, feature_cols)

# Anomaly detection
df_train = detect_anomalies(df_train, feature_cols)

# Risk
df_train = compute_risk(df_train)

print("Pipeline completed successfully ")