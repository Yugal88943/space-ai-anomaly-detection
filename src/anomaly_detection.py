from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def detect_anomalies(df, sensors):
    normal_data = df[df['RUL'] > 80][sensors]

    iso_model = IsolationForest(contamination=0.06, random_state=42)
    iso_model.fit(normal_data)

    df['iforest_score'] = iso_model.decision_function(df[sensors])

    scaler = MinMaxScaler()
    df['iforest_score'] = scaler.fit_transform(df[['iforest_score']])

    df['sensor_score'] = df[sensors].std(axis=1)
    df['sensor_score'] = scaler.fit_transform(df[['sensor_score']])

    df['anomaly_score'] = (
        0.6 * (1 - df['iforest_score']) +
        0.2 * df['sensor_score'] +
        0.2 * (1 / (df['RUL'] + 1))
    )

    threshold = df['anomaly_score'].quantile(0.75)
    df['anomaly'] = (df['anomaly_score'] > threshold).astype(int)

    return df

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_anomaly(df):
    # Ground truth (based on RUL threshold)
    df['true_label'] = (df['RUL'] < 40).astype(int)

    y_true = df['true_label']
    y_pred = df['anomaly']

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    acc = accuracy_score(y_true, y_pred)

    return cm, report, acc