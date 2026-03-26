def compute_risk(df):
    df['risk_score'] = (
        0.5 * df['anomaly_score'] +
        0.3 * (1 / (df['RUL'] + 1)) +
        0.2 * df['sensor_score']
    )
    return df


def decision(row):
    if row['risk_score'] > 0.8:
        return "CRITICAL"
    elif row['risk_score'] > 0.6:
        return "WARNING"
    else:
        return "NORMAL"