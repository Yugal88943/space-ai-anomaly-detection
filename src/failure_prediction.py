from xgboost import XGBRegressor

def train_model(df, feature_cols):
    X = df[feature_cols]
    y = df['RUL']

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)
    return model


def predict(model, df, feature_cols):
    return model.predict(df[feature_cols])