import pandas as pd
from sklearn.preprocessing import MinMaxScaler

important_sensors = [
    'sensor_2','sensor_3','sensor_4','sensor_7',
    'sensor_11','sensor_12','sensor_15',
    'sensor_17','sensor_20','sensor_21'
]

def load_data(train_file, test_file, rul_file):
    df_train = pd.read_csv(train_file, sep=r"\s+", header=None).iloc[:, :26]
    df_test = pd.read_csv(test_file, sep=r"\s+", header=None).iloc[:, :26]
    y_true = pd.read_csv(rul_file, header=None)[0]

    columns = ['engine_id', 'cycle'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    df_train.columns = columns
    df_test.columns = columns

    return df_train, df_test, y_true


def preprocess(df_train, df_test):
    # Create RUL
    max_cycle = df_train.groupby('engine_id')['cycle'].max().reset_index()
    max_cycle.columns = ['engine_id', 'max_cycle']
    df_train = df_train.merge(max_cycle, on='engine_id')
    df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
    df_train['RUL'] = df_train['RUL'].clip(upper=125)

    # Drop op settings
    df_train = df_train.drop(columns=[c for c in df_train.columns if 'op_setting' in c])
    df_test = df_test.drop(columns=[c for c in df_test.columns if 'op_setting' in c])

    # Scaling
    scaler = MinMaxScaler()
    df_train[important_sensors] = scaler.fit_transform(df_train[important_sensors])
    df_test[important_sensors] = scaler.transform(df_test[important_sensors])

    return df_train, df_test


def create_features(df):
    for col in important_sensors:
        df[f'{col}_ma'] = df.groupby('engine_id')[col].transform(lambda x: x.rolling(5).mean())
        df[f'{col}_std'] = df.groupby('engine_id')[col].transform(lambda x: x.rolling(5).std())
        df[f'{col}_trend'] = df.groupby('engine_id')[col].transform(lambda x: x.diff())
        df[f'{col}_lag1'] = df.groupby('engine_id')[col].shift(1)
        df[f'{col}_lag2'] = df.groupby('engine_id')[col].shift(2)

    df['cycle_norm'] = df['cycle'] / df.groupby('engine_id')['cycle'].transform('max')

    return df.bfill()