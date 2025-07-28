import pandas as pd
from neuralforecast.models import TFT
from neuralforecast import NeuralForecast

def prepare_tft_dataframe(df):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['抽せん日'])
    df['unique_id'] = 'loto'
    df['y'] = df['本数字'].apply(lambda x: sum(x))
    return df[['unique_id', 'ds', 'y']].sort_values('ds')

def train_tft_model(df):
    df_tft = prepare_tft_dataframe(df)
    model = NeuralForecast(models=[TFT(input_size=20, h=1)], freq='W')
    model.fit(df_tft)
    return model

def predict_tft(model):
    return model.predict().tail(1)['TFT'].values[0]