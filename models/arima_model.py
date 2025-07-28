from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    fitted_model = model.fit()
    return fitted_model

def predict_arima(fitted_model, steps=1):
    return fitted_model.forecast(steps=steps)
