import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

lookback = 100      # number of days to fit ARIMA on
position_size = 10  # number of shares to trade
n_inst = 50

# Suppress ARIMA warnings (they'll happen a lot)
warnings.filterwarnings("ignore")

def forecast_arima(return_series):
    """
    Fits ARIMA and forecasts the next value.
    """
    try:
        model = ARIMA(return_series, order=(1, 0, 0))  # AR(1)
        model_fit = model.fit()
        forecast = model_fit.forecast()[0]
        return forecast
    except:
        return 0.0  # fallback if model fails

def getMyPosition(prices: np.ndarray) -> np.ndarray:
    n_inst, n_days = prices.shape
    if n_days < lookback + 2:
        return np.zeros(n_inst, dtype=int)

    log_returns = np.diff(np.log(prices), axis=1)  # shape: (n_inst, n_days-1)
    positions = np.zeros(n_inst, dtype=int)

    for inst in range(n_inst):
        returns = log_returns[inst, -lookback:]  # most recent 100 days
        predicted_return = forecast_arima(returns)

        if predicted_return > 0:
            positions[inst] = position_size
        elif predicted_return < 0:
            positions[inst] = -position_size
        # else: stay flat

    return positions
