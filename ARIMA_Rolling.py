import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")  # Suppress ARIMA warnings

# Fetch Dogecoin data function
def fetch_dogecoin_data(start_date, end_date):
    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int(pd.Timestamp(end_date).timestamp())

    url = "https://api.coingecko.com/api/v3/coins/dogecoin/market_chart/range"
    params = {"vs_currency": "usd", "from": start_timestamp, "to": end_timestamp}

    response = requests.get(url, params=params)
    data = response.json()

    if "prices" not in data:
        raise ValueError("API response does not contain 'prices' key")

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.asfreq("D").ffill()  # Ensure daily data

    return df

# Define date range (6 months ago)
#results for 6 months: Mean Absolute Error (MAE): 0.0189; Mean Absolute Percentage Error (MAPE): 6.66%
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(months=6)

df = fetch_dogecoin_data(start_date, end_date)

# Rolling ARIMA Model with 30-Day Window & AIC Optimization
train_window = 30  # Number of days used for training
predictions = []
actuals = []
dates = []

# Grid of (p, d, q) values to test
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)

# Function to find best ARIMA order using AIC
def find_best_arima(train_data):
    best_aic = float("inf")
    best_order = None
    best_model = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = sm.tsa.ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except:
                    continue

    return best_model, best_order

# Iterate over the test period, rolling 1 day at a time
for i in range(train_window, len(df) - 1):
    train_data = df.iloc[i - train_window:i]  # Last 30 days
    test_data = df.iloc[i + 1]  # Next day's actual price

    try:
        # Find best ARIMA model based on AIC
        best_model, best_order = find_best_arima(train_data)

        if best_model is not None:
            # Predict next day's price
            pred = best_model.forecast(steps=1)[0]
            predictions.append(pred)
        else:
            predictions.append(np.nan)

        actuals.append(test_data["price"])
        dates.append(df.index[i + 1])  # Store dates for plotting

    except Exception as e:
        print(f"Error on day {df.index[i]}: {e}")
        predictions.append(np.nan)
        actuals.append(test_data["price"])
        dates.append(df.index[i + 1])

# Convert to NumPy arrays for accuracy measurement
predictions = np.array(predictions)
actuals = np.array(actuals)

# Compute accuracy metrics
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(dates, actuals, label="Actual Price", color="blue")
plt.plot(dates, predictions, label="Predicted Price", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Dogecoin Price (USD)")
plt.title("Actual vs. Predicted Dogecoin Price using Rolling ARIMA")
plt.legend()
plt.grid(True)
plt.show()
