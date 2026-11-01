import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("data/Stock_Prices.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index().dropna(subset=["Close"])

print(df.head())
df.info()

df["Close"].plot(title="Closing Price Over Time", figsize=(10, 4))
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

df["MA30"] = df["Close"].rolling(30).mean()
df["MA90"] = df["Close"].rolling(90).mean()
df[["Close", "MA30", "MA90"]].plot(title="Moving Averages", figsize=(10, 4))
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

result = seasonal_decompose(df["Close"], model="additive", period=21)
result.plot()
plt.show()

print("Correlation Close vs Volume:", round(df["Close"].corr(df["Volume"]), 3))

train = df["Close"][:-30]
test = df["Close"][-30:]
model = ARIMA(train, order=(5, 1, 0)).fit()
forecast = model.forecast(steps=30)

rmse = np.sqrt(((test.values - forecast.values) ** 2).mean())
print("Forecast RMSE:", round(rmse, 2))

plt.figure(figsize=(10, 4))
plt.plot(train[-100:], label="Train")
plt.plot(test, label="Actual")
plt.plot(test.index, forecast.values, label="Forecast")
plt.title("ARIMA(5,1,0) Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
