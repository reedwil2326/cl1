import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("City_Air_Quality.csv")
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['AQI'], color='black')
plt.title("AQI over time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(df['Date'], df['PM2.5'], color='red')
ax[1].plot(df['Date'], df['PM10'], color='blue')
ax[2].plot(df['Date'], df['CO'], color='green')
plt.suptitle("Pollutant Levels Over Time")
plt.xlabel("Date")
plt.show()

df_monthly = df.groupby(df['Date'].dt.month)['AQI'].mean()
plt.figure(figsize=(8, 5))
df_monthly.plot(kind='bar', color='orange')
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([df['PM2.5'].dropna(), df['PM10'].dropna(), df['CO'].dropna()])
plt.title("Pollutant Distributions (Box Plot)")
plt.ylabel("Concentration")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['PM2.5'], df['AQI'], alpha=0.5)
plt.title("AQI vs PM2.5 (bubble size = CO)")
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.show()

