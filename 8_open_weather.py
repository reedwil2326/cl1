import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

API_KEY = "d6d400c80987e7eef9a1fd8b30a7a10d"
CITY = "Pune"
BASE = "https://api.openweathermap.org/data/2.5"

if API_KEY:
    url = f"{BASE}/forecast?q={CITY}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    print(data)
else:
    data = json.load(open("data/weather_sample.json"))

rows = []
for item in data["list"]:
    rows.append({
        "time":     item["dt_txt"],
        "temp":     item["main"]["temp"],
        "humidity": item["main"]["humidity"],
        "wind":     item["wind"]["speed"],
        "rain":     item.get("rain", {}).get("3h", 0),  
    })
df = pd.DataFrame(rows)

df["time"] = pd.to_datetime(df["time"])
df = df.dropna().set_index("time")

print(df.describe().round(2))

daily = df.resample("D").agg({"temp": ["mean", "max", "min"],
                              "rain": "sum",
                              "wind": "mean"})
print(daily.head())

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["temp"].plot(ax=ax[0], title="Temperature (°C)")                  
daily["rain"].plot.bar(ax=ax[1], title="Daily rain (mm)", xticks=[])
ax[2].scatter(df["temp"], df["humidity"])                            
ax[2].set(title="Temp vs Humidity", xlabel="Temp", ylabel="Humidity")
plt.tight_layout()
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation")
plt.show()
