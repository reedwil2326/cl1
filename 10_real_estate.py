import pandas as pd

df = pd.read_csv("RealEstate_Prices.csv")
df.columns = (df.columns.str.lower()
                        .str.replace(r"[^a-z0-9]+", "_", regex=True)   # spaces/symbols -> _
                        .str.strip("_"))
print(df.columns.tolist()) 

df["area_sqft"] = df["area_sqft"].fillna(df["area_sqft"].median())
df = df.dropna()
df["sale_date"] = pd.to_datetime(df["sale_date"])

info = pd.DataFrame({
    "neighborhood": ["Kothrud", "Baner", "Hinjewadi", "Wakad"],
    "metro_km":     [2, 5, 8, 7],
})
df = df.merge(info, on="neighborhood", how="left")

houses = df[(df["sale_date"] >= "2023-01-01") & (df["property_type"] == "House")]
print("Houses sold since 2023:", len(houses))

print(df.groupby("neighborhood")["sale_price"].mean().round())
print(df.groupby("property_type")["sale_price"].mean().round())

df = pd.get_dummies(df, columns=["property_type"])

q1 = df["sale_price"].quantile(0.25)
q3 = df["sale_price"].quantile(0.75)
iqr = q3 - q1
df = df[df["sale_price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
print("Rows after outlier removal:", len(df))
