import pandas as pd
import matplotlib.pyplot as plt

csv = pd.read_csv("sales.csv")
xls = pd.read_excel("sales.xlsx")      
js  = pd.read_json("sales.json")

for name, d in [("CSV", csv), ("Excel", xls), ("JSON", js)]:
    print(name, d.shape,
          "| missing:", d.isna().sum().sum(),
          "| duplicates:", d.duplicated().sum())
print(csv.head())                            
print(csv.dtypes)

df = pd.concat([csv, xls, js], ignore_index=True).drop_duplicates()
df["UnitPrice"] = df["UnitPrice"].fillna(df["UnitPrice"].median())   
df["Category"]  = df["Category"].str.strip().str.title()            
df["Date"]      = pd.to_datetime(df["Date"])  

df["Sales"] = df["Quantity"] * df["UnitPrice"]
df["Month"] = df["Date"].dt.to_period("M").astype(str)

print(df[["Quantity", "UnitPrice", "Sales"]].describe())   
print("Total sales     :", round(df["Sales"].sum(), 2))
print("Avg order value :", round(df["Sales"].mean(), 2))
print(df.groupby("Category")["Sales"].sum().round(2)) 

cat_sales = df.groupby("Category")["Sales"].sum()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cat_sales.plot.bar(ax=ax[0], title="Sales by Category")
cat_sales.plot.pie(ax=ax[1], autopct="%1.0f%%", title="Share of Sales")
df.boxplot("Sales", by="Category", ax=ax[2])
plt.tight_layout()
plt.show()
