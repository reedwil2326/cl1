import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Retail_Sales_Data.csv")

print(df.head())
df.info()

by_region = df.groupby("Region")["SalesAmount"].sum().sort_values(ascending=False)
print(by_region.round(2))
print("Top region:", by_region.idxmax())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
by_region.plot.bar(ax=ax[0], title="Sales by Region")
ax[0].set_ylabel("Total Sales")
by_region.plot.pie(ax=ax[1], autopct="%1.0f%%", title="Share of Sales")
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()

combo = df.groupby(["Region", "ProductCategory"])["SalesAmount"].sum().unstack()
print(combo.round(2))

combo.plot.bar(title="Region vs Category (Grouped)")
plt.ylabel("Total Sales")
plt.show()

combo.plot.bar(stacked=True, title="Region vs Category (Stacked)")
plt.ylabel("Total Sales")
plt.show()
