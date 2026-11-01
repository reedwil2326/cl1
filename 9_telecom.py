import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Telecom_Customer_Churn.csv")

print(df.shape)
df.info()
print(df.isna().sum())

df["gender"] = df["gender"].str.strip().str.title()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
df["TotalCharges"]   = df["TotalCharges"].fillna(df["TotalCharges"].median())

df = df.drop_duplicates()

q1 = df["MonthlyCharges"].quantile(0.25)
q3 = df["MonthlyCharges"].quantile(0.75)
iqr = q3 - q1
df = df[df["MonthlyCharges"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTerm"] = (df["Contract"] != "Month-to-month").astype(int)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df = pd.get_dummies(df.drop(columns="customerID"), drop_first=True)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth"]
df[num_cols] = StandardScaler().fit_transform(df[num_cols])

X = df.drop(columns="Churn")
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Train / Test:", X_train.shape, X_test.shape)

df.to_csv("Telecom_Cleaned.csv", index=False)
print("Saved Telecom_Cleaned.csv")
