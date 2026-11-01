import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("Iris.csv")

X, y = df.drop(columns="species"), df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

pred = lda.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

new = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("Predicted species:", lda.predict(new)[0])
