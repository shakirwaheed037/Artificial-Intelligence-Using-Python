# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score



# df = pd.read_csv("titanic.csv")

# df["Age"] = df["Age"].fillna(df["Age"].median())
# df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
# y = df["Survived"]

# X = X.copy()
# X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y)

# model = SVC()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))