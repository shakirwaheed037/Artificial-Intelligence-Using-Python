import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex")
plt.title("Survival Rate by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x="Fare", bins=30, kde=True)
plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival and Non-Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()


gender_survival = df.groupby("Sex")["Survived"].mean()

print("Survival Rate by Gender:")
print(gender_survival)