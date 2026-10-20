import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

print("Missing values before cleaning:")
print(df.isnull().sum())



df["Age"] = df["Age"].fillna(df["Age"].median())



if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])



if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())



print("\nMissing values after cleaning:")
print(df.isnull().sum())