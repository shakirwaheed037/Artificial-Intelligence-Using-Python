import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# 1 Load of the Titanic dataset

df = pd.read_csv("titanic.csv")

print("========== DATASET INFORMATION ==========")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])



# 2 Descriptive Statistics

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(df.describe())


# 3 Identify Numerical Variables

print("\n========== NUMERICAL VARIABLES ==========")

numerical_columns = df.select_dtypes(include=np.number).columns

print("Numerical Variables:")
for column in numerical_columns:
    print("-", column)



# 4 Identify Categorical Variables

print("\n========== CATEGORICAL VARIABLES ==========")

categorical_columns = df.select_dtypes(include="object").columns

print("Categorical Variables:")
for column in categorical_columns:
    print("-", column)


# 5 Analyze Sex Variable

print("\n========== SEX ANALYSIS ==========")

print("Unique values in Sex:")
print(df["Sex"].unique())

print("\nFrequency distribution of Sex:")
print(df["Sex"].value_counts())



# 6 Analyze Passenger Class

print("\n========== PASSENGER CLASS ANALYSIS ==========")

print("Unique values in Pclass:")
print(df["Pclass"].unique())

print("\nFrequency distribution of Pclass:")
print(df["Pclass"].value_counts().sort_index())


# 7 Summary of Data Types

print("\n========== DATA TYPES ==========")
print(df.dtypes)