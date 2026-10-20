import pandas as pd

# Load the dataset
df = pd.read_csv("toxic_comments_dataset.csv")
print("Dataset loaded successfully!")
# 1. First 10 records
print("\nFirst 10 Records:")
print(df.head(10))

# Last 8 records
print("\nLast 8 Records:")
print(df.tail(8))

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# 3. Statistical summary
print("\nStatistical Summary of Numerical Columns:")
print(df.describe())