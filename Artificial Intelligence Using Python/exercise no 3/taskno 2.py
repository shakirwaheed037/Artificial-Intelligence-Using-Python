import pandas as pd
import re


# Load the dataset
df = pd.read_csv("toxic_comments_dataset.csv")

print("Dataset loaded successfully!")


# Check column names

print("\nColumn Names:")
print(df.columns.tolist())


# 1. Check missing values

print("\nMissing Values in Each Column:")
print(df.isnull().sum())


# 2. Check duplicate records

print("\nNumber of Duplicate Records:")
print(df.duplicated().sum())


# Handle missing values
# Remove rows where comment is missing
df = df.dropna(subset=["comment"])

# Fill missing target labels if any
# with the most frequent label
if df["label"].isnull().sum() > 0:
    df["label"] = df["label"].fillna(df["label"].mode()[0])


# Remove duplicate comments

df = df.drop_duplicates(subset=["comment"])

print("\nDataset after handling missing values and duplicates:")
print(df.shape)


# 3. Text Cleaning Function


def clean_text(text):

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning function
df["clean_comment"] = df["comment"].apply(clean_text)


# Display original and cleaned comments

print("\nOriginal and Cleaned Comments:")
print(df[["comment", "clean_comment"]].head(10).to_string(index=False))


# Final missing value check

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# Final duplicate check

print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())


# Target / Label Distribution

print("\nToxicity Label Distribution:")
print(df["label"].value_counts())

print("\nTask 2 completed successfully!")