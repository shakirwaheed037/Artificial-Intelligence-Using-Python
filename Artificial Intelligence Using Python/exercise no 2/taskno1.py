import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("titanic.csv")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 7 rows:")

print(df.head(7))
