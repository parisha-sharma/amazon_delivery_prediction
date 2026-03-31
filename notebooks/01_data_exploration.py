# We are loading our dataset and looking at what's insid
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery.csv')

print("=" * 50)
print("DATASET SHAPE (rows, columns):")
print(df.shape)

print("\n" + "=" * 50)
print("FIRST 5 ROWS OF DATA:")
print(df.head())

# Printing all column names
print("\n" + "=" * 50)
print("COLUMN NAMES:")
print(df.columns.tolist())

# data types of each column
print("\n" + "=" * 50)
print("DATA TYPES OF EACH COLUMN:")
print(df.dtypes)

# count of missing values in each column
print("\n" + "=" * 50)
print("MISSING VALUES IN EACH COLUMN:")
print(df.isnull().sum())

# basic statistics (min, max, average, etc.)
print("\n" + "=" * 50)
print("BASIC STATISTICS:")
print(df.describe())
