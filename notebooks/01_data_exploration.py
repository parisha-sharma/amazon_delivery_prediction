# ============================================================
# STEP 1: DATA EXPLORATION
# We are loading our dataset and looking at what's inside
# ============================================================

# Import pandas - our tool for working with data tables
import pandas as pd

# Import numpy - our tool for math calculations
import numpy as np

# ============================================================
# LOAD THE DATASET
# pd.read_csv() reads our CSV file and stores it as a "DataFrame"
# A DataFrame is like an Excel spreadsheet in Python
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery.csv')

# ============================================================
# BASIC EXPLORATION
# Let's look at what our data looks like
# ============================================================

# Print how many rows and columns we have
print("=" * 50)
print("DATASET SHAPE (rows, columns):")
print(df.shape)  # Example output: (45000, 13)

# Print the first 5 rows of data
print("\n" + "=" * 50)
print("FIRST 5 ROWS OF DATA:")
print(df.head())

# Print all column names
print("\n" + "=" * 50)
print("COLUMN NAMES:")
print(df.columns.tolist())

# Print data types of each column
print("\n" + "=" * 50)
print("DATA TYPES OF EACH COLUMN:")
print(df.dtypes)

# Print count of missing values in each column
print("\n" + "=" * 50)
print("MISSING VALUES IN EACH COLUMN:")
print(df.isnull().sum())

# Print basic statistics (min, max, average, etc.)
print("\n" + "=" * 50)
print("BASIC STATISTICS:")
print(df.describe())
