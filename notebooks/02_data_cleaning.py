# ============================================================
# STEP 2: DATA CLEANING
# We fix messy, missing, or incorrect data here
# Think of this like erasing mistakes on a worksheet
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# LOAD THE DATASET AGAIN
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery.csv')

print("Shape BEFORE cleaning:", df.shape)

# ============================================================
# FIX 1: REMOVE DUPLICATE ROWS
# Sometimes the same order appears twice — we remove extras
# ============================================================
df = df.drop_duplicates()
print("Shape AFTER removing duplicates:", df.shape)

# ============================================================
# FIX 2: FILL MISSING VALUES
# Agent_Rating: fill missing with the AVERAGE rating
# Why average? Because it's the most "neutral" guess
# ============================================================
average_rating = df['Agent_Rating'].mean()
df['Agent_Rating'] = df['Agent_Rating'].fillna(average_rating)
print(f"\nFilled missing Agent_Rating with average: {average_rating:.2f}")

# ============================================================
# Weather: fill missing with the MOST COMMON weather
# Why most common? Because it's the most likely value
# ============================================================
most_common_weather = df['Weather'].mode()[0]
df['Weather'] = df['Weather'].fillna(most_common_weather)
print(f"Filled missing Weather with most common: {most_common_weather}")

# ============================================================
# FIX 3: CLEAN TEXT COLUMNS
# Remove extra spaces from text columns
# Example: " Urban " becomes "Urban"
# ============================================================
text_columns = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
for col in text_columns:
    df[col] = df[col].str.strip()  # Remove leading/trailing spaces

print("\nCleaned text columns (removed extra spaces)")

# ============================================================
# FIX 4: CONVERT DATE AND TIME COLUMNS
# Right now Order_Date is just text — we convert it to a real date
# This lets us extract useful info like day of week, month, etc.
# ============================================================
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True)
print("\nConverted Order_Date to datetime format")

# ============================================================
# FIX 5: EXTRACT USEFUL TIME FEATURES
# From the date, we pull out extra useful information
# ============================================================

# What day of the week was the order? (0=Monday, 6=Sunday)
df['Day_of_Week'] = df['Order_Date'].dt.dayofweek

# What month was the order?
df['Month'] = df['Order_Date'].dt.month

# What day of the month?
df['Day'] = df['Order_Date'].dt.day

print("Extracted Day_of_Week, Month, Day from Order_Date")

# ============================================================
# FIX 6: EXTRACT HOUR FROM ORDER TIME AND PICKUP TIME
# Convert "10:30 AM" into just the hour number (10)
# ============================================================
df['Order_Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.hour
df['Pickup_Hour'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce').dt.hour

print("Extracted Order_Hour and Pickup_Hour")

# ============================================================
# VERIFY: CHECK MISSING VALUES AGAIN
# Should show all zeros now!
# ============================================================
print("\n" + "="*50)
print("MISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# ============================================================
# SAVE THE CLEANED DATA
# We save it as a new file so we don't touch the original
# ============================================================
df.to_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_cleaned.csv', index=False)
print("\n" + "="*50)
print("✅ Cleaned data saved to: data/amazon_delivery_cleaned.csv")
print("Final shape:", df.shape)