# ============================================================
# STEP 3: FEATURE ENGINEERING
# We create NEW useful columns from existing data
# This helps our ML model make better predictions
# ============================================================

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ============================================================
# LOAD THE CLEANED DATASET
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_cleaned.csv')

print("Loaded cleaned data. Shape:", df.shape)

# ============================================================
# NEW FEATURE 1: DISTANCE BETWEEN STORE AND CUSTOMER
# We use the "Haversine Formula" to calculate real-world
# distance between two GPS points on Earth
# Think of it like Google Maps measuring distance! 🗺️
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    This function takes 4 GPS numbers and returns
    the distance in kilometers between two points
    """
    # Convert degrees to radians (math requirement)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # Earth's radius in kilometers
    radius = 6371
    distance = radius * c
    return distance

# Apply the function to every row in our dataset
# This creates a new column called 'Distance_km'
df['Distance_km'] = df.apply(
    lambda row: calculate_distance(
        row['Store_Latitude'],
        row['Store_Longitude'],
        row['Drop_Latitude'],
        row['Drop_Longitude']
    ),
    axis=1  # axis=1 means apply to each ROW
)

print("\n✅ Created 'Distance_km' column")
print("Sample distances (km):", df['Distance_km'].head().values)

# ============================================================
# NEW FEATURE 2: PICKUP WAIT TIME
# How many minutes between Order Time and Pickup Time?
# Longer wait = possibly longer delivery
# ============================================================
df['Order_Time_dt'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')
df['Pickup_Time_dt'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')

# Calculate difference in minutes
df['Pickup_Wait_Minutes'] = (
    df['Pickup_Time_dt'] - df['Order_Time_dt']
).dt.total_seconds() / 60

# Some pickups might show negative (midnight crossover) — fix those
df['Pickup_Wait_Minutes'] = df['Pickup_Wait_Minutes'].abs()

print("\n✅ Created 'Pickup_Wait_Minutes' column")
print("Sample wait times (mins):", df['Pickup_Wait_Minutes'].head().values)

# ============================================================
# NEW FEATURE 3: IS IT A WEEKEND?
# Weekend deliveries might take longer due to more orders
# 1 = Weekend (Saturday/Sunday), 0 = Weekday
# ============================================================
df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
print("\n✅ Created 'Is_Weekend' column (1=Weekend, 0=Weekday)")

# ============================================================
# NEW FEATURE 4: TIME OF DAY CATEGORY
# Morning, Afternoon, Evening, Night
# Rush hours affect delivery times!
# ============================================================
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['Time_of_Day'] = df['Order_Hour'].apply(get_time_of_day)
print("✅ Created 'Time_of_Day' column (Morning/Afternoon/Evening/Night)")

# ============================================================
# DROP COLUMNS WE NO LONGER NEED
# These were only used to create new features
# Keeping them would confuse our ML model
# ============================================================
df = df.drop(columns=[
    'Order_Time_dt',
    'Pickup_Time_dt',
    'Order_Date',       # We extracted Day, Month, Day_of_Week from this
    'Order_Time',       # We extracted Order_Hour from this
    'Pickup_Time',      # We extracted Pickup_Hour from this
    'Order_ID',         # Just an ID number, not useful for prediction
    'Store_Latitude',   # We combined these into Distance_km
    'Store_Longitude',
    'Drop_Latitude',
    'Drop_Longitude'
])

print("\n✅ Dropped unnecessary columns")

# ============================================================
# SHOW FINAL COLUMNS
# ============================================================
print("\n" + "="*50)
print("FINAL COLUMNS IN OUR DATASET:")
for col in df.columns:
    print(f"  → {col}")

print("\nFinal shape:", df.shape)

# ============================================================
# SAVE THE FEATURE-ENGINEERED DATA
# ============================================================
df.to_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_features.csv', index=False)
print("\n✅ Saved to: data/amazon_delivery_features.csv")
