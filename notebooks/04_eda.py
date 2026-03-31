# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# We draw charts to understand our data better
# Think of this as being a detective looking for patterns!
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# SETTINGS
# Make our charts look nice and professional
# ============================================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create a folder to save our charts
os.makedirs('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts', exist_ok=True)

# ============================================================
# LOAD THE FEATURE-ENGINEERED DATASET
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_features.csv')
print("Data loaded! Shape:", df.shape)

# ============================================================
# CHART 1: DISTRIBUTION OF DELIVERY TIME
# How are delivery times spread out?
# Most deliveries take how long?
# ============================================================
plt.figure()
sns.histplot(df['Delivery_Time'], bins=30, color='steelblue', kde=True)
plt.title('Distribution of Delivery Times', fontsize=16)
plt.xlabel('Delivery Time (hours)')
plt.ylabel('Number of Orders')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\01_delivery_time_distribution.png')
plt.close()
print("✅ Chart 1 saved: Delivery Time Distribution")

# ============================================================
# CHART 2: DELIVERY TIME BY WEATHER
# Does rain/fog make deliveries slower?
# ============================================================
plt.figure()
sns.boxplot(data=df, x='Weather', y='Delivery_Time', palette='Set2')
plt.title('Delivery Time by Weather Condition', fontsize=16)
plt.xlabel('Weather')
plt.ylabel('Delivery Time (hours)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\02_delivery_time_by_weather.png')
plt.close()
print("✅ Chart 2 saved: Delivery Time by Weather")

# ============================================================
# CHART 3: DELIVERY TIME BY TRAFFIC
# Does heavy traffic slow deliveries down?
# ============================================================
plt.figure()
sns.boxplot(data=df, x='Traffic', y='Delivery_Time', palette='Set3')
plt.title('Delivery Time by Traffic Condition', fontsize=16)
plt.xlabel('Traffic')
plt.ylabel('Delivery Time (hours)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\03_delivery_time_by_traffic.png')
plt.close()
print("✅ Chart 3 saved: Delivery Time by Traffic")

# ============================================================
# CHART 4: DISTANCE VS DELIVERY TIME
# Does farther distance = longer delivery?
# ============================================================
plt.figure()
sns.scatterplot(data=df, x='Distance_km', y='Delivery_Time',
                alpha=0.3, color='coral')
plt.title('Distance vs Delivery Time', fontsize=16)
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (hours)')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\04_distance_vs_delivery_time.png')
plt.close()
print("✅ Chart 4 saved: Distance vs Delivery Time")

# ============================================================
# CHART 5: DELIVERY TIME BY VEHICLE TYPE
# Bikes vs Cars vs Scooters — which is fastest?
# ============================================================
plt.figure()
sns.boxplot(data=df, x='Vehicle', y='Delivery_Time', palette='pastel')
plt.title('Delivery Time by Vehicle Type', fontsize=16)
plt.xlabel('Vehicle')
plt.ylabel('Delivery Time (hours)')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\05_delivery_time_by_vehicle.png')
plt.close()
print("✅ Chart 5 saved: Delivery Time by Vehicle")

# ============================================================
# CHART 6: AGENT RATING VS DELIVERY TIME
# Do higher-rated agents deliver faster?
# ============================================================
plt.figure()
sns.scatterplot(data=df, x='Agent_Rating', y='Delivery_Time',
                alpha=0.3, color='mediumseagreen')
plt.title('Agent Rating vs Delivery Time', fontsize=16)
plt.xlabel('Agent Rating')
plt.ylabel('Delivery Time (hours)')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\06_agent_rating_vs_delivery_time.png')
plt.close()
print("✅ Chart 6 saved: Agent Rating vs Delivery Time")

# ============================================================
# CHART 7: DELIVERY TIME BY AREA
# Urban vs Metropolitan — which area is faster?
# ============================================================
plt.figure()
sns.boxplot(data=df, x='Area', y='Delivery_Time', palette='muted')
plt.title('Delivery Time by Area Type', fontsize=16)
plt.xlabel('Area')
plt.ylabel('Delivery Time (hours)')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\07_delivery_time_by_area.png')
plt.close()
print("✅ Chart 7 saved: Delivery Time by Area")

# ============================================================
# CHART 8: CORRELATION HEATMAP
# Which numbers are most related to delivery time?
# Darker color = stronger relationship
# ============================================================
plt.figure(figsize=(12, 8))
# Only use numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\08_correlation_heatmap.png')
plt.close()
print("✅ Chart 8 saved: Correlation Heatmap")


# ============================================================
# CHART 9: AVERAGE DELIVERY TIME BY CATEGORY
# Which product category takes longest to deliver?
# ============================================================
plt.figure(figsize=(12, 6))
category_avg = df.groupby('Category')['Delivery_Time'].mean().sort_values(ascending=False)
sns.barplot(x=category_avg.index, y=category_avg.values, palette='viridis')
plt.title('Average Delivery Time by Product Category', fontsize=16)
plt.xlabel('Category')
plt.ylabel('Average Delivery Time (hours)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\09_delivery_time_by_category.png')
plt.close()
print("✅ Chart 9 saved: Delivery Time by Category")

# ============================================================
# CHART 10: DELIVERY TIME BY TIME OF DAY
# Morning vs Afternoon vs Evening vs Night
# ============================================================
plt.figure()
order = ['Morning', 'Afternoon', 'Evening', 'Night']
sns.boxplot(data=df, x='Time_of_Day', y='Delivery_Time',
            order=order, palette='Set1')
plt.title('Delivery Time by Time of Day', fontsize=16)
plt.xlabel('Time of Day')
plt.ylabel('Delivery Time (hours)')
plt.tight_layout()
plt.savefig('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\notebooks\\charts\\10_delivery_time_by_time_of_day.png')
plt.close()
print("✅ Chart 10 saved: Delivery Time by Time of Day")

# ============================================================
# PRINT KEY INSIGHTS
# ============================================================
print("\n" + "="*50)
print("📊 KEY INSIGHTS FROM EDA:")
print("="*50)
print(f"Average Delivery Time: {df['Delivery_Time'].mean():.2f} hours")
print(f"Shortest Delivery: {df['Delivery_Time'].min():.2f} hours")
print(f"Longest Delivery: {df['Delivery_Time'].max():.2f} hours")
print(f"\nMost Common Weather: {df['Weather'].mode()[0]}")
print(f"Most Common Traffic: {df['Traffic'].mode()[0]}")
print(f"Most Common Vehicle: {df['Vehicle'].mode()[0]}")
print(f"\nAverage Distance: {df['Distance_km'].mean():.2f} km")
print(f"Average Agent Rating: {df['Agent_Rating'].mean():.2f}")
print("\n✅ All 10 charts saved in: notebooks/charts/")
