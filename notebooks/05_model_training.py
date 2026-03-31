# ============================================================
# STEP 5: MODEL TRAINING
# We teach 3 different ML models to predict delivery time
# Then we compare which one is best!
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD THE FEATURE-ENGINEERED DATASET
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_features.csv')
print("Data loaded! Shape:", df.shape)

# ============================================================
# FIX: HANDLE ANY REMAINING MISSING VALUES
# Some columns may still have NaN values hiding inside
# We fill them so our models can work properly
# ============================================================

# Fill missing numeric columns with their median value
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill missing text columns with their most common value
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Replace any 'nan' strings (text "nan") with most common value
for col in text_cols:
    df[col] = df[col].replace('nan', df[col].mode()[0])

print("✅ Fixed all remaining missing/NaN values")
print("Missing values remaining:", df.isnull().sum().sum())

# ============================================================
# STEP A: ENCODE CATEGORICAL COLUMNS
# ML models only understand NUMBERS, not text
# So we convert text like "Fog" → 0, "Rain" → 1, etc.
# This is called "Label Encoding"
# ============================================================
categorical_columns = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Time_of_Day']

# We save the encoders so our app can use them later
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # Save encoder for this column
    print(f"✅ Encoded '{col}': {list(le.classes_)}")

# ============================================================
# STEP B: SEPARATE FEATURES (X) AND TARGET (y)
# X = all the columns we use to PREDICT
# y = the column we want to PREDICT (Delivery_Time)
# ============================================================
X = df.drop(columns=['Delivery_Time'])  # Everything except Delivery_Time
y = df['Delivery_Time']                 # Only Delivery_Time

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns used:\n{list(X.columns)}")

# ============================================================
# STEP C: SPLIT DATA INTO TRAINING AND TESTING SETS
# We give 80% of data to the model to LEARN from
# We keep 20% hidden to TEST how well it learned
# Like giving a student 80% of past exam questions to study,
# then testing them on the remaining 20%!
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # Makes results reproducible
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# ============================================================
# HELPER FUNCTION: EVALUATE A MODEL
# This function takes a trained model and prints its scores
# ============================================================
def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\n{'='*40}")
    print(f"📊 {name} Results:")
    print(f"  RMSE  (lower=better): {rmse:.2f} hours")
    print(f"  MAE   (lower=better): {mae:.2f} hours")
    print(f"  R²    (higher=better): {r2:.4f}")
    return rmse, mae, r2

# ============================================================
# MODEL 1: LINEAR REGRESSION
# The simplest model — draws a straight line through data
# ============================================================
print("\n" + "="*40)
print("🔵 Training Model 1: Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # TRAIN the model
lr_rmse, lr_mae, lr_r2 = evaluate_model(
    "Linear Regression", lr_model, X_test, y_test)

# ============================================================
# MODEL 2: RANDOM FOREST
# Uses many decision trees and averages their predictions
# Usually much better than Linear Regression!
# ============================================================
print("\n" + "="*40)
print("🟢 Training Model 2: Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,   # Use 100 trees
    random_state=42
)
rf_model.fit(X_train, y_train)  # TRAIN the model
rf_rmse, rf_mae, rf_r2 = evaluate_model(
    "Random Forest", rf_model, X_test, y_test)

# ============================================================
# MODEL 3: XGBOOST
# A powerful model that learns from its mistakes step by step
# Often the best performer!
# ============================================================
print("\n" + "="*40)
print("🟡 Training Model 3: XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=100,   # 100 rounds of learning
    learning_rate=0.1,  # How fast it learns
    random_state=42
)
xgb_model.fit(X_train, y_train)  # TRAIN the model
xgb_rmse, xgb_mae, xgb_r2 = evaluate_model(
    "XGBoost", xgb_model, X_test, y_test)

# ============================================================
# COMPARE ALL MODELS
# ============================================================
print("\n" + "="*50)
print("🏆 MODEL COMPARISON SUMMARY:")
print("="*50)
results = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE': [lr_rmse, rf_rmse, xgb_rmse],
    'MAE': [lr_mae, rf_mae, xgb_mae],
    'R2': [lr_r2, rf_r2, xgb_r2]
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best model based on lowest RMSE
best_idx = results_df['RMSE'].idxmin()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n🥇 BEST MODEL: {best_model_name}")

# ============================================================
# SAVE ALL MODELS AND ENCODERS
# We save them so our Streamlit app can use them later
# pickle is like putting the model in a zip file
# ============================================================
os.makedirs('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models', exist_ok=True)

with open('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models\\linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models\\random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models\\xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models\\encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\models\\feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\n✅ All models saved in: models/")
print("✅ Encoders saved in: models/encoders.pkl")
print("✅ Feature columns saved in: models/feature_columns.pkl")
