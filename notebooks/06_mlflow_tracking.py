# ============================================================
# STEP 6: MLFLOW TRACKING
# MLflow is like a lab notebook that records all our
# experiments automatically — models, settings, and scores!
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND PREPARE DATA (same as before)
# ============================================================
df = pd.read_csv('C:\\Users\\paris\\OneDrive\\Desktop\\GUVI\\amazon_delivery_prediction\\data\\amazon_delivery_features.csv')

# Fix missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
text_cols = df.select_dtypes(include=['object']).columns
for col in text_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].replace('nan', df[col].mode()[0])

# Encode categorical columns
categorical_columns = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category', 'Time_of_Day']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split features and target
X = df.drop(columns=['Delivery_Time'])
y = df['Delivery_Time']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data prepared and split successfully!")
print(f"Training: {X_train.shape[0]} rows | Testing: {X_test.shape[0]} rows")

# ============================================================
# SET UP MLFLOW
# We tell MLflow to save everything in a local folder
# called "mlruns" — it creates this automatically
# ============================================================
mlflow.set_tracking_uri("../mlruns")          # Where to save logs
mlflow.set_experiment("Amazon_Delivery_Prediction")  # Experiment name
print("\n✅ MLflow experiment set up!")

# ============================================================
# HELPER FUNCTION: TRAIN + LOG ONE MODEL
# This function trains a model AND records everything in MLflow
# ============================================================
def train_and_log(model_name, model, params):
    """
    Train a model and log everything to MLflow
    model_name: name to display in MLflow
    model: the ML model object
    params: dictionary of settings we used
    """
    print(f"\n{'='*45}")
    print(f"🔄 Training + Logging: {model_name}")

    # mlflow.start_run() opens a new "page" in our lab notebook
    with mlflow.start_run(run_name=model_name):

        # TRAIN the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # CALCULATE scores
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # LOG parameters (settings we used)
        # Think of this like writing down your recipe ingredients
        mlflow.log_params(params)

        # LOG metrics (scores we got)
        # Think of this like writing down your exam scores
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # LOG the model itself
        # This saves the actual trained model inside MLflow
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        print(f"  ✅ RMSE : {rmse:.2f} hours")
        print(f"  ✅ MAE  : {mae:.2f} hours")
        print(f"  ✅ R²   : {r2:.4f}")
        print(f"  ✅ Logged to MLflow!")

    return rmse, mae, r2

# ============================================================
# LOG MODEL 1: LINEAR REGRESSION
# ============================================================
lr_params = {
    "model_type": "Linear Regression",
    "fit_intercept": True
}
lr_rmse, lr_mae, lr_r2 = train_and_log(
    "Linear_Regression",
    LinearRegression(),
    lr_params
)

# ============================================================
# LOG MODEL 2: RANDOM FOREST
# ============================================================
rf_params = {
    "model_type": "Random Forest",
    "n_estimators": 100,
    "random_state": 42
}
rf_rmse, rf_mae, rf_r2 = train_and_log(
    "Random_Forest",
    RandomForestRegressor(n_estimators=100, random_state=42),
    rf_params
)

# ============================================================
# LOG MODEL 3: XGBOOST
# ============================================================
xgb_params = {
    "model_type": "XGBoost",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": 42
}
xgb_rmse, xgb_mae, xgb_r2 = train_and_log(
    "XGBoost",
    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    xgb_params
)

# ============================================================
# PRINT FINAL SUMMARY
# ============================================================
print("\n" + "="*50)
print("🏆 MLFLOW TRACKING COMPLETE!")
print("="*50)
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-"*50)
print(f"{'Linear Regression':<25} {lr_rmse:>8.2f} {lr_mae:>8.2f} {lr_r2:>8.4f}")
print(f"{'Random Forest':<25} {rf_rmse:>8.2f} {rf_mae:>8.2f} {rf_r2:>8.4f}")
print(f"{'XGBoost':<25} {xgb_rmse:>8.2f} {xgb_mae:>8.2f} {xgb_r2:>8.4f}")

print("\n✅ All experiments logged to MLflow!")
print("👉 Run this command to open MLflow dashboard:")
print("   mlflow ui --backend-store-uri ../mlruns")
