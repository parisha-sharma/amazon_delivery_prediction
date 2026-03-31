# 🚚 Amazon Delivery Time Prediction

<div align="center">

![Delivery Prediction](https://img.shields.io/badge/Amazon-Delivery%20Prediction-667eea?style=for-the-badge&logo=amazon&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.13+-764ba2?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-667eea?style=for-the-badge&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-764ba2?style=for-the-badge&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-667eea?style=for-the-badge&logo=xgboost&logoColor=white)

### *An end-to-end Machine Learning pipeline to predict Amazon delivery times*

[Overview](#-overview) • [Features](#-features) • [Tech Stack](#-tech-stack) • [Dataset](#-dataset) • [Models](#-model-performance) • [App](#-streamlit-app) • [Key Insights](#-key-insights) • [How to Run](#-how-to-run)

</div>

---

## 📊 Overview

This project builds a complete **end-to-end Machine Learning pipeline** that predicts delivery times for Amazon e-commerce orders — built as a **GUVI | HCL Data Science Project**.

Using a dataset of **43,000+ real delivery records**, the pipeline covers everything from raw data cleaning to a beautiful, user-friendly **Streamlit web application** where anyone can get an instant delivery time estimate.

### 🎯 Project Goals

- Clean and preprocess 43,000+ Amazon delivery records using Python
- Engineer powerful new features like GPS-based distance and pickup wait time
- Train and compare 3 regression models to find the best predictor
- Track all experiments and metrics using **MLflow**
- Deploy a beautiful, pastel-themed **Streamlit web app** for real-time predictions

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Data & ML Pipeline
- 43,000+ real Amazon delivery records
- Full data cleaning and preprocessing
- Haversine formula for GPS distance calculation
- Feature engineering (distance, wait time, time of day)
- 3 regression models trained and compared
- MLflow experiment tracking and model logging

</td>
<td width="50%">

### 🌐 Streamlit Web App
- Beautiful pastel-themed user interface
- Step-by-step guided form (4 sections)
- Real-time delivery time prediction
- Color-coded results (Express / Standard / Extended)
- Smart delivery tips based on conditions
- Model performance stats displayed

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<div align="center">

| Tool | Purpose | Badge |
|------|---------|-------|
| Python | Core programming language | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| Pandas & NumPy | Data manipulation | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) |
| Scikit-learn | ML model building | ![Sklearn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) |
| XGBoost | Best performing model | ![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat&logo=xgboost&logoColor=white) |
| MLflow | Experiment tracking | ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white) |
| Streamlit | Web application | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| Matplotlib & Seaborn | Data visualization | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white) |
| Geopy | GPS distance calculation | ![Geopy](https://img.shields.io/badge/Geopy-34A853?style=flat&logo=googlemaps&logoColor=white) |

</div>

---

## 📁 Dataset

<div align="center">

| Metric | Value |
|--------|-------|
| **Data Source** | Amazon Delivery Records |
| **Total Records** | 43,739 deliveries |
| **Features** | 16 input features |
| **Target Variable** | Delivery Time (hours) |
| **Average Delivery** | ~125 hours (5 days) |
| **Fastest Delivery** | 10 hours |
| **Slowest Delivery** | 270 hours |
| **Average Distance** | 38.56 km |

</div>

### 📋 Dataset Features

| Feature | Description |
|---------|-------------|
| `Agent_Age` | Age of the delivery agent |
| `Agent_Rating` | Star rating of delivery agent (1–5) |
| `Store_Lat/Long` | GPS coordinates of the store |
| `Drop_Lat/Long` | GPS coordinates of delivery address |
| `Order_Date/Time` | When the order was placed |
| `Pickup_Time` | When agent picked up the order |
| `Weather` | Weather conditions (Sunny/Fog/Stormy etc.) |
| `Traffic` | Traffic level (Low/Medium/High/Jam) |
| `Vehicle` | Delivery vehicle (Motorcycle/Bike/Van) |
| `Area` | Delivery area type (Urban/Metropolitan) |
| `Category` | Product category (Electronics/Grocery etc.) |
| `Delivery_Time` | ⭐ Target variable — actual delivery time |

---

## 🔧 Feature Engineering

New features created to boost model performance:

| New Feature | How it's made | Why it helps |
|-------------|--------------|--------------|
| `Distance_km` | Haversine formula on GPS coords | Direct measure of delivery distance |
| `Pickup_Wait_Minutes` | Order time → Pickup time difference | Longer wait = longer total delivery |
| `Is_Weekend` | Extracted from Order_Date | Weekends have higher delivery demand |
| `Time_of_Day` | Extracted from Order_Hour | Rush hours affect delivery speed |
| `Order_Hour` | Extracted from Order_Time | Time of day affects traffic |
| `Day_of_Week` | Extracted from Order_Date | Weekly delivery patterns |

---

## 📈 Model Performance

<div align="center">

| Model | RMSE ⬇️ | MAE ⬇️ | R² Score ⬆️ | Result |
|-------|---------|--------|------------|--------|
| Linear Regression | 43.92 hrs | 33.60 hrs | 0.28 | ❌ Too simple |
| Random Forest | 22.81 hrs | 17.67 hrs | 0.80 | ✅ Good |
| **XGBoost** | **22.13 hrs** | **17.33 hrs** | **0.82** | 🥇 **Best!** |

</div>

> 🏆 **XGBoost** was selected as the final model with **82% accuracy (R²: 0.82)**
> All experiments tracked and logged using **MLflow**

---

## 🌐 Streamlit App

The prediction app features:

- 📊 **Stats Banner** — Shows 43K+ deliveries analyzed, 82% accuracy
- 🌤️ **Step 2** — Weather & traffic conditions input
- 📦 **Step 3** — Order details (vehicle, category, area, distance)
- 🧑‍💼 **Step 4** — Agent details (age, rating)
- 🔮 **Predict Button** — Instant delivery time estimate
- 🎨 **Color-coded results:**
  - ⚡ Green — Express Delivery (≤48 hrs)
  - 📦 Purple — Standard Delivery (≤120 hrs)
  - 🐢 Pink — Extended Delivery (>120 hrs)

---

## 📊 Key Insights from EDA

- ⏱️ **Average Delivery Time:** 124.91 hours (~5 days)
- ⚡ **Fastest Delivery:** 10 hours (same day!)
- 🐢 **Slowest Delivery:** 270 hours (~11 days)
- 🌫️ **Most Common Weather:** Fog
- 🚦 **Most Common Traffic:** Low
- 🏍️ **Most Used Vehicle:** Motorcycle
- 📏 **Average Distance:** 38.56 km
- ⭐ **Average Agent Rating:** 4.63 / 5.0

---

## 📁 Project Structure
```
amazon_delivery_prediction/
│
├── data/
│   ├── amazon_delivery.csv              # Raw dataset
│   ├── amazon_delivery_cleaned.csv      # After cleaning
│   └── amazon_delivery_features.csv     # After feature engineering
│
├── notebooks/
│   ├── 01_data_exploration.py           # Initial data exploration
│   ├── 02_data_cleaning.py              # Data cleaning
│   ├── 03_feature_engineering.py        # Feature creation
│   ├── 04_eda.py                        # Charts & visualizations
│   ├── 05_model_training.py             # Train & compare models
│   ├── 06_mlflow_tracking.py            # MLflow experiment logging
│   └── charts/                          # Saved EDA charts
│
├── models/
│   ├── linear_regression.pkl            # Saved Linear Regression
│   ├── random_forest.pkl                # Saved Random Forest
│   ├── xgboost.pkl                      # Saved XGBoost (best)
│   ├── encoders.pkl                     # Label encoders
│   └── feature_columns.pkl             # Feature column list
│
├── app/
│   └── app.py                           # Streamlit web app
│
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Files to ignore
└── README.md                            # Project documentation
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/amazon_delivery_prediction.git
cd amazon_delivery_prediction
```

### 2️⃣ Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the full pipeline (in order)
```bash
python notebooks/01_data_exploration.py
python notebooks/02_data_cleaning.py
python notebooks/03_feature_engineering.py
python notebooks/04_eda.py
python notebooks/05_model_training.py
python notebooks/06_mlflow_tracking.py
```

### 5️⃣ Launch the Streamlit app
```bash
streamlit run app/app.py
```

### 6️⃣ View MLflow dashboard (optional)
```bash
mlflow ui --backend-store-uri notebooks/mlruns
```
Then open: `http://127.0.0.1:5000`

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🔨 Create a new branch (`git checkout -b feature/improvement`)
3. 💾 Commit your changes (`git commit -am 'Add new feature'`)
4. 📤 Push to the branch (`git push origin feature/improvement`)
5. 🔃 Create a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

*Parisha Sharma*

- GitHub: [parisha-sharma](https://github.com/parisha-sharma)
- LinkedIn: [parishasharma15](https://www.linkedin.com/in/parishasharma15)

---

## 🌟 Acknowledgments

- Dataset: Amazon Delivery Records
- Built as **GUVI | HCL Data Science Project**
- Powered by XGBoost, MLflow & Streamlit

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

![Made with Love](https://img.shields.io/badge/Made%20with-❤️-667eea?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-ML%20Project-764ba2?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-667eea?style=for-the-badge&logo=streamlit)

**Happy Learning! 🚀**

</div>
