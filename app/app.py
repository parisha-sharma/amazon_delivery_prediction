# ============================================================
# STREAMLIT APP: AMAZON DELIVERY TIME PREDICTOR
# Beautiful, user-friendly version with pastel colors!
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="📦 Delivery Time Estimator",
    page_icon="🚚",
    layout="centered"
)

# ============================================================
# CUSTOM CSS — PASTEL COLORS & BEAUTIFUL STYLING
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #f0f4ff; }

    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #3a3a5c;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #7a7a9d;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background: white;
        border-radius: 20px;
        padding: 25px 30px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #3a3a5c;
        margin-bottom: 5px;
    }
    .section-desc {
        font-size: 0.88rem;
        color: #9a9ab0;
        margin-bottom: 15px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: 700;
        width: 100%;
    }
    .tip-box {
        background: #fff8e7;
        border-left: 4px solid #ffc107;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #7a6500;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 30px;
    }
    .stat-box {
        background: #f8f0ff;
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
    }
    .stat-num {
        font-size: 1.5rem;
        font-weight: 800;
        color: #764ba2;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #9a9ab0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS AND ENCODERS
# ============================================================
@st.cache_resource
def load_models():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    with open(os.path.join(base_path, 'linear_regression.pkl'), 'rb') as f:
        lr_model = pickle.load(f)
    with open(os.path.join(base_path, 'random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(base_path, 'xgboost.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(base_path, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(base_path, 'feature_columns.pkl'), 'rb') as f:
        feature_columns = pickle.load(f)
    return lr_model, rf_model, xgb_model, encoders, feature_columns

lr_model, rf_model, xgb_model, encoders, feature_columns = load_models()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🚚 Delivery Time Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Find out when your Amazon order will arrive!</div>', unsafe_allow_html=True)

# ============================================================
# QUICK STATS BANNER (fills the first empty white card!)
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 16px; padding: 16px 25px; margin: 10px 0;
            display: flex; align-items: center; gap: 15px;">
    <div style="font-size:2rem;">📊</div>
    <div>
        <div style="color:white; font-weight:700; font-size:1rem;">
            Step 1 of 4 — Review AI Stats</div>
        <div style="color:rgba(255,255,255,0.75); font-size:0.85rem;">
            See how our AI model works before filling in details</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 About This Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Our AI has analyzed thousands of real deliveries to give you accurate estimates.</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-num">43K+</div>
        <div class="stat-label">Deliveries Analyzed</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-num">82%</div>
        <div class="stat-label">Prediction Accuracy</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-num">3</div>
        <div class="stat-label">AI Models Used</div>
    </div>""", unsafe_allow_html=True)
with s4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-num">16</div>
        <div class="stat-label">Factors Considered</div>
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚙️ Preferences")
st.sidebar.markdown("**Prediction Accuracy**")

accuracy_choice = st.sidebar.radio(
    "Choose accuracy level:",
    ["🥇 Highest Accuracy", "⚡ Fast & Accurate", "🔵 Basic Estimate"],
    help="Higher accuracy = more precise delivery estimate"
)

model_map = {
    "🥇 Highest Accuracy": xgb_model,
    "⚡ Fast & Accurate": rf_model,
    "🔵 Basic Estimate": lr_model
}
selected_model = model_map[accuracy_choice]

tips = {
    "🥇 Highest Accuracy": "✨ Best choice for accurate delivery estimates!",
    "⚡ Fast & Accurate": "👍 Great balance of speed and accuracy!",
    "🔵 Basic Estimate": "💡 Quick rough estimate only."
}
st.sidebar.markdown(f'<div class="tip-box">{tips[accuracy_choice]}</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 How it works")
st.sidebar.markdown("""
1. Fill in your delivery details
2. Click **Predict**
3. Get your estimated delivery time!

Our AI model has learned from **43,000+ real deliveries** to give you accurate estimates.
""")

# ============================================================
# SECTION 1 — DELIVERY CONDITIONS
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb, #f5576c);
            border-radius: 16px; padding: 16px 25px; margin: 10px 0;
            display: flex; align-items: center; gap: 15px;">
    <div style="font-size:2rem;">🌤️</div>
    <div>
        <div style="color:white; font-weight:700; font-size:1rem;">
            Step 2 of 4 — Delivery Conditions</div>
        <div style="color:rgba(255,255,255,0.75); font-size:0.85rem;">
            Weather and traffic affect your delivery time</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌤️ Delivery Conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Tell us about the weather and traffic where the delivery is happening.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

weather_labels = {
    'Sunny': '☀️ Sunny',
    'Cloudy': '☁️ Cloudy',
    'Fog': '🌫️ Foggy',
    'Stormy': '⛈️ Stormy',
    'Sandstorms': '🌪️ Sandstorm',
    'Windy': '💨 Windy'
}
traffic_labels = {
    'Low': '🟢 Light Traffic',
    'Medium': '🟡 Moderate Traffic',
    'High': '🟠 Heavy Traffic',
    'Jam': '🔴 Traffic Jam'
}

with col1:
    weather_display = list(weather_labels.values())
    weather_keys    = list(weather_labels.keys())
    weather_sel     = st.selectbox("Current Weather", options=weather_display, index=1)
    weather         = weather_keys[weather_display.index(weather_sel)]

with col2:
    traffic_display = list(traffic_labels.values())
    traffic_keys    = list(traffic_labels.keys())
    traffic_sel     = st.selectbox("Traffic Conditions", options=traffic_display, index=0)
    traffic         = traffic_keys[traffic_display.index(traffic_sel)]

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 2 — ORDER DETAILS (fills second empty white card!)
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #56ab2f, #a8e063);
            border-radius: 16px; padding: 16px 25px; margin: 10px 0;
            display: flex; align-items: center; gap: 15px;">
    <div style="font-size:2rem;">📦</div>
    <div>
        <div style="color:white; font-weight:700; font-size:1rem;">
            Step 3 of 4 — Order Details</div>
        <div style="color:rgba(255,255,255,0.75); font-size:0.85rem;">
            Tell us about your product and delivery location</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📦 Order Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Tell us about the product, delivery vehicle, and how far it needs to travel.</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

vehicle_labels = {
    'motorcycle': '🏍️ Motorcycle',
    'scooter':    '🛵 Scooter',
    'bicycle':    '🚲 Bicycle',
    'van':        '🚐 Van'
}
area_labels = {
    'Urban':         '🏙️ Urban City',
    'Metropolitian': '🌆 Metropolitan',
    'Semi-Urban':    '🏘️ Semi-Urban',
    'Other':         '🌄 Other'
}

with col3:
    vehicle_display = list(vehicle_labels.values())
    vehicle_keys    = list(vehicle_labels.keys())
    vehicle_sel     = st.selectbox("Delivery Vehicle", options=vehicle_display, index=0)
    vehicle         = vehicle_keys[vehicle_display.index(vehicle_sel)]

    category_options = list(encoders['Category'].classes_)
    category         = st.selectbox("Product Category 📦", options=category_options, index=0)

with col4:
    area_display = list(area_labels.values())
    area_keys    = list(area_labels.keys())
    area_sel     = st.selectbox("Delivery Area", options=area_display, index=0)
    area         = area_keys[area_display.index(area_sel)]

    distance_km = st.number_input(
        "📏 Distance to deliver (km)",
        min_value=0.5, max_value=200.0, value=10.0, step=0.5
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 3 — AGENT DETAILS
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #f7971e, #ffd200);
            border-radius: 16px; padding: 16px 25px; margin: 10px 0;
            display: flex; align-items: center; gap: 15px;">
    <div style="font-size:2rem;">🧑‍💼</div>
    <div>
        <div style="color:white; font-weight:700; font-size:1rem;">
            Step 4 of 4 — Agent Details</div>
        <div style="color:rgba(255,255,255,0.75); font-size:0.85rem;">
            Agent experience and rating impact delivery speed</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧑‍💼 Delivery Agent Details</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Agent experience and rating can affect how quickly your order is delivered.</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    agent_age = st.slider("Agent Age 👤", min_value=18, max_value=60, value=30)
with col6:
    agent_rating = st.slider("Agent Rating ⭐", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

stars = "⭐" * int(agent_rating)
st.markdown(f"**Current Rating:** {stars} ({agent_rating})")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SECTION 4 — TIME DETAILS (collapsible)
# ============================================================
with st.expander("🕐 Time Details (Optional — click to expand)"):
    st.markdown("*These details help improve accuracy. Default values work fine if you're unsure!*")
    col7, col8 = st.columns(2)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    day_names   = ["Monday","Tuesday","Wednesday",
                   "Thursday","Friday","Saturday","Sunday"]

    with col7:
        order_hour   = st.slider("Order Hour (0=midnight, 12=noon) 🕐", 0, 23, 10)
        pickup_hour  = st.slider("Pickup Hour 🕑", 0, 23, 11)
        pickup_wait  = st.number_input("Pickup Wait Time (minutes) ⏱️", 0, 120, 15)

    with col8:
        day_of_week  = st.selectbox("Day of Week 📅", options=list(range(7)),
                                    format_func=lambda x: day_names[x])
        # FIX: use selectbox instead of slider for month (avoids format_func error)
        month_sel    = st.selectbox("Month 📆", options=list(range(1, 13)),
                                    format_func=lambda x: month_names[x-1])
        month        = month_sel
        day          = st.slider("Day of Month", 1, 31, 15)

# Derived features
is_weekend = 1 if day_of_week >= 5 else 0

def get_time_of_day(hour):
    if 6 <= hour < 12:   return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else:                 return 'Night'

time_of_day = get_time_of_day(order_hour)

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Estimate My Delivery Time!")

if predict_clicked:

    # Encode inputs
    weather_enc  = encoders['Weather'].transform([weather])[0]
    traffic_enc  = encoders['Traffic'].transform([traffic])[0]
    vehicle_enc  = encoders['Vehicle'].transform([vehicle])[0]
    area_enc     = encoders['Area'].transform([area])[0]
    category_enc = encoders['Category'].transform([category])[0]
    tod_enc      = encoders['Time_of_Day'].transform([time_of_day])[0]

    # Build input
    input_data = pd.DataFrame([[
        agent_age, agent_rating,
        weather_enc, traffic_enc, vehicle_enc,
        area_enc, category_enc,
        day_of_week, month, day,
        order_hour, pickup_hour,
        distance_km, pickup_wait,
        is_weekend, tod_enc
    ]], columns=feature_columns)

    # Predict
    prediction = max(0, selected_model.predict(input_data)[0])
    days = prediction / 24

    # Color theme based on speed
    if prediction <= 48:
        emoji = "⚡"
        label = "Express Delivery!"
        color = "linear-gradient(135deg, #56ab2f, #a8e063)"
    elif prediction <= 120:
        emoji = "📦"
        label = "Standard Delivery"
        color = "linear-gradient(135deg, #667eea, #764ba2)"
    else:
        emoji = "🐢"
        label = "Extended Delivery"
        color = "linear-gradient(135deg, #f093fb, #f5576c)"

    # Result card
    st.markdown(f"""
    <div style="background:{color}; border-radius:20px; padding:35px;
                text-align:center; color:white; margin:20px 0;
                box-shadow:0 8px 25px rgba(0,0,0,0.15);">
        <div style="font-size:3.5rem;">{emoji}</div>
        <div style="font-size:1rem; opacity:0.9; margin-bottom:5px;">
            Estimated Delivery Time
        </div>
        <div style="font-size:3rem; font-weight:900;">
            {prediction:.0f} hours
        </div>
        <div style="font-size:1.3rem; margin-top:5px;">
            (~{days:.1f} days)
        </div>
        <div style="font-size:1rem; margin-top:10px;
                    background:rgba(255,255,255,0.2); border-radius:20px;
                    padding:6px 20px; display:inline-block;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Delivery tips
    st.markdown("### 💡 Delivery Tips")
    tc1, tc2 = st.columns(2)
    with tc1:
        if traffic == 'Jam':
            st.warning("🚦 Traffic jam detected — expect delays.")
        elif traffic == 'Low':
            st.success("🟢 Light traffic — great delivery conditions!")
        if weather in ['Stormy', 'Sandstorms']:
            st.warning("⛈️ Severe weather may cause delays.")
        elif weather == 'Sunny':
            st.success("☀️ Clear weather — perfect for delivery!")
    with tc2:
        if agent_rating >= 4.5:
            st.success("⭐ Highly rated agent — expect great service!")
        if distance_km > 50:
            st.info(f"📏 Long distance ({distance_km}km) — allow extra time.")
        if is_weekend:
            st.info("📅 Weekend delivery — slightly higher demand.")

    # Full summary
    with st.expander("📋 See full order summary"):
        summary_data = {
            "Detail": ["Weather","Traffic","Vehicle","Area","Product",
                       "Distance","Agent Rating","Order Time","Day Type"],
            "Value":  [weather_sel, traffic_sel, vehicle_sel, area_sel,
                       category, f"{distance_km} km", f"{agent_rating} ⭐",
                       f"{order_hour}:00 ({time_of_day})",
                       "Weekend 🏖️" if is_weekend else "Weekday 💼"]
        }
        st.table(pd.DataFrame(summary_data))

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<div class="footer">🚚 Powered by AI | Trained on 43,000+ real deliveries</div>',
    unsafe_allow_html=True
)