
import streamlit as st
import pandas as pd
from utils import load_models, get_input_template, preprocess_input, recommend_agent, generate_heatmap, find_best_delivery_slots
import streamlit.components.v1 as components
import tempfile
import joblib

# Load models
classifier = joblib.load('models/logistics_delay_classifier.pkl')
regressor = joblib.load('models/delivery_time_regressor.pkl')

# Load sample dataset for heatmap
@st.cache_data
def load_sample_data():
    return pd.read_csv('data/our_cleaned_dataset.csv') 

# UI setup
st.set_page_config(page_title="LogiPredictor AI", layout="wide")
st.title("🚛 LogiPredictor AI Assistant")

st.sidebar.header("📥 Input Delivery Features")
user_input = get_input_template()

user_input['Agent_Age'] = st.sidebar.slider('Agent Age', 18, 60, user_input['Agent_Age'])
user_input['Agent_Rating'] = st.sidebar.slider('Agent Rating', 1.0, 5.0, user_input['Agent_Rating'])
user_input['order_hour'] = st.sidebar.slider('Order Hour', 0, 23, user_input['order_hour'])
user_input['order_dayofweek'] = st.sidebar.selectbox('Day of Week (0=Mon)', list(range(7)), index=user_input['order_dayofweek'])
user_input['is_weekend'] = st.sidebar.selectbox('Is Weekend?', [0, 1], index=user_input['is_weekend'])
user_input['distance_km'] = st.sidebar.slider('Distance (km)', 1.0, 50.0, user_input['distance_km'])

user_input['Weather'] = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Snowy', 'Cloudy'], index=0)
user_input['Traffic'] = st.sidebar.selectbox('Traffic', ['Low', 'Medium', 'High'], index=0)
user_input['Vehicle'] = st.sidebar.selectbox('Vehicle Type', ['Bike', 'Car', 'Truck', 'Van'], index=0)
user_input['Area'] = st.sidebar.selectbox('Area', ['Urban', 'Suburban', 'Rural'], index=0)
user_input['Category'] = st.sidebar.selectbox('Category', ['Food', 'Electronics', 'Clothing', 'Groceries', 'Furniture'], index=0)

input_df = preprocess_input(user_input)

# Delay Classifier
st.subheader("🚦 Delay Classifier")
if st.button("Predict Delay"):
    pred = classifier.predict(input_df)[0]
    proba = classifier.predict_proba(input_df)[0][1]
    st.success(f"Prediction: {'❌ Delayed' if pred == 1 else '✅ On Time'}")
    st.info(f"Delay Probability: {proba:.2%}")

# Delivery Time Estimator
st.subheader("⏱️ Delivery Time Estimator")
if st.button("Estimate Delivery Time"):
    predicted_time = regressor.predict(input_df)[0]
    st.success(f"Estimated Delivery Duration: {predicted_time:.2f} minutes")

# Agent Recommender
st.subheader("🧠 Agent Recommender")
if st.button("Recommend Agent Type"):
    agent = recommend_agent(user_input['distance_km'], user_input['Traffic'], user_input['Weather'], user_input['Area'])
    st.success(f"Recommended Agent Type: {agent}")

# Bottleneck Detector
st.subheader("📍 Bottleneck Heatmap")
if st.button("Show Delivery Bottlenecks"):
    df = load_sample_data()
    m = generate_heatmap(df)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        m.save(tmp.name)
        components.html(open(tmp.name, 'r').read(), height=600)

# Smart Planner
st.subheader("📅 Smart Delivery Planner")
if st.button("Find Best Time Slots"):
    time_slots = find_best_delivery_slots(classifier, input_df)
    best_hour, best_risk = time_slots[0]
    st.success(f"Best Hour to Deliver: {best_hour}:00 with only {best_risk:.2%} delay risk")
    st.write("---")
    st.subheader("📊 Hourly Risk Breakdown")
    for hour, risk in time_slots:
        st.write(f"🕒 {hour}:00 → Delay Risk: {risk:.2%}")
