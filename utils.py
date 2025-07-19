# utils.py

import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap

def load_models():
    classifier = joblib.load('models/logistics_delay_classifier.pkl')
    regressor = joblib.load('models/delivery_time_regressor.pkl')
    return classifier, regressor

def get_input_template():
    return {
        'Agent_Age': 30,
        'Agent_Rating': 4.5,
        'order_hour': 13,
        'order_dayofweek': 2,
        'is_weekend': 0,
        'distance_km': 10.0,
        'Weather': 'Sunny',
        'Traffic': 'Low',
        'Vehicle': 'Bike',
        'Area': 'Urban',
        'Category': 'Food'
    }

def preprocess_input(input_dict):
    return pd.DataFrame([input_dict])

def recommend_agent(distance_km, traffic, weather, area):
    if area == 'Urban':
        if traffic == 'High':
            return 'Bike'
        elif traffic == 'Medium':
            return 'Van'
        else:
            return 'Car'
    elif area == 'Rural':
        return 'Truck'
    elif weather in ['Snowy', 'Rainy']:
        return 'Truck'
    elif distance_km > 25:
        return 'Truck'
    else:
        return 'Car'

def generate_heatmap(df):
    m = folium.Map(location=[df['Drop_Latitude'].mean(), df['Drop_Longitude'].mean()], zoom_start=6)
    heat_data = [[row['Drop_Latitude'], row['Drop_Longitude'], row['Delivery_Time']] for _, row in df.iterrows()]
    HeatMap(heat_data).add_to(m)
    return m

def find_best_delivery_slots(model, base_input, hours=range(8, 20)):
    results = []
    for h in hours:
        input_copy = base_input.copy()
        input_copy['order_hour'] = h
        prob = model.predict_proba(input_copy)[0][1]
        results.append((h, prob))
    return sorted(results, key=lambda x: x[1])
