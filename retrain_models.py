# retrain_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('data/amazon_delivery.csv')

# --- Step 1: Preprocess the data ---

# Drop rows with missing values (optional: customize this)
df.dropna(inplace=True)

# Feature Engineering: extract useful time-based features
df['Order_Time'] = pd.to_datetime(df['Order_Time'])
df['order_hour'] = df['Order_Time'].dt.hour
df['order_dayofweek'] = df['Order_Time'].dt.dayofweek
df['is_weekend'] = df['order_dayofweek'].isin([5, 6]).astype(int)

# Encode categorical features
categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Optional: save for inverse transform later

# Select features
features = ['Agent_Age', 'Agent_Rating', 'order_hour', 'order_dayofweek', 'is_weekend',
            'distance_km', 'Weather', 'Traffic', 'Vehicle', 'Area', 'Category']

X = df[features]

# --- Step 2A: Train Delay Classifier ---
y_class = df['Delayed']  # Make sure this column exists: 0 (on time), 1 (delayed)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_cls, y_train_cls)

# Save classifier model
joblib.dump(clf, 'models/logistics_delay_classifier1.pkl')

# --- Step 2B: Train Delivery Time Regressor ---
y_reg = df['Delivery_Time']  # This should be a numeric column (e.g., in minutes)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)

# Save regressor model
joblib.dump(reg, 'models/delivery_time_regressor1.pkl')

print("✅ Models retrained and saved to 'models/' folder.")
