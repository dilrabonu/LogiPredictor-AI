import zipfile
import pandas as pd
import os

# 1. Unzip the data
zip_path = 'amazon_delivery.csv.zip'
extract_to = 'data/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ File extracted.")

# 2. Load CSV
df = pd.read_csv('data/amazon_delivery.csv')

# 3. Drop rows with missing critical values
df = df.dropna(subset=['Order_Date', 'Order_Time', 'Pickup_Time'])

# 4. Convert Date and Time safely
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')

# 5. Drop rows where conversion failed
df = df.dropna(subset=['Order_Date', 'Order_Time', 'Pickup_Time'])

# 6. Feature Engineering
df['Order_Hour'] = df['Order_Time'].dt.hour
df['Pickup_Hour'] = df['Pickup_Time'].dt.hour
df['Order_DayOfWeek'] = df['Order_Date'].dt.dayofweek
df['Is_Weekend'] = df['Order_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# 7. Delivery Duration in Minutes
order_datetime = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Order_Time'].dt.strftime('%H:%M:%S'))
pickup_datetime = pd.to_datetime(df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Pickup_Time'].dt.strftime('%H:%M:%S'))
df['Delivery_duration'] = (pickup_datetime - order_datetime).dt.total_seconds() / 60.0

# 8. Binary Classification Target
df['Delayed'] = df['Delivery_duration'].apply(lambda x: 1 if x > 60 else 0)

# 9. Save cleaned dataset
df.to_csv('data/our_cleaned_dataset.csv', index=False)
print("✅ Cleaned dataset saved to 'data/our_cleaned_dataset.csv'")
