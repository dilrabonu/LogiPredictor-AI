

# 🚚 LogiPredictor-AI

**LogiPredictor-AI** is an intelligent AI assistant built with Streamlit and machine learning models to help optimize delivery operations for logistics companies like Amazon. The system predicts delivery delays, estimates delivery times, recommends agent types, and visualizes bottlenecks – all in one powerful and interactive dashboard.

---

## 💡 Features

🔹 **Delay Classifier**  
Predicts the likelihood of a delivery delay using classification models.

🔹 **Delivery Time Estimator**  
Estimates the expected delivery duration using regression techniques.

🔹 **Agent Recommender**  
Recommends the optimal type of delivery agent (e.g., bike, truck, van) based on conditions.

🔹 **Bottleneck Heatmap**  
Displays delivery congestion areas on a map using geospatial data.

🔹 **Smart Delivery Planner**  
Finds the best delivery time slots with the lowest delay risk based on historical trends.

---

## 🧠 Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Scikit-learn (Logistic Regression, Random Forest, etc.)
- **Visualization**: Folium, Seaborn, Matplotlib
- **Data Processing**: Pandas, NumPy
- **Language**: Python 3.10

---

## 📁 Project Structure

```bash
LogiPredictor-AI/
│
├── data/                 # CSV dataset for training & inference

├── models/               # Saved ML models

├── app.py                # Main Streamlit app

├── retrain_models.py     # Script to retrain models with new data

├── utils.py              # Helper functions for preprocessing & prediction

├── requirements.txt      # Dependencies

└── README.md             # You're here!

1. Clone the Repository

git clone https://github.com/dilrabonu/LogiPredictor-AI.git
cd LogiPredictor-AI
2. Create & Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
3. Install Dependencies

pip install -r requirements.txt
4. Run the App

streamlit run app.py
🔁 Retrain Models
If you want to retrain the models using updated data (amazon_delivery.csv), run:

python retrain_models.py
This will automatically preprocess the data and save new models in the /models directory.

📊 Sample Inputs
You can test predictions using delivery features such as:

Agent Age & Rating

Distance (km)

Vehicle Type

Weather & Traffic Conditions

Day of Week / Is Weekend

Order Hour

🎯 Use Cases
Last-mile delivery route optimization

Delay forecasting for logistics operations

Real-time agent assignment system

Delivery time analysis for urban vs rural areas

Visual delivery bottleneck diagnostics

👤 Author
Dilrabonu Khidirova
Data Scientist & AI Engineer
https://www.linkedin.com/in/dilrabo-khidirova-3144b8244/

⭐ Contributions & Feedback
Feel free to fork this repo, submit pull requests, or open issues for improvements.
If you like this project, give it a ⭐ to support the work!

