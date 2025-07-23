import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
@st.cache_data
def load_data():
    df = pd.read_csv("lucknow.csv") 
     # replace with your file path
    df['carpet_area']=pd.to_numeric(df['carpet_area'], errors='coerce')
    si=SimpleImputer(missing_values=np.nan, strategy='mean')
    df['carpet_area']=si.fit_transform(df[['carpet_area']])

    df['status'] = df['status'].map({'StatusReady': 1, 'StatusUnder': 0})
    return df

df = load_data()

# Define feature categories
numeric_features = ['bhk', 'area_sq_ft', 'carpet_area', 'bathrooms']
categorical_features = ['type', 'location']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')  # Keeps 'status'

# Split data
X = df.drop('price_lakh', axis=1)
y = df['price_lakh']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
st.title("🏠 House Price Prediction App (Lucknow)")
st.header("Enter  Property details of Lucknow")
st.header("Enter Property Details:")
bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
property_type = st.selectbox("Property Type", df['type'].unique())
area = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1100)
carpet = st.number_input("Carpet Area (sq ft)", min_value=100, max_value=10000, value=800)
location = st.selectbox("Location", df['location'].unique())
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
status = st.selectbox("Status", ['StatusReady', 'StatusUnder'])
status = 1 if status == 'StatusReady' else 0
input_df = pd.DataFrame([{
    'bhk': bhk,
    'type': property_type,
    'area_sq_ft': area,
    'location': location,
    'carpet_area': carpet,
    'status': status,
    'bathrooms': bathrooms
}])

# Predict
if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"🏷️ Estimated Price: ₹ {round(price, 2)} Lakh")
    
