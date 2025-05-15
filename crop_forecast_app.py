import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model, scaler, and encoders
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('area_encoder.pkl', 'rb') as f:
    area_encoder = pickle.load(f)
with open('crop_encoder.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)

# Streamlit app setup
st.title('Global Crop Production Forecast')
st.write('Predict crop production (in tons) based on agricultural factors. Select a region, crop, and input the year, area harvested, and yield.')

# User inputs
area = st.selectbox('Select Region', area_encoder.classes_)
crop = st.selectbox('Select Crop', crop_encoder.classes_)
year = st.number_input('Year', min_value=2019, max_value=2030, value=2023)
area_harvested = st.number_input('Area Harvested (hectares)', min_value=0.0, value=1000.0)
yield_kg_ha = st.number_input('Yield (kg/ha)', min_value=0.0, value=1000.0)

# Prediction
if st.button('Predict Production'):
    # Encode area and crop
    area_encoded = area_encoder.transform([area])[0]
    crop_encoded = crop_encoder.transform([crop])[0]
    # Create input array
    input_data = np.array([[area_encoded, crop_encoded, year, area_harvested, yield_kg_ha]])
    # Scale input
    input_scaled = scaler.transform(input_data)
    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f'Predicted Production for {crop} in {area} in {year}: {prediction:.2f} tons')

# Additional information
st.markdown('### About')
st.write('This app uses a Random Forest Regressor trained on FAOSTAT data to forecast crop production globally. The model considers region, crop type, year, area harvested, and yield as inputs.')