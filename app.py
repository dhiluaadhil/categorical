import streamlit as st
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- STEP 1: TRAIN THE MODEL (Copy-pasted from your notebook) ---
weather = ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy']
play = ['No', 'No', 'Yes', 'Yes', 'Yes']

le_weather = LabelEncoder()
le_play = LabelEncoder()

X = le_weather.fit_transform(weather).reshape(-1, 1)
y = le_play.fit_transform(play)

model = CategoricalNB()
model.fit(X, y)

# --- STEP 2: STREAMLIT INTERFACE ---
st.title("Weather Play Predictor")
st.write("This model predicts if you should play based on the weather.")

# User selection
weather_input = st.selectbox("Select the weather:", ['Sunny', 'Rainy'])

if st.button("Predict"):
    # Transform input and predict
    test_encoder = le_weather.transform([weather_input]).reshape(-1, 1)
    prediction_id = model.predict(test_encoder)
    
    # Convert ID back to 'Yes' or 'No'
    result = le_play.inverse_transform(prediction_id)
    
    # Show result
    st.subheader(f"Prediction: Should you play? **{result[0]}**")
