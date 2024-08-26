
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your pre-trained model (assuming it's saved as 'model.h5')
model = tf.keras.models.load_model('deep_learning_model.h5')

# Create a simple UI
st.title("Mandate Prediction App")
st.write("This app predicts the number of mandates based on input data.")

# Input fields for user to enter data
time = st.text_input("Time")
total_mandates = st.number_input("Total Mandates")
num_parishes = st.number_input("Number of Parishes")
blank_votes = st.number_input("Blank Votes")
null_votes = st.number_input("Null Votes")
voters_percentage = st.number_input("Voters Percentage")
party = st.text_input("Party")
percentage = st.number_input("Percentage")

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    input_data = np.array([[total_mandates, num_parishes, blank_votes, null_votes, voters_percentage, percentage]])
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction
    st.write(f"Predicted Mandates: {prediction[0][0]}")
