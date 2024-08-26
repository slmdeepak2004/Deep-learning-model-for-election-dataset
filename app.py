
import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load the trained model
model = tf.keras.models.load_model('deep_learning_model.h5')

# Define preprocessing steps
numeric_features = ['numParishes', 'blankVotes', 'nullVotes', 'votersPercentage', 'Percentage']
categorical_features = ['Party']

# Create ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define feature names for OneHotEncoder
onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = numeric_features + list(onehot_feature_names)

# Streamlit UI
st.title("Election Mandates Prediction")

st.write("Enter the details below to predict the total mandates:")

# Input fields
numParishes = st.number_input('Number of Parishes', min_value=0)
blankVotes = st.number_input('Blank Votes', min_value=0)
nullVotes = st.number_input('Null Votes', min_value=0)
votersPercentage = st.number_input('Voters Percentage', min_value=0.0, max_value=100.0)
Party = st.selectbox('Party', options=['Party1', 'Party2', 'Party3'])  # Adjust this list based on actual categories
Percentage = st.number_input('Percentage', min_value=0.0, max_value=100.0)

# Predict button
if st.button('Predict'):
    # Create a DataFrame for preprocessing
    input_df = pd.DataFrame([{
        'numParishes': numParishes,
        'blankVotes': blankVotes,
        'nullVotes': nullVotes,
        'votersPercentage': votersPercentage,
        'Party': Party,
        'Percentage': Percentage
    }])
    
    # Preprocess input data
    input_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display prediction
    st.write(f"Predicted Total Mandates: {prediction[0][0]:.2f}")

