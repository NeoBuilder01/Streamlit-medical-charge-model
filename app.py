# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:44:26 2024

@author: Dell
"""
import numpy as np
import pandas as pd
import joblib  # Import this if you are using joblib to save/load models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

# Change directory if necessary (usually not required when running Streamlit apps)
# import os
# os.chdir(r"D:\MLLLL")

# Load and preprocess the dataset
medical_df = pd.read_csv("insurance (4).csv")

# Encode categorical variables
medical_df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_df.replace({'region': {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}}, inplace=True)

# Split data
X = medical_df.drop('charges', axis=1)
y = medical_df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# Train the model
lg = LinearRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
print("R^2 Score:", r2_score(y_test, y_pred))  # Display R^2 score in the console

# Streamlit app
st.title("Medical Insurance Prediction Model")

# Collect user input
age = st.slider('Age', min_value=10, max_value=100, value=33, step=1)
sex = st.selectbox('Biological sex', ['male', 'female'])
bmi = st.slider('Body mass index during encounter', min_value=0, max_value=60, value=26, step=1)
children = st.slider('Number of children', min_value=0, max_value=25, value=1, step=1)
smoker = st.selectbox('Smoking status', ['yes', 'no'])
region = st.selectbox('Region', ['northwest', 'southwest', 'northeast', 'southeast'])

# Encode user input
sex_encoded = 0 if sex == 'male' else 1
smoker_encoded = 0 if smoker == 'yes' else 1
region_encoded = {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}[region]

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex_encoded],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker_encoded],
    'region': [region_encoded]
})

# Add a submit button
if st.button('Submit'):
    try:
        # Make prediction
        prediction = lg.predict(input_data)
        st.write(f"Medical Insurance for this person is: ${np.round(prediction[0], 2)}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
