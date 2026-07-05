import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load model and encoders
model = joblib.load("models/salary_model.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")
education_encoder = joblib.load("models/education_encoder.pkl")

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼"
)

st.title("💼 Employee Salary Prediction")

st.write("Enter employee information to predict salary.")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=70,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=5.0
)

if st.button("Predict Salary"):

    gender_encoded = gender_encoder.transform([gender])[0]
    education_encoded = education_encoder.transform([education])[0]

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_encoded],
        "Education Level": [education_encoded],
        "Years of Experience": [experience]
    })

    prediction = model.predict(input_df)

    st.success(f"💰 Predicted Salary: ${prediction[0]:,.2f}")



st.markdown("## 📊 Model Evaluation Charts")

col1, col2 = st.columns(2)

with col1:
    st.image("assets/actual_vs_predicted.png",
             caption="Actual vs Predicted Salary",
             use_container_width=True)

with col2:
    st.image("assets/feature_importance.png",
             caption="Feature Importance",
             use_container_width=True)

import os
st.write("Files in assets:", os.listdir("assets"))