import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
pipeline_rf = joblib.load(r'C:\Users\vanig\lokeshh\employee_salary_prediction.pkl')

# Load the dataset to get unique values for dropdowns
df = pd.read_csv(r'C:\Users\vanig\lokeshh\employee_salary.csv')
df = df.dropna()
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df = df.dropna(subset=['Salary'])

# Streamlit UI
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict employee salary based on details.")

# User inputs
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", df['Gender'].unique())
education = st.selectbox("Education Level", df['Education Level'].unique())
job_title = st.selectbox("Job Title", df['Job Title'].unique())
experience = st.slider("Years of Experience", 0, 40, 5)

# Feature engineering
exp_to_age = experience / age if age > 0 else 0

# Create input
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience],
    'Exp_to_Age': [exp_to_age]
})

if st.button("Predict Salary"):
    prediction = pipeline_rf.predict(input_df)[0]
    st.success(f"💰 Estimated Salary: ₹{prediction:,.2f}")
