import streamlit as st
import pandas as pd
import pickle
from scipy.special import inv_boxcox

# Load the model and lambda
with open("final_best_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open("boxcox_lambda.pkl", "rb") as f:
    lam = pickle.load(f)

# App Title
st.title("💡 SmartPremium: Insurance Premium Predictor")

st.markdown("Enter the customer details below to predict their estimated insurance premium.")

# Input Fields
age = st.slider("Age", 18, 100, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Annual Income (₹)", value=500000)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents = st.slider("Number of Dependents", 0, 10, 1)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
health_score = st.slider("Health Score", 0, 100, 75)
location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
claims = st.slider("Previous Claims", 0, 10, 1)
vehicle_age = st.slider("Vehicle Age (Years)", 0, 20, 5)
credit_score = st.slider("Credit Score", 300, 900, 600)
duration = st.slider("Insurance Duration (Years)", 1, 10, 3)
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])
property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

# Collect input
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Annual Income": income,
    "Marital Status": marital_status,
    "Number of Dependents": dependents,
    "Education Level": education,
    "Occupation": occupation,
    "Health Score": health_score,
    "Location": location,
    "Policy Type": policy_type,
    "Previous Claims": claims,
    "Vehicle Age": vehicle_age,
    "Credit Score": credit_score,
    "Insurance Duration": duration,
    "Smoking Status": smoking,
    "Exercise Frequency": exercise,
    "Property Type": property_type
}])

# Predict on submit
if st.button("Predict Premium"):
    pred_boxcox = model.predict(input_data)
    predicted_premium = inv_boxcox(pred_boxcox, lam)[0]
    st.success(f"💸 Predicted Insurance Premium: ₹{predicted_premium:,.2f}")
