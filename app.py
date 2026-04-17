import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("health_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Diabetes Prediction System")

st.write("Enter patient details:")

# Inputs
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 60.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")
