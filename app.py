import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("health_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(preg, glucose, bp, skin, insulin, bmi, dpf, age):
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
    )
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    return "⚠️ Diabetic" if prediction[0] == 1 else "✅ Not Diabetic"

# UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        "number","number","number","number",
        "number","number","number","number"
    ],
    outputs="text",
    title="Diabetes Prediction System"
)

demo.launch()
