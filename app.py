from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open('health_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)

    return render_template('index.html', prediction_text=f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
