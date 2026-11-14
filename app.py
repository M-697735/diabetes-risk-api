# app.py  -  the API (web service) that uses the saved model
# Run this after main.py:  python app.py

import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the model that main.py saved
if not os.path.exists("diabetes_model.pkl"):
    raise SystemExit("diabetes_model.pkl not found. Run:  python main.py  first.")
with open("diabetes_model.pkl", "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
features = saved["features"]
accuracy = saved["accuracy"]
recall = saved["recall"]
eda = saved["eda"]

history = []   # remembers the last 10 predictions (cleared when the app restarts)


# Endpoint 1: check that the API is running
@app.route("/")
def home():
    return jsonify({"message": "Diabetes Risk Prediction API is running"})


# Endpoint 2: health check
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# Endpoint 3: information about the model
@app.route("/model-info")
def model_info():
    return jsonify({
        "model": "RandomForestClassifier",
        "features": features,
        "accuracy_percent": round(accuracy * 100, 2),
        "recall_per_class": recall
    })


# Endpoint 4: the EDA summary (made by main.py)
@app.route("/eda")
def get_eda():
    return jsonify(eda)


# Endpoint 5: predict the risk for one patient
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Check that the user sent JSON
    if not data:
        return jsonify({"error": "Please send the patient data as JSON"}), 400

    # Check that no field is missing
    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    # Check that every value is a number
    try:
        values = [float(data[f]) for f in features]
    except (ValueError, TypeError):
        return jsonify({"error": "All values must be numbers"}), 400

    # Ask the model for the answer
    patient = pd.DataFrame([values], columns=features)
    risk = str(model.predict(patient)[0])
    chances = model.predict_proba(patient)[0]
    probabilities = {str(c): round(float(p), 3) for c, p in zip(model.classes_, chances)}

    # Save in history (keep only the last 10)
    history.append({"input": data, "risk": risk})
    if len(history) > 10:
        history.pop(0)

    return jsonify({"risk": risk, "probabilities": probabilities})


# Endpoint 6: see the last predictions
@app.route("/history")
def get_history():
    return jsonify(history)


if __name__ == "__main__":
    # Render gives the port in the PORT variable; on your PC it is 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
