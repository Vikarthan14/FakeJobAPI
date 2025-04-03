from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load the trained model and vectorizer (using correct filenames)
model = joblib.load("job_fake_detection_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route("/", methods=["GET"])
def home():
    return "Fake Job Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        job_description = data.get("job_description")

        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        # Transform input text using the trained vectorizer
        input_tfidf = vectorizer.transform([job_description])
        
        # Make prediction (0 = Real, 1 = Fake)
        prediction = model.predict(input_tfidf)[0]
        confidence = model.predict_proba(input_tfidf).max() * 100  # Get confidence %

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
