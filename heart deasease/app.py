from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model and scaler
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        # Scale input data
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Display result
        result = "Heart Disease Detected" if prediction[0] == 0 else "No Heart Disease"
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid input data")

if __name__ == "__main__":
    app.run(debug=True)
