
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import json
import os
import io

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(
    os.path.join(MODEL_DIR, "best_svm_model.pkl")
)

with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Tomato Leaf Disease Detection API",
        "status": "running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Upload an image using the 'image' field"
        }), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((64, 64))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, -1)

        prediction = int(model.predict(image_array)[0])

        return jsonify({
            "predicted_class": class_names[prediction],
            "class_index": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
