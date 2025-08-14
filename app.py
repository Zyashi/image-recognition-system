import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

# Create upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["image"]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Preprocess image for MobileNetV2
        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)

        if not decoded or len(decoded[0]) == 0:
            raise ValueError("Model could not produce predictions.")

        results = []
        for item in decoded[0]:
            if len(item) == 3:  # ('id', 'label', probability)
                results.append(item)

        return render_template("result.html", predictions=results, image_path=file_path)

    except Exception as e:
        return f"Prediction failed: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
