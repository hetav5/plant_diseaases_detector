import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model(r'F:\codes\plant_diseases_detector\models\mobilenet_model.h5')

# Load class names (label map)
with open(r'F:\codes\plant_diseases_detector\models\class_names.json', 'r') as f:
    label_map = json.load(f)
idx_to_class = {v: k for k, v in label_map.items()}

# Upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocess the uploaded image
    img = tf.keras.utils.load_img(filepath, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    
    # Predict
    preds = model.predict(img_array)
    predicted_idx = np.argmax(preds, axis=1)[0]
    predicted_class = idx_to_class[predicted_idx]
    confidence = np.max(preds) * 100  # Get confidence in %

    # Return prediction
    return render_template('result.html', prediction=predicted_class, confidence=f"{confidence:.2f}%")

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

