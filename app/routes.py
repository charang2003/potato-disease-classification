from flask_app import request, jsonify, render_template
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from app import app
from app.utils import preprocess_image

# Load the trained model
model_version = 2
model_path = os.path.join('app/models', '2.keras')
model = tf.keras.models.load_model(model_path)

# Class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Replace with your actual class names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img_file = request.files['image']
    img_path = os.path.join('uploads', img_file.filename)
    img_file.save(img_path)

    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    # Clean up
    os.remove(img_path)

    return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
