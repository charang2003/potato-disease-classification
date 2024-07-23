import os
from flask_app import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_version = 2
model_path = os.path.join('app/models', f'{model_version}.keras')
model = tf.keras.models.load_model(model_path)

# Class names (replace with your actual class names)
class_names = ['Early Blight', 'Late Blight', 'Healthy']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch dimension
    img_array = img_array / 255.0  # Rescale to [0,1]
    return img_array

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

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
