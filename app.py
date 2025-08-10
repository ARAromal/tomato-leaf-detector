# app.py

import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename

# --- INITIALIZATION ---
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('tomato_leaf_disease_detector.keras')

# Define the class names in the correct order
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- HELPER FUNCTION ---
def predict_disease(image_path):
    """Load and preprocess the image, then make a prediction."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# --- ROUTES ---
@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handle file upload and display the result."""
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        predicted_class, confidence = predict_disease(filepath)

        return render_template('result.html', prediction=predicted_class, confidence=f"{confidence:.2f}", image_file=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file to be displayed on the result page."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- MAIN ---
if __name__ == '__main__':
    app.run(debug=True)