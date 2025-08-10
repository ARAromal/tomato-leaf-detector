# predict.py
import tensorflow as tf
import numpy as np
import cv2 # OpenCV for image handling
import sys

# --- 1. LOAD THE TRAINED MODEL AND CLASS NAMES ---
model = tf.keras.models.load_model('tomato_leaf_disease_detector.keras')

# You must have the same class names in the same order as during training
class_names = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 2. GET IMAGE PATH FROM COMMAND LINE ---
if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

# --- 3. LOAD AND PREPROCESS THE IMAGE ---
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert from BGR (OpenCV default) to RGB
img = cv2.resize(img, (224, 224)) # Resize to match model's input size
img_array = tf.expand_dims(img, 0) # Create a batch

# --- 4. MAKE PREDICTION ---
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# --- 5. PRINT THE RESULT ---
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"\nPrediction: This leaf is most likely '{predicted_class}' with {confidence:.2f}% confidence.")