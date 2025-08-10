# Tomato Leaf Disease Detector 🍅

A web application built using Flask and TensorFlow to classify 10 different types of tomato leaf diseases from an uploaded image. This project utilizes a Convolutional Neural Network (CNN) with transfer learning from the MobileNetV2 model.

---

## Features

-   **Web Interface:** Easy-to-use page for uploading leaf images.
-   **Real-time Prediction:** Uses a trained Keras/TensorFlow model to provide instant predictions.
-   **10-Class Classification:** Identifies 9 common tomato leaf diseases plus a "healthy" category.
-   **Confidence Score:** Displays the model's confidence in its prediction.

---

## Technology Stack

-   **Backend:** Python, Flask
-   **Deep Learning:** TensorFlow, Keras
-   **Image Processing:** OpenCV-Python
-   **Core Libraries:** NumPy, Matplotlib

---

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/ARAromal/tomato-leaf-detector.git](https://github.com/ARAromal/tomato-leaf-detector.git)
cd tomato-leaf-detector

2. Download the Dataset (Important)
This repository does not include the dataset due to its large size.

Download the dataset from this Kaggle link.

Unzip the file. You should have a tomato folder that contains another tomato folder inside it. Place this top-level tomato folder inside your project directory.

The final structure should be: .../tomato-leaf-detector/tomato/tomato/train.

3. Set Up the Environment
# Create a virtual environment
py -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install the required packages
pip install -r requirements.txt

4. Run the Web Application
python app.py

5. Use the App
Open your web browser and navigate to the following address:
[http://127.0.0.1:5000](http://127.0.0.1:5000)
