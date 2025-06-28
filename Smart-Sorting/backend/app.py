from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
from werkzeug.utils import secure_filename
from datetime import datetime

# Define paths relative to this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'fruit_veg_disease_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'feedback_data', 'feedbacks.json')

# App init with external static/templates
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100

    result = "✅ Good to Eat" if "__Healthy" in predicted_class else "❌ Don't Eat"

    return render_template('result.html',
                           prediction=predicted_class,
                           confidence=round(confidence, 2),
                           result=result,
                           image_path=f'uploads/{filename}')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_data = {
        "name": request.form.get("name"),
        "email": request.form.get("email"),
        "accurate": request.form.get("accurate"),
        "correct_label": request.form.get("correct_label"),
        "comments": request.form.get("comments"),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(feedback_data)

    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    return "✅ Thank you for your feedback!"

if __name__ == '__main__':
    app.run(debug=True)
