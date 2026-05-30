from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load both models
print("Loading models...")
leaf_model = tf.keras.models.load_model("crop_disease_model.keras")
fruit_model = tf.keras.models.load_model("fruit_disease_model.keras")
print("✅ Both models loaded!")

# Load fruit class names
with open("fruit_class_names.json", "r") as f:
    FRUIT_CLASSES = json.load(f)

# Leaf class names (38 classes)
LEAF_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

LEAF_DISEASE_INFO = {
    'Apple___Apple_scab': {'severity': 'Moderate', 'treatment': 'Apply fungicides containing myclobutanil or captan. Remove and destroy infected leaves.', 'prevention': 'Plant resistant varieties. Ensure good air circulation.'},
    'Apple___Black_rot': {'severity': 'High', 'treatment': 'Remove infected fruit and branches. Apply copper-based fungicide.', 'prevention': 'Prune dead wood. Keep orchard clean.'},
    'Apple___Cedar_apple_rust': {'severity': 'Moderate', 'treatment': 'Apply fungicides early in the season.', 'prevention': 'Plant resistant apple varieties.'},
    'Apple___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Continue regular monitoring.'},
    'Blueberry___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Maintain proper soil pH.'},
    'Cherry_(including_sour)___Powdery_mildew': {'severity': 'Moderate', 'treatment': 'Apply sulfur-based fungicide.', 'prevention': 'Ensure good air circulation.'},
    'Cherry_(including_sour)___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Regular pruning and monitoring.'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'severity': 'High', 'treatment': 'Apply foliar fungicide. Rotate crops.', 'prevention': 'Use resistant hybrids.'},
    'Corn_(maize)___Common_rust_': {'severity': 'Moderate', 'treatment': 'Apply fungicide if severe.', 'prevention': 'Plant early. Use rust-resistant hybrids.'},
    'Corn_(maize)___Northern_Leaf_Blight': {'severity': 'High', 'treatment': 'Apply triazole fungicide.', 'prevention': 'Use resistant varieties.'},
    'Corn_(maize)___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Maintain balanced fertilization.'},
    'Grape___Black_rot': {'severity': 'High', 'treatment': 'Apply mancozeb fungicide. Remove mummified fruit.', 'prevention': 'Prune for air circulation.'},
    'Grape___Esca_(Black_Measles)': {'severity': 'High', 'treatment': 'No cure. Remove infected vines.', 'prevention': 'Protect pruning wounds.'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'severity': 'Moderate', 'treatment': 'Apply copper-based fungicide.', 'prevention': 'Ensure good drainage.'},
    'Grape___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Regular pruning and monitoring.'},
    'Orange___Haunglongbing_(Citrus_greening)': {'severity': 'Critical', 'treatment': 'No cure. Remove infected trees.', 'prevention': 'Control Asian citrus psyllid vector.'},
    'Peach___Bacterial_spot': {'severity': 'Moderate', 'treatment': 'Apply copper-based bactericide.', 'prevention': 'Plant resistant varieties.'},
    'Peach___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Regular pruning and pest monitoring.'},
    'Pepper,_bell___Bacterial_spot': {'severity': 'Moderate', 'treatment': 'Apply copper hydroxide spray.', 'prevention': 'Use disease-free seed.'},
    'Pepper,_bell___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Crop rotation and balanced watering.'},
    'Potato___Early_blight': {'severity': 'Moderate', 'treatment': 'Apply mancozeb fungicide every 7-10 days.', 'prevention': 'Crop rotation. Remove plant debris.'},
    'Potato___Late_blight': {'severity': 'Critical', 'treatment': 'Apply metalaxyl fungicide immediately.', 'prevention': 'Use certified disease-free seed.'},
    'Potato___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Use certified seed potatoes.'},
    'Raspberry___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Prune old canes.'},
    'Soybean___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Maintain proper plant spacing.'},
    'Squash___Powdery_mildew': {'severity': 'Moderate', 'treatment': 'Apply potassium bicarbonate or neem oil.', 'prevention': 'Plant resistant varieties.'},
    'Strawberry___Leaf_scorch': {'severity': 'Moderate', 'treatment': 'Apply copper-based fungicide.', 'prevention': 'Avoid overhead irrigation.'},
    'Strawberry___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Regular renovation and mulching.'},
    'Tomato___Bacterial_spot': {'severity': 'Moderate', 'treatment': 'Apply copper bactericide.', 'prevention': 'Use disease-free seed.'},
    'Tomato___Early_blight': {'severity': 'Moderate', 'treatment': 'Apply chlorothalonil fungicide.', 'prevention': 'Mulch soil. Avoid overhead watering.'},
    'Tomato___Late_blight': {'severity': 'Critical', 'treatment': 'Apply metalaxyl fungicide immediately.', 'prevention': 'Avoid overhead irrigation.'},
    'Tomato___Leaf_Mold': {'severity': 'Moderate', 'treatment': 'Apply copper-based fungicide.', 'prevention': 'Reduce humidity.'},
    'Tomato___Septoria_leaf_spot': {'severity': 'Moderate', 'treatment': 'Apply mancozeb fungicide.', 'prevention': 'Avoid overhead watering.'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'severity': 'Moderate', 'treatment': 'Apply miticide or neem oil.', 'prevention': 'Avoid water stress.'},
    'Tomato___Target_Spot': {'severity': 'Moderate', 'treatment': 'Apply azoxystrobin fungicide.', 'prevention': 'Crop rotation.'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'severity': 'High', 'treatment': 'No cure. Control whitefly vectors.', 'prevention': 'Use resistant varieties.'},
    'Tomato___Tomato_mosaic_virus': {'severity': 'High', 'treatment': 'No cure. Remove infected plants.', 'prevention': 'Use virus-free seed.'},
    'Tomato___healthy': {'severity': 'None', 'treatment': 'No treatment needed.', 'prevention': 'Regular monitoring and proper watering.'}
}

def preprocess_image(file):
    img = Image.open(file.stream).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/leaf', methods=['POST'])
def predict_leaf():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    arr = preprocess_image(request.files['image'])
    predictions = leaf_model.predict(arr)[0]
    top3 = np.argsort(predictions)[-3:][::-1]
    results = []
    for i in top3:
        name = LEAF_CLASSES[i]
        info = LEAF_DISEASE_INFO.get(name, {'severity': 'Unknown', 'treatment': 'Consult an expert.', 'prevention': 'Monitor regularly.'})
        results.append({
            'name': name.replace('___', ' → ').replace('_', ' '),
            'confidence': round(float(predictions[i]) * 100, 2),
            'severity': info['severity'],
            'treatment': info['treatment'],
            'prevention': info['prevention']
        })
    return jsonify({'predictions': results})

@app.route('/predict/fruit', methods=['POST'])
def predict_fruit():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    arr = preprocess_image(request.files['image'])
    predictions = fruit_model.predict(arr)[0]
    top3 = np.argsort(predictions)[-3:][::-1]
    results = []
    for i in top3:
        name = FRUIT_CLASSES[i]
        is_healthy = 'Healthy' in name
        results.append({
            'name': name.replace('__', ' → ').replace('_', ' '),
            'confidence': round(float(predictions[i]) * 100, 2),
            'severity': 'None' if is_healthy else 'High',
            'treatment': 'No treatment needed. Fruit looks healthy!' if is_healthy else 'Remove rotten parts immediately. Store in cool dry place. Check surrounding fruits.',
            'prevention': 'Continue proper storage and handling.' if is_healthy else 'Store fruits properly. Maintain hygiene. Check regularly for early signs.'
        })
    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=True)