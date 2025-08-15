import io, os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = "../saved_models/best_model"  # ajustar
IMG_SIZE = (160,160)  # ajustar al modelo

model = tf.keras.models.load_model(MODEL_PATH)
# Cargar mapeo de labels (exporta class_indices desde entrenamiento)
import json
with open('../saved_models/class_indices.json','r') as f:
    class_indices = json.load(f)
inv_map = {v:k for k,v in class_indices.items()}

def prepare_image(image_bytes, target_size):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'no file'}), 400
    file = request.files['file']
    img_bytes = file.read()
    x = prepare_image(img_bytes, IMG_SIZE)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    label = inv_map[idx]
    prob = float(np.max(preds))
    return jsonify({'label': label, 'probability': round(prob,4)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
