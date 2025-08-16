import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from utils import preprocess_image

# Cargar modelo
model = tf.keras.models.load_model("saved_models/best_model")

# Cargar class_indices
with open("saved_models/class_indices.json", "r") as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

st.title("🍎🍌🍊 Clasificador de Frutas")
st.write("Sube una imagen y el modelo intentará adivinar qué fruta es.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesar
    img_array = preprocess_image(image, target_size=(160,160))  # tamaño máximo

    # Predicción
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.write(f"🔮 Predicción: **{classes[pred_class]}** con confianza {confidence:.2f}")
