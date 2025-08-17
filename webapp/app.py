import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from utils import preprocess_image
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(
    page_title="Clasificador de Frutas Futurista",
    page_icon="🍎",
    layout="centered",
)

# ==============================
# CSS PERSONALIZADO (Tema Futurista)
# ==============================
st.markdown(
    """
    <style>
    body {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0d0d, #1a1a1a);
        color: #ffffff;
        font-family: 'Trebuchet MS', sans-serif;
    }
    h1, h2, h3 {
        text-align: center;
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc;
    }
    .stFileUploader {
        border: 2px dashed #00ffcc;
        padding: 20px;
        border-radius: 15px;
        background-color: #1a1a1a;
    }
    .stButton>button {
        background: #00ffcc;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #ff00ff;
        color: white;
        box-shadow: 0px 0px 15px #ff00ff;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #999999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Cargar modelo y clases
# ==============================
model = tf.keras.models.load_model("../saved_models/best_model.keras")

with open("../saved_models/class_indices.json", "r") as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

# ==============================
# Interfaz
# ==============================
st.title("🍎🍌🍊 Clasificador de Frutas")
st.markdown("Sube una imagen y nuestra **IA con redes neuronales convolucionales (CNN)** intentará adivinar qué fruta es.")

st.markdown("### 📖 ¿Cómo funciona?")
st.markdown("""
1. 🔍 **Preprocesamiento:** La imagen se redimensiona y normaliza.  
2. 🧠 **Predicción CNN:** El modelo analiza patrones como colores, texturas y bordes.  
3. 🎯 **Clasificación:** Se asigna la clase con la mayor probabilidad.  
""")

uploaded_file = st.file_uploader("📤 Sube una imagen de una fruta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Imagen subida", use_column_width=True)

    # Preprocesar
    img_array = preprocess_image(image, target_size=(160,160))

    # Predicción
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.markdown("### 🔮 Resultado de la predicción")
    st.write(f"**Fruta detectada:** {classes[pred_class]}")
    st.progress(float(confidence))  # Barra de confianza
    st.write(f"Confianza: **{confidence:.2f}**")

# ==============================
# Footer
# ==============================
st.markdown(
    """
    <div class="footer">
        Proyecto de Inteligencia Artificial 🤖 | Desarrollado por <b>Brayan and Ramiro</b> <br>
        Universidad Politecnica del estado de Nayarit - 2025
    </div>
    """,
    unsafe_allow_html=True,
)
