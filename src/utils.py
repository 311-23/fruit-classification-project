from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(100,100), model_name="default"):
    """
    Preprocesa una imagen para un modelo de Keras.

    Args:
        image (PIL.Image): Imagen RGB.
        target_size (tuple): Tamaño (alto, ancho) esperado por el modelo.
        model_name (str): Nombre del modelo si requiere preprocess_input especial.
            Por ejemplo: "mobilenet", "vgg16", etc.
    Returns:
        np.array: Imagen lista para predecir, shape=(1, H, W, 3)
    """
    # Redimensionar
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32")

    # Normalizar según el modelo
    if model_name.lower() == "mobilenet":
        img_array = mobilenet_preprocess(img_array)
    else:
        img_array /= 255.0

    # Añadir dimensión batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
