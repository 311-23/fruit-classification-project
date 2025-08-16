import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(160,160)):
    """
    Convierte una imagen en un array listo para predecir con el modelo.
    """
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0  # normalizar
    image_array = np.expand_dims(image_array, axis=0)  # batch de 1
    return image_array
