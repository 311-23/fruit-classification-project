# src/prepare_dataset_final.py

import os
from pathlib import Path
import random
from PIL import Image

# Frutas deseadas
FRUITS = ["Banana", "Orange", "Apple", "Strawberry", "Pineapple", "Grapes", "Cherries", "Mango", "Watermelon"]

# Mapear al nombre exacto de la carpeta en Fruits-360
FRUIT_FOLDER_MAP = {
    "Banana": "Banana",
    "Orange": "Orange",
    "Apple": "Apple Red 1", 
    "Strawberry": "Strawberry",
    "Pineapple": "Pineapple",
    "Grapes": "Grape White",  
    "Cherries": "Cherry Wax Red",     
    "Mango": "Mango",
    "Watermelon": "Watermelon"
}

# Directorios del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Ruta del dataset original Fruits-360 (modifica si está en otra ubicación)
EXTRACT_DIR = BASE_DIR / "Fruit-Images-Dataset-master"
SRC_TRAIN = EXTRACT_DIR / "Training"
SRC_TEST = EXTRACT_DIR / "Test"

# Configuración
TOTAL_IMAGES = 180  # total aproximado
IMG_SIZE = (100, 100)

def create_folders():
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
        for fruit in FRUITS:
            (folder / fruit).mkdir(exist_ok=True)

def copy_and_resize_images():
    images_per_fruit = TOTAL_IMAGES // len(FRUITS)

    for fruit in FRUITS:
        folder_name = FRUIT_FOLDER_MAP[fruit]
        src_folder_train = SRC_TRAIN / folder_name
        src_folder_test = SRC_TEST / folder_name

        if src_folder_train.exists():
            src_folder = src_folder_train
        elif src_folder_test.exists():
            src_folder = src_folder_test
        else:
            print(f"[AVISO] No se encontraron imágenes para {fruit}. Carpeta creada vacía.")
            continue

        images = list(src_folder.glob("*.jpg"))
        if not images:
            print(f"[AVISO] No se encontraron imágenes .jpg en {src_folder} para {fruit}")
            continue

        random.shuffle(images)
        selected_images = images[:images_per_fruit]

        n_train = int(0.7 * len(selected_images))
        n_val = int(0.15 * len(selected_images))
        n_test = len(selected_images) - n_train - n_val

        # Función para copiar y redimensionar
        def resize_and_save(img_path, dest_folder):
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(dest_folder / img_path.name)

        for img_path in selected_images[:n_train]:
            resize_and_save(img_path, TRAIN_DIR / fruit)
        for img_path in selected_images[n_train:n_train+n_val]:
            resize_and_save(img_path, VAL_DIR / fruit)
        for img_path in selected_images[n_train+n_val:]:
            resize_and_save(img_path, TEST_DIR / fruit)

        print(f"[INFO] {fruit}: {len(selected_images)} imágenes seleccionadas -> "
              f"{n_train} train, {n_val} val, {n_test} test")

def main():
    create_folders()
    copy_and_resize_images()
    print(f"[LISTO] Preparación de dataset completada. Revisa las carpetas en {DATA_DIR}")

if __name__ == "__main__":
    main()
