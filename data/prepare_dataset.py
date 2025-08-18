import os
import random
import shutil
from simple_image_download import simple_image_download as simp 

# ================================
# CONFIGURACIÓN
# ================================
BASE_DIR = "data"
CLASSES = ["manzana", "platano", "naranja", "uva"]   # frutas que quieres
IMAGES_PER_CLASS = 100                             # total por clase
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}  # 70/15/15

# ================================
# DESCARGAR IMÁGENES
# ================================
print("📥 Descargando imágenes...")

response = simp.Downloader()

for fruit in CLASSES:
    print(f"   🔎 Descargando {IMAGES_PER_CLASS} imágenes de {fruit}...")
    response.download(fruit + " fruit png", IMAGES_PER_CLASS)

print("✅ Descarga completa\n")

# ================================
# CREAR CARPETAS Y ORGANIZAR
# ================================
print("📂 Organizándolas en train/val/test...")

for split in SPLIT_RATIOS.keys():
    for fruit in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, split, fruit), exist_ok=True)

for fruit in CLASSES:
    # Carpeta donde se guardaron las imágenes crudas
    raw_dir = os.path.join("simple_images", fruit + " fruit png")
    if not os.path.exists(raw_dir):
        print(f"⚠️ No se encontraron imágenes para {fruit}, revisa descarga.")
        continue

    all_images = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train+n_val],
        "test": all_images[n_train+n_val:]
    }

    # Copiar a destino
    for split, files in splits.items():
        for f in files:
            dest = os.path.join(BASE_DIR, split, fruit, os.path.basename(f))
            shutil.copy(f, dest)

print("✅ Organización completa\n")

# ================================
# VERIFICACIÓN DEL DATASET
# ================================
print("📊 Verificación del dataset:\n")

for split in SPLIT_RATIOS.keys():
    split_dir = os.path.join(BASE_DIR, split)
    print(f"📂 {split.upper()}:")
    total_split = 0
    for fruit in CLASSES:
        fruit_dir = os.path.join(split_dir, fruit)
        count = len([f for f in os.listdir(fruit_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        print(f"   🍎 {fruit}: {count} imágenes")
        total_split += count
    print(f"   ➡️ Total en {split}: {total_split} imágenes\n")

print("🎉 Dataset listo para entrenamiento.")
