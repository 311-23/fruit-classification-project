import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from models import build_model_a, build_model_b, build_model_c

# -------------------
# Configuración
# -------------------
DATA_DIR = '../data'
IMG_SIZE_A = (128, 128)
IMG_SIZE_C = (160, 160)
BATCH = 32
EPOCHS = 30

# -------------------
# Generadores de datos con aumentos
# -------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.12,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.15
)

train_flow = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE_A,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training'
)
val_flow = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE_A,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_flow.class_indices)
print("Clases:", train_flow.class_indices)

# -------------------
# Función para entrenar y devolver mejor val_accuracy + history
# -------------------
def compile_and_train(model, target_size, name):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    es = EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss')
    rl = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)

    if target_size != IMG_SIZE_A:
        flow = train_gen.flow_from_directory(
            os.path.join(DATA_DIR, 'train'),
            target_size=target_size,
            batch_size=BATCH,
            class_mode='categorical',
            subset='training'
        )
        vflow = train_gen.flow_from_directory(
            os.path.join(DATA_DIR, 'train'),
            target_size=target_size,
            batch_size=BATCH,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
    else:
        flow, vflow = train_flow, val_flow

    history = model.fit(
        flow,
        epochs=EPOCHS,
        validation_data=vflow,
        callbacks=[es, rl]
    )

    best_val_acc = max(history.history['val_accuracy'])
    return model, best_val_acc, history

# -------------------
# Entrenar modelos
# -------------------
print("Entrenando modelo A...")
m_a, acc_a, h_a = compile_and_train(build_model_a((IMG_SIZE_A[0], IMG_SIZE_A[1], 3), num_classes), IMG_SIZE_A, 'model_a')

print("Entrenando modelo B...")
m_b, acc_b, h_b = compile_and_train(build_model_b((IMG_SIZE_A[0], IMG_SIZE_A[1], 3), num_classes), IMG_SIZE_A, 'model_b')

print("Entrenando modelo C (fase 1)...")
m_c, acc_c, h_c = compile_and_train(build_model_c((IMG_SIZE_C[0], IMG_SIZE_C[1], 3), num_classes), IMG_SIZE_C, 'model_c')

# Fine-tuning modelo C
m_c.layers[1].trainable = True
for layer in m_c.layers[1].layers[:-25]:
    layer.trainable = False
m_c.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
ft_flow = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE_C,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training'
)
ft_vflow = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE_C,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
history_ft = m_c.fit(ft_flow, epochs=10, validation_data=ft_vflow)
acc_c_ft = max(history_ft.history['val_accuracy'])
if acc_c_ft > acc_c:
    acc_c = acc_c_ft
    # concatenar histories
    for k in h_c.history.keys():
        h_c.history[k] += history_ft.history[k]

# -------------------
# Crear carpeta para gráficas
# -------------------
os.makedirs('../saved_models/plots', exist_ok=True)

def plot_history(histories, labels, metric, filename):
    plt.figure(figsize=(8,6))
    for hist, label in zip(histories, labels):
        plt.plot(hist.history[metric], label=f"{label} train")
        plt.plot(hist.history[f"val_{metric}"], linestyle='--', label=f"{label} val")
    plt.title(f"Comparación de {metric}")
    plt.xlabel("Épocas")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../saved_models/plots/{filename}", dpi=300)
    plt.close()

plot_history([h_a, h_b, h_c], ["Modelo A", "Modelo B", "Modelo C"], "accuracy", "comparacion_accuracy.png")
plot_history([h_a, h_b, h_c], ["Modelo A", "Modelo B", "Modelo C"], "loss", "comparacion_loss.png")

print("📊 Gráficos guardados en ../saved_models/plots/")

# -------------------
# Selección automática del mejor modelo
# -------------------
scores = {
    "model_a": acc_a,
    "model_b": acc_b,
    "model_c": acc_c
}
best_name = max(scores, key=scores.get)
print(f"Mejor modelo: {best_name} con val_accuracy = {scores[best_name]:.4f}")

if best_name == "model_a":
    best_model = m_a
    best_img_size = IMG_SIZE_A
elif best_name == "model_b":
    best_model = m_b
    best_img_size = IMG_SIZE_A
else:
    best_model = m_c
    best_img_size = IMG_SIZE_C

# -------------------
# Guardar modelo y class_indices
# -------------------
os.makedirs('../saved_models', exist_ok=True)
tf.keras.models.save_model(best_model, '../saved_models/best_model', save_format='tf')

with open('../saved_models/class_indices.json', 'w') as f:
    json.dump(train_flow.class_indices, f)

print("✅ Modelo y class_indices guardados en ../saved_models/")

# -------------------
# Matriz de confusión del mejor modelo
# -------------------
test_gen = ImageDataGenerator(rescale=1./255)
test_flow = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=best_img_size,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

y_true = test_flow.classes
y_pred = np.argmax(best_model.predict(test_flow), axis=1)
labels = list(test_flow.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de Confusión - {best_name}")
plt.savefig("../saved_models/plots/matriz_confusion.png", dpi=300)
plt.close()

print("📌 Matriz de confusión guardada en ../saved_models/plots/matriz_confusion.png")
print(classification_report(y_true, y_pred, target_names=labels))
