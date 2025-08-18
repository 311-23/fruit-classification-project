import os
import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_any_model(model_path):
    """Carga un modelo sin importar si es .keras, .h5 o carpeta SavedModel"""
    if os.path.isdir(model_path):  # Carpeta (SavedModel)
        print(f"📂 Cargando modelo desde carpeta: {model_path}")
        return tf.keras.models.load_model(model_path)
    elif model_path.endswith(".keras") or model_path.endswith(".h5"):  # Archivo Keras o H5
        print(f"📄 Cargando modelo desde archivo: {model_path}")
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"❌ Formato no soportado: {model_path}. Usa .keras, .h5 o carpeta SavedModel.")

def evaluate_model(model_path, data_dir, class_indices_path, batch_size=2):
    # Cargar modelo
    model = load_any_model(model_path)
    print("✅ Modelo cargado.")

    # Cargar class_indices
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
    labels = list(class_indices.keys())

    # Generador de test
    test_gen = ImageDataGenerator(rescale=1./255)
    test_flow = test_gen.flow_from_directory(
        data_dir,
        target_size=(100, 100),   # usa el tamaño mayor para seguridad
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluar
    loss, acc = model.evaluate(test_flow, verbose=1)
    print(f"📊 Test accuracy = {acc:.4f}, Test loss = {loss:.4f}")

    # Predicciones
    y_true = test_flow.classes
    y_pred = np.argmax(model.predict(test_flow), axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - Test")
    os.makedirs("../saved_models/plots", exist_ok=True)
    plt.savefig("../saved_models/plots/matriz_confusion_test.png", dpi=300)
    plt.close()
    print("✅ Matriz de confusión guardada en ../saved_models/plots/")

    print("\n📌 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../saved_models/best_model.h5",)
    parser.add_argument("--data", type=str, default="../data/test")
    parser.add_argument("--classes", type=str, default="../saved_models/class_indices.json")
    args = parser.parse_args()

    evaluate_model(args.model, args.data, args.classes)
    print("✅ Evaluación completada.")