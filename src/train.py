import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from models import build_model_a, build_model_b, build_model_c

def plot_history(history, save_dir):
    """Guardar curvas de accuracy y loss"""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    
    plt.subplot(1,2,2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend()
    plt.title("Loss")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=300)
    plt.close()

def train_model(data_dir, model_type="a", batch_size=16, epochs=20, img_size=(100,100)):
    # -------------------------------
    # Generadores con augmentación
    # -------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_flow = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_flow = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_flow.class_indices)

    # -------------------------------
    # Guardar class_indices.json
    # -------------------------------
    save_dir = os.path.join(os.path.dirname(__file__), "../saved_models")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "class_indices.json"), "w") as f:
        json.dump(train_flow.class_indices, f, indent=4)

    print(f"✅ class_indices.json guardado en {save_dir}")

    # -------------------------------
    # Seleccionar modelo
    # -------------------------------
    if model_type == "a":
        model = build_model_a(input_shape=img_size+(3,), num_classes=num_classes)
    elif model_type == "b":
        model = build_model_b(input_shape=img_size+(3,), num_classes=num_classes)
    elif model_type == "c":
        model = build_model_c(input_shape=img_size+(3,), num_classes=num_classes)
    else:
        raise ValueError("Modelo no válido. Usa: a, b o c.")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # -------------------------------
    # Callbacks
    # -------------------------------
    checkpoint_path = os.path.join(save_dir, "best_model.h5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    # -------------------------------
    # Entrenamiento
    # -------------------------------
    history = model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=epochs,
        callbacks=callbacks
    )

    # -------------------------------
    # Guardar curvas
    # -------------------------------
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_history(history, plots_dir)
    print(f"✅ Curvas de entrenamiento guardadas en {plots_dir}/training_curves.png")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--model", type=str, default="a", help="a, b o c")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_model(args.data, model_type=args.model, batch_size=args.batch, epochs=args.epochs)
