import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

from models import build_model_c

def finetune_model(data_dir, batch_size=16, epochs=10, img_size=(100,100)):
    # -------------------------------
    # Generadores
    # -------------------------------
    datagen = ImageDataGenerator(
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

    train_flow = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_flow = datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    num_classes = len(train_flow.class_indices)

    # -------------------------------
    # Cargar modelo base MobileNetV2
    # -------------------------------
    model = build_model_c(input_shape=img_size+(3,), num_classes=num_classes)

    # Descongelar últimas capas para fine-tuning
    base_model = model.layers[1]  # MobileNetV2
    base_model.trainable = True

    # 🔑 Descongelar solo últimas 30 capas
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # LR más bajo
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # -------------------------------
    # Callbacks
    # -------------------------------
    save_dir = os.path.join(os.path.dirname(__file__), "../saved_models")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, "best_model_finetuned.h5")
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

    print(f"✅ Modelo fine-tuned guardado en {checkpoint_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    finetune_model(args.data, batch_size=args.batch, epochs=args.epochs)
