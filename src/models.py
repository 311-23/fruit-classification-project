import tensorflow as tf
from tensorflow.keras import layers, models, applications

def build_model_a(input_shape=(128,128,3), num_classes=3):
    # Modelo A: CNN simple (baseline)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inputs, outputs, name='model_a')

def build_model_b(input_shape=(128,128,3), num_classes=3):
    # Modelo B: deeper + batchnorm + dropout
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256,activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return models.Model(inputs, outputs, name='model_b')

def build_model_c(input_shape=(160,160,3), num_classes=3):
    # Modelo C: Transfer learning (MobileNetV2)
    base = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False  # primera fase: congelado
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs, name='model_c')
    return model
