import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def evaluate_model(model_path, data_dir='../data/test', img_size=(128,128), batch=32):
    model = tf.keras.models.load_model(model_path)
    gen = ImageDataGenerator(rescale=1./255)
    test_flow = gen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch, class_mode='categorical', shuffle=False)
    preds = model.predict(test_flow, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_flow.classes
    labels = list(test_flow.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.show()
