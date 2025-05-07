import sys
import numpy as np
import tensorflow as tf
from utils.preprocess import load_and_preprocess_image
from dataset.lfw_dataset import load_dataset

model = tf.keras.models.load_model('output/model-output/face_cnn_model.h5')
_, _, _, _, label_dict = load_dataset('data/lfw-deepfunneled')
id_to_label = {v: k for k, v in label_dict.items()}

def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    if img is None:
        print("Error loading image.")
        return
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    predicted_id = np.argmax(pred)
    print(f"Predicted person: {id_to_label.get(predicted_id, 'Unknown')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict.py path_to_image")
    else:
        predict_image(sys.argv[1])
