import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import load_and_preprocess_image

def load_dataset(data_dir, min_images_per_person=10):
    images, labels = [], []
    label_dict = {}
    label_id = 0

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if len(os.listdir(person_dir)) < min_images_per_person:
            continue

        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(label_dict[person_name])

    images = np.array(images)
    labels = np.array(labels)

    return train_test_split(images, labels, test_size=0.2, random_state=42), label_dict
