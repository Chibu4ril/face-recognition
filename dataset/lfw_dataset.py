import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess import load_and_preprocess_image

def load_dataset(data_dir, min_images_per_person=1):
    images, labels = [], []
    label_dict = {}
    label_id = 0

    # Traverse through all subdirectories (nested structure)
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        
        # Ensure it's a directory and not a file
        if not os.path.isdir(person_dir):
            continue

        # We need to handle potential nested directories
        # Check if the directory contains enough images
        image_files = []
        for root, _, files in os.walk(person_dir):  # os.walk() to go through subfolders
            for file in files:
                if file.endswith('.jpg'):  # You can modify this for other image formats
                    image_files.append(os.path.join(root, file))

        # Skip if there are fewer images than the minimum required
        if len(image_files) < min_images_per_person:
            continue

        # Add person name to label_dict if not already there
        if person_name not in label_dict:
            label_dict[person_name] = label_id
            label_id += 1

        # Load and preprocess images
        for img_path in image_files:
            img = load_and_preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(label_dict[person_name])

    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training and test sets
    return train_test_split(images, labels, test_size=0.2, random_state=42), label_dict
