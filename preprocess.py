# import cv2

# def load_and_preprocess_image(image_path, target_size=(100, 100)):
#     image = cv2.imread(image_path)
#     if image is None:
#         return None
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, target_size)
#     image = image / 255.0  # Normalize
#     return image


# Face Recognition for Class Attendance using OpenCV + CNN (LFW Dataset, Recursive Loading)

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# --- Configuration ---
dataset_path = 'data/lfw-deepfunneled'  # Update to your dataset root folder
img_size = (100, 100)
attendance_file = 'attendance.csv'
model_file = 'face_recognition_model.h5'

# --- Recursive Data Loading and Preprocessing ---
data = []
labels = []
print("Loading images recursively...")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            person_name = os.path.basename(os.path.dirname(img_path))  # Use folder name as label
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, img_size)
            data.append(img)
            labels.append(person_name)

data = np.array(data).reshape(-1, img_size[0], img_size[1], 1) / 255.0
le = LabelEncoder()
labels_enc = le.fit_transform(labels)
labels_cat = to_categorical(labels_enc)

print(f"Total samples: {len(data)} | Classes: {len(le.classes_)}")

# --- Data Visualization ---
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(data[i].reshape(img_size), cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.suptitle('Sample Faces')
plt.show()

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(data, labels_cat, test_size=0.2, random_state=42)

# --- Model Definition ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# --- Training ---
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save(model_file)

# --- Plot Training History ---
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()

# --- Evaluation Script ---
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

unique_labels = np.unique(np.concatenate([y_true, y_pred]))
unique_class_names = le.inverse_transform(unique_labels)

print("Classification Report:")
print(classification_report(y_true, y_pred, labels=unique_labels, target_names=unique_class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- Initialize Attendance File ---
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Present'])
    df.to_csv(attendance_file, index=False)

# --- Real-time Attendance ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print("Starting real-time attendance... Press 'q' to quit.")
marked_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, img_size).reshape(1, img_size[0], img_size[1], 1) / 255.0
        pred = model.predict(face_img)
        class_idx = np.argmax(pred)
        confidence = np.max(pred)
        name = le.inverse_transform([class_idx])[0]

        if confidence > 0.7 and name not in marked_students:
            date_str = datetime.now().strftime('%Y-%m-%d')
            df = pd.read_csv(attendance_file)
            df.loc[len(df)] = {'Name': name, 'Date': date_str, 'Present': 'Yes'}
            df.to_csv(attendance_file, index=False)
            marked_students.add(name)
            print(f"Marked {name} as present.")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- Mark Absentees ---
date_str = datetime.now().strftime('%Y-%m-%d')
all_students = set(le.classes_)
present_students = marked_students
absent_students = all_students - present_students

if absent_students:
    df = pd.read_csv(attendance_file)
    for student in absent_students:
        df.loc[len(df)] = {'Name': student, 'Date': date_str, 'Present': 'No'}
    df.to_csv(attendance_file, index=False)
    print(f"Marked absent students: {', '.join(absent_students)}")