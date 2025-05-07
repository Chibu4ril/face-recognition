import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from dataset.lfw_dataset import load_dataset

(data_train, data_test, labels_train, labels_test), label_dict = load_dataset('data/lfw-deepfunneled')
model = load_model('output/model-output/face_cnn_model.h5')

y_pred_probs = model.predict(data_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")
print(classification_report(labels_test, y_pred, target_names=list(label_dict.keys())))

# Confusion Matrix
cm = confusion_matrix(labels_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
