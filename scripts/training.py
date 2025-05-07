from tensorflow import keras
from dataset.lfw_dataset import load_dataset
from model.cnn_model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# --- Load dataset ---
(data_train, data_test, labels_train, labels_test), label_dict = load_dataset('data/lfw-deepfunneled')

input_shape = data_train.shape[1:]
num_classes = len(set(labels_train))

print(f"Input shape: {input_shape}, Number of classes: {num_classes}")

# --- Build CNN model ---
model = build_cnn_model(input_shape, num_classes)
model.summary()

# --- Setup callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True  # Restores best weights, avoids overfitting
)

checkpoint_path = 'output/model-output/best_face_cnn_model.keras'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

model_checkpoint = ModelCheckpoint(
    checkpoint_path, 
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)

# --- Train the model ---
history = model.fit(
    data_train, labels_train,
    epochs=10,
    batch_size=32,
    validation_data=(data_test, labels_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# --- Save final model (legacy HDF5 + recommended Keras format) ---
final_h5_path = 'output/model-output/face_cnn_model.h5'
final_keras_path = 'output/model-output/face_cnn_model.keras'

model.save(final_h5_path)
model.save(final_keras_path)

print(f"Model saved at {final_h5_path} and {final_keras_path}")
