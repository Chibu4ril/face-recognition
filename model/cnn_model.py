from tensorflow import keras
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Add extra dropout for regularization

        Dense(num_classes, activation='softmax') if num_classes > 1 else Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    return model

