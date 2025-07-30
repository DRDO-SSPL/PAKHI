import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_cnn_model(X_train, y_train, input_shape=None, num_classes=None, epochs=10):
    if input_shape is None:
        input_shape = X_train.shape[1:]  # e.g. (28, 28, 1)
    if num_classes is None:
        num_classes = len(set(y_train))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 Training CNN Model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    
    return model
