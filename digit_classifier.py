import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib

def prepare_data():
    # Load and preprocess MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize and reshape images
    train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0
    
    # One-hot encode the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return train_images, train_labels, test_images, test_labels

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_save_model():
    # Prepare data
    train_images, train_labels, test_images, test_labels = prepare_data()
    
    # Create and train model
    model = create_model()
    history = model.fit(train_images, train_labels, 
                        epochs=10, 
                        validation_split=0.2, 
                        batch_size=64)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    
    # Save the model
    model.save('mnist_classifier.h5')
    
    return model, history

# Run training and save model
if __name__ == '__main__':
    model, history = train_and_save_model()