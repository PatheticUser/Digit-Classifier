import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np


def prepare_data():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Reshape and normalize
    train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
    test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

    # One-hot encode labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def create_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_and_save_model():
    # Prepare data
    train_images, train_labels, test_images, test_labels = prepare_data()

    # Create and train model
    model = create_model()
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_split=0.2,
        batch_size=64,
        verbose=1,
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Save the model
    model.save("mnist_classifier.h5")

    return model


# Run training and save model
if __name__ == "__main__":
    train_and_save_model()
