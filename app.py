import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Load and cache the trained model to improve efficiency
@st.cache_resource
def load_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, verbose=0)  # Train without console logs
    return model

model = load_model()

# Function to preprocess and classify the digit
def classify_digit(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (28, 28))  # Resize to MNIST format
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28)  # Reshape for model input
    
    prediction = model.predict(img)[0]
    return prediction

# Streamlit UI
st.title("🖌️ Handwritten Digit Classifier")
st.write("Upload an image of a digit (0-9) to classify it.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Read image

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Digit"):
        prediction = classify_digit(image)
        
        # Display the results
        st.subheader("🔢 Prediction")
        predicted_digit = np.argmax(prediction)
        st.write(f"**Predicted Digit:** {predicted_digit}")
        
        # Show prediction confidence
        st.bar_chart(prediction)
