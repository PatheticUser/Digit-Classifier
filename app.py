import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained CNN model
model = tf.keras.models.load_model("cnn_digit_classifier.h5")

# UI customization
st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="centered")

# Custom CSS for UI (Minimal White, Yellow, Black Theme)
st.markdown("""
    <style>
        body {background-color: white; color: black;}
        .stApp { background-color: white; }
        .stButton>button { background-color: black; color: white; border-radius: 5px; }
        .stFileUploader { background-color: #F7C200; }
    </style>
""", unsafe_allow_html=True)

st.title("🔢 Digit Classifier (CNN Model)")
st.markdown("### Upload a hand-written digit image (28x28 pixels)")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert to grayscale and resize
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # CNN expects 4D input

    # Show Image with fixed size
    st.image(img, caption="Uploaded Image", width=200)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_digit = np.argmax(prediction)

    # Display results
    st.markdown(f"### **Predicted Digit: `{predicted_digit}`**")

    # Show confidence scores
    st.bar_chart(prediction)
