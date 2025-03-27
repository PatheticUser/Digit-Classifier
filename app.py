import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def load_model():
    try:
        return tf.keras.models.load_model("mnist_classifier.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Invert colors (MNIST dataset has white digits on black background)
    image = ImageOps.invert(image)

    # Convert to numpy array and normalize
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    return image_array


def predict_digit(model, image_array):
    if model is None:
        return None, None

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence


def main():
    # Set page config with custom favicon and title
    st.set_page_config(
        page_title="Digit Recognizer",
        page_icon="https://i.imgur.com/6NvUVst.png",
        layout="centered",
    )

    # Custom CSS for styling
    st.markdown(
        """
    <style>
    .main {
        background-color: white;
        color: black;
    }
    .stButton>button {
        background-color: #FFD700;
        color: black;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFC107;
        transform: scale(1.05);
    }
    h1 {
        color: black;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Title
    st.title("Digit Recognizer")

    # Load pre-trained model
    model = load_model()

    if model is None:
        st.warning(
            "Please train the model first by running the mnist_classifier.py script."
        )
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Preprocess and predict
        processed_image = preprocess_image(image)

        # Prediction button with cool hover effect
        if st.button("Predict Digit"):
            # Make prediction
            predicted_class, confidence = predict_digit(model, processed_image)

            if predicted_class is not None:
                # Display results with animation
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Predicted Digit", value=predicted_class)

                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")

                # Visualization of prediction probabilities
                st.subheader("Prediction Probabilities")
                predictions = model.predict(processed_image)[0]
                fig, ax = plt.subplots()
                ax.bar(range(10), predictions, color="#FFD700")
                ax.set_xlabel("Digits")
                ax.set_ylabel("Probability")
                ax.set_title("Digit Prediction Probabilities")
                ax.set_xticks(range(10))
                st.pyplot(fig)


if __name__ == "__main__":
    main()
