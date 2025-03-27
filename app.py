import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io


def load_model():
    """Load the pre-trained MNIST classifier model."""
    try:
        return tf.keras.models.load_model("mnist_classifier.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image):
    """Preprocess the uploaded image for model prediction."""
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
    """Predict the digit and confidence."""
    if model is None:
        return None, None

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence


def main():
    # Set page config with original favicon and title
    st.set_page_config(
        page_title="Digit Recognizer",
        page_icon="https://i.imgur.com/6NvUVst.png",
        layout="centered",
    )

    # Minimal custom CSS
    st.markdown(
        """
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stButton>button {
        background-color: #FFD700;
        color: black;
        border: 2px solid black;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: black;
        color: #FFD700;
        transform: scale(1.05);
    }
    
    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        border: 2px dashed #FFD700;
        background-color: #FFFAF0;
    }
    
    .stContainer, .stMetric {
        background-color: #F5F5F5;
        border: 1px solid #FFD700;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .stMetric div:first-child {
        color: #666;
    }
    
    .stMetric div:last-child {
        color: black;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Original title
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
        "Upload a handwritten digit image",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a handwritten digit",
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)

        # Prediction button
        if st.button("Predict Digit"):
            # Preprocess and predict
            processed_image = preprocess_image(image)
            predicted_class, confidence = predict_digit(model, processed_image)

            if predicted_class is not None:
                # Display metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(label="Predicted Digit", value=predicted_class)

                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2f}%")

                # Predictions visualization
                st.subheader("Prediction Probabilities")
                predictions = model.predict(processed_image)[0]

                # Create matplotlib figure
                plt.figure(figsize=(8, 4), facecolor="#F5F5F5")
                plt.bar(range(10), predictions, color="#FFD700", edgecolor="black")
                plt.title("Digit Prediction Probabilities", fontsize=12)
                plt.xlabel("Digits")
                plt.ylabel("Probability")
                plt.xticks(range(10))
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # Save plot to a buffer
                buf = io.BytesIO()
                plt.savefig(
                    buf, format="png", bbox_inches="tight", dpi=300, facecolor="#F5F5F5"
                )
                buf.seek(0)

                # Display the plot
                st.image(buf)
                plt.close()


if __name__ == "__main__":
    main()
