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
    # Set page config with custom favicon and title
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="🔢",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for advanced styling
    st.markdown(
        """
    <style>
    /* Body and container styles */
    body {
        background-color: #f4f4f4;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header styles */
    .stMarkdown h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 600;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card-like container for file uploader and prediction */
    .stContainer {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Button styles */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric card styles */
    .stMetric {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    .stMetric > div {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .stMetric div:first-child {
        color: #7f8c8d;
        margin-bottom: 5px;
    }
    
    .stMetric div:last-child {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Title with gradient
    st.markdown("<h1>MNIST Digit Recognizer</h1>", unsafe_allow_html=True)

    # Load pre-trained model
    model = load_model()

    if model is None:
        st.warning(
            "Please train the model first by running the mnist_classifier.py script."
        )
        return

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)

        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "Upload a handwritten digit image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear, high-contrast image of a handwritten digit",
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
                    # Display metrics in a visually appealing way
                    st.markdown("### Prediction Results")

                    col_pred, col_conf = st.columns(2)
                    with col_pred:
                        st.metric(label="Predicted Digit", value=predicted_class)

                    with col_conf:
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Predictions visualization
        if uploaded_file is not None:
            st.markdown('<div class="stContainer">', unsafe_allow_html=True)

            st.markdown("### Prediction Probabilities")

            # Only show probability graph if a prediction was made
            if "processed_image" in locals():
                predictions = model.predict(processed_image)[0]

                # Create a more visually appealing matplotlib figure
                plt.figure(figsize=(8, 4), facecolor="#ecf0f1")
                plt.bar(range(10), predictions, color="#3498db", alpha=0.7)
                plt.title(
                    "Digit Prediction Probabilities", fontsize=15, color="#2c3e50"
                )
                plt.xlabel("Digits", fontsize=12)
                plt.ylabel("Probability", fontsize=12)
                plt.xticks(range(10))
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # Save plot to a buffer to improve rendering
                buf = io.BytesIO()
                plt.savefig(
                    buf, format="png", bbox_inches="tight", dpi=300, facecolor="#ecf0f1"
                )
                buf.seek(0)

                # Display the plot
                st.image(buf)
                plt.close()

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
