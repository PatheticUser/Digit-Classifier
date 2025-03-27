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
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .stButton>button {
            background-color: #FFD700;
            color: black;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FFC107;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        h1 {
            color: black;
            text-align: center;
            font-weight: 800;
            padding: 20px 0;
            border-bottom: 3px solid #FFD700;
            margin-bottom: 30px;
        }
        h2, h3 {
            color: black;
            border-left: 4px solid #FFD700;
            padding-left: 10px;
            margin: 20px 0;
        }
        .stFileUploader {
            padding: 20px;
            border: 2px dashed #FFD700;
            border-radius: 10px;
            margin: 20px 0;
        }
        .css-1v3fvcr {
            background-color: white;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 5px;
            background-color: #f8f8f8;
        }
        .metric-card {
            background-color: #f8f8f8;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            text-align: center;
            margin: 10px 0;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: black;
        }
        .metric-label {
            font-size: 16px;
            color: #555;
            margin-bottom: 5px;
        }
        .prediction-container {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header with logo
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("✏️ Digit Recognizer")
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 30px;">
                <p>Upload a handwritten digit image and let AI recognize it!</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Load pre-trained model
    with st.spinner("Loading model..."):
        model = load_model()

    if model is None:
        st.warning(
            "⚠️ Please train the model first by running the mnist_classifier.py script."
        )
        return

    # Create a card-like container for the upload section
    st.markdown(
        """
        <div style="background-color: #f8f8f8; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="margin-top: 0;">Upload Your Image</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a handwritten digit image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Create columns for better layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)

        with col2:
            st.markdown(
                """
                <div style="height: 50px;"></div>
                <div style="background-color: #f8f8f8; padding: 20px; border-radius: 10px;">
                    <p>Your image has been uploaded successfully! Click the button below to analyze it.</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Preprocess image
            processed_image = preprocess_image(image)

            # Prediction button with enhanced styling
            predict_btn = st.button("🔍 Recognize Digit")

            if predict_btn:
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence = predict_digit(model, processed_image)

                if predicted_class is not None:
                    # Display results with animation and better styling
                    st.markdown(
                        """
                        <div class="prediction-container">
                            <h3>Recognition Results</h3>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Create metric cards for results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Predicted Digit</div>
                                <div class="metric-value">{predicted_class}</div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{confidence:.1f}%</div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Visualization of prediction probabilities with improved styling
                    st.markdown(
                        "<h3>Prediction Probabilities</h3>", unsafe_allow_html=True
                    )
                    predictions = model.predict(processed_image)[0]

                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(range(10), predictions, color="#FFD700")

                    # Highlight the predicted digit
                    bars[predicted_class].set_color("#FFC107")

                    ax.set_xlabel("Digits", fontsize=12)
                    ax.set_ylabel("Probability", fontsize=12)
                    ax.set_title("Digit Recognition Confidence Levels", fontsize=14)
                    ax.set_xticks(range(10))
                    ax.set_ylim(0, 1)
                    ax.grid(axis="y", linestyle="--", alpha=0.7)

                    # Add value labels on top of bars
                    for i, v in enumerate(predictions):
                        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

                    fig.tight_layout()
                    st.pyplot(fig)
    else:
        # Display placeholder when no image is uploaded
        st.markdown(
            """
            <div style="background-color: #f8f8f8; padding: 30px; border-radius: 10px; text-align: center; margin-top: 20px;">
                <img src="https://i.imgur.com/6NvUVst.png" width="100" style="opacity: 0.5;">
                <p style="margin-top: 15px;">Upload an image of a handwritten digit to get started</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>Digit Recognizer • Powered by TensorFlow and Streamlit</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
