import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# 1. Page Configuration & Dynamic Path Setup
st.set_page_config(page_title="Multimodal House Price Predictor", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "multimodal_house_model.h5")
SCALER_MEAN_PATH = os.path.join(BASE_DIR, "saved_model", "scaler_mean.npy")
SCALER_VAR_PATH = os.path.join(BASE_DIR, "saved_model", "scaler_var.npy")


# Cache resources to avoid reloading model on every interaction
@st.cache_resource
def load_saved_resources():
    # Adding custom_objects to bypass Keras 3 deserialization bug
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "mse": tf.keras.losses.MeanSquaredError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError()
        }
    )
    mean = np.load(SCALER_MEAN_PATH)
    var = np.load(SCALER_VAR_PATH)
    return model, mean, var


try:
    model, scaler_mean, scaler_var = load_saved_resources()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}")
    model_loaded = False


# Helper function to manually scale input tabular features
def scale_input_features(features, mean, var):
    std = np.sqrt(var)
    return (features - mean) / std


# 2. UI Layout
st.title("🏡 Multimodal House Price Predictor")
st.write("Predicting real estate value using both **Tabular Parameters** and **House Images** simultaneously.")
st.markdown("---")

if model_loaded:
    # Creating layout grids
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Tabular Parameters")
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, step=1)
        area_sqft = st.slider("Area (Square Feet)", min_value=500, max_value=5000, value=1500, step=50)
        age_years = st.slider("House Age (Years)", min_value=0, max_value=100, value=10, step=1)

    with col2:
        st.subheader("🖼️ House Image input")
        uploaded_file = st.file_uploader("Upload an image of the house...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded House Image", use_container_width=True)
        else:
            st.info("Please upload a house photo to trigger multimodal prediction.")

    st.markdown("---")

    # 3. Prediction Pipeline execution
    if st.button("🚀 Predict Housing Value", use_container_width=True):
        if uploaded_file is None:
            st.warning("⚠️ Multimodal model requires both structured tabular data and a visual house image.")
        else:
            with st.spinner("Processing tabular features and image matrices..."):
                try:
                    # Preprocess Tabular Input
                    raw_tabular = np.array([[rooms, bathrooms, area_sqft, age_years]], dtype=np.float32)
                    scaled_tabular = scale_input_features(raw_tabular, scaler_mean, scaler_var)

                    # Preprocess Image Input (Resizing to 224x224x3 and normalizing)
                    img_processed = image.resize((224, 224)).convert("RGB")
                    img_array = np.array(img_processed, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension -> (1, 224, 224, 3)

                    # Execute Prediction using both inputs
                    prediction = model.predict([scaled_tabular, img_array])
                    predicted_price = float(prediction[0][0])

                    # Formatting and displaying the output
                    st.balloons()
                    st.success("🎉 Prediction Completed!")
                    st.markdown(f"### 🏷️ Estimated Housing Price: **${predicted_price:,.2f}**")

                except Exception as ex:
                    st.error(f"An error occurred during prediction logic: {ex}")
else:
    st.info("Verify your training artifacts in the 'saved_model/' directory.")