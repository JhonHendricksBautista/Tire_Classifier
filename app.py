import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 320
CLASS_NAMES = ["defective", "good"]
BASE_DIR = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    print("Files in directory:", os.listdir(BASE_DIR))  # debug

    baseline = tf.keras.models.load_model(os.path.join(BASE_DIR, "baseModel.keras"))
    hypertuned = tf.keras.models.load_model(os.path.join(BASE_DIR, "bestModel.keras"))
    yolo = YOLO(os.path.join(BASE_DIR, "yoloToh.pt"))  # ✅ correct filename

    return baseline, hypertuned, yolo


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_cnn(model, image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return CLASS_NAMES[class_idx], confidence

def predict_yolo(model, image):
    results = model(image)
    if len(results[0].boxes) == 0:
        return "No defect detected", 0.0
    
    cls = int(results[0].boxes.cls[0])
    conf = float(results[0].boxes.conf[0])
    
    return model.names[cls], conf

# =========================
# UI DESIGN
# =========================
st.set_page_config(page_title="Tire Defect Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🛞 Tire Defect Classifier</h1>
    <p style='text-align: center;'>Upload an image and choose a model to classify tire defects</p>
""", unsafe_allow_html=True)

# MODEL SELECTION
model_choice = st.selectbox(
    "Choose Model",
    ["YOLOv8", "Baseline CNN", "HyperTuned CNN"]
)

# IMAGE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Tire Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # PREDICT BUTTON
    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):

            if model_choice == "Baseline CNN":
                label, conf = predict_cnn(baseline_model, image)

            elif model_choice == "HyperTuned CNN":
                label, conf = predict_cnn(hypertuned_model, image)

            else:  # YOLO
                label, conf = predict_yolo(yolo_model, image)

        # =========================
        # OUTPUT
        # =========================
        st.markdown("### 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", label)

        with col2:
            st.metric("Confidence", f"{conf:.2%}")

        # Optional: YOLO visualization
        if model_choice == "YOLOv8":
            results = yolo_model(image)
            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result", use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with Streamlit</p>",
    unsafe_allow_html=True
)