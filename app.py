import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Set page config
st.set_page_config(page_title="Agri-AID", page_icon="🌿", layout="centered")

# Load model and class map
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\Admin\Desktop\Python Project\Agri-AID\plant_disease_model_V2.keras")

@st.cache_data
def load_label_map():
    with open("class_to_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

model = load_model()
idx_to_class = load_label_map()

# Session state to store prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# App UI
st.title("🌿 Agri-AID - Plant Disease Classifier")
st.write("Upload a leaf image to detect the plant and its disease with confidence.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width =True)

    # Preprocess image
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        pred = model.predict(img_batch)[0]
        pred_idx = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        pred_class = idx_to_class[pred_idx]
        plant, disease = pred_class.split("___")

        st.success(f"🪴 **Plant:** {plant}")
        st.error(f"🦠 **Disease:** {disease}")
        st.info(f"📈 **Confidence:** {confidence:.2f}%")

        # Save to history
        st.session_state.history.append({
            "plant": plant,
            "disease": disease,
            "confidence": confidence,
            "image": image.copy()
        })

# Image Gallery of Predictions
if st.session_state.history:
    st.subheader("🖼️ Prediction Gallery")
    cols = st.columns(3)
    for i, entry in enumerate(reversed(st.session_state.history[-6:])):  # show last 6
        with cols[i % 3]:
            st.image(entry["image"].resize((150, 150)))
            st.caption(f"🌱 {entry['plant']} - 🦠 {entry['disease']} ({entry['confidence']:.1f}%)")
