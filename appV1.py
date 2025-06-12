import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import base64
import google.generativeai as genai

genai.configure(api_key="")

def set_theme_adaptive_styles():
    st.markdown("""
        <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Light Mode */
        @media (prefers-color-scheme: light) {
            .themed-container {
                background-color: rgba(255, 255, 255, 0.8);
                color: #000;
            }
            .themed-box {
                background-color: #f0f0f0;
                color: #333;
                border: 1px solid #ccc;
            }
        }

        /* Dark Mode */
        @media (prefers-color-scheme: dark) {
            .themed-container {
                background-color: rgba(20, 20, 20, 0.7);
                color: #fff;
            }
            .themed-box {
                background-color: #2b2b2b;
                color: #ddd;
                border: 1px solid #555;
            }
        }

        .themed-box {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def get_disease_plan(plant, disease):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    I have detected a disease on a plant.

    Plant: {plant}
    Disease: {disease}

    Please provide:
    1. A short description of the disease
    2. Possible causes
    3. Preventive measures
    4. Treatment and medication plans (organic and chemical if possible)

    Respond in a clean bullet-point format.
    """

    response = model.generate_content(prompt)
    return response.text.strip()

# PAGE CONFIG
st.set_page_config(page_title="Agri-AID", page_icon="🌿", layout="wide")

# BACKGROUND IMAGE
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    .stButton > button {{
        background-color: #3f7652;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {{
        color: #f4f4f4;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        backdrop-filter: blur(6px);
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 15px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_from_local(r"D:\Python Projects\Agri AID - Plant Disease Identification\agriculture-healthy-food.jpg")

# LOAD MODEL & LABELS 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\Python Projects\Agri AID - Plant Disease Identification\plant_disease_model_V2.keras")

@st.cache_data
def load_label_map():
    with open("class_to_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class

model = load_model()
idx_to_class = load_label_map()

if "history" not in st.session_state:
    st.session_state.history = []
set_theme_adaptive_styles()
# UI HEADER
st.markdown("<h1 style='text-align:center;'>🌿 Agri-AID: Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload a plant leaf image to identify the plant and its disease.</p>", unsafe_allow_html=True)

# IMAGE UPLOAD
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # st.image(image, caption="📷 Uploaded Image", use_container_width =False, width=350)
    # --- SIDE BY SIDE LAYOUT ---
    col1, col2 = st.columns([1, 2])  # Ratio of width: image : response

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width =True, width=350)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        pred = model.predict(img_batch)[0]
        pred_idx = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        pred_class = idx_to_class[pred_idx]
        plant, disease = pred_class.split("___")

        with col2:
            with st.spinner("🤖 Consulting Agri-AID AI for treatment suggestions..."):
                treatment_info = get_disease_plan(plant, disease)
            st.markdown(f"<h3>🪴 Plant: <span style='color:#a0fca2'>{plant}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<h3>🦠 Disease: <span style='color:#f78c8c'>{disease}</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<h4>📈 Confidence: <span style='color:#f7d98c'>{confidence:.2f}%</span></h4>", unsafe_allow_html=True)
            st.markdown("### 💊 Treatment & Precaution Plan")
            st.markdown(treatment_info)

        st.session_state.history.append({
            "plant": plant,
            "disease": disease,
            "confidence": confidence,
            "image": image.copy()
        })

# GALLERY
if st.session_state.history:
    st.markdown("<h2 style='margin-top:40px;'>🖼️ Prediction Gallery</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, entry in enumerate(reversed(st.session_state.history[-6:])):
        with cols[i % 3]:
            st.image(entry["image"].resize((180, 180)), use_container_width=False)
            st.markdown(f"""
                <div style='padding:8px; background-color:#1f1f1f; border-radius:10px; margin-top:5px;'>
                    <b>🌱 {entry['plant']}</b><br>
                    🦠 <span style='color:#ffaaaa'>{entry['disease']}</span><br>
                    📊 Confidence: {entry['confidence']:.1f}%
                </div>
            """, unsafe_allow_html=True)
