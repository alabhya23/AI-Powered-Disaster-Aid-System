# ============================================================
# 🌍 AI-Powered Disaster Detection System (Unified App)
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

# ✅ Streamlit Page Config (must be first)
st.set_page_config(page_title="AI-Powered Disaster Detection System", layout="centered")

# ============================================================
# LOAD MODELS & TOKENIZER
# ============================================================

@st.cache_resource
def load_models():
    cnn_model = load_model("disaster_cnn_mobilenet_clean.h5")
    text_model = load_model("disaster_text_lstm_clean.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return cnn_model, text_model, tokenizer

try:
    cnn_model, text_model, tokenizer = load_models()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error("⚠️ Model load error: " + str(e))

# ============================================================
# PAGE HEADER
# ============================================================

st.title("🌍 AI-Powered Disaster Detection System")
st.write("Analyze **images** and **messages** to detect potential disasters in real time.")

st.divider()

# ============================================================
# USER INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🖼️ Image Analysis")
    uploaded_image = st.file_uploader("Upload an Image for Disaster Detection", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("💬 Text Analysis")
    user_text = st.text_area("Enter a message or tweet to classify", placeholder="Type your message here...")

st.divider()

# ============================================================
# IMAGE PREDICTION FUNCTION
# ============================================================

def predict_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = cnn_model.predict(img_array)
    class_labels = ['Earthquake', 'Fire', 'Flood', 'Non-Damage', 'Storm']
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return pred_class, confidence, img

# ============================================================
# TEXT PREDICTION FUNCTION
# ============================================================

def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=60, padding='post')
    preds = text_model.predict(padded)
    disaster_prob = preds[0][0]
    if disaster_prob > 0.5:
        return "🚨 Disaster", disaster_prob * 100
    else:
        return "✅ Safe", (1 - disaster_prob) * 100

# ============================================================
# EXECUTION LOGIC
# ============================================================

analyze = st.button("🔍 Analyze")

if analyze:
    if not uploaded_image and not user_text.strip():
        st.warning("⚠️ Please upload an image or enter text to analyze.")
    else:
        if uploaded_image:
            try:
                pred_class, conf, img = predict_image(uploaded_image)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                st.success(f"🖼️ **{pred_class}** detected with **{conf:.2f}% confidence.**")
            except Exception as e:
                st.error(f"Image analysis failed: {str(e)}")

        if user_text.strip():
            try:
                result, conf = predict_text(user_text)
                st.info(f"💬 {result} ({conf:.2f}% confidence)")
            except Exception as e:
                st.error(f"Text analysis failed: {str(e)}")

st.divider()
st.caption("Developed using TensorFlow, Keras, and Streamlit 🌱")

