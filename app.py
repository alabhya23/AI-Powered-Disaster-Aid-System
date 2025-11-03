import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re

# -------------------------------
# 🌍 PAGE SETUP — FIRST LINE
# -------------------------------
st.set_page_config(page_title="AI-Powered Disaster Detection System", layout="centered")

st.title("🌍 AI-Powered Disaster Detection System")
st.write("Analyze **images** and **messages** to detect potential disasters in real time.")

# -------------------------------
# 🧠 CACHE MODELS
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    cnn = load_model("disaster_cnn_mobilenet_clean.h5")
    text = load_model("disaster_text_lstm_clean.h5")
    return cnn, text

cnn_model, text_model = load_models()

# -------------------------------
# 🧹 TEXT PREPROCESSING
# -------------------------------
@st.cache_data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")

@st.cache_data
def tokenize_text(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=60, padding='post')

# -------------------------------
# 🖼️ IMAGE PREPROCESSING
# -------------------------------
@st.cache_data
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------------
# 🎛️ MAIN APP INTERFACE
# -------------------------------
st.divider()
st.header("🖼️ Image or 💬 Text Analysis")

col1, col2 = st.columns(2)

# -------------------------------
# 🖼️ IMAGE SECTION
# -------------------------------
with col1:
    uploaded_image = st.file_uploader("Upload an image for disaster detection", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        preprocessed = preprocess_image(img)
        preds = cnn_model.predict(preprocessed)[0]
        label = np.argmax(preds)
        confidence = np.max(preds) * 100

        categories = ["Non-Damage", "Fire Disaster", "Flood", "Earthquake", "Other"]
        result = categories[label]

        st.success(f"**{result} ({confidence:.1f}%)**")

# -------------------------------
# 💬 TEXT SECTION
# -------------------------------
with col2:
    input_text = st.text_area("Enter a message or tweet to classify")
    if input_text:
        cleaned = clean_text(input_text)
        padded = tokenize_text([cleaned])
        pred = text_model.predict(padded)[0][0]
        label = "🚨 Disaster" if pred > 0.5 else "✅ Safe"
        st.info(f"**{label} ({pred*100:.2f}%)**")

st.divider()
st.caption("Built with ❤️ using Streamlit + TensorFlow")

