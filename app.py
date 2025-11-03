import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from PIL import Image
import io
import re

# ---------------------------
# Load models
# ---------------------------
@st.cache_resource
def load_models():
    text_model = load_model("disaster_text_bilstm.h5")
    cnn_model = load_model("disaster_cnn_mobilenet_clean.h5")
    return cnn_model, text_model

cnn_model, text_model = load_models()

# ---------------------------
# Title and layout
# ---------------------------
st.title("🌍 AI-Powered Disaster Detection System")
st.write("Analyze **images** and **messages** to detect potential disasters in real time.")

tab1, tab2 = st.tabs(["🖼️ Image Analysis", "💬 Text Analysis"])

# ---------------------------
# IMAGE ANALYSIS TAB
# ---------------------------
with tab1:
    st.subheader("Upload an Image for Disaster Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess for MobileNetV2
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = cnn_model.predict(img_array)
        pred_class = np.argmax(preds)
        confidence = float(np.max(preds)) * 100

        # Labels — match your training set
        labels = ['Damaged_Infrastructure', 'Fire_Disaster', 'Human_Damage', 'Land_Disaster', 'Non_Damage', 'Water_Disaster']
        result = labels[pred_class]

        st.success(f"**Prediction:** {result} ({confidence:.2f}%)")

# ---------------------------
# TEXT ANALYSIS TAB
# ---------------------------
with tab2:
    st.subheader("Enter a message or tweet to classify")
    user_text = st.text_area("Type your message here...")

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    if st.button("Analyze Text"):
        if user_text.strip():
            # Prepare tokenizer (recreate same config as training)
            tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
            text_seq = tokenizer.texts_to_sequences([clean_text(user_text)])
            padded = pad_sequences(text_seq, maxlen=60, padding='post')

            pred = text_model.predict(padded)[0][0]
            label = "🚨 Disaster" if pred > 0.5 else "✅ Safe"
            confidence = float(pred if pred > 0.5 else 1 - pred) * 100
            st.success(f"{label} ({confidence:.2f}%)")
        else:
            st.warning("Please enter some text to analyze.")

st.caption("Developed for AI Disaster Detection Project — Phase 5")
