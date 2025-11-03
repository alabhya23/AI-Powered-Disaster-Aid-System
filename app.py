import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image, text, sequence
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI-Powered Disaster Detection System", layout="centered")

st.title("🌍 AI-Powered Disaster Detection System")
st.write("Analyze images and messages to detect potential disasters in real time.")
# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    cnn_model = load_model("disaster_cnn_mobilenet_clean.h5")
    text_model = load_model("disaster_text_bilstm.h5")
    return cnn_model, text_model

cnn_model, text_model = load_models()

# =========================
# CONFIGURATION
# =========================

st.markdown("Analyze images and text messages to detect potential disasters in real-time.")

# =========================
# SIDEBAR INFO
# =========================
st.sidebar.header("🧠 About")
st.sidebar.info("This app analyzes uploaded images and/or text messages to classify whether they indicate a **disaster** situation or not.")

# =========================
# INPUT SECTION
# =========================
st.subheader("📥 Upload or Enter Data")

uploaded_image = st.file_uploader("Upload an image (optional):", type=["jpg", "jpeg", "png"])
user_text = st.text_area("Enter a message or tweet (optional):", placeholder="Example: Heavy flooding has destroyed roads in the city.")

# =========================
# PROCESSING FUNCTIONS
# =========================
def predict_image(img_file):
    try:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = cnn_model.predict(img_array)
        classes = ['Cyclone', 'Earthquake', 'Flood', 'Fire', 'Non-Damage']
        result = classes[np.argmax(preds)]
        confidence = np.max(preds) * 100
        return result, confidence
    except Exception as e:
        return "Error", str(e)

def predict_text(message):
    try:
        tokenizer = text.Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts([message])
        text_seq = tokenizer.texts_to_sequences([message])
        padded = sequence.pad_sequences(text_seq, maxlen=60, padding='post')
        preds = text_model.predict(padded)
        labels = ["Safe", "Disaster"]
        result = labels[np.argmax(preds)]
        confidence = np.max(preds) * 100
        return result, confidence
    except Exception as e:
        return "Error", str(e)

# =========================
# DETECTION LOGIC
# =========================
if st.button("🚨 Run Disaster Detection"):
    if not uploaded_image and not user_text.strip():
        st.warning("⚠️ Please upload an image or enter a text message.")
    else:
        st.markdown("### 🔍 Analyzing...")

        # IMAGE ANALYSIS
        if uploaded_image:
            img_result, img_conf = predict_image(uploaded_image)
            st.success(f"🖼️ **Image Result:** {img_result} ({img_conf:.2f}%)")

        # TEXT ANALYSIS
        if user_text.strip():
            txt_result, txt_conf = predict_text(user_text)
            st.success(f"💬 **Text Result:** {txt_result} ({txt_conf:.2f}%)")

        # Combined Interpretation (like GCC)
        if (uploaded_image and img_result != "Non-Damage") or (user_text.strip() and txt_result == "Disaster"):
            st.error("🚨 ALERT: Potential Disaster detected! ⚠️")
            st.markdown("**📡 Ground Control Center (GCC):** Emergency teams have been notified.")
        else:
            st.info("✅ All clear! No disaster detected at this time.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed using TensorFlow, MobileNetV2, and Streamlit • © 2025 Disaster AI System")
