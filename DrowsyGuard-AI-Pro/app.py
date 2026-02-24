import streamlit as st
import streamlit.components.v1 as components
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
import time
from collections import deque
import base64
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import tempfile

# ===============================
# KONFIGURASI
# ===============================
st.set_page_config(
    page_title="DrowsyGuard AI Pro MAX",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "drowsy_model.keras")
SOUND_PATH = os.path.join(BASE_DIR, "score.mp3")
PROFILE_IMAGE_PATH = os.path.join(BASE_DIR, "fotosaya.jpeg")

REPORT_DIR = os.path.join(BASE_DIR, "reports")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model_cached()

# ===============================
# STYLE FUTURISTIK
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
.title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    background: linear-gradient(90deg,#00f2fe,#4facfe,#00f2fe);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.card {
    padding:25px;
    border-radius:25px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    box-shadow:0 0 30px rgba(0,255,255,0.3);
}
.profile-card {
    padding:20px;
    border-radius:20px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    box-shadow:0 0 20px rgba(0,255,255,0.2);
}
.stButton>button {
    border-radius:30px;
    height:3em;
    font-weight:bold;
    font-size:16px;
    background: linear-gradient(90deg,#00f2fe,#4facfe);
    color:black;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="title">🛡️ DrowsyGuard AI Pro MAX</div>', unsafe_allow_html=True)
st.write("### 🚘 AI Real-Time Driver Monitoring + Smart Reporting System")

# ===============================
# PROFILE SECTION
# ===============================
st.markdown("---")
st.subheader("👤 About The Developer")

col1, col2 = st.columns([1, 2])

with col1:
    if os.path.exists(PROFILE_IMAGE_PATH):
        st.image(PROFILE_IMAGE_PATH, width=220)
    else:
        st.warning("File fotosaya.jpg tidak ditemukan.")

with col2:
    st.markdown("""
    <div class="profile-card">
    <h3>Ericson Chandra Sihombing</h3>
    🎓 Data Science Student — Institut Teknologi Sumatera (ITERA) <br>
    🤖 AI & Machine Learning Enthusiast <br><br>

    Saya percaya bahwa <b>data bukan hanya angka, tetapi cerita yang menunggu untuk diungkap.</b><br><br>

    <b>Spesialisasi:</b><br>
    • Machine Learning <br>
    • Deep Learning <br>
    • Computer Vision <br>
    • Data Analytics <br>
    • NLP <br><br>

    📬 Email: sihombingericson@gmail.com <br>
    🔗 LinkedIn: https://www.linkedin.com/in/ericsonchandrasihombing <br>
    📸 Instagram: @ericsonchandra99
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===============================
# UTIL
# ===============================
def preprocess(frame):
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224,224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def play_sound():
    if os.path.exists(SOUND_PATH):
        with open(SOUND_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        components.html(f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """, height=0)

def save_evidence(frame):
    filename = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(EVIDENCE_DIR, filename)
    cv2.imwrite(path, frame)
    return path

def generate_report(data):
    df = pd.DataFrame(data)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(REPORT_DIR, filename)
    df.to_csv(path, index=False)
    return path

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Ambang Bahaya (%)", 30, 95, 65)
    smoothing = st.slider("Smoothing Buffer", 1, 15, 5)
    alarm_on = st.toggle("🔔 Aktifkan Alarm", True)
    mode = st.radio("Mode Monitoring", ["Webcam", "Upload Media"])
    st.success("Model Status: Loaded ✅")

# ===============================
# SESSION
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []

# ===============================
# WEBCAM MODE
# ===============================
if mode == "Webcam":

    if st.button("▶️ START MONITORING"):
        st.session_state.run = True

    if st.button("⛔ STOP"):
        st.session_state.run = False

    if st.session_state.get("run", False):

        frame_window = st.empty()
        status_box = st.empty()

        cap = cv2.VideoCapture(0)
        buffer = deque(maxlen=smoothing)

        while st.session_state.get("run", False):

            ret, frame = cap.read()
            if not ret:
                break

            preds = model.predict(preprocess(frame), verbose=0)[0]
            buffer.append(preds)
            avg = np.mean(buffer, axis=0)

            danger_score = float((avg[0] + avg[1]) * 100)
            danger = danger_score >= threshold

            label = "⚠️ DROWSY DETECTED" if danger else "✅ DRIVER ALERT"
            color = "#ff0000" if danger else "#00ff88"

            status_box.markdown(f"""
            <div class="card" style="border-left:10px solid {color};">
            <h2 style="color:{color};">{label}</h2>
            <h3>Risk Level: {danger_score:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            if danger:
                evidence_path = save_evidence(frame)
                st.session_state.report_data.append({
                    "timestamp": datetime.now(),
                    "risk_level": danger_score,
                    "evidence": evidence_path
                })
                if alarm_on:
                    play_sound()

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.03)

        cap.release()

# ===============================
# UPLOAD MODE
# ===============================
if mode == "Upload Media":

    uploaded_file = st.file_uploader(
        "Upload Image / Video",
        type=["jpg","jpeg","png","mp4","avi","mov","mkv"]
    )

    if uploaded_file:

        file_type = uploaded_file.type

        if "image" in file_type:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            preds = model.predict(preprocess(frame), verbose=0)[0]
            danger_score = float((preds[0] + preds[1]) * 100)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.success(f"Risk Level: {danger_score:.1f}%")

        elif "video" in file_type:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                preds = model.predict(preprocess(frame), verbose=0)[0]
                danger_score = float((preds[0] + preds[1]) * 100)

                if danger_score >= threshold:
                    evidence_path = save_evidence(frame)
                    st.session_state.report_data.append({
                        "timestamp": datetime.now(),
                        "risk_level": danger_score,
                        "evidence": evidence_path
                    })

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(0.03)

            cap.release()

# ===============================
# REPORT SYSTEM UPGRADE
# ===============================
st.markdown("---")
st.subheader("📄 Smart Report & Evidence Viewer")

if st.session_state.report_data:

    total_events = len(st.session_state.report_data)
    st.info(f"Total Drowsy Events Terdeteksi: {total_events}")

    # ===============================
    # PILIH JUMLAH GAMBAR
    # ===============================
    max_images = st.slider(
        "Pilih jumlah gambar evidence yang ingin ditampilkan:",
        min_value=1,
        max_value=total_events,
        value=min(5, total_events)
    )

    st.markdown("### 🖼️ Evidence Preview")

    selected_data = st.session_state.report_data[-max_images:]

    cols = st.columns(3)

    for i, data in enumerate(selected_data):
        col = cols[i % 3]
        with col:
            st.image(data["evidence"], caption=f"{data['timestamp']} | {data['risk_level']:.1f}%")
            
            with open(data["evidence"], "rb") as file:
                st.download_button(
                    label="⬇ Download",
                    data=file,
                    file_name=os.path.basename(data["evidence"]),
                    key=f"download_{i}"
                )

    st.markdown("---")

    # ===============================
    # DOWNLOAD CSV
    # ===============================
    if st.button("📥 Download Full Report (CSV)"):
        path = generate_report(st.session_state.report_data)
        with open(path, "rb") as f:
            st.download_button(
                "Download CSV Report",
                f,
                file_name=os.path.basename(path)
            )

else:
    st.warning("Belum ada data drowsy yang tersimpan.")