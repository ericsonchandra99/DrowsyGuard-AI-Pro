import streamlit as st
import streamlit.components.v1 as components
import os

# Matikan log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2
import time
import base64
import pandas as pd
import tempfile
from collections import deque
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ===============================
# 1. ENV & CONFIG
# ===============================
st.set_page_config(
    page_title="DrowsyGuard AI Pro MAX",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

MODEL_NAME = "drowsy_model.keras"
SOUND_NAME = "score.mp3"
PROFILE_NAMES = ["fotosaya.jpeg", "fotosaya.jpg", "fotosaya.JPG", "FOTOSAYA.JPG"]

# Inisialisasi Session State
if "report_data" not in st.session_state:
    st.session_state.report_data = []
if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0
if "is_drowsy" not in st.session_state:
    st.session_state.is_drowsy = False

# ===============================
# 2. LOAD MODEL
# ===============================
@st.cache_resource
def load_ai_model():
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    return None

model = load_ai_model()

# ===============================
# 3. STYLE FUTURISTIK
# ===============================
st.markdown("""
<style>
.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white; }
.title-text {
    text-align:center; font-size:clamp(30px, 5vw, 55px); font-weight:800;
    background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
.profile-card {
    padding:25px; border-radius:20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border:1px solid rgba(255,255,255,0.1);
}
.metric-card {
    background: rgba(0,242,254,0.1);
    padding:20px; border-radius:15px;
    border-left:5px solid #4facfe;
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. HEADER & PROFILE
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

with st.container():
    col_img, col_info = st.columns([1, 3])
    with col_img:
        found_profile = next((os.path.join(BASE_DIR, n) for n in PROFILE_NAMES if os.path.exists(os.path.join(BASE_DIR, n))), None)
        st.image(found_profile if found_profile else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=230)
    with col_info:
        st.markdown(f"""
        <div class="profile-card">
            <h2 style="margin-top:0; color:#00f2fe;">Ericson Chandra Sihombing</h2>
            <p>🎓 <b>Data Science Student</b> — Institut Teknologi Sumatera (ITERA)</p>
            <p>🤖 <i>"Data bukan hanya angka, tetapi cerita yang menunggu untuk diungkap."</i></p>
            <div style="display:flex; gap:20px; font-size:0.9em;">
                <span>📧 sihombingericson@gmail.com</span>
                <span>🔗 <a href="https://www.linkedin.com/in/ericsonchandrasihombing" target="_blank" style="color:#4facfe;">LinkedIn</a></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ===============================
# 5. HELPER FUNCTIONS
# ===============================
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def play_alarm_ui():
    """Fungsi untuk memicu suara di sisi Client"""
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)

# ===============================
# 6. WEBRTC TRANSFORMER
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)
        self.danger = False
        self.score = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if model:
            input_data = preprocess_frame(img)
            preds = model.predict(input_data, verbose=0)[0]
            self.buffer.append(preds[0])
            self.score = np.mean(self.buffer) * 100
        else:
            self.score = 0

        self.danger = self.score >= self.threshold
        
        # Gambar info di frame
        color = (0, 0, 255) if self.danger else (0, 255, 0)
        label = "DROWSY" if self.danger else "ALERT"
        cv2.putText(img, f"{label}: {self.score:.1f}%", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return img

# ===============================
# 7. SIDEBAR & LAYOUT
# ===============================
with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5)
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    mode = st.radio("Monitoring Mode", ["🌐 Live Cloud Camera", "📂 Upload Media"])

    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()

col_viz, col_status = st.columns([2, 1])
alarm_placeholder = st.empty() # Placeholder khusus audio

# ===============================
# 8. PROCESSOR
# ===============================
if mode == "🌐 Live Cloud Camera":
    with col_viz:
        webrtc_ctx = webrtc_streamer(
            key="drowsy-guard",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing),
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

    status_placeholder = col_status.empty()

    if webrtc_ctx.video_transformer:
        # Loop UI untuk update status secara real-time
        score = webrtc_ctx.video_transformer.score
        is_danger = webrtc_ctx.video_transformer.danger
        
        color = "#ff4b4b" if is_danger else "#00ff88"
        status_text = "⚠️ DROWSY" if is_danger else "✅ ALERT"

        status_placeholder.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}; background: {color}22;">
                <h3 style="color:{color}">{status_text}</h3>
                <h1 style="color:white;">{score:.1f}%</h1>
                <p>Status: {'Bahaya!' if is_danger else 'Aman'}</p>
            </div>
        """, unsafe_allow_html=True)

        if is_danger:
            # Jalankan alarm jika aktif
            if alarm_on:
                now = time.time()
                if now - st.session_state.last_alarm_time > 3: # Jeda 3 detik
                    with alarm_placeholder:
                        play_alarm_ui()
                    st.session_state.last_alarm_time = now
            
            # Log Data (Hanya jika belum tercatat di detik yang sama)
            timestamp = datetime.now().strftime("%H:%M:%S")
            if not st.session_state.report_data or st.session_state.report_data[-1]["Timestamp"] != timestamp:
                st.session_state.report_data.append({
                    "Timestamp": timestamp,
                    "Risk": f"{score:.1f}%",
                    "Evidence": "Live Stream" # Untuk live, kita tidak simpan file tiap frame agar ringan
                })

elif mode == "📂 Upload Media":
    uploaded = col_viz.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        
        frame_win = col_viz.empty()
        status_win = col_status.empty()
        buffer = deque(maxlen=smoothing)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            preds = model.predict(preprocess_frame(frame), verbose=0)[0] if model else [0]
            buffer.append(preds[0])
            avg_score = np.mean(buffer) * 100
            
            is_danger = avg_score >= threshold
            color = "#ff4b4b" if is_danger else "#00ff88"
            
            status_win.markdown(f"""
                <div class="metric-card" style="border-left-color:{color}">
                    <h3 style="color:{color}">{'⚠️ DROWSY' if is_danger else '✅ ALERT'}</h3>
                    <h1>{avg_score:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)

            if is_danger and alarm_on:
                play_alarm_ui()

            frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

# ===============================
# 9. REPORT
# ===============================
st.divider()
st.subheader("📊 Monitoring Log Reports")
if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export CSV", csv, "drowsy_report.csv", "text/csv")
else:
    st.info("Sistem siap. Mulai kamera untuk mendeteksi.")

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Ericson Chandra Sihombing</center>", unsafe_allow_html=True)
