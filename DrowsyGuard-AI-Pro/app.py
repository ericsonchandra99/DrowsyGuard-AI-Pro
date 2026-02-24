import streamlit as st
import streamlit.components.v1 as components
import os

# Matikan log TensorFlow agar tidak mengotori UI
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
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

MODEL_NAME = "drowsy_model.keras"
SOUND_NAME = "score.mp3"
PROFILE_NAMES = ["fotosaya.jpeg", "fotosaya.jpg", "fotosaya.JPG", "FOTOSAYA.JPG"]

# ===============================
# 2. CACHED ASSETS
# ===============================
@st.cache_resource
def load_ai_model():
    model_path = os.path.join(BASE_DIR, MODEL_NAME)
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
    return None

model = load_ai_model()

# ===============================
# 3. STYLE FUTURISTIK & GLASSMORPHISM
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
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-card {
        background: rgba(0, 242, 254, 0.1);
        padding: 20px; border-radius: 15px;
        border-left: 5px solid #4facfe;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. HEADER & PROFILE SECTION
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

with st.container():
    col_img, col_info = st.columns([1, 3])
    with col_img:
        found_profile = next((os.path.join(BASE_DIR, n) for n in PROFILE_NAMES if os.path.exists(os.path.join(BASE_DIR, n))), None)
        if found_profile: st.image(found_profile, width=230)
        else: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=230)
    with col_info:
        st.markdown(f"""<div class="profile-card">
            <h2 style="margin-top:0; color:#00f2fe;">Ericson Chandra Sihombing</h2>
            <p>🎓 <b>Data Science Student</b> — Institut Teknologi Sumatera (ITERA)</p>
            <p>🤖 <i>"Data bukan hanya angka, tetapi cerita yang menunggu untuk diungkap."</i></p>
            <div style="display: flex; gap: 20px; font-size: 0.9em;">
                <span>📧 sihombingericson@gmail.com</span>
                <span>🔗 <a href="https://www.linkedin.com/in/ericsonchandrasihombing" target="_blank" style="color:#4facfe;">LinkedIn</a></span>
            </div><br><b>Core Expertise:</b> Machine Learning • Computer Vision • Data Analytics</div>""", unsafe_allow_html=True)

st.divider()

# ===============================
# 5. HELPER FUNCTIONS
# ===============================
def trigger_alarm():
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.components.v1.html(f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>""", height=0)

# ===============================
# 6. WEBRTC ENGINE (The Secret Sauce)
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing, alarm_on):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)
        self.alarm_on = alarm_on
        self.last_alarm_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Logic
        res = cv2.resize(img, (224, 224))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        res = tf.keras.applications.mobilenet_v2.preprocess_input(res.astype(np.float32))
        res = np.expand_dims(res, axis=0)

        if model:
            preds = model.predict(res, verbose=0)[0]
            self.buffer.append(preds[0])
            avg_score = np.mean(self.buffer) * 100
        else:
            avg_score = 0

        danger = avg_score >= self.threshold
        
        if danger:
            # Note: Storing data from within WebRTC thread requires care
            # We'll handle logging via session_state in the main loop instead
            cv2.putText(img, f"DROWSY! {avg_score:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(img, f"SAFE: {avg_score:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return img

# ===============================
# 7. SIDEBAR & LOGIC CONTROL
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5)
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    mode = st.radio("Monitoring Mode", ["🌐 Live Cloud Camera", "📂 Upload Media"])
    
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()

# ===============================
# 8. MONITORING UI
# ===============================
col_viz, col_status = st.columns([2, 1])

if mode == "🌐 Live Cloud Camera":
    webrtc_ctx = webrtc_streamer(
        key="drowsy-guard",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing, alarm_on),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
    if webrtc_ctx.state.playing:
        col_status.success("Kamera Berjalan! AI sedang memantau...")
        # Note: Auto-alarm & logging for WebRTC can be complex due to threading. 
        # For Cloud stability, focus on the visual risk indicator provided in the stream.

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
            
            # Prediction
            res = cv2.resize(frame, (224, 224))
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            res = tf.keras.applications.mobilenet_v2.preprocess_input(res.astype(np.float32))
            res = np.expand_dims(res, axis=0)
            
            preds = model.predict(res, verbose=0)[0] if model else [0]
            buffer.append(preds[0])
            avg_score = np.mean(buffer) * 100
            danger = avg_score >= threshold
            
            color = "#ff4b4b" if danger else "#00ff88"
            status_win.markdown(f'<div class="metric-card" style="border-left-color: {color}"><h3>{"⚠️ DROWSY" if danger else "✅ SAFE"}</h3><h1>{avg_score:.1f}%</h1></div>', unsafe_allow_html=True)
            
            if danger:
                img_path = os.path.join(EVIDENCE_DIR, f"ev_{int(time.time())}.jpg")
                cv2.imwrite(img_path, frame)
                st.session_state.report_data.append({"Timestamp": datetime.now().strftime("%H:%M:%S"), "Risk": f"{avg_score:.1f}%", "Evidence": img_path})
                if alarm_on: trigger_alarm()

            frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

# ===============================
# 9. SMART REPORT SECTION
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")
if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    tab_gal, tab_data = st.tabs(["🖼️ Evidence Gallery", "📊 Data Logs"])
    with tab_gal:
        recent_items = st.session_state.report_data[-6:]
        cols = st.columns(3)
        for i, item in enumerate(recent_items):
            with cols[i % 3]: st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")
    with tab_data:
        st.dataframe(df, use_container_width=True)
else:
    st.warning("Belum ada data insiden yang terekam.")

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Developed by Ericson Chandra</center>", unsafe_allow_html=True)
