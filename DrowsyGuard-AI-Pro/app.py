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
import queue
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

# Queue Global untuk komunikasi AI -> UI
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

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
# 3. STYLE FUTURISTIK & CSS
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
# 4. HEADER & PROFILE (SEMUA INFO DIKEMBALIKAN)
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

with st.container():
    col_img, col_info = st.columns([1, 3])
    with col_img:
        found_profile = next((os.path.join(BASE_DIR, n) for n in PROFILE_NAMES if os.path.exists(os.path.join(BASE_DIR, n))), None)
        if found_profile: st.image(found_profile, width=230)
        else: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=230)
            
    with col_info:
        st.markdown(f"""
        <div class="profile-card">
            <h2 style="margin-top:0; color:#00f2fe;">Ericson Chandra Sihombing</h2>
            <p>🎓 <b>Data Science Student</b> — Institut Teknologi Sumatera (ITERA)</p>
            <p>🤖 <i>"Data bukan hanya angka, tetapi cerita yang menunggu untuk diungkap."</i></p>
            <div style="display: flex; gap: 20px; font-size: 0.9em;">
                <span>📧 sihombingericson@gmail.com</span>
                <span>🔗 <a href="https://www.linkedin.com/in/ericsonchandrasihombing" target="_blank" style="color:#4facfe;">LinkedIn</a></span>
            </div>
            <br>
            <b>Core Expertise:</b> Machine Learning • Computer Vision • Data Analytics
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

def trigger_alarm():
    # Menggunakan file score.mp3 yang sudah ada di folder BASE_DIR
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        components.html(f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>""", height=0)

# ===============================
# 6. WEBRTC ENGINE (BEBAS LAG)
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Prediction (Hanya dijalankan jika model ada)
        if model:
            # Downscale sedikit agar proses model lebih cepat (mengurangi lag)
            input_data = preprocess_frame(img)
            preds = model.predict(input_data, verbose=0)[0]
            self.buffer.append(preds[0])
            avg_score = np.mean(self.buffer) * 100
        else:
            avg_score = 0

        danger = avg_score >= self.threshold
        
        # Masukkan hasil ke Queue untuk diproses di Main Thread UI
        st.session_state.result_queue.put({
            "danger": danger, 
            "score": avg_score, 
            "frame": img.copy() if danger else None
        })

        color = (0, 0, 255) if danger else (0, 255, 0)
        label = "DROWSY" if danger else "ALERT"
        cv2.putText(img, f"{label}: {avg_score:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# ===============================
# 7. MAIN LOGIC & SIDEBAR
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

col_viz, col_status = st.columns([2, 1])

# --- MODE 1: LIVE CLOUD (WEBRTC) ---
if mode == "🌐 Live Cloud Camera":
    with col_viz:
        webrtc_ctx = webrtc_streamer(
            key="drowsy-guard",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing),
            async_processing=True, # Kunci untuk menghilangkan lag
        )

    status_placeholder = col_status.empty()
    
    # Loop untuk mengambil data dari Queue AI
    if webrtc_ctx.state.playing:
        while True:
            try:
                res = st.session_state.result_queue.get(timeout=0.1)
                color = "#ff4b4b" if res["danger"] else "#00ff88"
                status_placeholder.markdown(f"""
                    <div class="metric-card" style="border-left-color: {color}">
                        <h3>{'⚠️ DROWSY' if res["danger"] else '✅ ALERT'}</h3>
                        <h1>{res["score"]:.1f}%</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                if res["danger"]:
                    if alarm_on: trigger_alarm()
                    if res["frame"] is not None:
                        path = os.path.join(EVIDENCE_DIR, f"ev_{int(time.time()*100)}.jpg")
                        cv2.imwrite(path, res["frame"])
                        st.session_state.report_data.append({
                            "Timestamp": datetime.now().strftime("%H:%M:%S"),
                            "Risk": f"{res['score']:.1f}%",
                            "Evidence": path
                        })
            except queue.Empty:
                if not webrtc_ctx.state.playing: break
                continue

# --- MODE 2: UPLOAD MEDIA ---
elif mode == "📂 Upload Media":
    uploaded = col_viz.file_uploader("Upload Video untuk Analisis AI", type=["mp4", "avi", "mov"])
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
            danger = avg_score >= threshold
            
            color = "#ff4b4b" if danger else "#00ff88"
            status_win.markdown(f'<div class="metric-card" style="border-left-color: {color}"><h3>{"⚠️ DROWSY" if danger else "✅ ALERT"}</h3><h1>{avg_score:.1f}%</h1></div>', unsafe_allow_html=True)
            
            if danger:
                img_path = os.path.join(EVIDENCE_DIR, f"ev_{int(time.time()*100)}.jpg")
                cv2.imwrite(img_path, frame)
                st.session_state.report_data.append({"Timestamp": datetime.now().strftime("%H:%M:%S"), "Risk": f"{avg_score:.1f}%", "Evidence": img_path})
                if alarm_on: trigger_alarm()

            frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        cap.release()

# ===============================
# 8. SMART REPORT & HISTORY
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")

if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    
    # Fitur History Slider dikembalikan
    num_show = st.slider("Tampilkan N insiden terakhir:", 1, len(df), min(6, len(df)))
    
    tab_gal, tab_data = st.tabs(["🖼️ Evidence Gallery", "📊 Data Logs"])
    
    with tab_gal:
        recent_items = st.session_state.report_data[-num_show:]
        cols = st.columns(3)
        for i, item in enumerate(recent_items):
            with cols[i % 3]:
                if os.path.exists(item["Evidence"]):
                    st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")
                else:
                    st.error("Gambar hilang.")

    with tab_data:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Report", csv, "report_drowsy.csv", "text/csv")
else:
    st.warning("Belum ada data insiden yang terekam.")

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Developed by Ericson Chandra Sihombing</center>", unsafe_allow_html=True)
