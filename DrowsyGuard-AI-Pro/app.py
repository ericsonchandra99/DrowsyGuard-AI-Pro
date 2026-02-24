import streamlit as st
import streamlit.components.v1 as components
import os
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
st.set_page_config(page_title="DrowsyGuard AI Pro MAX", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

MODEL_NAME = "drowsy_model.keras"
SOUND_NAME = "score.mp3"
PROFILE_NAMES = ["fotosaya.jpeg", "fotosaya.jpg", "fotosaya.JPG", "FOTOSAYA.JPG"]

# Queue untuk mengirim data dari Camera Thread ke Main Thread
# Ini kunci agar Alarm dan Laporan bisa berfungsi
result_queue = queue.Queue()

# ===============================
# 2. CACHED ASSETS
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
    .main { background: #0f0c29; color: white; }
    .title-text { 
        text-align:center; font-size:45px; font-weight:800; 
        background: linear-gradient(90deg, #00f2fe, #4facfe); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
    }
    .profile-card {
        padding:20px; border-radius:15px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-card { 
        background: rgba(0, 242, 254, 0.1); 
        padding: 15px; border-radius: 15px; 
        border-left: 5px solid #4facfe; 
        margin-top: 10px; 
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
        if found_profile: st.image(found_profile, width=220)
        else: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=220)
    
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
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ===============================
# 5. HELPER FUNCTIONS
# ===============================
def trigger_alarm():
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        components.html(f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>""", height=0)

# ===============================
# 6. WEBRTC ENGINE
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Processing
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
        
        # Kirim data ke Main Thread via Queue
        result_queue.put({
            "danger": danger, 
            "score": avg_score, 
            "frame": img.copy() if danger else None
        })

        # Visual Feedback pada Stream
        color = (0, 0, 255) if danger else (0, 255, 0)
        cv2.putText(img, f"RISK: {avg_score:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# ===============================
# 7. MAIN APP LOGIC
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5)
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()
    
    if model: st.success("AI Model: Active ✅")
    else: st.error("AI Model: Not Found ❌")

col_viz, col_status = st.columns([2, 1])

with col_viz:
    webrtc_ctx = webrtc_streamer(
        key="drowsy-guard",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing),
        media_stream_constraints={"video": True, "audio": False},
    )

# Sinkronisasi Queue ke UI & Session State
status_placeholder = col_status.empty()

while webrtc_ctx.state.playing:
    try:
        # Ambil data dari transformer (max wait 0.1 detik)
        result = result_queue.get(timeout=0.1)
        
        # 1. Update UI Status
        color = "#ff4b4b" if result["danger"] else "#00ff88"
        status_placeholder.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <h3>{'⚠️ DROWSY' if result["danger"] else '✅ SAFE'}</h3>
                <h1>{result["score"]:.1f}%</h1>
                <p>Status: Monitoring...</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. Jalankan Alarm
        if result["danger"] and alarm_on:
            trigger_alarm()

        # 3. Simpan Evidence ke Report
        if result["danger"] and result["frame"] is not None:
            img_name = f"ev_{int(time.time())}.jpg"
            img_path = os.path.join(EVIDENCE_DIR, img_name)
            cv2.imwrite(img_path, result["frame"])
            
            st.session_state.report_data.append({
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Risk": f"{result['score']:.1f}%",
                "Evidence": img_path
            })
    except queue.Empty:
        continue

# ===============================
# 8. SMART REPORT SECTION
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")

if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    tab_gal, tab_data = st.tabs(["🖼️ Evidence Gallery", "📊 Data Logs"])
    
    with tab_gal:
        recent = st.session_state.report_data[-6:]
        cols = st.columns(3)
        for i, item in enumerate(recent):
            with cols[i % 3]:
                st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")
    
    with tab_data:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report", csv, "drowsy_report.csv", "text/csv")
else:
    st.info("Belum ada data deteksi. Mulai kamera untuk memantau.")

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Developed by Ericson Chandra</center>", unsafe_allow_html=True)
