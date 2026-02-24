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

# Queue Global agar tidak hancur saat re-run
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

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
# 3. STYLE & PROFILE
# ===============================
st.markdown("""
<style>
    .main { background: #0f0c29; color: white; }
    .title-text { text-align:center; font-size:45px; font-weight:800; background: linear-gradient(90deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .profile-card { padding:20px; border-radius:15px; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
    .metric-card { background: rgba(0, 242, 254, 0.1); padding: 15px; border-radius: 15px; border-left: 5px solid #4facfe; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

# Profile Section
with st.container():
    col_img, col_info = st.columns([1, 3])
    with col_img:
        found_profile = next((os.path.join(BASE_DIR, n) for n in PROFILE_NAMES if os.path.exists(os.path.join(BASE_DIR, n))), None)
        if found_profile: st.image(found_profile, width=220)
        else: st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=220)
    with col_info:
        st.markdown(f'<div class="profile-card"><h2>Ericson Chandra Sihombing</h2><p>🎓 <b>Data Science Student</b> — ITERA 21</p><span>📧 sihombingericson@gmail.com</span></div>', unsafe_allow_html=True)

st.divider()

# ===============================
# 4. UTILS
# ===============================
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def trigger_alarm():
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        components.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>', height=0)

# ===============================
# 5. WEBRTC ENGINE
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        res = preprocess_frame(img)
        preds = model.predict(res, verbose=0)[0] if model else [0]
        self.buffer.append(preds[0])
        avg_score = np.mean(self.buffer) * 100
        danger = avg_score >= self.threshold
        
        # Kirim ke Queue
        st.session_state.result_queue.put({"danger": danger, "score": avg_score, "frame": img.copy() if danger else None})
        
        color = (0, 0, 255) if danger else (0, 255, 0)
        cv2.putText(img, f"RISK: {avg_score:.1f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# ===============================
# 6. APP STATE & SIDEBAR
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5)
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    mode = st.radio("Mode", ["🌐 Live Cloud", "📂 Upload"])
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()

col_viz, col_status = st.columns([2, 1])

# --- PROSES DATA ---
def save_incident(score, frame):
    path = os.path.join(EVIDENCE_DIR, f"ev_{int(time.time()*1000)}.jpg")
    cv2.imwrite(path, frame)
    st.session_state.report_data.append({
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Risk": f"{score:.1f}%",
        "Evidence": path
    })

# --- LOGIC RUN ---
if mode == "🌐 Live Cloud":
    with col_viz:
        webrtc_ctx = webrtc_streamer(
            key="drowsy-guard",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing),
            async_processing=True,
        )
    
    status_placeholder = col_status.empty()
    # Script ini tetap berjalan selama WebRTC aktif
    while webrtc_ctx.state.playing:
        try:
            res = st.session_state.result_queue.get(timeout=0.1)
            color = "#ff4b4b" if res["danger"] else "#00ff88"
            status_placeholder.markdown(f'<div class="metric-card" style="border-left-color: {color}"><h3>{"⚠️ DROWSY" if res["danger"] else "✅ SAFE"}</h3><h1>{res["score"]:.1f}%</h1></div>', unsafe_allow_html=True)
            
            if res["danger"]:
                if alarm_on: trigger_alarm()
                if res["frame"] is not None:
                    save_incident(res["score"], res["frame"])
        except queue.Empty: continue

elif mode == "📂 Upload":
    uploaded = col_viz.file_uploader("Upload", type=["mp4", "avi", "mov", "jpg", "png"])
    if uploaded:
        if uploaded.type.startswith('image'):
            # Logic Image (sama seperti sebelumnya)
            pass
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded.read())
            cap = cv2.VideoCapture(tfile.name)
            st_frame = col_viz.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                # AI Logic & Save Incident (Sama seperti sebelumnya)
                # ... 
            cap.release()

# ===============================
# 7. REPORT & EVIDENCE (FIXED PERSISTENCE)
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")

# Gunakan data dari session_state
if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    
    # Slider untuk mengatur histori
    num_to_show = st.slider("Tampilkan N foto terakhir:", 1, len(df), min(6, len(df)))
    
    tab_gal, tab_data = st.tabs(["🖼️ Gallery Bukti", "📊 Log Tabel"])
    
    with tab_gal:
        # Mengambil data dari akhir (terbaru)
        display_data = st.session_state.report_data[-num_to_show:]
        cols = st.columns(3)
        for i, item in enumerate(display_data):
            with cols[i % 3]:
                if os.path.exists(item["Evidence"]):
                    st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")
                    with open(item["Evidence"], "rb") as f:
                        st.download_button("Download", f, os.path.basename(item["Evidence"]), key=f"btn_{i}_{time.time()}")

    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Export CSV", df.to_csv(index=False), "report.csv")
else:
    st.info("Belum ada data insiden. Pastikan AI mendeteksi 'DROWSY' (mata tertutup) agar data tersimpan.")
