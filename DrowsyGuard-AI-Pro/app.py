import streamlit as st
import streamlit.components.v1 as components
import os
import tensorflow as tf
import numpy as np
import cv2
import time
import base64
import pandas as pd
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

# Queue untuk sinkronisasi data antar thread
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
# 3. UTILS & STYLE
# ===============================
def trigger_alarm():
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        # Menggunakan height=0 dan key unik agar tidak mengganggu UI
        components.html(
            f'<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>', 
            height=0
        )

st.markdown("""
<style>
    .main { background: #0f0c29; color: white; }
    .title-text { text-align:center; font-size:40px; font-weight:800; background: linear-gradient(90deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .metric-card { background: rgba(0, 242, 254, 0.1); padding: 15px; border-radius: 15px; border-left: 5px solid #4facfe; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. WEBRTC ENGINE (OPTIMIZED)
# ===============================
class DrowsyTransformer(VideoTransformerBase):
    def __init__(self, threshold, smoothing):
        self.threshold = threshold
        self.buffer = deque(maxlen=smoothing)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perkecil resolusi input model untuk mengurangi LAG
        small_img = cv2.resize(img, (224, 224))
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        input_data = tf.keras.applications.mobilenet_v2.preprocess_input(small_img.astype(np.float32))
        input_data = np.expand_dims(input_data, axis=0)

        if model:
            preds = model.predict(input_data, verbose=0)[0]
            self.buffer.append(preds[0])
            avg_score = np.mean(self.buffer) * 100
        else:
            avg_score = 0

        danger = avg_score >= self.threshold
        
        # Kirim data ke Main Thread
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
# 5. HEADER & SIDEBAR
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 10, 3) # Dikurangi agar lebih responsif
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()

# ===============================
# 6. MAIN DISPLAY & LOGIC
# ===============================
col_viz, col_status = st.columns([2, 1])

with col_viz:
    webrtc_ctx = webrtc_streamer(
        key="drowsy-guard",
        mode=WebRtcMode.SENDRECV,
        # STUN Server Google untuk mengatasi masalah "Connection Taking Longer"
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
        },
        video_transformer_factory=lambda: DrowsyTransformer(threshold, smoothing),
        async_processing=True, # Menghilangkan lag pada video stream
    )

status_placeholder = col_status.empty()

# LOOP PEMROSESAN DATA (MAIN THREAD)
if webrtc_ctx.state.playing:
    while True:
        try:
            # Ambil data dari antrean AI
            res = st.session_state.result_queue.get(timeout=0.1)
            
            # Tampilkan Status
            color = "#ff4b4b" if res["danger"] else "#00ff88"
            status_placeholder.markdown(f"""
                <div class="metric-card" style="border-left-color: {color}">
                    <h2 style="color:{color};">{'⚠️ DROWSY' if res["danger"] else '✅ ALERT'}</h2>
                    <h1>{res["score"]:.1f}%</h1>
                </div>
            """, unsafe_allow_html=True)

            if res["danger"]:
                # Pemicu Alarm
                if alarm_on: trigger_alarm()
                
                # Simpan Laporan jika ada frame
                if res["frame"] is not None:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    img_path = os.path.join(EVIDENCE_DIR, f"ev_{int(time.time()*100)}.jpg")
                    cv2.imwrite(img_path, res["frame"])
                    
                    st.session_state.report_data.append({
                        "Timestamp": timestamp,
                        "Risk": f"{res['score']:.1f}%",
                        "Evidence": img_path
                    })
        except queue.Empty:
            if not webrtc_ctx.state.playing: break
            continue

# ===============================
# 7. SMART REPORT & EVIDENCE
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")

if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    tab_gal, tab_data = st.tabs(["🖼️ Gallery Bukti", "📊 Log Tabel"])
    
    with tab_gal:
        # Menampilkan 6 terakhir secara default
        display_items = st.session_state.report_data[-6:]
        cols = st.columns(3)
        for i, item in enumerate(display_items):
            with cols[i % 3]:
                st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")
    
    with tab_data:
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Export CSV", df.to_csv(index=False), "drowsy_report.csv")
else:
    st.info("Belum ada data. Pastikan AI mendeteksi status 'DROWSY' agar sistem menyimpan bukti.")

st.markdown(f"<br><center>© {datetime.now().year} Ericson Chandra Sihombing — ITERA Data Science</center>", unsafe_allow_html=True)
