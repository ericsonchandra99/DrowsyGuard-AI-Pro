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

# ===============================
# 1. ENV & CONFIG
# ===============================
st.set_page_config(
    page_title="DrowsyGuard AI Pro MAX",
    page_icon="🛡️",
    layout="wide"
)

# Fix Path untuk Deployment (Linux/Windows Friendly)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pastikan folder penting ada
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Nama file aset (Pastikan sudah di-upload ke GitHub)
MODEL_NAME = "drowsy_model.keras"
SOUND_NAME = "score.mp3"
# Kita buat list kemungkinan nama file profil (Case-Sensitive fix)
PROFILE_NAMES = ["fotosaya.jpeg", "fotosaya.jpg", "fotosaya.JPG", "FOTOSAYA.JPG"]

# ===============================
# 2. CACHED ASSETS (Model & UI)
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
    .stButton>button {
        border-radius:30px; font-weight:bold;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        color: black; border: none; transition: 0.3s;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. HEADER & PROFILE SECTION (FIXED)
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

with st.container():
    col_img, col_info = st.columns([1, 3])
    
    with col_img:
        # Mencari file profil yang tersedia
        found_profile = None
        for name in PROFILE_NAMES:
            p_path = os.path.join(BASE_DIR, name)
            if os.path.exists(p_path):
                found_profile = p_path
                break
        
        if found_profile:
            st.image(found_profile, width=230)
        else:
            # Placeholder jika foto tidak ketemu agar tidak error
            st.warning("📸 Foto profil tidak ditemukan. Pastikan file ada di GitHub.")
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=230)
            
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
    sound_path = os.path.join(BASE_DIR, SOUND_NAME)
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.components.v1.html(f"""<audio autoplay><source src="data:audio/mp3;base64,{b64}"></audio>""", height=0)

# ===============================
# 6. SIDEBAR & LOGIC CONTROL
# ===============================
if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    st.header("⚙️ Control Panel")
    threshold = st.slider("Danger Threshold (%)", 30, 95, 65)
    smoothing = st.slider("Buffer Smoothing", 1, 15, 5)
    alarm_on = st.toggle("🔔 Enable Alarm", True)
    mode = st.radio("Monitoring Mode", ["Live Webcam (Local Only)", "Upload Media"])
    
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()
    
    if model:
        st.success("AI Model: Active ✅")
    else:
        st.error("AI Model: Not Found ❌")

# ===============================
# 7. MONITORING ENGINE
# ===============================
col_viz, col_status = st.columns([2, 1])

def run_engine(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        col_viz.error("Gagal membuka sumber video.")
        return

    buffer = deque(maxlen=smoothing)
    frame_win = col_viz.empty()
    status_win = col_status.empty()
    
    # Gunakan session state untuk kontrol loop
    stop_button = col_status.button("🛑 STOP MONITORING")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break

        # AI Prediction
        if model:
            preds = model.predict(preprocess_frame(frame), verbose=0)[0]
            buffer.append(preds[0]) 
            avg_score = np.mean(buffer) * 100
        else:
            avg_score = 0

        danger = avg_score >= threshold
        color = "#ff4b4b" if danger else "#00ff88"
        label = "⚠️ DROWSY DETECTED" if danger else "✅ DRIVER ALERT"
        
        status_win.markdown(f"""
            <div class="metric-card" style="border-left-color: {color}">
                <h3 style="color:{color}; margin:0;">{label}</h3>
                <h1 style="margin:10px 0;">{avg_score:.1f}%</h1>
                <p style="color:gray;">Risk Level Indicator</p>
            </div>
        """, unsafe_allow_html=True)

        if danger:
            timestamp = datetime.now().strftime('%H%M%S')
            img_filename = f"ev_{timestamp}.jpg"
            img_path = os.path.join(EVIDENCE_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            
            # Simpan ke log
            st.session_state.report_data.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Risk": f"{avg_score:.1f}%",
                "Evidence": img_path
            })
            if alarm_on: trigger_alarm()

        frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(0.01)

    cap.release()
    st.rerun()

# ===============================
# 8. EXECUTION
# ===============================
if mode == "Live Webcam (Local Only)":
    st.info("Catatan: Webcam mode hanya bekerja di lingkungan Lokal. Untuk Cloud, gunakan 'Upload Media'.")
    if col_viz.button("▶️ START CAMERA"):
        run_engine(0)

else:
    uploaded = col_viz.file_uploader("Upload Video untuk Analisis AI", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())
        if col_viz.button("🔍 ANALYZE VIDEO"):
            run_engine(tfile.name)

# ===============================
# 9. SMART REPORT SECTION
# ===============================
st.divider()
st.subheader("📄 Smart Report & Evidence Viewer")

if st.session_state.report_data:
    df = pd.DataFrame(st.session_state.report_data)
    tab_gal, tab_data = st.tabs(["🖼️ Evidence Gallery", "📊 Data Logs"])
    
    with tab_gal:
        num_show = st.slider("Tampilkan N insiden terakhir:", 1, len(df), min(6, len(df)))
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

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Developed by Ericson Chandra</center>", unsafe_allow_html=True)
