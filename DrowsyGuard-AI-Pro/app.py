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
from collections import deque
from datetime import datetime

# ===============================
# 1. ENV & CONFIG
# ===============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="DrowsyGuard AI Pro MAX",
    page_icon="🛡️",
    layout="wide"
)

# Setup Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "model": os.path.join(BASE_DIR, "drowsy_model.keras"),
    "sound": os.path.join(BASE_DIR, "score.mp3"),
    "profile": os.path.join(BASE_DIR, "fotosaya.jpeg"),
    "reports": os.path.join(BASE_DIR, "reports"),
    "evidence": os.path.join(BASE_DIR, "evidence")
}

for p in [PATHS["reports"], PATHS["evidence"]]:
    os.makedirs(p, exist_ok=True)

# ===============================
# 2. CACHED ASSETS (Model & UI)
# ===============================
@st.cache_resource
def load_ai_model():
    if os.path.exists(PATHS["model"]):
        return tf.keras.models.load_model(PATHS["model"], compile=False)
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
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 15px #00f2fe; }
</style>
""", unsafe_allow_html=True)

# ===============================
# 4. HEADER & PROFILE SECTION
# ===============================
st.markdown('<h1 class="title-text">🛡️ DrowsyGuard AI Pro MAX</h1>', unsafe_allow_html=True)

with st.container():
    col_img, col_info = st.columns([1, 3])
    with col_img:
        if os.path.exists(PATHS["profile"]):
            st.image(PATHS["profile"], width=230, use_container_width=False)
        else:
            st.info("📸 Foto Profil")
            
    with col_info:
        st.markdown(f"""
        <div class="profile-card">
            <h2 style="margin-top:0; color:#00f2fe;">Ericson Chandra Sihombing</h2>
            <p>🎓 <b>Data Science Student</b> — Institut Teknologi Sumatera (ITERA)</p>
            <p>🤖 <i>"Data bukan hanya angka, tetapi cerita yang menunggu untuk diungkap."</i></p>
            <div style="display: flex; gap: 20px; font-size: 0.9em;">
                <span>📧 sihombingericson@gmail.com</span>
                <span>🔗 <a href="https://www.linkedin.com/in/ericsonchandrasihombing" style="color:#4facfe;">LinkedIn</a></span>
                <span>📸 @ericsonchandra99</span>
            </div>
            <br>
            <b>Core Expertise:</b> Machine Learning • Computer Vision • NLP • Data Analytics
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
    if os.path.exists(PATHS["sound"]):
        with open(PATHS["sound"], "rb") as f:
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
    mode = st.radio("Monitoring Mode", ["Live Webcam", "Upload Media"])
    
    if st.button("🗑️ Reset Reports"):
        st.session_state.report_data = []
        st.rerun()
    
    st.success("AI Model: Active ✅")

# ===============================
# 7. MONITORING ENGINE
# ===============================
col_viz, col_status = st.columns([2, 1])

def run_engine(source):
    cap = cv2.VideoCapture(source)
    buffer = deque(maxlen=smoothing)
    frame_win = col_viz.empty()
    status_win = col_status.empty()
    
    # Tombol Stop
    if col_status.button("🛑 STOP MONITORING", use_container_width=True):
        st.session_state.run = False
        st.rerun()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # AI Prediction
        preds = model.predict(preprocess_frame(frame), verbose=0)[0]
        buffer.append(preds[0]) # Sesuaikan index dengan output model anda
        avg_score = np.mean(buffer) * 100
        danger = avg_score >= threshold

        # UI Update
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
            # Save Evidence
            img_path = os.path.join(PATHS["evidence"], f"ev_{datetime.now().strftime('%H%M%S')}.jpg")
            cv2.imwrite(img_path, frame)
            st.session_state.report_data.append({
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Risk": f"{avg_score:.1f}%",
                "Evidence": img_path
            })
            if alarm_on: trigger_alarm()

        # Render Video
        frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(0.01)

    cap.release()

# ===============================
# 8. EXECUTION
# ===============================
if mode == "Live Webcam":
    if col_viz.button("▶️ START CAMERA"):
        run_engine(0)

else:
    uploaded = col_viz.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
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
        num_show = st.slider("Show last N incidents:", 1, len(df), min(6, len(df)))
        recent_items = st.session_state.report_data[-num_show:]
        cols = st.columns(3)
        for i, item in enumerate(recent_items):
            with cols[i % 3]:
                st.image(item["Evidence"], caption=f"{item['Timestamp']} | {item['Risk']}")

    with tab_data:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV Report", csv, "report.csv", "text/csv")
else:
    st.warning("Belum ada data insiden yang terekam.")

st.markdown(f"<br><center>© {datetime.now().year} DrowsyGuard AI Pro — Developed by Ericson Chandra</center>", unsafe_allow_html=True)
