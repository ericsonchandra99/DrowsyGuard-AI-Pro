import streamlit as st
import streamlit.components.v1 as components
import os

# WAJIB: Memaksa penggunaan Keras 2 (Legacy) agar model .h5 tidak error di Keras 3 (Cloud)
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

import tensorflow as tf
import numpy as np
import cv2
import time
from collections import deque

# --- PERBAIKAN IMPORT DISINI ---
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model
# -------------------------------

import base64
import pandas as pd
import plotly.graph_objects as go
import tempfile
from PIL import Image

# =========================================================
# 1. KONFIGURASI APP & UI
# =========================================================
st.set_page_config(page_title="DrowsyGuard AI Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stImage > img { width: 100% !important; border-radius: 15px; border: 2px solid #3e4150; }
    .status-card { 
        padding: 4vh 2vw; 
        border-radius: 20px; 
        text-align: center; 
        color: white; 
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .status-card h1 { font-size: calc(20px + 1.5vw); margin: 0; }
    .status-card h3 { font-size: calc(12px + 0.5vw); opacity: 0.9; }
    .sidebar-info { padding: 15px; border-radius: 12px; background-color: #161b22; margin-top: 15px; border: 1px solid #3e4150; }
    .manual-box { background-color: #1e2130; padding: 25px; border-radius: 15px; border-left: 5px solid #00D166; margin-bottom: 20px; }
    .contact-link { text-decoration: none; color: #00D166; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. INISIALISASI SESSION STATE
# =========================================================
if 'history_score' not in st.session_state:
    st.session_state['history_score'] = deque(maxlen=50)
if 'alert_log' not in st.session_state:
    st.session_state['alert_log'] = []

# Path Relatif untuk Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_9_final.h5")
SOUND_PATH = os.path.join(BASE_DIR, "score.mp3")

@st.cache_resource
def load_drowsiness_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan!")
        return None
    try:
        # --- PERBAIKAN LOAD MODEL DISINI (safe_mode=False) ---
        return load_model(MODEL_PATH, compile=False, safe_mode=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_drowsiness_model()

# =========================================================
# 3. UTILITIES
# =========================================================
def get_audio_html(is_active):
    if is_active and os.path.exists(SOUND_PATH):
        with open(SOUND_PATH, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"""
                <div style="display:none;">
                    <audio id="alarm-audio" autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>
                    <script>document.getElementById("alarm-audio").play();</script>
                </div>
            """
    return "<div></div>"

def preprocess_frame(frame):
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return np.expand_dims(img_array, axis=0)

# =========================================================
# 4. SIDEBAR & DOWNLOAD LOG
# =========================================================
with st.sidebar:
    st.title("🛡️ Pro Control")
    st.subheader("👨‍🎓 Developer Info")
    st.markdown(f"""
    <div class="sidebar-info">
        <b>Ericson Chandra Sihombing</b><br>
        🆔 121450026 | ITERA
    </div>
    <div class="sidebar-info">
        📫 sihombingericson@gmail.com<br>
        🔗 <a class="contact-link" href="https://linkedin.com/in/username">LinkedIn</a> | 📸 <a class="contact-link" href="https://instagram.com/ericsonchandra99">@ericsonchandra99</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("⚙️ Opsi Peringatan")
    enable_sound = st.toggle("🔔 Nyalakan Suara Peringatan", value=False)
    
    st.markdown("---")
    buffer_val = st.slider("Smoothing (Stabilitas)", 1, 15, 5)
    conf_threshold = st.slider("Ambang Bahaya (%)", 30, 95, 65)

    if st.session_state['alert_log']:
        st.markdown("---")
        df_log = pd.DataFrame(st.session_state['alert_log'], columns=["Timestamp Alert"])
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data Alert (CSV)", data=csv, file_name='history_alert.csv', mime='text/csv', use_container_width=True)
    
    if st.button("🚨 STOP & EXIT SYSTEM", use_container_width=True):
        os._exit(0)

# =========================================================
# 5. MAIN INTERFACE (TABS)
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Real-time Detection", "🎞️ Video Analysis", "🖼️ Image Check", "📖 Manual Book"])

with tab1:
    col_cam, col_info = st.columns([1.6, 1], gap="large")
    with col_cam:
        st.subheader("📷 Live Monitoring")
        run_cam = st.checkbox("Aktifkan Kamera Utama", value=True)
        FRAME_WINDOW = st.image([])
    with col_info:
        st.subheader("📊 Analisis Risiko")
        status_ui = st.empty()
        chart_placeholder = st.empty()
        audio_placeholder = st.empty() 
        st.metric("Total Alert Sesi Ini", len(st.session_state['alert_log']))

    if run_cam and model:
        camera = cv2.VideoCapture(0)
        pred_buffer = deque(maxlen=buffer_val)
        while run_cam:
            ret, frame = camera.read()
            if not ret or frame is None:
                st.error("Kamera tidak terdeteksi. Silakan berikan izin akses kamera di browser.")
                break
            
            preds = model.predict(preprocess_frame(frame), verbose=0)[0]
            pred_buffer.append(preds)
            avg_preds = np.mean(pred_buffer, axis=0)
            prob_danger = (avg_preds[0] + avg_preds[1]) * 100
            st.session_state['history_score'].append(prob_danger)
            
            is_danger = prob_danger >= conf_threshold
            bg_color = "#FF4B4B" if is_danger else "#00D166"
            label = "⚠️ BERBAHAYA" if is_danger else "✅ NORMAL"
            
            status_ui.markdown(f'<div class="status-card" style="background-color:{bg_color};"><h1>{label}</h1><h3>Skor: {prob_danger:.1f}%</h3></div>', unsafe_allow_html=True)
            
            if enable_sound:
                with audio_placeholder: components.html(get_audio_html(is_danger), height=0)
            else: audio_placeholder.empty()

            fig = go.Figure(go.Scatter(y=list(st.session_state['history_score']), mode='lines', fill='tozeroy', line=dict(color=bg_color, width=3)))
            fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            if is_danger:
                ct = time.strftime("%H:%M:%S")
                if not st.session_state['alert_log'] or st.session_state['alert_log'][-1] != ct:
                    st.session_state['alert_log'].append(ct)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        camera.release()

with tab2:
    uvid = st.file_uploader("Upload Video", type=["mp4", "avi"])
    v_win = st.image([])
    if uvid and model:
        tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uvid.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            p = model.predict(preprocess_frame(frame), verbose=0)[0]
            sc = (p[0] + p[1]) * 100
            cv2.putText(frame, f"{'BAHAYA' if sc > 50 else 'NORMAL'} ({sc:.1f}%)", (30, 50), 2, 1, (0,0,255), 2)
            v_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

with tab3:
    uimg = st.file_uploader("Upload Foto", type=["jpg", "png"])
    if uimg and model:
        img_pil = Image.open(uimg); p = model.predict(preprocess_frame(np.array(img_pil)), verbose=0)[0]
        i_sc = (p[0] + p[1]) * 100
        st.image(img_pil, caption=f"Risk: {i_sc:.1f}%", use_container_width=True)

with tab4:
    st.header("📖 Manual Book: DrowsyGuard AI")
    st.markdown("---")
    st.info("📌 **Metodologi:** Model MobileNetV2 digunakan untuk klasifikasi biner 'Normal' dan 'Mengantuk'. Smoothing Moving Average diterapkan untuk menjaga stabilitas deteksi.")
    st.markdown("""<div class="manual-box"><h4>💡 Cara Kerja</h4><ul><li><b>Threshold:</b> Sesuaikan sensitivitas (65% direkomendasikan).</li><li><b>Alert Log:</b> Gunakan sidebar untuk mengunduh riwayat deteksi bahaya.</li></ul></div>""", unsafe_allow_html=True)
