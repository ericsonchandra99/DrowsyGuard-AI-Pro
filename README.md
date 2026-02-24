<p align="center">
  <h1 align="center">🛡️ DrowsyGuard AI Pro</h1>
  <p align="center">
    Sistem Deteksi Kantuk Real-Time Berbasis Deep Learning
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/OpenCV-ComputerVision-green?logo=opencv"/>
  <img src="https://img.shields.io/badge/Status-ProductionReady-success"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

---

## 🚀 Overview

**DrowsyGuard AI Pro** adalah sistem monitoring kantuk berbasis Artificial Intelligence yang dirancang untuk mendeteksi tingkat kelelahan pengguna secara real-time menggunakan webcam, video, maupun gambar statis.

Sistem ini tidak hanya melakukan klasifikasi, tetapi juga menghasilkan:

- 🎯 Risk Score (%)
- 🚨 Sistem Alarm Otomatis
- 🖼️ Evidence Capture (penyimpanan frame berisiko)
- 📊 Dashboard Monitoring Interaktif
- 📄 Sistem Report & Export CSV

Dirancang ringan sehingga dapat berjalan secara real-time tanpa GPU.

---

## 🎥 Demo Aplikasi

<p align="center">
  <img src="assets/demo.gif" width="700"/>
</p>

> Ganti `assets/demo.gif` dengan hasil screen recording aplikasi kamu.

---

## 🧠 Arsitektur AI

- Model: MobileNetV2 (Transfer Learning)
- Framework: TensorFlow / Keras
- Input: 224x224
- Output: 3 kelas (Softmax)
- Post-processing: Moving Average Smoothing
- Logika Alert: Threshold-based Risk Activation

### Klasifikasi Model

| Kelas | Deskripsi |
|-------|-----------|
| 0 | Mengantuk Tanpa Menguap |
| 1 | Mengantuk dan Menguap |
| 2 | Tidak Mengantuk |

Untuk sistem monitoring:

- ⚠️ BERBAHAYA → Kelas 0 & 1  
- ✅ NORMAL → Kelas 2  

---

## 📊 Project Metrics

| Komponen | Spesifikasi |
|----------|-------------|
| Arsitektur | MobileNetV2 |
| Resolusi Input | 224x224 |
| Real-time FPS | ±20–30 FPS (CPU) |
| Deployment | Streamlit |
| Inference Device | CPU Compatible |
| Sistem Alert | Threshold + Smoothing |
| Evidence Logging | Otomatis saat risiko tinggi |

---

## 🔥 Fitur Utama

### 🎥 Real-Time Detection
- Monitoring langsung via webcam
- Visualisasi Risk Score
- Status indikator (NORMAL / BERBAHAYA)
- Alarm otomatis
- Smoothing prediksi untuk stabilitas

### 🎞️ Analisis Video
- Input file video (.mp4 / .avi)
- Frame-by-frame classification
- Logging event berisiko

### 🖼️ Deteksi Gambar
- Klasifikasi gambar statis
- Output probabilitas tiap kelas

### 📄 Smart Report System
- Penyimpanan event kantuk
- Preview evidence gambar
- Download per gambar
- Export full report CSV

---

## 🏗️ Alur Sistem

1. Webcam menangkap frame
2. Frame di-resize menjadi 224x224
3. Preprocessing sesuai MobileNetV2
4. Model melakukan prediksi
5. Risk score dihitung
6. Moving average diterapkan
7. Jika melebihi threshold:
   - Status berubah menjadi BERBAHAYA
   - Alarm aktif
   - Evidence disimpan
   - Data masuk ke report system

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- Pandas
- Plotly
- PIL

---

## 📂 Struktur Proyek

```
DrowsyGuard-AI-Pro/
│── app.py
│── model_9_final.h5
│── score.mp3
│── evidence/
│── reports/
│── requirements.txt
│── assets/demo.gif
```

---

## ⚙️ Cara Menjalankan

### 1️⃣ Clone Repository

```bash
git clone https://github.com/username/DrowsyGuard-AI-Pro.git
cd DrowsyGuard-AI-Pro
```

### 2️⃣ Install Dependency

```bash
pip install -r requirements.txt
```

### 3️⃣ Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## 🎯 Use Case

- Driver Monitoring System
- Monitoring Keselamatan Industri
- Riset Fatigue Detection
- Human Attention Monitoring
- Prototype AI Safety System

---

# 👨‍💻 Developer

**Ericson Chandra Sihombing**  
Mahasiswa Sains Data 2021  
Institut Teknologi Sumatera (ITERA)

📧 Email: sihombingericson@gmail.com  
🔗 LinkedIn: https://linkedin.com/in/ericsonchandrasihombing  

---

# 🎓 Portfolio AI / ML Engineer

Project ini menunjukkan kemampuan dalam:

- ✅ Implementasi Deep Learning end-to-end
- ✅ Computer Vision real-time pipeline
- ✅ Optimasi model ringan untuk CPU
- ✅ Desain Risk Scoring System
- ✅ Sistem Alert Engineering
- ✅ Logging & Monitoring System
- ✅ Deployment model ke dashboard interaktif

Bukan hanya melatih model, tetapi membangun sistem AI yang siap digunakan.

---

## 📜 Lisensi

MIT License

---

<p align="center">
  Dibangun dengan ❤️ menggunakan Deep Learning & Computer Vision
</p>
