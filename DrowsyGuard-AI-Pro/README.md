
---

# 🛡️ DrowsyGuard AI Pro

**DrowsyGuard AI Pro** adalah sistem deteksi kantuk berbasis Artificial Intelligence yang dirancang untuk memonitor kondisi pengguna secara real-time menggunakan webcam, video, maupun gambar.

Aplikasi ini menggunakan model Deep Learning berbasis **MobileNetV2** yang ringan namun akurat, sehingga cocok untuk implementasi real-time.

Deploy dilakukan menggunakan **Streamlit** untuk menghasilkan dashboard interaktif.

---

## 🚀 Features

✅ Real-time drowsiness detection via webcam
✅ Video file analysis
✅ Image-based detection
✅ Risk scoring system
✅ Alarm notification system
✅ Smoothing prediction untuk stabilitas
✅ Interactive dashboard visualization

---

## 🧠 AI Model

Model CNN berbasis **MobileNetV2** digunakan untuk klasifikasi kondisi pengguna ke dalam 3 kelas:

* Mengantuk Tanpa Menguap
* Mengantuk dan Menguap
* Tidak Mengantuk dan Tidak Menguap

Untuk sistem monitoring, output digabung menjadi:

| Status       | Kondisi                                         |
| ------------ | ----------------------------------------------- |
| ⚠️ BERBAHAYA | Mengantuk Tanpa Menguap + Mengantuk dan Menguap |
| ✅ NORMAL     | Tidak Mengantuk                                 |

---

## 🖥️ Interface Modules

Aplikasi terdiri dari 4 mode utama:

### 🎥 Real-time Detection

* Monitoring langsung dari webcam
* Risk score ditampilkan dalam grafik
* Alarm otomatis saat kondisi berbahaya

### 🎞️ Video Analysis

* Analisis video (.mp4 / .avi)
* Frame-by-frame classification

### 🖼️ Image Check

* Deteksi kondisi dari gambar statis

### 📖 Manual Book

* Penjelasan sistem
* Panduan konfigurasi

---

## ⚙️ Configuration Options

User dapat mengatur:

* 🔔 Alarm suara
* 📉 Smoothing stabilitas deteksi
* 🎯 Threshold tingkat bahaya

---

## 📂 Project Structure

```
DrowsyGuard AI Pro/
│── app.py
│── model_9_final.h5
│── score.mp3
│── requirements.txt
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* NumPy
* Plotly
* PIL

---

## ▶️ How to Run

### 1. Activate Environment

```
conda activate sidang
```

### 2. Run Application

```
python -m streamlit run app.py
```

---

## 📊 How It Works

1. Webcam menangkap frame pengguna
2. Frame di-resize menjadi **224x224**
3. Preprocessing menggunakan MobileNetV2
4. Model memprediksi probabilitas kantuk
5. Moving average digunakan untuk stabilisasi
6. Jika melewati threshold → status BERBAHAYA + alarm aktif

---

## 📌 Use Cases

* Driver monitoring system
* Safety monitoring
* Fatigue detection research
* Human behavior analysis

---

## 👨‍💻 Developer

**Ericson Chandra Sihombing**
NIM: 121450026
Institut Teknologi Sumatera (ITERA)

📫 Email: [sihombingericson@gmail.com](mailto:sihombingericson@gmail.com)

---

## 📜 Notes

Model dirancang ringan agar tetap dapat berjalan secara real-time tanpa kebutuhan GPU.

