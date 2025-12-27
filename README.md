# Bike Sharing Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-KMeans-orange)
![Pandas](https://img.shields.io/badge/Data-Pandas-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview

Proyek ini merupakan analisis data mendalam (*End-to-End Data Analysis Project*) yang bertujuan untuk menggali wawasan dari dataset sistem berbagi sepeda (*Bike Sharing System*). Proyek ini tidak hanya berfokus pada analisis deskriptif masa lalu, tetapi juga mengimplementasikan **Machine Learning (Clustering)** untuk memberikan solusi strategis bagi manajemen stok.

Hasil akhir dari proyek ini adalah sebuah **Dashboard Interaktif berbasis Web** (menggunakan Streamlit) yang memungkinkan pemangku kepentingan (*stakeholders*) untuk memantau performa bisnis, menganalisis tren, dan melihat segmentasi jam operasional secara real-time.

### Business Questions
Analisis ini difokuskan untuk menjawab 6 pertanyaan strategis berikut:

**A. Analisis Deskriptif (Exploratory Data Analysis)**
1.  **Pola Waktu (Time Series):** Bagaimana karakteristik perbedaan pola penyewaan antara Hari Kerja (*Working Day*) dan Hari Libur (*Holiday*)?
2.  **Pertumbuhan Bisnis (Growth):** Bagaimana tren performa penyewaan sepeda pada tahun 2011 dibandingkan tahun 2012?
3.  **Faktor Musim (Seasonality):** Musim apa yang menjadi "Masa Panen" (*Peak Season*) dengan rata-rata penyewaan tertinggi?
4.  **Sensitivitas Cuaca (Environment):** Seberapa signifikan dampak kondisi cuaca buruk (hujan/salju) terhadap penurunan jumlah penyewaan?
5.  **Segmentasi Pelanggan (User Behavior):** Bagaimana perbedaan perilaku penggunaan sepeda antara Pengguna Terdaftar (*Registered*) dan Pengguna Kasual (*Casual*)?

**B. Analisis Lanjutan (Advanced Modeling)**
6.  **Segmentasi Jam Sibuk (Clustering):** Bagaimana kita dapat mengelompokkan jam operasional ke dalam kategori secara otomatis menggunakan algoritma K-Means untuk membantu manajemen stok?

---

## Struktur Direktori (Project Structure)

Proyek ini disusun dengan struktur folder yang rapi untuk memudahkan pengembangan dan *deployment*:

```text
submission_bike_sharing/
│
├── dashboard/
│   ├── dashboard.py             # Source code utama aplikasi Streamlit
│   └── hour.csv                 # Dataset dashboard
│
├── data/                        # Folder penyimpanan data mentah (Raw Data)
│   ├── day.csv
│   └── hour.csv
│
├── models/                      # Penyimpanan Model Machine Learning (AI)
│   ├── kmeans_bike_cluster.joblib
│   └── scaler_bike_cluster.joblib
│
├── notebooks/                   # Jupyter Notebook (Analisis Lengkap)
│   └── ML_Romi_Aprilian_Mustafa_without_output.ipynb
│
├── requirements.txt             # Daftar dependensi library
└── README.md                    # Dokumentasi Proyek

```

> **⚠️ Catatan Penting (Notebook):**
> Output visualisasi pada file notebook (`notebooks/ML_Romi_Aprilian_Mustafa_without_output.ipynb`) telah dibersihkan untuk menjaga ukuran file agar dapat ditampilkan di GitHub.
> **Untuk melihat hasil analisis dan grafik secara lengkap:**
> 1. Buka file notebook tersebut di **Google Colab**.
> 2. Pilih menu **Runtime** > **Run all** (Jalankan semua).
> 
> 

---

## Teknologi yang Digunakan (Tech Stack)

* **Bahasa Pemrograman:** Python 3.9+
* **Pengolahan Data:** Pandas, NumPy
* **Visualisasi Data:** Matplotlib, Seaborn, Plotly Express (Interaktif)
* **Machine Learning:** Scikit-Learn (K-Means Clustering, StandardScaler)
* **Deployment & Dashboard:** Streamlit
* **Model Persistence:** Joblib

---

## Panduan Instalasi & Menjalankan (Getting Started)

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer lokal Anda. Panduan ini kompatibel dengan Windows (WSL/CMD), macOS, dan Linux.

### 1. Clone Repository

Unduh kode sumber proyek ini dari GitHub:

```bash
git clone [https://github.com/romiaprilian7406/submission-bike-sharing.git](https://github.com/romiaprilian7406/submission-bike-sharing.git)
cd submission-bike-sharing

```

### 2. Setup Virtual Environment (Disarankan)

Sangat disarankan menggunakan *Virtual Environment* agar library tidak bentrok dengan sistem utama.

* **Windows (Command Prompt):**

```bash
python -m venv venv
venv\Scripts\activate

```

* **macOS / Linux / WSL:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Library (Dependencies)

Install semua library yang dibutuhkan sekaligus:

```bash
pip install -r requirements.txt

```

### 4. Menjalankan Dashboard

Jalankan aplikasi Streamlit dengan perintah berikut:

```bash
streamlit run dashboard/dashboard.py

```

* Jika berhasil, browser akan otomatis terbuka di alamat `http://localhost:8501`.
* Jika tidak terbuka otomatis, salin URL tersebut ke browser Anda.

---

## Fitur Dashboard

Dashboard ini dirancang dengan antarmuka yang ramah pengguna (*user-friendly*) dan memiliki fitur:

1. **Sidebar Filter Pintar:**
* **Rentang Tanggal:** Filter data berdasarkan periode waktu tertentu.
* **Filter Musim & Cuaca:** Menggunakan *Checkbox* interaktif untuk memilih kondisi spesifik.


2. **Key Performance Indicators (KPI):** Menampilkan metrik utama seperti Total Sewa, Rata-rata Harian, dan Rekor Tertinggi secara real-time mengikuti filter.
3. **Tab 1 - Business Insights:** Berisi visualisasi data interaktif (Heatmap Korelasi, Tren Harian, Pola Jam, Dampak Cuaca).
4. **Tab 2 - Cluster Visualization:** Peta visualisasi hasil algoritma AI (K-Means) yang menunjukkan pembagian zona waktu operasional.
5. **Tab 3 - Conclusion:** Ringkasan eksekutif dan rekomendasi bisnis.

---

## Hasil Analisis & Kesimpulan (Insights)

Berdasarkan analisis data menyeluruh, berikut adalah temuan kunci yang diperoleh:

### A. Jawaban Analisis Deskriptif (Exploratory Data Analysis)

**1. Pola Waktu: Bagaimana perbedaan karakteristik antara Hari Kerja vs Hari Libur?**

* **Hari Kerja (*Working Day*):** Memiliki pola "Bimodal" (Dua Puncak) yang sangat jelas pada pukul **08:00** dan **17:00-18:00**. Ini mengindikasikan penggunaan dominan untuk aktivitas komuter (pergi-pulang kerja/sekolah).
* **Hari Libur (*Holiday/Weekend*):** Memiliki pola "Unimodal" (Satu Puncak) yang landai, dimulai dari pukul **10:00 hingga 16:00**. Ini mengindikasikan penggunaan untuk rekreasi atau olahraga santai.

**2. Pertumbuhan Bisnis: Bagaimana tren performa dari tahun 2011 ke 2012?**

* Bisnis mengalami **pertumbuhan yang signifikan**. Total penyewaan pada tahun 2012 jauh lebih tinggi dibandingkan tahun 2011 di hampir seluruh bulan. Hal ini menunjukkan bahwa strategi ekspansi pasar berhasil dan popularitas *bike sharing* semakin meningkat.

**3. Faktor Musim: Musim apa yang menjadi "Masa Panen" (*Peak Season*)?**

* **Musim Gugur (*Fall*)** adalah masa puncak penyewaan tertinggi, diikuti oleh Musim Panas (*Summer*).
* Sebaliknya, **Musim Semi (*Spring*)** memiliki tingkat penyewaan terendah. Strategi pemasaran harus lebih agresif di musim semi untuk mendongkrak angka penggunaan.

**4. Sensitivitas Cuaca: Seberapa besar dampak cuaca buruk?**

* Kondisi cuaca memiliki dampak **sangat signifikan**. Penyewaan tertinggi terjadi saat cuaca cerah (*Clear*).
* Saat cuaca memburuk menjadi Hujan Ringan/Salju (*Light Snow/Rain*), terjadi penurunan drastis pada jumlah penyewa. Ini adalah risiko operasional yang tidak bisa dihindari namun bisa diprediksi.

**5. Segmentasi Pelanggan: Bagaimana perilaku Pengguna Terdaftar vs Casual?**

* **Pengguna Terdaftar (*Registered*):** Pola penggunaan sangat stabil dan mengikuti jam kerja (Senin-Jumat). Mereka adalah basis pelanggan setia yang menggunakan sepeda sebagai moda transportasi utama.
* **Pengguna Kasual (*Casual*):** Pola penggunaan lebih fluktuatif, meningkat tajam di akhir pekan dan sangat sensitif terhadap cuaca. Mereka adalah target pasar potensial untuk sektor pariwisata.

### B. Jawaban Analisis Lanjutan (Advanced Modeling)

**6. Segmentasi Jam Sibuk: Bagaimana hasil pengelompokan otomatis menggunakan K-Means?**
Algoritma K-Means berhasil mengelompokkan jam operasional menjadi 3 kategori strategi yang berbeda:

| Label Cluster | Jam Dominan | Action Plan (Rekomendasi Bisnis) |
| --- | --- | --- |
| **High Demand** | 08:00 & 17:00-18:00 | **Wajib Full Stock.** Prioritas utama staf lapangan. |
| **Medium Demand** | 10:00-16:00 & 19:00-22:00 | **Operasional Normal.** Waktu tepat untuk *maintenance* ringan di stasiun. |
| **Low Demand** | 00:00-06:00 | **Efisiensi Biaya.** Kurangi shift malam dan lakukan distribusi ulang (*rebalancing*) sepeda menggunakan truk. |

---

## Profil Penulis (Author)

**Romi Aprilian Mustafa**

* **Program:** Google Developer Groups on Campus (GDGoC) Universitas Sriwijaya - Machine Learning Division
* **Project:** Bike Sharing Analytics
* **Email:** [romiaprilian7406@gmail.com](mailto:romiaprilian7406@gmail.com)
* **LinkedIn:** [Romi Aprilian Mustafa](https://www.linkedin.com/in/romiapr/)

---

*Dokumentasi ini dibuat untuk melengkapi Submission Proyek GDGoC Universitas Sriwijaya - Machine Learning.*

```

```
