import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Bike Sharing Analytics",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. LOAD DATA & MODEL
@st.cache_data
def load_data():
    # Setup Path (Aman untuk dijalankan dari mana saja)
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'hour.csv')
    
    if not os.path.exists(file_path):
        file_path = os.path.join(base_path, '../data/hour.csv')
    
    # 1. Load Hour Data
    hour_df = pd.read_csv(file_path)
    
    # Cleaning & Mapping for HOUR
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    # Mapping Dictionary
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    year_map = {0: 2011, 1: 2012}
    working_map = {0: 'Holiday/Weekend', 1: 'Working Day'}
    weather_map = {1: 'Clear', 2: 'Mist', 3: 'Light Snow/Rain', 4: 'Heavy Rain'}

    hour_df['season_label'] = hour_df['season'].map(season_map)
    hour_df['year_label'] = hour_df['yr'].map(year_map)
    hour_df['workingday_label'] = hour_df['workingday'].map(working_map)
    hour_df['weather_label'] = hour_df['weathersit'].map(weather_map)

    # 2. Create Day Data (Aggregating from Hour on the fly)
    day_df = hour_df.groupby('dteday').agg({
        'season': 'first',
        'yr': 'first',
        'mnth': 'first',
        'holiday': 'first',
        'weekday': 'first',
        'workingday': 'first',
        'weathersit': 'max', 
        'temp': 'mean',
        'atemp': 'mean',
        'hum': 'mean',
        'windspeed': 'mean',
        'casual': 'sum',
        'registered': 'sum',
        'cnt': 'sum'
    }).reset_index()
    
    # Mapping for DAY
    day_df['season_label'] = day_df['season'].map(season_map)
    day_df['year_label'] = day_df['yr'].map(year_map)
    day_df['weather_label'] = day_df['weathersit'].map(weather_map)
    
    return hour_df, day_df

@st.cache_resource
def load_model():
    try:
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, '../models/kmeans_bike_cluster.joblib')
        scaler_path = os.path.join(base_path, '../models/scaler_bike_cluster.joblib')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        return None, None

# Load Resources
hour_df_org, day_df_org = load_data()
model, scaler = load_model()

# 3. SIDEBAR (FILTER CHECKBOX)
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1485965120184-e220f721d03e?q=80&w=1000&auto=format&fit=crop", caption="Bike Sharing Dashboard")
    
    st.header("⚙️ Filter Data")
    
    # --- 1. Date Range ---
    min_date = day_df_org['dteday'].min()
    max_date = day_df_org['dteday'].max()
    
    start_date, end_date = st.date_input(
        "Rentang Tanggal:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("---")

    # --- 2. Season Filter (CHECKBOX STYLE) ---
    st.subheader("Filter Musim")
    season_options = ['Spring', 'Summer', 'Fall', 'Winter']
    season_filter = []
    
    # Buat checkbox untuk setiap musim
    for season in season_options:
        if st.checkbox(season, value=True): # Default True (Terisi)
            season_filter.append(season)

    st.markdown("---")

    # --- 3. Weather Filter (CHECKBOX STYLE) ---
    st.subheader("Filter Cuaca")
    weather_options = ['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain']
    weather_filter = []
    
    # Buat checkbox untuk setiap cuaca
    for weather in weather_options:
        # Cek apakah data cuaca tersebut ada di dataset agar tidak membuat checkbox kosong
        if weather in day_df_org['weather_label'].unique():
            if st.checkbox(weather, value=True):
                weather_filter.append(weather)
    
    st.caption("Created by: **Romi Aprilian Mustafa**")

# 4. FILTERING LOGIC
# Filter Day Data
day_df = day_df_org[
    (day_df_org['dteday'].dt.date >= start_date) &
    (day_df_org['dteday'].dt.date <= end_date) &
    (day_df_org['season_label'].isin(season_filter)) &
    (day_df_org['weather_label'].isin(weather_filter))
]

# Filter Hour Data
hour_df = hour_df_org[
    (hour_df_org['dteday'].dt.date >= start_date) &
    (hour_df_org['dteday'].dt.date <= end_date) &
    (hour_df_org['season_label'].isin(season_filter)) &
    (hour_df_org['weather_label'].isin(weather_filter))
]

# 5. MAIN CONTENT
st.title("📊 Bike Sharing Business Analytics")
st.markdown(f"Data Analysis Period: **{start_date}** to **{end_date}**")

# KPI Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Penyewaan", f"{day_df['cnt'].sum():,.0f}")
col2.metric("Rata-rata Harian", f"{day_df['cnt'].mean():,.0f}")
col3.metric("Hari Tersibuk", f"{day_df['cnt'].max()} Unit")
col4.metric("Registered Users", f"{day_df['registered'].sum():,.0f}")

st.markdown("---")

# TABS
tab1, tab2, tab3 = st.tabs(["📈 Business Insights (EDA)", "🧩 Cluster Visualization", "📝 Conclusion"])

# TAB 1: BUSINESS INSIGHTS (MATPLOTLIB/SEABORN GRID)
with tab1:
    st.header("Exploratory Data Analysis")
    st.write("Visualisasi menyeluruh terkait korelasi, tren waktu, musim, dan perilaku pengguna.")
    
    if len(day_df) > 0 and len(hour_df) > 0:
        with st.container():
            # Setup Figure
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

            # PLOT 0: KORELASI HEATMAP
            numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
            corr_matrix = day_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax[0, 0])
            ax[0, 0].set_title('Korelasi Matriks: Hubungan Antar Variabel', fontsize=18)

            # GRAFIK 1: Pola Jam Kerja vs Libur 
            sns.pointplot(data=hour_df, x='hr', y='cnt', hue='workingday_label',
                        palette='coolwarm', errorbar=None, ax=ax[0, 1])
            ax[0, 1].set_title('Pola Jam Sewa (Commuter vs Leisure)', fontsize=14)
            ax[0, 1].set_xlabel('Jam')
            ax[0, 1].grid(True)

            # GRAFIK 2: Tren Tahunan
            sns.lineplot(data=day_df, x='mnth', y='cnt', hue='year_label',
                        palette='viridis', marker='o', ax=ax[1, 0])
            ax[1, 0].set_title('Pertumbuhan Bisnis (2011 vs 2012)', fontsize=14)
            ax[1, 0].set_xlabel('Bulan')
            ax[1, 0].set_xticks(range(1, 13))
            ax[1, 0].grid(True)

            # GRAFIK 3: Musim (Seasonality) 
            sns.barplot(data=day_df, x='season_label', y='cnt',
                        palette='autumn', order=['Spring', 'Summer', 'Fall', 'Winter'], ax=ax[1, 1])
            ax[1, 1].set_title('Rata-rata Penyewaan per Musim', fontsize=14)
            ax[1, 1].set_xlabel('Musim')

            # GRAFIK 4: Dampak Cuaca (Boxplot) 
            sns.boxplot(data=day_df, x='weather_label', y='cnt', hue='weather_label', palette='Set2',
                        order=['Clear', 'Mist', 'Light Snow/Rain', 'Heavy Rain'], legend=False, ax=ax[2, 0])
            ax[2, 0].set_title('Sebaran Penyewaan Berdasarkan Cuaca', fontsize=14)
            ax[2, 0].set_xlabel('Kondisi Cuaca')

            # GRAFIK 5: User Registered vs Casual (Hourly Pattern)
            user_hour = hour_df.groupby('hr')[['casual', 'registered']].mean().reset_index()
            user_hour_melt = user_hour.melt(id_vars='hr', var_name='User Type', value_name='Count')

            sns.lineplot(data=user_hour_melt, x='hr', y='Count', hue='User Type',
                        palette='magma', linewidth=2.5, ax=ax[2, 1])
            ax[2, 1].set_title('Perilaku User (Registered vs Casual)', fontsize=14)
            ax[2, 1].set_xlabel('Jam (0-23)')
            ax[2, 1].grid(True)

            st.pyplot(fig)
    else:
        st.warning("Data kosong. Mohon pilih minimal satu Filter Musim dan Cuaca di Sidebar.")

# TAB 2: CLUSTER VISUALIZATION
with tab2:
    st.header("Visualisasi Hasil Clustering (K-Means)")
    st.info("Peta persebaran data historis yang telah dikelompokkan oleh AI berdasarkan pola Waktu dan Jumlah Sewa.")
    
    if model is not None:
        if len(hour_df) > 0:
            # Sampling data jika terlalu banyak agar ringan
            if len(hour_df) > 2000:
                sample_df = hour_df.sample(2000, random_state=42).copy()
            else:
                sample_df = hour_df.copy()
            
            # Predict Cluster
            X_vis = sample_df[['hr', 'cnt']]
            X_vis_scaled = scaler.transform(X_vis)
            sample_df['cluster'] = model.predict(X_vis_scaled)
            
            # Mapping Label Cluster
            cluster_means = sample_df.groupby('cluster')['cnt'].mean().sort_values()
            
            label_map = {}
            # Dinamis mapping agar tidak error jika cluster berkurang krn filter
            if len(cluster_means) >= 1: label_map[cluster_means.index[0]] = 'Low Demand'
            if len(cluster_means) >= 2: label_map[cluster_means.index[1]] = 'Medium Demand'
            if len(cluster_means) >= 3: label_map[cluster_means.index[2]] = 'High Demand'
            
            sample_df['label'] = sample_df['cluster'].map(label_map)
            
            # Scatter Plot Plotly
            fig_cluster = px.scatter(
                sample_df, x='hr', y='cnt', color='label',
                color_discrete_map={'Low Demand': 'green', 'Medium Demand': 'orange', 'High Demand': 'red'},
                title="Segmentasi Jam Operasional (Low, Medium, High)",
                labels={'hr': 'Jam (0-23)', 'cnt': 'Jumlah Sewa'}
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Data kosong untuk Clustering. Periksa filter Anda.")
    else:
        st.error("Model tidak ditemukan. Pastikan folder 'models/' berisi file .joblib")

# TAB 3: CONCLUSION
with tab3:
    st.markdown("""
**Conclusion**

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

---

### B. Jawaban Analisis Lanjutan (Advanced Modeling)

**6. Segmentasi Jam Sibuk: Bagaimana hasil pengelompokan otomatis menggunakan K-Means?**
Algoritma K-Means berhasil mengelompokkan jam operasional menjadi 3 kategori strategi yang berbeda:
* **High Demand (Jam Sibuk):** Teridentifikasi pada jam **08:00** dan **17:00-18:00**.
    * *Action:* Stok sepeda harus penuh 100%. Prioritas utama staf lapangan.
* **Medium Demand (Jam Normal):** Teridentifikasi pada siang hari (**10:00-16:00**) dan malam hari (**19:00-22:00**).
    * *Action:* Operasional normal. Waktu yang tepat untuk *maintenance* ringan di stasiun.
* **Low Demand (Jam Sepi):** Teridentifikasi dominan pada dini hari (**00:00-06:00**).
    * *Action:* Efisiensi biaya (pengurangan shift malam) dan waktu terbaik untuk distribusi ulang (*rebalancing*) sepeda antar stasiun menggunakan truk.
    """)