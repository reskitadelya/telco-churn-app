"""app.py - titik masuk & navigasi. Jalankan: streamlit run app.py"""
import streamlit as st

from core import inject_css

st.set_page_config(page_title="Analisis & Prediksi Customer Churn Telco", page_icon="📉", layout="centered")
inject_css()

pages = {
    "Pengantar": [
        st.Page("views/beranda.py", title="Beranda", icon=":material/home:", default=True),
        st.Page("views/dataset.py", title="Tentang Dataset", icon=":material/database:"),
        st.Page("views/eda.py", title="Eksplorasi Data (EDA)", icon=":material/bar_chart:"),
    ],
    "Pengolahan & Model": [
        st.Page("views/preprocessing.py", title="Preprocessing", icon=":material/tune:"),
        st.Page("views/pca.py", title="Reduksi Dimensi (PCA)", icon=":material/scatter_plot:"),
        st.Page("views/modelling.py", title="Modelling & Evaluasi", icon=":material/model_training:"),
    ],
    "Penerapan": [
        st.Page("views/prediksi_satu.py", title="Prediksi Satu Pelanggan", icon=":material/person_search:"),
        st.Page("views/prediksi_massal.py", title="Prediksi Massal (CSV)", icon=":material/description:"),
        st.Page("views/roi.py", title="Kalkulator Untung-Rugi", icon=":material/paid:"),
        st.Page("views/kesimpulan.py", title="Kesimpulan & Saran", icon=":material/task_alt:"),
    ],
}
pg = st.navigation(pages)
st.sidebar.divider()
st.sidebar.caption("Tugas Machine Learning - Kelompok 8 - Teknik Informatika, Universitas Pamulang")
pg.run()
