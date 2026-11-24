import streamlit as st

from core import ANGGOTA, DOSEN, card, get_art, get_df

df, art = get_df(), get_art()
churn_rate = (df["Churn"] == "Yes").mean()

st.title("Analisis & Prediksi Customer Churn Telco")
st.caption("Implementasi Machine Learning berbasis web dengan Streamlit")
st.markdown('<div class="chips"><span class="chip">Python</span><span class="chip">scikit-learn</span>'
            '<span class="chip">SMOTE</span><span class="chip">PCA</span><span class="chip">Streamlit</span></div>',
            unsafe_allow_html=True)
st.markdown(
    "Website ini menyajikan seluruh tahapan proyek analisis dan prediksi pelanggan yang "
    "berpotensi berhenti berlangganan (*customer churn*) menggunakan dataset *Telco Customer "
    "Churn*. Proses yang ditampilkan meliputi :blue[pengenalan data], :blue[eksplorasi data], "
    ":orange[preprocessing], :orange[reduksi dimensi menggunakan PCA], serta perbandingan "
    "performa :green[lima algoritma klasifikasi]. Selain menampilkan hasil analisis dan evaluasi "
    "model, website ini juga dilengkapi **fitur prediksi langsung** yang memungkinkan pengguna "
    "memasukkan data pelanggan untuk mengetahui potensi *churn*. Dengan tampilan yang terstruktur "
    "dan interaktif, website ini memberikan gambaran penerapan *machine learning* dalam "
    "menganalisis dan memprediksi *customer churn*."
)
st.markdown("**Tujuan proyek**")
st.markdown(
    f"- Mengidentifikasi faktor yang paling memengaruhi keputusan pelanggan untuk berhenti.\n"
    f"- Membangun model yang andal meski hanya **{churn_rate:.1%}** pelanggan yang churn.\n"
    f"- Menyediakan alat prediksi untuk satu pelanggan maupun banyak pelanggan sekaligus (CSV).")

best = art["best_model"] if art else "-"
auc = f"{art['results'][best]['tuned']['roc_auc']:.3f}" if art else "-"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah pelanggan", f"{len(df):,}")
c2.metric("Churn rate", f"{churn_rate:.1%}")
c3.metric("Model terbaik (F1)", best)
c4.metric("ROC-AUC model terbaik", auc)

st.divider()
st.subheader("Tahapan Analisis")
n_cols = art["prep_info"]["n_cols_total"] if art else df.shape[1] - 1
pca_text = "Reduksi dimensi dengan PCA."
if art:
    p = art["pca_info"]
    pca_text = (f"{p['n_features']} fitur dipadatkan menjadi {p['k']} komponen "
                f"(selisih F1 vs tanpa PCA: {p['delta_f1_best']:+.3f}).")
cols = st.columns(4)
steps = [
    ("1. Dataset", f"Karakteristik data: {len(df):,} baris, {n_cols} kolom, klasifikasi biner."),
    ("2. EDA & Preprocessing", "Menangani data hilang, data leakage, ketidakseimbangan kelas, dan encoding."),
    ("3. Reduksi PCA", pca_text),
    ("4. Modelling", "Lima algoritma dibandingkan, di-tuning, lalu dipakai untuk prediksi."),
]
for col, (t, b) in zip(cols, steps):
    col.markdown(card(t, b), unsafe_allow_html=True)

st.markdown("**Mulai dari sini**")
l1, l2, l3 = st.columns(3)
l1.page_link("views/eda.py", label="Jelajahi data", icon=":material/bar_chart:")
l2.page_link("views/modelling.py", label="Lihat hasil model", icon=":material/model_training:")
l3.page_link("views/prediksi_satu.py", label="Coba prediksi", icon=":material/person_search:")

st.divider()
st.subheader("Anggota Kelompok 8")
cols = st.columns(3)
for col, (nama, nim) in zip(cols, ANGGOTA):
    col.markdown(card(nama, "", f"NIM {nim}"), unsafe_allow_html=True)
st.markdown(f"Dosen pengampu: **{DOSEN}**")
st.caption("Program Studi Informatika, Fakultas Ilmu Komputer, Universitas Pamulang")

with st.expander("Mengapa memprediksi churn penting?"):
    st.markdown(
    f"- **Mempertahankan pelanggan** umumnya lebih efisien dibandingkan terus mencari pelanggan baru, sehingga prediksi *churn* dapat membantu mengurangi potensi kerugian.\n"
    f"- **Prediksi lebih awal** memungkinkan tim retensi mengidentifikasi pelanggan berisiko dan memberikan penawaran atau strategi yang lebih tepat sasaran.\n"
    f"- **Data churn tidak seimbang**, dengan pelanggan yang berhenti berlangganan hanya sekitar **{churn_rate:.1%}**. Oleh karena itu, performa model tidak cukup dinilai dari akurasi, tetapi juga menggunakan **Recall, F1-Score, dan ROC-AUC** untuk memberikan evaluasi yang lebih menyeluruh."
)
