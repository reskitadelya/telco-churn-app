import streamlit as st

from core import get_df, require_art

art, df = require_art(), get_df()
best = art["best_model"]
t = art["results"][best]["tuned"]
imp = art["importance"].head(3)["feature"].tolist()
rate = (df["Churn"] == "Yes").mean()
p = art["pca_info"]

st.title("Kesimpulan & Saran")

st.subheader("Kesimpulan")
st.markdown(
    f"1. Dataset memuat **{len(df):,} pelanggan** dengan churn rate **{rate:.1%}** (kelas tidak seimbang).\n"
    f"2. Kolom leakage (mis. *Satisfaction Score*, *Churn Score*, *CLTV*) dibuang agar model valid untuk prediksi nyata.\n"
    f"3. PCA memadatkan **{p['n_features']} fitur menjadi {p['k']} komponen** (95% varians), dengan selisih F1 "
    f"{p['delta_f1_best']:+.3f} pada model terbaik dibanding tanpa PCA.\n"
    f"4. Dari lima algoritma, **{best}** terbaik menurut F1 validasi silang. Pada data uji: "
    f"Accuracy **{t['accuracy']:.3f}**, Precision **{t['precision']:.3f}**, Recall **{t['recall']:.3f}**, "
    f"F1 **{t['f1']:.3f}**, ROC-AUC **{t['roc_auc']:.3f}**.\n"
    f"5. Faktor paling berpengaruh: **{', '.join(imp)}**.")

st.subheader("Saran bisnis")
st.markdown("- Dorong pelanggan *Month-to-Month* ke kontrak jangka panjang lewat insentif.\n"
            "- Prioritaskan retensi pada pelanggan baru (masa langganan pendek) dan segmen berisiko tinggi.\n"
            "- Tinjau harga dan kualitas layanan pada segmen dengan churn rate tertinggi (lihat halaman EDA).\n"
            "- Gunakan Prediksi Massal secara berkala agar tim retensi menghubungi pelanggan berisiko lebih dulu.")

st.subheader("Keterbatasan & saran pengembangan")
st.markdown("- Data berasal dari satu wilayah dan satu periode; model perlu dilatih ulang bila perilaku pelanggan berubah.\n"
            "- SMOTE menghasilkan data sintetis, sehingga Precision dapat turun saat Recall naik.\n"
            "- Coba XGBoost/LightGBM, penyetelan ambang keputusan sesuai biaya bisnis, dan interpretasi SHAP.\n"
            "- Tambahkan data kualitas jaringan atau keluhan pelanggan sebagai fitur baru.")
