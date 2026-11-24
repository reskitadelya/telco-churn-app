import numpy as np
import plotly.express as px
import streamlit as st

from core import get_df, require_art, risk_label

art, df = require_art(), get_df()
feats = art["features"]

st.title("Prediksi Massal (CSV)")
st.caption("Prediksi banyak pelanggan sekaligus dan urutkan berdasarkan risiko")

template = df[feats].head(5)
st.download_button("Unduh template CSV", template.to_csv(index=False).encode("utf-8"), "template_prediksi.csv", "text/csv")
st.markdown("File harus memuat kolom fitur seperti template. Kolom lain (mis. Customer ID) boleh ada dan akan diabaikan.")

up = st.file_uploader("Unggah file CSV", type="csv")
demo = st.checkbox("Gunakan 200 baris contoh dari dataset")
data = None
if up is not None:
    import pandas as pd
    data = pd.read_csv(up)
    if "Offer" in data:
        data["Offer"] = data["Offer"].fillna("No Offer")
    if "Internet Type" in data:
        data["Internet Type"] = data["Internet Type"].fillna("No Internet")
elif demo:
    data = df.sample(200, random_state=7).drop(columns=["Churn"])

if data is not None:
    missing = [c for c in feats if c not in data.columns]
    if missing:
        st.error(f"Kolom berikut tidak ditemukan: {missing}")
        st.stop()
    with st.spinner("Memproses prediksi massal..."):
        proba = art["pipeline"].predict_proba(data[feats])[:, 1]
    out = data.copy()
    out["Probabilitas Churn"] = proba.round(4)
    out["Prediksi"] = np.where(proba >= 0.5, "Churn", "Bertahan")
    out["Level Risiko"] = [risk_label(p)[0] for p in proba]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah pelanggan", f"{len(out):,}")
    c2.metric("Diprediksi churn", f"{(out['Prediksi'] == 'Churn').sum():,}")
    c3.metric("Risiko tinggi", f"{(out['Level Risiko'] == 'TINGGI').sum():,}")
    st.plotly_chart(px.histogram(out, x="Probabilitas Churn", nbins=30, color="Prediksi",
                                 color_discrete_map={"Churn": "#0B1F3A", "Bertahan": "#35C6F4"}), width="stretch")
    show = out.sort_values("Probabilitas Churn", ascending=False)
    st.dataframe(show, width="stretch")
    st.download_button("Unduh hasil (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "hasil_prediksi_churn.csv", "text/csv")
