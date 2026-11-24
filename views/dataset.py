import pandas as pd
import streamlit as st

from core import get_art, get_df
from train_model import DROP_REASON, TARGET

df = get_df().drop(columns="Churn")
art = get_art()

st.title("Tentang Dataset")
st.caption("IBM Telco Customer Churn - data pelanggan telekomunikasi di California, Amerika Serikat")
st.markdown("Dataset berisi profil demografis, layanan yang dipakai, informasi kontrak dan tagihan, "
            "serta status churn tiap pelanggan. **Target** prediksi adalah kolom `Churn Label` (Yes/No).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baris", f"{len(df):,}")
c2.metric("Kolom", df.shape[1])
c3.metric("Fitur dipakai model", art["prep_info"]["n_features_raw"] if art else "-")
c4.metric("Kolom dibuang", len(DROP_REASON))

st.subheader("Kamus Data")
rows = []
for c in df.columns:
    if c == TARGET:
        role, ket = "Target", "Label churn (Yes = berhenti berlangganan)"
    elif c in DROP_REASON:
        role, ket = "Dibuang", DROP_REASON[c]
    else:
        role, ket = "Fitur", "Dipakai sebagai input model"
    rows.append({"Kolom": c, "Tipe": "Numerik" if pd.api.types.is_numeric_dtype(df[c]) else "Kategorikal",
                 "Nilai Unik": df[c].nunique(), "Peran": role, "Keterangan": ket})
dd = pd.DataFrame(rows)
filt = st.multiselect("Filter peran", ["Fitur", "Dibuang", "Target"], default=["Fitur", "Dibuang", "Target"])
st.dataframe(dd[dd["Peran"].isin(filt)], width="stretch", hide_index=True, height=420)

st.subheader("Cuplikan Data")
st.dataframe(df.head(10), width="stretch")
st.subheader("Statistik Deskriptif (numerik)")
st.dataframe(df.describe().T.round(2), width="stretch")
