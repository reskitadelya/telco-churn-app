import pandas as pd
import plotly.express as px
import streamlit as st

from core import get_df, require_art
from train_model import DROP_REASON

art, df = require_art(), get_df()
pi = art["prep_info"]

st.title("Preprocessing")
st.caption("Menyiapkan data agar valid dan siap dimodelkan")

st.subheader("1. Data hilang (missing values)")
miss = pd.DataFrame({"Kolom": list(pi["missing"]), "Jumlah kosong": list(pi["missing"].values())})
miss["Penanganan"] = miss["Kolom"].map({
    "Offer": "Kosong = tidak ada penawaran -> diisi 'No Offer'",
    "Internet Type": "Kosong = tidak berlangganan internet -> diisi 'No Internet'",
    "Churn Category": "Kosong = pelanggan tidak churn -> kolom dibuang (leakage)",
    "Churn Reason": "Kosong = pelanggan tidak churn -> kolom dibuang (leakage)"})
st.dataframe(miss, hide_index=True, width="stretch")
st.caption("Nilai kosong pada kolom ini bermakna (bukan data rusak), sehingga diberi label eksplisit, bukan dihapus.")

st.subheader("2. Data leakage")
st.markdown("*Leakage* terjadi bila fitur memuat informasi yang baru diketahui **setelah** pelanggan churn. "
            "Model tampak sangat akurat, tetapi tidak berguna untuk prediksi nyata. Contoh paling jelas:")
sat = (df.groupby("Satisfaction Score")["Churn"].apply(lambda s: (s == "Yes").mean())
       .reset_index(name="Churn Rate"))
fig = px.bar(sat, x="Satisfaction Score", y="Churn Rate", text=sat["Churn Rate"].map("{:.0%}".format),
             title="Churn rate per Satisfaction Score", color_discrete_sequence=["#1677FF"])
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig, width="stretch")
leak = [(c, r) for c, r in DROP_REASON.items() if r.startswith("Leakage")]
st.dataframe(pd.DataFrame(leak, columns=["Kolom dibuang", "Alasan"]), hide_index=True, width="stretch")

st.subheader("3. Seleksi fitur")
st.markdown(f"Dari **{pi['n_cols_total']} kolom**, dibuang **{len(DROP_REASON)}** kolom "
            f"(leakage, identitas, konstan, lokasi granular, dan turunan). Tersisa "
            f"**{pi['n_features_raw']} fitur** ({len(pi['num_cols'])} numerik, {len(pi['cat_cols'])} kategorikal).")
with st.expander("Lihat alasan seluruh kolom yang dibuang"):
    st.dataframe(pd.DataFrame(DROP_REASON.items(), columns=["Kolom", "Alasan"]), hide_index=True, width="stretch")

st.subheader("4. Pembagian data")
st.markdown(f"Stratified split **80 : 20** (`random_state=42`): **{pi['n_train']:,}** data latih dan "
            f"**{pi['n_test']:,}** data uji. Proporsi churn dijaga sama pada kedua bagian.")

st.subheader("5. Encoding & scaling")
st.markdown(f"- **Numerik** ({len(pi['num_cols'])} kolom): `StandardScaler`\n"
            f"- **Kategorikal** ({len(pi['cat_cols'])} kolom): `OneHotEncoder` -> total **{art['pca_info']['n_features']} fitur**\n"
            "- Dilakukan di dalam `Pipeline` sehingga hanya *belajar* dari data latih (tanpa kebocoran ke data uji).")

st.subheader("6. Ketidakseimbangan kelas (SMOTE)")
sm = pi["smote"]
d = pd.DataFrame({"Tahap": ["Sebelum SMOTE"] * 2 + ["Sesudah SMOTE"] * 2,
                  "Kelas": ["Tidak churn", "Churn"] * 2, "Jumlah": sm["before"] + sm["after"]})
st.plotly_chart(px.bar(d, x="Tahap", y="Jumlah", color="Kelas", barmode="group", text="Jumlah",
                       color_discrete_sequence=["#35C6F4", "#0B1F3A"]), width="stretch")
st.markdown("SMOTE membuat sampel sintetis kelas churn hanya pada **data latih** (di dalam validasi silang). "
            "Data uji tidak disentuh sehingga evaluasi tetap jujur.")

with st.expander("Kode pipeline"):
    st.code("""Pipeline([
    ("prep",  ColumnTransformer([("num", StandardScaler(), num_cols),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])),
    ("pca",   PCA(n_components=k)),
    ("smote", SMOTE(random_state=42)),
    ("clf",   classifier),
])""", language="python")
