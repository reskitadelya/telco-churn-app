import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import COLORS, get_df, require_art

art, df = require_art(), get_df()
p = art["pca_info"]
evr = np.array(p["evr"])
cum = np.cumsum(evr)

st.title("Reduksi Dimensi (PCA)")
st.caption("Memadatkan fitur hasil encoding menjadi komponen utama yang saling bebas")
st.markdown("*Principal Component Analysis* mengubah fitur berkorelasi menjadi komponen utama (PC) yang "
            "terurut menurut varians. Dengan mempertahankan sebagian besar varians, jumlah fitur dapat dikurangi.")

c1, c2, c3 = st.columns(3)
c1.metric("Fitur setelah encoding", p["n_features"])
c2.metric(f"Komponen ({p['threshold']:.0%} varians)", p["k"])
c3.metric("Reduksi", f"{1 - p['k'] / p['n_features']:.0%}")

st.subheader("Varians yang dijelaskan")
fig = go.Figure()
fig.add_bar(x=list(range(1, len(evr) + 1)), y=evr, name="Per komponen", marker_color="#1677FF")
fig.add_scatter(x=list(range(1, len(evr) + 1)), y=cum, name="Kumulatif", mode="lines+markers",
                line=dict(color="#35C6F4"))
fig.add_hline(y=p["threshold"], line_dash="dash", line_color="grey", annotation_text=f"{p['threshold']:.0%}")
fig.add_vline(x=p["k"], line_dash="dot", line_color="grey", annotation_text=f"k = {p['k']}")
fig.update_layout(xaxis_title="Komponen utama", yaxis_title="Proporsi varians", yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch")

st.subheader("Proyeksi 2 komponen pertama")
pipe = art["pipeline"]
sample = df.sample(2000, random_state=3)
Z = pipe.named_steps["pca"].transform(pipe.named_steps["prep"].transform(sample[art["features"]]))
proj = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "Churn": sample["Churn"].values})
st.plotly_chart(px.scatter(proj, x="PC1", y="PC2", color="Churn", opacity=.55, color_discrete_map=COLORS),
                width="stretch")
st.caption("Dua komponen pertama belum memisahkan kelas sepenuhnya, jadi model non-linear dan komponen lain tetap diperlukan.")

st.subheader("Fitur yang paling membentuk komponen")
pc = st.selectbox("Komponen", [f"PC{i}" for i in range(1, 6)])
load = pd.Series(pipe.named_steps["pca"].components_[int(pc[2:]) - 1], index=p["feature_names"])
top = load.reindex(load.abs().sort_values(ascending=False).head(10).index).reset_index()
top.columns = ["Fitur", "Loading"]
fig = px.bar(top, x="Loading", y="Fitur", orientation="h", color_discrete_sequence=["#1677FF"])
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
st.plotly_chart(fig, width="stretch")

st.subheader("Apakah PCA menurunkan kinerja?")
rows = []
for name, r in art["results"].items():
    a, b = r["baseline"], r["baseline_nopca"]
    rows.append({"Model": name, "F1 dengan PCA": a["f1"], "F1 tanpa PCA": b["f1"], "Selisih F1": a["f1"] - b["f1"],
                 "AUC dengan PCA": a["roc_auc"], "AUC tanpa PCA": b["roc_auc"]})
tbl = pd.DataFrame(rows).set_index("Model")
st.dataframe(tbl.style.format("{:.3f}").format("{:+.3f}", subset=["Selisih F1"]), width="stretch")
st.caption("Perbandingan memakai hyperparameter bawaan pada data uji. Selisih mendekati nol berarti informasi penting "
           "tetap terjaga setelah reduksi; selisih negatif besar berarti PCA membuang sinyal yang berguna.")
