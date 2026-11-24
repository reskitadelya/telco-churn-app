import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import metrics_table, require_art

art = require_art()
res, best = art["results"], art["best_model"]

st.title("Modelling & Evaluasi")
st.caption("Lima algoritma dibandingkan, di-tuning dengan RandomizedSearchCV (3-fold, skor F1)")

st.info(f"Model terbaik berdasarkan **F1 validasi silang**: **{best}** "
        f"(CV-F1 = {res[best]['cv_f1']:.3f}). Pemilihan memakai validasi silang, bukan data uji.")

st.subheader("Perbandingan model")
tab_t, tab_b = st.tabs(["Setelah tuning", "Sebelum tuning (default)"])
fmt = "{:.3f}"
with tab_t:
    t = metrics_table(art, "tuned")
    st.dataframe(t.style.format(fmt).highlight_max(axis=0, color="#CDEEFB"), width="stretch")
with tab_b:
    st.dataframe(metrics_table(art, "baseline").style.format(fmt).highlight_max(axis=0, color="#CDEEFB"),
                 width="stretch")

long = t.reset_index().melt("Model", var_name="Metrik", value_name="Nilai")
st.plotly_chart(px.bar(long, x="Metrik", y="Nilai", color="Model", barmode="group",
                       title="Metrik pada data uji (setelah tuning)"), width="stretch")

st.subheader("Kestabilan model (validasi silang 3-fold)")
stab = pd.DataFrame({"Model": list(res),
                     "CV-F1 rata-rata": [r["cv_f1"] for r in res.values()],
                     "Std": [r["cv_f1_std"] for r in res.values()]})
fig = px.bar(stab, x="Model", y="CV-F1 rata-rata", error_y="Std", text=stab["CV-F1 rata-rata"].round(3),
             color_discrete_sequence=["#1677FF"], title="CV-F1 (batang error = 1 simpangan baku)")
fig.update_yaxes(range=[0.5, 0.75])
st.plotly_chart(fig, width="stretch")
mirip = [n for n in res if n != best and res[best]["cv_f1"] - res[n]["cv_f1"] <= res[best]["cv_f1_std"]]
if mirip:
    st.info(f"**{', '.join(mirip)}** berada dalam satu simpangan baku dari **{best}**. Artinya selisihnya "
            f"belum bermakna secara praktis: kinerja model-model tersebut dapat dianggap setara.")
else:
    st.success(f"**{best}** unggul lebih dari satu simpangan baku dibanding model lain.")
with st.expander("Hyperparameter terbaik tiap model"):
    st.dataframe(pd.DataFrame({"Model": list(res), "Parameter": [str(r["best_params"]) for r in res.values()]})
                 .set_index("Model"), width="stretch")

st.subheader(f"Studi ablasi ({best})")
st.caption("Mengukur kontribusi tiap tahap: PCA dan SMOTE dinyalakan/dimatikan (hyperparameter sama, CV 5-fold).")
ab = pd.DataFrame(art["ablation"])
st.dataframe(ab.style.format({c: "{:.3f}" for c in ab.columns if c not in ("PCA", "SMOTE")}),
             hide_index=True, width="stretch")
nn, ns = ab[(ab.PCA == "Tidak") & (ab.SMOTE == "Tidak")].iloc[0], ab[(ab.PCA == "Tidak") & (ab.SMOTE == "Ya")].iloc[0]
pn, ps = ab[(ab.PCA == "Ya") & (ab.SMOTE == "Tidak")].iloc[0], ab[(ab.PCA == "Ya") & (ab.SMOTE == "Ya")].iloc[0]
st.markdown(
    f"- **Pengaruh SMOTE** (tanpa PCA): Test F1 {nn['Test F1']:.3f} -> {ns['Test F1']:.3f}, "
    f"Recall {nn['Test Recall']:.3f} -> {ns['Test Recall']:.3f}.\n"
    f"- **Pengaruh PCA** (dengan SMOTE): Test F1 {ns['Test F1']:.3f} -> {ps['Test F1']:.3f}. "
    f"Tanpa SMOTE: {nn['Test F1']:.3f} -> {pn['Test F1']:.3f}.")

st.subheader("Kurva ROC")
fig = go.Figure()
for n, r in res.items():
    fig.add_scatter(x=r["tuned"]["roc"]["fpr"], y=r["tuned"]["roc"]["tpr"], mode="lines",
                    name=f"{n} (AUC {r['tuned']['roc_auc']:.3f})", line=dict(width=4 if n == best else 2))
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak", line=dict(dash="dash", color="grey"))
fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, width="stretch")

st.subheader("Confusion matrix")
pick = st.selectbox("Model", list(res), index=list(res).index(best))
cm = np.array(res[pick]["tuned"]["cm"])
a, b = st.columns([3, 2])
a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["Prediksi: Tidak", "Prediksi: Churn"], y=["Aktual: Tidak", "Aktual: Churn"]),
               width="stretch")
tn, fp, fn, tp = cm.ravel()
b.markdown(f"- **TP** {tp}: churn terdeteksi\n- **FN** {fn}: churn terlewat\n- **FP** {fp}: salah alarm\n- **TN** {tn}: benar bertahan")
b.caption("Dalam retensi, FN (pelanggan churn tak terdeteksi) umumnya lebih mahal daripada FP, sehingga Recall penting.")

st.subheader(f"Fitur terpenting ({best})")
imp = art["importance"].head(12)
fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=["#1677FF"],
             labels={"importance": "Penurunan ROC-AUC saat diacak", "feature": ""})
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=440)
st.plotly_chart(fig, width="stretch")
