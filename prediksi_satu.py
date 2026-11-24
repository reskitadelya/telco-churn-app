import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import require_art, risk_label

art = require_art()
meta, feats = art["feature_meta"], art["features"]
pipe = art["pipeline"]

st.title("Prediksi Satu Pelanggan")
st.caption(f"Isi profil pelanggan, lalu model **{art['best_model']}** menghitung probabilitas churn")

GROUPS = {
    "Profil": ["Gender", "Age", "Married", "Dependents", "Number of Dependents", "Referred a Friend",
               "Number of Referrals"],
    "Langganan": ["Tenure in Months", "Contract", "Offer", "Paperless Billing", "Payment Method", "Monthly Charge"],
    "Layanan": ["Phone Service", "Multiple Lines", "Internet Service", "Internet Type", "Online Security",
                "Online Backup", "Device Protection Plan", "Premium Tech Support", "Streaming TV",
                "Streaming Movies", "Streaming Music", "Unlimited Data"],
    "Penggunaan & Tagihan": ["Avg Monthly GB Download", "Avg Monthly Long Distance Charges", "Total Refunds",
                             "Total Extra Data Charges"],
}
# Atribut yang dapat diubah perusahaan (dipakai simulator "bagaimana jika")
ACTIONABLE = ["Contract", "Offer", "Premium Tech Support", "Online Security", "Online Backup",
              "Device Protection Plan", "Payment Method", "Paperless Billing", "Unlimited Data", "Monthly Charge"]


def widget(f, container):
    m = meta[f]
    if m["type"] == "cat":
        return container.selectbox(f, m["options"], index=m["options"].index(m["mode"]), key=f"in_{f}")
    if m["is_int"]:
        return container.number_input(f, int(m["min"]), int(m["max"] * 1.5), int(m["median"]), step=1, key=f"in_{f}")
    return container.number_input(f, float(m["min"]), float(m["max"] * 1.5), float(m["median"]), step=1.0,
                                  key=f"in_{f}")


def proba_of(values: dict) -> float:
    return float(pipe.predict_proba(pd.DataFrame([values])[feats])[0, 1])


with st.form("form"):
    values = {}
    for tab, (gname, gfeats) in zip(st.tabs(list(GROUPS)), GROUPS.items()):
        with tab:
            cols = st.columns(2)
            for i, f in enumerate(x for x in gfeats if x in feats):
                values[f] = widget(f, cols[i % 2])
    go_btn = st.form_submit_button("Prediksi", type="primary")

if go_btn:
    if values["Internet Service"] == "No":
        values["Internet Type"] = "No Internet"
    # Disimpan di session_state agar tidak hilang saat widget simulator mengubah halaman
    with st.spinner("Menghitung probabilitas churn..."):
        st.session_state["pred"] = {"values": values, "p": proba_of(values)}
    st.toast("Prediksi selesai", icon="✅")

pred = st.session_state.get("pred")
if pred:
    values, p = pred["values"], pred["p"]
    level, icon = risk_label(p)
    a, b = st.columns(2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"}, title={"text": "Probabilitas churn"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0B1F3A"},
               "steps": [{"range": [0, 40], "color": "#E3F4FC"}, {"range": [40, 70], "color": "#B9E6F9"},
                         {"range": [70, 100], "color": "#8AD5F3"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    a.plotly_chart(fig, width="stretch")
    b.subheader(f"{icon} Risiko {level}")
    b.markdown("Diprediksi **CHURN**" if p >= 0.5 else "Diprediksi **BERTAHAN**")
    b.caption("Ambang keputusan 50% (garis hitam pada gauge).")
    tips = []
    if values["Contract"] == "Month-to-Month":
        tips.append("Tawarkan insentif migrasi ke kontrak 1 atau 2 tahun.")
    if values["Internet Type"] == "Fiber Optic":
        tips.append("Tinjau harga dan kualitas layanan Fiber Optic untuk segmen ini.")
    if values["Premium Tech Support"] == "No" and values["Internet Service"] == "Yes":
        tips.append("Tawarkan percobaan gratis Premium Tech Support.")
    if values["Tenure in Months"] <= 12:
        tips.append("Pelanggan baru: jalankan onboarding dan follow-up dini.")
    if values["Offer"] == "No Offer":
        tips.append("Pertimbangkan penawaran retensi khusus.")
    if p >= 0.5 and tips:
        b.markdown("**Rekomendasi retensi:**")
        for t in tips:
            b.markdown(f"- {t}")

    # ------------------------------------------------------------ simulator "bagaimana jika"
    st.divider()
    st.subheader("Simulator: bagaimana jika ...")
    st.caption("Ubah satu hal pada pelanggan ini dan lihat perubahan risikonya. Hasil adalah **perkiraan model** "
               "(asosiasi pada data), bukan jaminan sebab-akibat.")
    feat = st.selectbox("Apa yang ingin diubah?", [f for f in ACTIONABLE if f in feats], key="sim_feat")
    now = values[feat]
    if meta[feat]["type"] == "cat":
        opts = meta[feat]["options"]
        new = st.selectbox(f"Nilai baru untuk {feat} (saat ini: {now})", opts,
                           index=opts.index(now), key=f"sim_val_{feat}")
        with st.spinner("Menghitung skenario..."):
            scen = pd.DataFrame({"Pilihan": opts,
                                 "Probabilitas": [proba_of({**values, feat: o}) for o in opts]})
        scen["Status"] = ["Saat ini" if o == now else ("Pilihan simulasi" if o == new else "Alternatif") for o in opts]
        fig = px.bar(scen, x="Pilihan", y="Probabilitas", color="Status", text=scen["Probabilitas"].map("{:.1%}".format),
                     color_discrete_map={"Saat ini": "#718096", "Pilihan simulasi": "#1677FF", "Alternatif": "#BFE9FA"},
                     title=f"Probabilitas churn untuk setiap pilihan {feat}")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch")
    else:
        step = 1.0
        new = st.number_input(f"Nilai baru untuk {feat} (saat ini: {now})", 0.0, float(meta[feat]["max"] * 1.5),
                              float(now), step=step, key=f"sim_val_{feat}")
    p_new = proba_of({**values, feat: new})
    m1, m2 = st.columns(2)
    m1.metric("Probabilitas saat ini", f"{p:.1%}")
    m2.metric("Probabilitas setelah perubahan", f"{p_new:.1%}", f"{(p_new - p) * 100:+.1f} poin", delta_color="inverse")
    if new == now:
        st.info("Pilih nilai yang berbeda dari kondisi saat ini untuk melihat dampaknya.")
    elif p_new < p:
        st.success(f"Perubahan ini **menurunkan** risiko churn sebesar {(p - p_new) * 100:.1f} poin persentase.")
    elif p_new > p:
        st.warning(f"Perubahan ini **menaikkan** risiko churn sebesar {(p_new - p) * 100:.1f} poin persentase.")
