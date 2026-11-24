import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import get_df, require_art

art, df = require_art(), get_df()
proba = np.array(art["test_scores"]["proba"])
y = np.array(art["test_scores"]["y"])
n = len(y)

st.title("Kalkulator Untung-Rugi Retensi")
st.caption("Menentukan pelanggan mana yang layak diberi promo agar keuntungan maksimal")
with st.expander("Cara kerja (sederhana)", expanded=False):
    st.markdown(
        "Model memberi tiap pelanggan **skor risiko churn**. Pelanggan dengan skor di atas *threshold* "
        "akan diberi promo.\n\n"
        "- Jika yang diberi promo memang akan churn dan **promo berhasil**, perusahaan menyelamatkan nilai pelanggan.\n"
        "- Jika yang diberi promo ternyata tidak akan churn (**salah alarm**), biaya promo terbuang.\n\n"
        "Threshold rendah menjangkau lebih banyak pelanggan tetapi banyak salah alarm; threshold tinggi "
        "lebih hemat tetapi banyak churn terlewat. Kalkulator mencari titik tengah yang paling menguntungkan.")

st.subheader("Asumsi bisnis (ubah sesuai kondisi)")
avg_m = float(df["Monthly Charge"].mean())
c1, c2, c3, c4 = st.columns(4)
cost = c1.number_input("Biaya promo per pelanggan ($)", 0.0, 1000.0, 20.0, 5.0)
monthly = c2.number_input("Tagihan bulanan pelanggan ($)", 1.0, 500.0, round(avg_m, 2), 1.0,
                          help="Bawaan: rata-rata Monthly Charge pada dataset.")
months = c3.number_input("Lama bertahan jika berhasil (bulan)", 1, 60, 12)
succ = c4.slider("Peluang promo berhasil", 0.0, 1.0, 0.30, 0.05)
V = monthly * months
st.caption(f"Nilai satu pelanggan yang berhasil dipertahankan: **${V:,.0f}** "
           f"(= ${monthly:,.2f} x {months} bulan).")


def outcome(t):
    flag = proba >= t
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    return flag.sum(), tp, fp, tp * (succ * V - cost) - fp * cost


ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
profits = np.array([outcome(t)[3] for t in ts])
t_best = float(ts[profits.argmax()])

st.subheader("Hasil")
t = st.slider("Threshold yang ingin dicoba", 0.05, 0.95, 0.50, 0.01)
called, tp, fp, profit = outcome(t)
per1000 = profit / n * 1000
all_called, all_tp, all_fp, all_profit = outcome(0.0)

a, b, c = st.columns(3)
a.metric("Pelanggan diberi promo", f"{called:,} dari {n:,}")
b.metric("Churn tertangkap / salah alarm", f"{tp} / {fp}")
c.metric("Keuntungan bersih", f"${profit:,.0f}", f"${per1000:,.0f} per 1.000 pelanggan", delta_color="off")

fig = go.Figure()
fig.add_scatter(x=ts, y=profits, mode="lines", name="Keuntungan", line=dict(color="#1677FF", width=3))
fig.add_scatter(x=[t_best], y=[profits.max()], mode="markers", name=f"Terbaik (t={t_best:.2f})",
                marker=dict(size=12, color="#35C6F4"))
fig.add_scatter(x=[t], y=[profit], mode="markers", name="Pilihan Anda", marker=dict(size=10, color="#0B1F3A", symbol="diamond"))
fig.add_hline(y=0, line_dash="dash", line_color="grey")
fig.update_layout(xaxis_title="Threshold", yaxis_title="Keuntungan bersih ($, data uji)")
st.plotly_chart(fig, width="stretch")

if profits.max() <= 0:
    st.warning("Dengan asumsi ini program promo **tidak menguntungkan** pada threshold mana pun. "
               "Coba turunkan biaya promo atau naikkan peluang keberhasilan.")
else:
    st.success(f"Threshold paling menguntungkan: **{t_best:.2f}**, dengan keuntungan **${profits.max():,.0f}** "
               f"pada {n:,} pelanggan uji.")
st.markdown(f"Pembanding: **memberi promo ke semua pelanggan** menghasilkan **${all_profit:,.0f}**; "
            f"**tanpa program** menghasilkan $0.")
st.caption("Angka bergantung penuh pada asumsi di atas (biaya, nilai pelanggan, peluang berhasil), yang bukan hasil "
           "dari data. Skor model dilatih dengan SMOTE sehingga threshold dipilih dari data, bukan dari nilai 0,5.")
