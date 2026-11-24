import pandas as pd
import plotly.express as px
import streamlit as st

from core import COLORS, get_df

df = get_df().copy()
st.title("Eksplorasi Data (EDA)")
st.caption("Pola churn pada tiap kelompok pelanggan")

t1, t2, t3, t4, t5 = st.tabs(["Target", "Kategorikal", "Numerik", "Korelasi", "Alasan Churn"])

with t1:
    cnt = df["Churn"].value_counts().reset_index()
    cnt.columns = ["Churn", "Jumlah"]
    a, b = st.columns(2)
    a.plotly_chart(px.pie(cnt, names="Churn", values="Jumlah", hole=0.5, color="Churn",
                          color_discrete_map=COLORS, title="Distribusi Churn"), width="stretch")
    b.dataframe(cnt.assign(Persen=(cnt["Jumlah"] / cnt["Jumlah"].sum()).map("{:.1%}".format)),
                hide_index=True, width="stretch")
    b.info("Data **tidak seimbang** (kelas churn minoritas), sehingga akurasi saja menyesatkan. "
           "Penanganannya dibahas di halaman Preprocessing (SMOTE).")

with t2:
    opts = ["Contract", "Internet Type", "Payment Method", "Offer", "Gender", "Married", "Dependents",
            "Paperless Billing", "Online Security", "Premium Tech Support", "Online Backup",
            "Streaming TV", "Unlimited Data", "Referred a Friend", "Multiple Lines"]
    col = st.selectbox("Variabel", opts)
    rate = (df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
            .reset_index(name="Churn Rate").sort_values("Churn Rate", ascending=False))
    a, b = st.columns(2)
    a.plotly_chart(px.bar(df.groupby([col, "Churn"]).size().reset_index(name="Jumlah"), x=col, y="Jumlah",
                          color="Churn", barmode="group", color_discrete_map=COLORS,
                          title="Jumlah pelanggan"), width="stretch")
    fig = px.bar(rate, x=col, y="Churn Rate", text=rate["Churn Rate"].map("{:.1%}".format),
                 title="Churn rate", color_discrete_sequence=["#1677FF"])
    fig.update_yaxes(tickformat=".0%")
    b.plotly_chart(fig, width="stretch")
    top, low = rate.iloc[0], rate.iloc[-1]
    st.success(f"**Insight:** churn tertinggi pada **{top[col]}** ({top['Churn Rate']:.1%}), "
               f"terendah pada **{low[col]}** ({low['Churn Rate']:.1%}).")

with t3:
    nums = ["Tenure in Months", "Monthly Charge", "Age", "Total Charges", "Avg Monthly GB Download",
            "Avg Monthly Long Distance Charges", "Number of Referrals", "Total Revenue"]
    col = st.selectbox("Variabel numerik", nums)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=col, color="Churn", barmode="overlay", opacity=.65, nbins=40,
                                color_discrete_map=COLORS, title=f"Distribusi {col}"), width="stretch")
    b.plotly_chart(px.box(df, x="Churn", y=col, color="Churn", color_discrete_map=COLORS,
                          title=f"Boxplot {col}"), width="stretch")
    st.dataframe(df.groupby("Churn")[col].describe().round(2), width="stretch")

with t4:
    num = df.select_dtypes("number").drop(columns=["Zip Code", "Latitude", "Longitude", "Churn Score",
                                                   "CLTV", "Satisfaction Score"], errors="ignore")
    num["Churn (1=Yes)"] = (df["Churn"] == "Yes").astype(int)
    fig = px.imshow(num.corr().round(2), text_auto=True, aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale=[[0, "#0B1F3A"], [0.5, "#F4F8FC"], [1, "#35C6F4"]], title="Matriks korelasi fitur numerik")
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")
    corr = num.corr()["Churn (1=Yes)"].drop("Churn (1=Yes)").sort_values(key=abs, ascending=False)
    st.caption("Korelasi terkuat dengan churn: " + ", ".join(f"**{k}** ({v:+.2f})" for k, v in corr.head(3).items()))

with t5:
    ch = df[df["Churn"] == "Yes"]
    a = ch["Churn Category"].value_counts().reset_index()
    a.columns = ["Kategori", "Jumlah"]
    fig = px.bar(a, x="Jumlah", y="Kategori", orientation="h", title="Kategori alasan churn",
                 color_discrete_sequence=["#1677FF"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    r = ch["Churn Reason"].value_counts().head(10).reset_index()
    r.columns = ["Alasan", "Jumlah"]
    fig = px.bar(r, x="Jumlah", y="Alasan", orientation="h", title="10 alasan churn terbanyak",
                 color_discrete_sequence=["#1677FF"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Kolom alasan churn hanya dipakai untuk eksplorasi, **tidak** untuk model (leakage).")
