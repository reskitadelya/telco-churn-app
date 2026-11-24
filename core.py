"""core.py - fungsi bersama untuk seluruh halaman Streamlit."""
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from train_model import DATA_PATH, MODEL_PATH, TARGET, load_data

ANGGOTA = [("Reskita Delya Rumaday", "241011402496"),
           ("Feni Iza Tulnisa", "241011400626"),
           ("Ifat Mardyah", "241011400632")]
DOSEN = "Farizi Ilham, M.Kom."
COLORS = {"Yes": "#0B1F3A", "No": "#35C6F4"}  # Churn = Midnight Blue, Tidak churn = Cyan

# ------------------------------------------------------------------ TEMA GRAFIK (Plotly)
# Semua grafik otomatis memakai palet Midnight Blue & Cyan, latar transparan, dan transisi halus.
pio.templates["telco"] = go.layout.Template(layout=dict(
    colorway=["#1677FF", "#35C6F4", "#0B1F3A", "#718096", "#5B9DFF", "#8FD6F3"],
    font=dict(color="#243247"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#E9EFF6", linecolor="#D5DFEB", zerolinecolor="#D5DFEB"),
    yaxis=dict(gridcolor="#E9EFF6", linecolor="#D5DFEB", zerolinecolor="#D5DFEB"),
    title=dict(font=dict(size=14, color="#0B1F3A")),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    transition=dict(duration=450, easing="cubic-in-out"),
))
pio.templates.default = "plotly_white+telco"

# ------------------------------------------------------------------ CSS GLOBAL
# Palet: Primary #0B1F3A | Secondary #1677FF | Accent #35C6F4 | Background #F4F8FC
#        Surface #FFFFFF | Text #243247 | Muted #718096
CSS = """
<style>
:root{
  --navy:#0B1F3A; --blue:#1677FF; --cyan:#35C6F4; --bg:#F4F8FC; --card:#FFFFFF;
  --text:#243247; --muted:#718096; --line:#E2EAF3;
  --shadow:0 1px 2px rgba(11,31,58,.04),0 4px 14px rgba(11,31,58,.06);
  --shadow-hover:0 12px 30px rgba(22,119,255,.18);
  --ease:cubic-bezier(.4,0,.2,1);
}

/* ---------- Animasi dasar ---------- */
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
@keyframes growBar{from{width:0}to{width:56px}}
@keyframes softPulse{0%,100%{box-shadow:0 0 0 0 rgba(53,198,244,.35)}50%{box-shadow:0 0 0 8px rgba(53,198,244,0)}}
@keyframes sheen{from{background-position:200% 0}to{background-position:-200% 0}}

/* ---------- Kanvas & tipografi ---------- */
.stApp{background:var(--bg)}
header[data-testid="stHeader"]{background:rgba(244,248,252,.82);backdrop-filter:blur(8px)}
.block-container{max-width:940px;padding-top:2.2rem;padding-bottom:3rem}
::selection{background:rgba(53,198,244,.30)}

h1,[data-testid="stHeading"] h1{font-size:1.9rem!important;line-height:1.25!important;font-weight:700!important;
  letter-spacing:-.02em;color:var(--navy)!important}
h1::after{content:"";display:block;height:4px;width:56px;margin-top:.55rem;border-radius:4px;
  background:linear-gradient(90deg,var(--blue),var(--cyan));animation:growBar .8s var(--ease) both}
h2,h3,[data-testid="stHeading"] h2,[data-testid="stHeading"] h3{color:var(--navy)!important;font-weight:650!important;letter-spacing:-.01em}
[data-testid="stCaptionContainer"]{color:var(--muted)}
hr{border:none!important;height:1px!important;margin:1.6rem 0!important;
  background:linear-gradient(90deg,transparent,#C9D8EA,transparent)!important}

/* ---------- Fade-in saat halaman / elemen tampil (berurutan) ---------- */
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*{animation:fadeUp .55s var(--ease) both}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(2){animation-delay:.05s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(3){animation-delay:.10s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(4){animation-delay:.15s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(5){animation-delay:.20s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(6){animation-delay:.25s}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]>*:nth-child(n+7){animation-delay:.30s}
[data-testid="stHorizontalBlock"]>*:nth-child(2){animation:fadeUp .55s var(--ease) .08s both}
[data-testid="stHorizontalBlock"]>*:nth-child(3){animation:fadeUp .55s var(--ease) .16s both}
[data-testid="stHorizontalBlock"]>*:nth-child(4){animation:fadeUp .55s var(--ease) .24s both}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0B1F3A 0%,#0E2748 100%)}
[data-testid="stSidebarNavLink"]{border-radius:10px;margin:2px 8px;transition:background .22s var(--ease),transform .22s var(--ease),box-shadow .22s var(--ease)}
[data-testid="stSidebarNavLink"]:hover{background:rgba(53,198,244,.13)!important;transform:translateX(4px)}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:rgba(53,198,244,.20)!important;box-shadow:inset 3px 0 0 var(--cyan);font-weight:600}
[data-testid="stSidebar"] hr{background:linear-gradient(90deg,transparent,rgba(255,255,255,.22),transparent)!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#9FB3CC}

/* ---------- Kartu statistik (st.metric) & kartu HTML (.card) ---------- */
[data-testid="stMetric"],.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow);position:relative;overflow:hidden;
  transition:transform .28s var(--ease),box-shadow .28s var(--ease),border-color .28s var(--ease)}
[data-testid="stMetric"]{padding:14px 18px 14px 22px}
[data-testid="stMetric"]::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;
  background:linear-gradient(180deg,var(--blue),var(--cyan))}
[data-testid="stMetric"]:hover,.card:hover{transform:translateY(-4px);box-shadow:var(--shadow-hover);border-color:#B7D4FF}
[data-testid="stMetricLabel"] p{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600}
[data-testid="stMetricValue"]{color:var(--navy);font-weight:700}

.card{padding:16px 18px;height:100%}
.card::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;
  background:linear-gradient(90deg,var(--blue),var(--cyan));opacity:.9}
.card h5{margin:0 0 .6rem;font-size:.88rem;font-weight:650;color:var(--navy)}
.card p{margin:0;font-size:.8rem;color:var(--text);line-height:1.6}
.card .sub{color:var(--muted);font-size:.74rem;margin-top:.5rem}

/* ---------- Chip teknologi (Beranda) ---------- */
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.1rem 0 1rem}
.chip{font-size:.72rem;font-weight:600;color:var(--navy);background:#E3F4FC;border:1px solid #BFE9FA;border-radius:999px;
  padding:.2rem .75rem;transition:background .2s var(--ease),color .2s var(--ease),transform .2s var(--ease)}
.chip:hover{background:var(--navy);color:#fff;transform:translateY(-1px)}

/* ---------- Tombol ---------- */
[data-testid^="stBaseButton"]{border-radius:10px!important;font-weight:600;
  transition:transform .2s var(--ease),box-shadow .2s var(--ease),background .2s var(--ease),border-color .2s var(--ease),color .2s var(--ease)}
button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(135deg,var(--blue),#0E5CD6)!important;border:none!important;color:#fff!important;
  box-shadow:0 4px 12px rgba(22,119,255,.30)}
button[data-testid="stBaseButton-primary"]:hover,button[data-testid="stBaseButton-primaryFormSubmit"]:hover{
  transform:translateY(-2px);box-shadow:0 10px 22px rgba(22,119,255,.38);filter:brightness(1.06)}
button[data-testid^="stBaseButton-secondary"]{background:#fff!important;border:1px solid #C9D8EA!important;color:var(--navy)!important}
button[data-testid^="stBaseButton-secondary"]:hover{border-color:var(--blue)!important;color:var(--blue)!important;
  transform:translateY(-2px);box-shadow:0 6px 16px rgba(22,119,255,.14)}
[data-testid^="stBaseButton"]:active{transform:translateY(0) scale(.98)}
[data-testid="stPageLink-NavLink"]{border:1px solid var(--line);border-radius:10px;background:#fff;
  transition:transform .2s var(--ease),border-color .2s var(--ease),box-shadow .2s var(--ease),background .2s var(--ease)}
[data-testid="stPageLink-NavLink"]:hover{transform:translateY(-2px);border-color:var(--blue);background:#F2F8FF;box-shadow:0 6px 16px rgba(22,119,255,.14)}

/* ---------- Tab ---------- */
[data-testid="stTabs"] button[role="tab"]{color:var(--muted);font-weight:600;border-radius:8px 8px 0 0;
  transition:color .2s var(--ease),background .2s var(--ease)}
[data-testid="stTabs"] button[role="tab"]:hover{color:var(--blue);background:rgba(22,119,255,.07)}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:var(--blue)}

/* ---------- Formulir & input ---------- */
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;box-shadow:var(--shadow)}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],[data-testid="stNumberInputContainer"]{
  border-radius:10px!important;transition:border-color .2s var(--ease),box-shadow .2s var(--ease)}
[data-baseweb="select"]>div:hover,[data-baseweb="input"]:hover{border-color:#9CC4FF!important}
[data-baseweb="select"]>div:focus-within,[data-baseweb="input"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within{
  border-color:var(--blue)!important;box-shadow:0 0 0 3px rgba(22,119,255,.16)!important}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed #9CC4FF;border-radius:14px;background:#F8FBFF;
  transition:background .25s var(--ease),border-color .25s var(--ease)}
[data-testid="stFileUploaderDropzone"]:hover{background:#EEF6FF;border-color:var(--blue)}

/* ---------- Expander, grafik, tabel, notifikasi ---------- */
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line)!important;border-radius:14px!important;
  box-shadow:var(--shadow);overflow:hidden;transition:box-shadow .25s var(--ease),border-color .25s var(--ease)}
[data-testid="stExpander"]:hover{border-color:#B7D4FF!important;box-shadow:0 6px 18px rgba(22,119,255,.10)}
[data-testid="stExpander"] summary{transition:background .2s var(--ease)}
[data-testid="stExpander"] summary:hover{background:#F2F8FF}
[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.5rem;
  box-shadow:var(--shadow);transition:box-shadow .28s var(--ease),border-color .28s var(--ease)}
[data-testid="stPlotlyChart"]:hover{box-shadow:0 8px 24px rgba(22,119,255,.12);border-color:#CFE0F7}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);
  transition:box-shadow .28s var(--ease)}
[data-testid="stDataFrame"]:hover{box-shadow:0 8px 24px rgba(22,119,255,.12)}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line);box-shadow:var(--shadow)}

/* ---------- Animasi loading ---------- */
[data-testid="stSpinner"]{background:#fff;border:1px solid var(--line);border-left:4px solid var(--cyan);
  border-radius:12px;padding:.75rem 1rem;box-shadow:var(--shadow);color:var(--navy);animation:softPulse 1.5s ease-in-out infinite}
[data-testid="stSpinner"] i,[data-testid="stSpinner"] svg{color:var(--blue)!important;border-top-color:var(--blue)!important}
[data-testid="stStatusWidget"]{color:var(--blue)}

/* ---------- Scrollbar ---------- */
*{scrollbar-width:thin;scrollbar-color:#B9CCE3 transparent}

/* ---------- Hormati preferensi "kurangi gerakan" ---------- */
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def card(title: str, body: str, sub: str = "") -> str:
    s = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><h5>{title}</h5><p>{body}</p>{s}</div>'


@st.cache_data(show_spinner="Memuat dataset...")
def get_df() -> pd.DataFrame:
    df = load_data(DATA_PATH)
    df["Churn"] = df[TARGET]
    return df


@st.cache_resource(show_spinner="Memuat model prediksi...")
def get_art():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:  # biasanya beda versi scikit-learn / imbalanced-learn
        st.error("File model gagal dibuka (kemungkinan beda versi library). "
                 "Jalankan ulang `python train_model.py` lalu muat ulang halaman.")
        st.caption(f"Detail: {e}")
        return None


def require_art():
    art = get_art()
    if art is None:
        st.error("Model belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()
    return art


def risk_label(p: float) -> tuple[str, str]:
    if p >= 0.7:
        return "TINGGI", "🔴"
    if p >= 0.4:
        return "SEDANG", "🟠"
    return "RENDAH", "🟢"


def metrics_table(art, key="tuned") -> pd.DataFrame:
    rows = []
    for name, r in art["results"].items():
        m = r[key]
        rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]})
    return pd.DataFrame(rows).set_index("Model")
