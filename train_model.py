"""
train_model.py
Pipeline Machine Learning prediksi churn (IBM Telco Customer Churn, 50 kolom).

Alur:
 1. Muat data & tangani data hilang
 2. Buang kolom data leakage / identitas / turunan
 3. Split stratified 80/20
 4. Pipeline: Scaling + One-Hot -> PCA -> SMOTE -> Classifier
 5. Bandingkan 5 algoritma (baseline dengan & tanpa PCA), lalu tuning (RandomizedSearchCV, F1)
 6. Simpan semua artefak ke model/churn_model.joblib
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE / "model" / "churn_model.joblib"
RANDOM_STATE = 42
TARGET = "Churn Label"
PCA_VARIANCE = 0.95  # PCA mempertahankan 95% varians

# Kolom yang dibuang beserta alasannya (dipakai juga di halaman Preprocessing)
DROP_REASON = {
    "Customer Status": "Leakage - status akhir pelanggan (berisi 'Churned')",
    "Churn Score": "Leakage - skor churn hasil perhitungan Telco",
    "Churn Category": "Leakage - kategori alasan, hanya terisi jika sudah churn",
    "Churn Reason": "Leakage - alasan, hanya terisi jika sudah churn",
    "Satisfaction Score": "Leakage - survei pasca-kejadian (skor 1-2 = 100% churn)",
    "CLTV": "Leakage - dihitung Telco dengan memperhitungkan status churn",
    "Customer ID": "Identitas unik, tidak punya daya prediksi",
    "Country": "Konstan (hanya 1 nilai)",
    "State": "Konstan (hanya 1 nilai)",
    "Quarter": "Konstan (hanya 1 nilai)",
    "City": "Kardinalitas tinggi (1.000+ kota)",
    "Zip Code": "Kardinalitas tinggi / identitas lokasi",
    "Latitude": "Koordinat lokasi granular",
    "Longitude": "Koordinat lokasi granular",
    "Population": "Atribut wilayah, bukan perilaku pelanggan",
    "Under 30": "Turunan dari Age (redundan)",
    "Senior Citizen": "Turunan dari Age (redundan)",
    "Total Charges": "Turunan: Monthly Charge x Tenure (redundan)",
    "Total Revenue": "Turunan dari kolom tagihan (redundan)",
    "Total Long Distance Charges": "Turunan: tarif x Tenure (redundan)",
}
DROP_COLS = list(DROP_REASON)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """'None' pada Offer / Internet Type terbaca NaN -> diberi label eksplisit."""
    df = pd.read_csv(path)
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")
    return df


def make_pipe(clf, num_cols, cat_cols, n_pca=None, use_smote=True):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    steps = [("prep", prep)]
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clone(clf)))  # clone: tiap pipeline punya model sendiri
    return ImbPipeline(steps)


def evaluate(pipe, X, y, full=False):
    proba = pipe.predict_proba(X)[:, 1]
    pred = pipe.predict(X)
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
           "recall": recall_score(y, pred), "f1": f1_score(y, pred),
           "roc_auc": roc_auc_score(y, proba)}
    if full:
        fpr, tpr, _ = roc_curve(y, proba)
        out["cm"] = confusion_matrix(y, pred).tolist()
        out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return out


def model_zoo():
    """(nama, estimator default, ruang pencarian hyperparameter)."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 10, 20]}),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                          {"clf__n_estimators": [200, 300], "clf__max_depth": [8, 12, None],
                           "clf__min_samples_leaf": [1, 3, 5]}),
        "Gradient Boosting": (HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [3, 4, 6],
                               "clf__max_iter": [150, 250]}),
    }


def main():
    raw = pd.read_csv(DATA_PATH)
    missing = {c: int(n) for c, n in raw.isna().sum().items() if n > 0}

    df = load_data()
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ---- Analisis PCA pada data latih
    prep_only = make_pipe(LogisticRegression(), num_cols, cat_cols).named_steps["prep"]
    Z = prep_only.fit_transform(X_tr)
    full_pca = PCA(random_state=RANDOM_STATE).fit(Z)
    evr = full_pca.explained_variance_ratio_
    k = int(np.searchsorted(np.cumsum(evr), PCA_VARIANCE) + 1)
    names = [n.split("__", 1)[1] for n in prep_only.get_feature_names_out()]
    print(f"Fitur setelah encoding: {Z.shape[1]} -> PCA {k} komponen ({PCA_VARIANCE:.0%} varians)")

    # ---- Distribusi kelas sebelum/sesudah SMOTE (data latih)
    _, y_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(Z, y_tr)
    smote_info = {"before": y_tr.value_counts().sort_index().tolist(),
                  "after": pd.Series(y_sm).value_counts().sort_index().tolist()}

    # ---- Baseline (PCA vs tanpa PCA) + tuning
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    results, tuned_models = {}, {}
    for name, (clf, grid) in model_zoo().items():
        base_pca = make_pipe(clf, num_cols, cat_cols, k).fit(X_tr, y_tr)
        base_raw = make_pipe(clf, num_cols, cat_cols, None).fit(X_tr, y_tr)
        n_iter = min(6, int(np.prod([len(v) for v in grid.values()])))
        search = RandomizedSearchCV(make_pipe(clf, num_cols, cat_cols, k), grid, n_iter=n_iter,
                                    scoring="f1", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_tr, y_tr)
        results[name] = {
            "baseline": evaluate(base_pca, X_te, y_te),
            "baseline_nopca": evaluate(base_raw, X_te, y_te),
            "tuned": evaluate(search.best_estimator_, X_te, y_te, full=True),
            "cv_f1": float(search.best_score_),
            "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "best_params": {k_.replace("clf__", ""): v for k_, v in search.best_params_.items()},
        }
        tuned_models[name] = search.best_estimator_
        t = results[name]["tuned"]
        print(f"{name:20s} CV-F1={search.best_score_:.3f} | test F1={t['f1']:.3f} AUC={t['roc_auc']:.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_f1"])  # dipilih via CV, bukan data uji
    best = tuned_models[best_name]

    # ---- Studi ablasi: kontribusi PCA dan SMOTE pada model terbaik (hyperparameter terbaik)
    cv5 = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    ablation = []
    for use_pca in (False, True):
        for use_smote in (False, True):
            pipe = make_pipe(best.named_steps["clf"], num_cols, cat_cols,
                             k if use_pca else None, use_smote)
            cvs = cross_val_score(pipe, X_tr, y_tr, cv=cv5, scoring="f1", n_jobs=-1)
            m = evaluate(pipe.fit(X_tr, y_tr), X_te, y_te)
            ablation.append({"PCA": "Ya" if use_pca else "Tidak", "SMOTE": "Ya" if use_smote else "Tidak",
                             "CV-F1": float(cvs.mean()), "CV-std": float(cvs.std()),
                             "Test F1": m["f1"], "Test Recall": m["recall"],
                             "Test Precision": m["precision"], "Test ROC-AUC": m["roc_auc"]})

    test_scores = {"proba": best.predict_proba(X_te)[:, 1].tolist(), "y": y_te.tolist()}

    pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    feature_meta = {}
    for c in num_cols:
        feature_meta[c] = {"type": "num", "min": float(X[c].min()), "max": float(X[c].max()),
                           "median": float(X[c].median()),
                           "is_int": bool(pd.api.types.is_integer_dtype(X[c]))}
    for c in cat_cols:
        feature_meta[c] = {"type": "cat", "options": sorted(X[c].unique().tolist()),
                           "mode": X[c].mode()[0]}

    base_b, base_r = results[best_name]["baseline"], results[best_name]["baseline_nopca"]
    artifact = {
        "pipeline": best, "best_model": best_name, "results": results,
        "ablation": ablation, "test_scores": test_scores,
        "importance": importance, "feature_meta": feature_meta, "features": X.columns.tolist(),
        "pca_info": {"n_features": int(Z.shape[1]), "k": k, "threshold": PCA_VARIANCE,
                     "evr": evr.tolist(), "feature_names": names,
                     "delta_f1_best": float(base_b["f1"] - base_r["f1"])},
        "prep_info": {"missing": missing, "smote": smote_info, "n_train": len(X_tr),
                      "n_test": len(X_te), "n_features_raw": X.shape[1],
                      "n_cols_total": raw.shape[1], "num_cols": num_cols, "cat_cols": cat_cols},
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel terbaik (CV-F1): {best_name} | disimpan di {MODEL_PATH}")


if __name__ == "__main__":
    main()
