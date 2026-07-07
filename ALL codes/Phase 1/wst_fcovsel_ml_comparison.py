#!/usr/bin/env python
"""
wst_fcovsel_ml_comparison.py
============================
Reads pre-calculated wavelength subsets from CSV files and evaluates them
with SVM-RBF and Random Forest classifiers on Salinas and Indian Pines,
using checkerboard spatial split.

Pre-calculated CSVs:
- WST methods (BOSS, CARS, GA-iPLS, GA-iPLS_BOSS)
- FCovSel (Salinas and Indian Pines)
"""

import os
import sys
import time
import math
import random
import ast
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# PATHS
# ============================================================================
DATASET_BASE   = Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_WST_Pipeline\dataset")
WAVELENGTHS_DIR= Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\CCARS_Tuned_Hyperparameters")
OUTPUT_DIR     = Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation\Core_Scripts\wst_fcovsel_ml_results_from_csv")

WST_CSV_PATH   = Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation\WST_Results\wst_selected_wavelengths_summary.csv")
IP_FCOV_CSV    = Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation\Core_Scripts\Thesis_9.4.2026_CNN_Results\Indian_pines\Indian_pines_all_FCovSel\Indian_pines_FCovSel_selected_wavelengths.csv")
SA_FCOV_CSV    = Path(r"C:\Users\hosse\Desktop\Thesis_Phase1_completed\HSI_Fresh_Adaptation\Core_Scripts\Thesis_9.4.2026_CNN_Results\Salinas_results\Salinas_all_FCovSel\Salinas_FCovSel_selected_wavelengths.csv")


# ============================================================================
# CONFIGURATION
# ============================================================================
BLOCK_SIZES = {"indian_pines": 8, "salinas": 16}

# Hyperparameters
SVM_C           = 100
SVM_GAMMA       = 0.01
RF_N_ESTIMATORS = 500
RANDOM_SEED     = 42

N_PCA_COMPONENTS= 10
CHI2_CONFIDENCE = 0.95

PREPROCESSING_COMBOS = [
    ("SG",  "SNV"),
    ("SG",  "MSC"),
    ("SG1", "SNV"),
    ("SG1", "MSC"),
]

DATASETS = {
    "indian_pines": {
        "cube_path": DATASET_BASE / "Indian_pines_corrected.mat",
        "gt_path":   DATASET_BASE / "Indian_pines_gt.mat",
        "cube_key":  "indian_pines_corrected",
        "gt_key":    "indian_pines_gt",
        "wl_csv":    WAVELENGTHS_DIR / "indianpines_wavelengths_200.csv",
    },
    "salinas": {
        "cube_path": DATASET_BASE / "Salinas_corrected.mat",
        "gt_path":   DATASET_BASE / "Salinas_gt.mat",
        "cube_key":  "salinas_corrected",
        "gt_key":    "salinas_gt",
        "wl_csv":    WAVELENGTHS_DIR / "wavelengths_salinas_corrected_204.csv",
    },
}

# ============================================================================
# CSV LOADING HELPERS
# ============================================================================
def load_wst_csv():
    if not WST_CSV_PATH.exists():
        print(f"WARNING: WST CSV not found at {WST_CSV_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(WST_CSV_PATH)
    return df

def load_fcov_csvs():
    dfs = []
    if IP_FCOV_CSV.exists(): dfs.append(pd.read_csv(IP_FCOV_CSV))
    if SA_FCOV_CSV.exists(): dfs.append(pd.read_csv(SA_FCOV_CSV))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def parse_wavelengths(val):
    if pd.isna(val): return []
    try:
        return ast.literal_eval(val)
    except:
        return []

def get_wst_pipeline_name(sm, nm):
    # WST CSV uses formats like "10_SG1_MSC" or "10_SG1_SVN"
    n_mapped = "SVN" if nm == "SNV" else nm
    return f"10_{sm}_{n_mapped}"

def get_fcov_pipeline_name(sm, nm):
    # FCovSel CSV uses formats like "SG MSC" or "SG1 SNV"
    return f"{sm} {nm}"

def wls_to_indices(target_wls, all_wls):
    indices = []
    for wl in target_wls:
        idx = np.argmin(np.abs(all_wls - wl))
        indices.append(idx)
    return np.unique(indices).tolist()

def get_selected_indices(method, dataset, sm, nm, all_wls, wst_df, fcov_df):
    dataset_formatted = "Indian_pines" if dataset == "indian_pines" else "Salinas"
    
    if method == "FCovSel":
        if fcov_df.empty: return None
        pipe_name = get_fcov_pipeline_name(sm, nm)
        row = fcov_df[(fcov_df["Dataset"] == dataset_formatted) & 
                      (fcov_df["Method"] == "FCovSel") & 
                      (fcov_df["Pipeline"] == pipe_name)]
        if row.empty: return None
        wls_str = row["Selected_Wavelengths"].values[0]
        
    else:
        if wst_df.empty: return None
        pipe_name = get_wst_pipeline_name(sm, nm)
        row = wst_df[(wst_df["Dataset"] == dataset_formatted) & 
                     (wst_df["Method"] == method) & 
                     (wst_df["Pipeline"] == pipe_name)]
        if row.empty: return None
        wls_str = row["Wavelengths"].values[0]
        
    wls = parse_wavelengths(wls_str)
    if not wls: return None
    return wls_to_indices(wls, all_wls)

# ============================================================================
# DATA PROCESSING
# ============================================================================
def load_cube_and_gt(dataset_name):
    cfg = DATASETS[dataset_name]
    cube_mat = loadmat(str(cfg["cube_path"]))
    gt_mat   = loadmat(str(cfg["gt_path"]))

    cube = None
    for key in [cfg["cube_key"], cfg["cube_key"].lower()]:
        if key in cube_mat:
            cube = cube_mat[key].astype(np.float32); break
    if cube is None:
        arrays = {k: v for k, v in cube_mat.items() if isinstance(v, np.ndarray) and v.ndim == 3}
        cube = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.float32)

    gt = None
    for key in [cfg["gt_key"], cfg["gt_key"].lower()]:
        if key in gt_mat:
            gt = gt_mat[key].astype(np.int32); break
    if gt is None:
        arrays = {k: v for k, v in gt_mat.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        gt = arrays[max(arrays, key=lambda k: arrays[k].size)].astype(np.int32)

    wl_df = pd.read_csv(str(cfg["wl_csv"]))
    wavelengths = wl_df.iloc[:, 0].values.astype(np.float64)

    return cube, gt, wavelengths

def checkerboard_split(cube, gt, block_size, max_retries=100):
    H, W, B = cube.shape
    all_classes = set(np.unique(gt)) - {0}
    best_split, best_score = None, -1

    for offset in range(max_retries):
        offset_r = offset % block_size
        offset_c = (offset // block_size) % block_size

        block_rows = (np.arange(H) + offset_r) // block_size
        block_cols = (np.arange(W) + offset_c) // block_size
        checkerboard = (block_rows[:, None] + block_cols[None, :]) % 2 == 0

        train_mask = checkerboard & (gt > 0)
        test_mask  = ~checkerboard & (gt > 0)

        train_classes = set(np.unique(gt[train_mask])) - {0}
        test_classes  = set(np.unique(gt[test_mask]))  - {0}
        common = train_classes & test_classes

        if len(common) > best_score:
            best_score = len(common)
            best_split = (train_mask, test_mask)

        if train_classes == all_classes and test_classes == all_classes:
            break

    train_mask, test_mask = best_split
    return cube[train_mask], cube[test_mask], gt[train_mask], gt[test_mask]

def sg_filter(X, deriv=0):
    wl = min(31, X.shape[1] - 1)
    if wl % 2 == 0: wl -= 1
    wl = max(5, wl)
    return np.array([savgol_filter(row, wl, 2, deriv=deriv) for row in X])

def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def msc(X):
    ref = X.mean(axis=0)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        m, b = np.polyfit(ref, X[i], 1)
        out[i] = (X[i] - b) / m if m != 0 else X[i]
    return out

def preprocess(X, smoothing, normalization):
    if smoothing.upper() == "SG":
        X = sg_filter(X, deriv=0)
    elif smoothing.upper() == "SG1":
        X = sg_filter(X, deriv=1)
    if normalization.upper() in ("SNV", "SVN"):
        X = snv(X)
    elif normalization.upper() == "MSC":
        X = msc(X)
    return X

def remove_outliers(X, n_components=N_PCA_COMPONENTS, confidence=CHI2_CONFIDENCE):
    pca = PCA(n_components=min(n_components, X.shape[1]))
    scores = pca.fit_transform(X)
    center = scores.mean(axis=0)
    cov = np.cov(scores, rowvar=False)
    try:
        inv_cov = np.linalg.inv(cov)
    except Exception:
        inv_cov = np.linalg.pinv(cov)
    dists = np.array([(s - center) @ inv_cov @ (s - center) for s in scores])
    threshold = chi2.ppf(confidence, df=pca.n_components_)
    return dists <= threshold

# ============================================================================
# CLASSIFIERS
# ============================================================================
def make_svm():
    return SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA,
               class_weight="balanced", random_state=RANDOM_SEED)

def make_rf():
    return RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                  class_weight="balanced",
                                  n_jobs=-1, random_state=RANDOM_SEED)

def evaluate(X_tr, y_tr, X_te, y_te, indices=None):
    Xtr = X_tr[:, indices] if indices is not None else X_tr
    Xte = X_te[:, indices] if indices is not None else X_te

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    results = {}
    for name, clf in [("SVM-RBF", make_svm()), ("RF", make_rf())]:
        clf.fit(Xtr_s, y_tr)
        pred = clf.predict(Xte_s)
        results[name] = {
            "OA":      accuracy_score(y_te, pred),
            "macroF1": f1_score(y_te, pred, average="macro", zero_division=0),
            "kappa":   cohen_kappa_score(y_te, pred),
            "n_bands": Xtr.shape[1],
        }
    return results

# ============================================================================
# PLOT RESULTS
# ============================================================================
def plot_comparison(df, dataset_name, output_path):
    methods    = ["ALL_BANDS", "BOSS", "CARS", "GA-iPLS", "GA-iPLS_BOSS", "FCovSel"]
    classifiers = ["SVM-RBF", "RF"]
    colors = {"SVM-RBF": "#2E86AB", "RF": "#A23B72"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"WST vs FCovSel — {dataset_name.capitalize()} (ML Classifiers)", fontsize=16, fontweight="bold")

    for ax, clf in zip(axes, classifiers):
        sub = df[df["classifier"] == clf].copy()
        x = np.arange(len(methods))
        
        oas = []
        n_bands_list = []
        
        for i, m in enumerate(methods):
            row = sub[sub["method"] == m]
            if row.empty:
                oas.append(0)
                n_bands_list.append(0)
                continue
            oas.append(row["OA_mean"].values[0])
            n_bands_list.append(int(row["n_bands_mean"].values[0]))
            
        bars = ax.bar(x, oas, color=colors[clf], alpha=0.85, edgecolor="black", linewidth=1.2)
        
        # Add percentage on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x)
        # Set x-tick labels with method name and n_bands under it
        xticklabels = [f"{m}\n(n={nb})" if nb > 0 else m for m, nb in zip(methods, n_bands_list)]
        ax.set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=11)
        
        ax.set_ylabel("Overall Accuracy", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_title(clf, fontsize=14, fontweight="bold")
        ax.grid(axis="y", linestyle='--', alpha=0.6)
        
        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_dataset(dataset_name, all_rows, wst_df, fcov_df):
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'='*70}")

    cube, gt, wavelengths = load_cube_and_gt(dataset_name)
    block_size = BLOCK_SIZES[dataset_name]

    for sm, nm in PREPROCESSING_COMBOS:
        combo = f"{sm}_{nm}"
        print(f"\n  --- Preprocessing: {combo} ---")

        out_dir = OUTPUT_DIR / dataset_name / combo
        out_dir.mkdir(parents=True, exist_ok=True)

        X_tr_raw, X_te_raw, y_tr, y_te = checkerboard_split(cube, gt, block_size)
        
        X_tr = preprocess(X_tr_raw.astype(np.float32), sm, nm)
        X_te = preprocess(X_te_raw.astype(np.float32), sm, nm)

        mask = remove_outliers(X_tr)
        X_tr, y_tr = X_tr[mask], y_tr[mask]

        combo_rows = []

        # ---- ALL BANDS (baseline) ----
        print(f"    [ALL_BANDS] ...", end=" ", flush=True)
        t0 = time.time()
        res = evaluate(X_tr, y_tr, X_te, y_te, indices=None)
        for clf_name, metrics in res.items():
            combo_rows.append({"dataset": dataset_name, "preprocessing": combo, "method": "ALL_BANDS",
                               "classifier": clf_name, **metrics})
        print(f"SVM OA={res['SVM-RBF']['OA']:.4f}, RF OA={res['RF']['OA']:.4f} ({time.time()-t0:.1f}s)")

        # ---- METHODS ----
        methods = ["BOSS", "CARS", "GA-iPLS", "GA-iPLS_BOSS", "FCovSel"]
        for m_name in methods:
            idx = get_selected_indices(m_name, dataset_name, sm, nm, wavelengths, wst_df, fcov_df)
            if not idx:
                print(f"    [{m_name}] ... SKIP (Not found in CSV)")
                continue

            print(f"    [{m_name}] ...", end=" ", flush=True)
            t0 = time.time()
            res = evaluate(X_tr, y_tr, X_te, y_te, indices=idx)
            for clf_name, metrics in res.items():
                combo_rows.append({"dataset": dataset_name, "preprocessing": combo, "method": m_name,
                                   "classifier": clf_name, **metrics})
            print(f"n_bands={len(idx)}, SVM OA={res['SVM-RBF']['OA']:.4f}, RF OA={res['RF']['OA']:.4f} ({time.time()-t0:.1f}s)")

        # Save per-combo
        df_combo = pd.DataFrame(combo_rows)
        df_combo.to_csv(out_dir / "results_summary.csv", index=False)
        all_rows.extend(combo_rows)

    return all_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["salinas", "indian_pines", "both"], default="both")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wst_df = load_wst_csv()
    fcov_df = load_fcov_csvs()

    datasets = []
    if args.dataset in ("salinas", "both"): datasets.append("salinas")
    if args.dataset in ("indian_pines", "both"): datasets.append("indian_pines")

    all_rows = []
    for ds in datasets:
        all_rows = run_dataset(ds, all_rows, wst_df, fcov_df)

    df_all = pd.DataFrame(all_rows)
    master_path = OUTPUT_DIR / "all_results_combined.csv"
    df_all.to_csv(master_path, index=False)
    print(f"\nMaster results saved: {master_path}")

    if not df_all.empty:
        agg = (df_all.groupby(["dataset", "method", "classifier"])
               .agg(OA_mean=("OA", "mean"), OA_std=("OA", "std"),
                    macroF1_mean=("macroF1", "mean"), kappa_mean=("kappa", "mean"),
                    n_bands_mean=("n_bands", "mean"))
               .reset_index())
        agg.to_csv(OUTPUT_DIR / "aggregated_results.csv", index=False)
        
        for ds in datasets:
            sub = agg[agg["dataset"] == ds]
            print(f"\n  {ds.upper()} SUMMARY")
            for method in ["ALL_BANDS","BOSS","CARS","GA-iPLS","GA-iPLS_BOSS","FCovSel"]:
                for clf in ["SVM-RBF", "RF"]:
                    row = sub[(sub["method"] == method) & (sub["classifier"] == clf)]
                    if row.empty: continue
                    print(f"  {method:<20} {clf:<10} OA={row['OA_mean'].values[0]:.4f}  bands={row['n_bands_mean'].values[0]:.1f}")
            plot_comparison(sub, ds, OUTPUT_DIR / f"comparison_{ds}.png")

if __name__ == "__main__":
    main()
