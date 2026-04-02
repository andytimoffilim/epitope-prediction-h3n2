#!/usr/bin/env python3
"""
Step 4: Extended ensemble model (MLP, RF, XGBoost, LightGBM, CatBoost) with ESM‑2 embeddings.
Uses only informative local features (7 features). Removed conservation and one-hot AA.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ========================
# 1. Load data
# ========================
df = pd.read_csv("h3n2_epitope_dataset.csv")

# Use only informative local features (7 features)
feature_cols = [
    "degree",
    "clustering",
    "avg_neighbor_degree",
    "coreness",
    "surface_score",
    "hydro_score",
    "approx_sasa",
]

# One-hot encoding for region labels (keep if useful, else remove)
region_dummies = pd.get_dummies(df["region_label"], prefix="region")
X_local = pd.concat([df[feature_cols], region_dummies], axis=1).values
y = df["epitope_label"].values
groups = df["pdb_id"].values

# ========================
# 2. Load ESM‑2 embeddings (optional)
# ========================
EMB_FILE = "new_ha_embeddings_enhanced.pkl"
emb_matrix = None
if Path(EMB_FILE).exists():
    with open(EMB_FILE, 'rb') as f:
        emb_dict = pickle.load(f)
    # Align embeddings with dataframe order
    emb_list = []
    missing = []
    for pid in df['pdb_id']:
        if pid in emb_dict:
            emb_list.append(emb_dict[pid])
        else:
            missing.append(pid)
            # fallback to zero vector of appropriate dimension
            if len(emb_list) == 0:
                dummy_dim = list(emb_dict.values())[0].shape[0]
            else:
                dummy_dim = emb_list[0].shape[0]
            emb_list.append(np.zeros(dummy_dim))
    if missing:
        print(f"Warning: embeddings missing for {missing}, filled with zeros.")
    emb_matrix = np.array(emb_list)
    print(f"Embedding dimension: {emb_matrix.shape[1]}")
else:
    print(f"File {EMB_FILE} not found. Experiments with embeddings will be skipped.")

# ========================
# 3. Experiment configurations
# ========================
experiments = [{"name": "no_emb", "n_components": None}]
if emb_matrix is not None:
    experiments.extend([
        {"name": "emb_1", "n_components": 1},
        {"name": "emb_2", "n_components": 2},
        {"name": "emb_4", "n_components": 4},
        {"name": "emb_8", "n_components": 8},
        {"name": "emb_16", "n_components": 16},
    ])

# ========================
# 4. Initialize boosters (check availability)
# ========================
XGB_AVAILABLE = False
LGBM_AVAILABLE = False
CAT_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available, will be skipped.")
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available, will be skipped.")
try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    print("CatBoost not available, will be skipped.")

all_results = {}

for exp in experiments:
    print(f"\n\n{'='*60}")
    print(f"Experiment: {exp['name']}")
    print('='*60)

    # Build feature matrix
    if exp['n_components'] is None:
        X = X_local.copy()
        print("Using only local features.")
    else:
        pca = PCA(n_components=exp['n_components'])
        emb_reduced = pca.fit_transform(emb_matrix)
        print(f"PCA {exp['n_components']}: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
        X = np.hstack([X_local, emb_reduced])

    # ========================
    # 5. Cross‑validation (LOSO)
    # ========================
    logo = LeaveOneGroupOut()
    f1_scores, roc_auc_scores, pr_auc_scores = [], [], []
    results_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_pdb = df.iloc[test_idx]["pdb_id"].iloc[0]
        print(f"\n--- Fold {fold+1}: test on {test_pdb} ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if np.sum(y_test) == 0:
            print("   No epitopes in test, skipping.")
            continue

        # Normalization (for MLP)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SMOTE for MLP
        smote = SMOTE(random_state=14, k_neighbors=6)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        # --- MLP ---
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            alpha=0.001,
            max_iter=500,
            random_state=23,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        mlp.fit(X_train_res, y_train_res)
        y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

        # --- Random Forest (no scaling) ---
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=442,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]

        prob_list = [y_prob_mlp, y_prob_rf]

        # --- XGBoost ---
        if XGB_AVAILABLE:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                scale_pos_weight=scale_pos_weight,
                random_state=242,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
            prob_list.append(y_prob_xgb)

        # --- LightGBM ---
        if LGBM_AVAILABLE:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=242,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
            prob_list.append(y_prob_lgb)

        # --- CatBoost ---
        if CAT_AVAILABLE:
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            class_weights = [1.0, neg_count / pos_count] if pos_count > 0 else [1.0, 1.0]
            cat_model = CatBoostClassifier(
                iterations=100,
                class_weights=class_weights,
                random_seed=342,
                verbose=0,
                allow_writing_files=False
            )
            cat_model.fit(X_train, y_train)
            y_prob_cat = cat_model.predict_proba(X_test)[:, 1]
            prob_list.append(y_prob_cat)

        # Ensemble average
        y_prob_ensemble = np.mean(prob_list, axis=0)
        y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob_ensemble)
        pr_auc = average_precision_score(y_test, y_prob_ensemble)
        f1 = f1_score(y_test, y_pred_ensemble)

        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)
        pr_auc_scores.append(pr_auc)

        print(f"   ROC‑AUC: {roc_auc:.3f}, PR‑AUC: {pr_auc:.3f}, F1: {f1:.3f}")

        results_per_fold.append({
            "test_pdb": test_pdb,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
        })

    # Save per‑fold results
    fold_df = pd.DataFrame(results_per_fold)
    fold_df.to_csv(f"results_{exp['name']}_folds.csv", index=False)

    # Summary statistics
    mean_roc = np.mean(roc_auc_scores) if roc_auc_scores else np.nan
    std_roc = np.std(roc_auc_scores) if roc_auc_scores else np.nan
    mean_pr = np.mean(pr_auc_scores) if pr_auc_scores else np.nan
    std_pr = np.std(pr_auc_scores) if pr_auc_scores else np.nan
    mean_f1 = np.mean(f1_scores) if f1_scores else np.nan
    std_f1 = np.std(f1_scores) if f1_scores else np.nan

    all_results[exp['name']] = {
        "ROC-AUC_mean": mean_roc,
        "ROC-AUC_std": std_roc,
        "PR-AUC_mean": mean_pr,
        "PR-AUC_std": std_pr,
        "F1_mean": mean_f1,
        "F1_std": std_f1,
    }

    print(f"\n>>> Results for {exp['name']}:")
    print(f"   ROC‑AUC: {mean_roc:.3f} ± {std_roc:.3f}")
    print(f"   PR‑AUC:  {mean_pr:.3f} ± {std_pr:.3f}")
    print(f"   F1:      {mean_f1:.3f} ± {std_f1:.3f}")

# ========================
# 6. Summary table
# ========================
summary_df = pd.DataFrame(all_results).T
summary_df = summary_df.round(3)
print("\n\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(summary_df)
summary_df.to_csv("experiment_summary.csv")
print("\nSummary table saved to experiment_summary.csv")