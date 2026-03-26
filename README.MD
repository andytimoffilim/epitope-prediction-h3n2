#Epitope Prediction for H3N2 Hemagglutinin

This repository contains the code for the article **"Prediction of Conformational Epitopes of H3N2 Hemagglutinin Using Interpretable Features and Ensemble Learning"** (Timofeev et al., 2026).

##Overview

The pipeline predicts conformational B‑cell epitopes on the hemagglutinin (HA) protein of the H3N2 influenza A virus. It builds residue interaction networks (RINs) from PDB structures, extracts interpretable features (degree, surface score, hydrophilicity, approximate SASA, conservation, one‑hot amino acids, antigenic regions), and trains an ensemble of classifiers. Two models are provided:

- **Baseline ensemble** – MLP + Random Forest using only local features.
- **Extended ensemble** – MLP + Random Forest + XGBoost + LightGBM + CatBoost, optionally enhanced with global sequence embeddings from ESM‑2 (reduced with PCA).

All experiments are validated with Leave‑One‑Structure‑Out (LOSO) cross‑validation on 15 structures (11 with epitopes, 4 negative examples).

##Pipeline Steps

1. **Generate dataset** – `01_generate_dataset.py`  
   Downloads PDB structures, builds RIN graphs, computes all features, and creates `h3n2_epitope_dataset.csv` and `h3n2_gnn_graphs.pt`.

2. **Extract ESM‑2 embeddings** – `02_extract_embeddings.py`  
   Uses the ESM‑2 model to produce 1280‑dimensional sequence embeddings. Saves them as `new_ha_embeddings_enhanced.pkl`.

3. **Baseline ensemble (MLP+RF)** – `03_baseline_ensemble.py`  
   Trains the baseline model on local features only. Outputs ROC‑AUC, PR‑AUC, and F1 scores.

4. **Extended ensemble with embeddings** – `04_ensemble_with_embeddings.py`  
   Performs systematic comparison of configurations with and without ESM‑2 embeddings (no_emb, emb_4, emb_8, emb_16). Generates per‑fold results and a summary table.

##Requirements

All dependencies are listed in `requirements.txt`. Install them with:


pip install -r requirements.txt
For the ESM‑2 embeddings, you need fair-esm (included in requirements.txt).
For the extended ensemble, install xgboost, lightgbm, and catboost if desired (they are optional; if missing, the script will skip them and still run MLP+RF).

**Usage**

Place the PDB IDs of interest in the H3N2_STRUCTURES dictionary inside 01_generate_dataset.py (the default list is already provided).

Run the scripts in order:

bash
* python 01_generate_dataset.py
* python 02_extract_embeddings.py
* python 03_baseline_ensemble.py
* python 04_ensemble_with_embeddings.py

**Output Files**

* h3n2_epitope_dataset.csv – tabular dataset (residues × features)
* h3n2_gnn_graphs.pt – graphs for GNN (not used in this work)
* new_ha_embeddings_enhanced.pkl – ESM‑2 embeddings
* mlp_rf_ensemble_results.csv – per‑fold results for the baseline model
* results_{no_emb,emb_4,emb_8,emb_16}_folds.csv – per‑fold results for each configuration
* experiment_summary.csv – summary of all configurations (Table 2 in the article)

**Data Availability**

The original PDB structures are publicly available from the RCSB Protein Data Bank. The epitope annotations used in this study are based on literature and are included in the script.

**License**

This project is licensed under the MIT License – see the LICENSE file for details.