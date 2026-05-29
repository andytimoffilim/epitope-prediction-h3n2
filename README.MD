# Epitope Prediction for H3N2 Hemagglutinin

This repository contains the code for the article **"Prediction of Conformational Epitopes of H3N2 Hemagglutinin Based on Interpretable Structural and Evolutionary Features"** (Timofeev et al., PeerJ Computer Science, 2026).

## Overview

The pipeline predicts conformational B‑cell epitopes on the hemagglutinin (HA) protein of the H3N2 influenza A virus. It builds residue interaction networks (RINs) from PDB structures, extracts interpretable features, and trains an ensemble of classifiers. Two models are provided:

- **Baseline ensemble** – MLP + Random Forest using only local features (7 interpretable features).
- **Extended ensemble** – MLP + Random Forest + XGBoost + LightGBM + CatBoost, optionally enhanced with global sequence embeddings from ESM‑2 (reduced with PCA).

All experiments are validated with **Leave‑One‑Structure‑Out (LOSO) cross‑validation** on 15 H3N2 structures (10 with epitopes, 5 negative examples), spanning 44 years of evolution (1979–2023).

## Dataset Information

The pipeline generates a tabular dataset `h3n2_epitope_dataset.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `pdb_id` | PDB identifier of the structure |
| `chain` | Chain identifier (typically 'A' for HA) |
| `res_id` | Residue number in the PDB file |
| `aa` | Amino acid type (one-letter code) |
| `epitope_label` | Binary label (1 = epitope, 0 = non‑epitope) |
| `degree` | Node degree in the RIN graph |
| `clustering` | Local clustering coefficient |
| `avg_neighbor_degree` | Average degree of neighbors |
| `coreness` | k‑core value |
| `surface_score` | Empirical surface exposure (calibrated) |
| `approx_sasa` | Approximate solvent accessibility (calibrated) |
| `hydro_score` | Hydrophilicity score (empirical scale) |
| `conservation` | Evolutionary conservation score |
| `region_label` | Antigenic region annotation |
| `aa_oh_*` | One‑hot encoding of amino acids (20 columns) |

The dataset contains **4,685 amino acid residues** after filtering incomplete entries.

## Code Information

| File | Description |
|------|-------------|
| `01_generate_dataset.py` | Downloads PDB structures, builds RIN graphs, computes all features, and generates `h3n2_epitope_dataset.csv` and `h3n2_gnn_graphs.pt`. |
| `02_extract_embeddings.py` | Uses ESM‑2 to produce 1280‑dimensional sequence embeddings. Saves as `new_ha_embeddings_enhanced.pkl`. |
| `03_baseline_ensemble.py` | Trains the baseline ensemble (MLP + Random Forest) on local features only. Outputs ROC‑AUC, PR‑AUC, and F1 scores with LOSO cross‑validation. |
| `04_ensemble_with_embeddings.py` | Compares configurations with/without ESM‑2 embeddings (no_emb, emb_1, emb_2, emb_4, emb_8, emb_16). Generates per‑fold results and a summary table. |
| `feature_importance.py` | Computes Random Forest feature importance (Gini impurity and permutation importance). |
| `inspect_dataset.py` | Validates the dataset structure, label distribution, and one‑hot encoding. |
| `visualize_epitopes.py` | Generates linear sequence diagrams and PyMOL scripts for epitope visualization. |

## Usage Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/andytimoffilim/epitope-prediction-h3n2.git
   cd epitope-prediction-h3n2
Install dependencies

bash
pip install -r requirements.txt
Generate dataset (PDB files are downloaded automatically)

bash
python 01_generate_dataset.py
Extract ESM‑2 embeddings (optional, for extended ensemble)

bash
python 02_extract_embeddings.py
Train baseline model

bash
python 03_baseline_ensemble.py
Train extended ensemble with embeddings

bash
python 04_ensemble_with_embeddings.py
Requirements
All dependencies are listed in requirements.txt. Key packages include:

Python 3.9+

numpy, pandas, scikit-learn

torch, matplotlib

xgboost, lightgbm, catboost (optional, for extended ensemble)

fair-esm (for ESM‑2 embeddings)

freesasa (for SASA calibration)

requests, biopython

Install with:

bash
pip install -r requirements.txt
Methodology
Brief summary of the approach as implemented in the code:

RIN construction: Undirected graph where vertices are residues; edges added when Cα–Cα distance < 8 Å.

Features (7 final): degree, clustering coefficient, avg_neighbor_degree, coreness, surface_score (calibrated), approx_sasa (calibrated), hydro_score.

Feature selection: Based on Random Forest importance analysis (zero‑importance features: one‑hot amino acids, region labels, conservation scores).

Class imbalance handling: SMOTE (k_neighbors=6) for MLP; class_weight='balanced' for Random Forest and boosting models.

Validation: Leave‑One‑Structure‑Out (LOSO) — each of the 10 epitope‑containing structures serves as the test set once.

Evaluation metrics: ROC‑AUC, PR‑AUC (justified due to extreme class imbalance), and F1‑score (threshold 0.5).

Output Files
File	Description
h3n2_epitope_dataset.csv	Tabular dataset (residues × features)
h3n2_gnn_graphs.pt	Graphs for GNN (not used in this work)
new_ha_embeddings_enhanced.pkl	ESM‑2 embeddings (1280‑dim)
mlp_rf_ensemble_results.csv	Per‑fold results for baseline model
results_{config}_folds.csv	Per‑fold results for each configuration
experiment_summary.csv	Summary of all configurations (Table 3 in the article)
*_sequence_epitopes.png	Linear sequence diagrams for visualization
*_view.pml	PyMOL scripts for 3D visualization
Data Availability
The original PDB structures are publicly available from the RCSB Protein Data Bank. The epitope annotations used in this study are based on literature sources and are included in the generation script.

Citation
If you use this code in your research, please cite:

Timofeev A., Anufriev A., Isakova-Sivak I. (2026). Prediction of Conformational Epitopes of H3N2 Hemagglutinin Based on Interpretable Structural and Evolutionary Features. PeerJ Computer Science (in press).

License
This project is licensed under the MIT License – see the LICENSE file for details.

Contribution Guidelines
Contributions are welcome. Please open an issue or submit a pull request for any improvements or bug fixes. For major changes, please open an issue first to discuss what you would like to change