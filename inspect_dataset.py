#!/usr/bin/env python3
"""
Inspect structure of h3n2_epitope_dataset.csv:
- columns and their data types
- first 10 rows
- label distribution (epitope / non-epitope)
- feature examples for several residues
"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("h3n2_epitope_dataset.csv")

print("=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)
print(f"Number of rows (samples): {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:")
print(df.dtypes.value_counts())

print("\n" + "=" * 80)
print("2. FIRST 10 ROWS")
print("=" * 80)
# Show only main columns for clarity
main_cols = ['pdb_id', 'chain', 'res_id', 'aa', 'epitope_label',
             'degree', 'surface_score', 'hydro_score', 'approx_sasa',
             'conservation', 'region_label']
print(df[main_cols].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("3. LABEL DISTRIBUTION (EPITOPE_LABEL)")
print("=" * 80)
label_counts = df['epitope_label'].value_counts()
print(label_counts)
print(f"Epitope fraction: {label_counts.get(1, 0) / len(df) * 100:.2f}%")

print("\n" + "=" * 80)
print("4. FEATURE EXAMPLES FOR EPITOPE AND NON-EPITOPE RESIDUES")
print("=" * 80)
feature_cols = ['degree', 'clustering', 'avg_neighbor_degree', 'coreness',
                'surface_score', 'hydro_score', 'approx_sasa', 'conservation']
print("Mean feature values:")
print(df.groupby('epitope_label')[feature_cols].mean().round(3))

print("\n" + "=" * 80)
print("5. ONE-HOT ENCODING CHECK (first 5 amino acids)")
print("=" * 80)
aa_oh_cols = [f'aa_oh_{i}' for i in range(20)]
# Verify that exactly one one-hot column is set to 1 per residue
sample_aa = df[['aa'] + aa_oh_cols].head(5)
for idx, row in sample_aa.iterrows():
    aa = row['aa']
    oh_sum = row[aa_oh_cols].sum()
    print(f"Residue {aa}: one-hot sum = {oh_sum} (should be 1)")

print("\n" + "=" * 80)
print("6. UNIQUE REGION LABELS")
print("=" * 80)
print(df['region_label'].value_counts())

print("\n" + "=" * 80)
print("7. STRUCTURES IN THE DATASET")
print("=" * 80)
structures = df.groupby('pdb_id')['chain'].unique()
for pdb, chains in structures.items():
    print(f"{pdb}: chains {', '.join(chains)}")