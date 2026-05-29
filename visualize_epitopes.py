#!/usr/bin/env python3
"""
Visualization of epitope annotations for the original dataset.
For a given PDB ID, loads the structure, colors epitope residues, and
shows their location on the sequence.

Requirements:
    pip install matplotlib biopython pandas
    (PyMOL requires a separate installer and license)

Usage:
    python visualize_epitopes.py --pdb 4FNK
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
import sys

# Colors for epitope and non-epitope residues
COLOR_EPITOPE = 'red'
COLOR_NON_EPITOPE = 'lightgray'

# Dictionary of epitope residues (copied from H3N2_STRUCTURES)
H3N2_EPITOPES = {
    '7RS1': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '6XPO': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '3WHE': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '5HMB': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '4FNK': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '3ZNZ': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '2YPG': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '3HMX': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '1TI8': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '2IBX': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '5FTG': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '6WXY': [133, 145, 157, 162, 172, 186, 189, 194, 219, 225],
    '6MZK': [],
    '6WXB': [],
    '9CXU': [],
}

def download_pdb(pdb_id, out_dir='.'):
    """Download a PDB file if not already present locally."""
    pdb_file = Path(out_dir) / f"{pdb_id}.pdb"
    if pdb_file.exists():
        print(f"Using local file: {pdb_file}")
        return pdb_file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id}...")
    r = requests.get(url)
    with open(pdb_file, 'wb') as f:
        f.write(r.content)
    return pdb_file

def plot_sequence_epitopes_(pdb_id, epitope_list, seq=None, title=None):
    """
    Generate a linear sequence diagram.
    If seq is not provided, attempts to extract the sequence from the PDB file.
    """
    if seq is None:
        # Simple PDB parsing to extract sequence (chain A only)
        pdb_file = download_pdb(pdb_id)
        seq = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == 'A':
                    resname = line[17:20].strip()
                    seq.append(resname)
    if not seq:
        print("Failed to extract sequence. Sequence visualization is not possible.")
        return

    epitope_set = set(epitope_list)
    try:
        df = pd.read_csv('h3n2_epitope_dataset.csv')
        df_pdb = df[(df['pdb_id'] == pdb_id) & (df['chain'] == 'A')].sort_values('res_id')
        res_ids = df_pdb['res_id'].values
        labels = df_pdb['epitope_label'].values

        fig, ax = plt.subplots(figsize=(15, 2))

        # Draw as a sequence of colored rectangles
        colors = [COLOR_EPITOPE if l == 1 else COLOR_NON_EPITOPE for l in labels]
        ax.bar(range(len(labels)), [1]*len(labels), color=colors, width=1, edgecolor='none')
        ax.set_xlim(0, len(labels))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Position in sequence (sorted by residue number)')
        ax.set_title(f'{pdb_id} – Epitope annotation (red = epitope)')

        # Add residue number ticks (every 20th residue)
        ticks = np.arange(0, len(labels), 20)
        ticklabels = [str(res_ids[i]) for i in ticks if i < len(res_ids)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{pdb_id}_sequence_epitopes.png', dpi=150)
        plt.show()
        print(f"Saved {pdb_id}_sequence_epitopes.png")
    except Exception as e:
        print(f"Failed to generate sequence plot: {e}")

def plot_sequence_epitopes(pdb_id, epitope_list, seq=None, title=None):
    """
    Generate a linear sequence diagram.
    If seq is not provided, attempts to extract the sequence from the PDB file.
    """
    if seq is None:
        pdb_file = download_pdb(pdb_id)
        seq = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == 'A':
                    resname = line[17:20].strip()
                    seq.append(resname)
    if not seq:
        print("Failed to extract sequence. Sequence visualization is not possible.")
        return

    epitope_set = set(epitope_list)
    try:
        df = pd.read_csv('h3n2_epitope_dataset.csv')
        df_pdb = df[(df['pdb_id'] == pdb_id) & (df['chain'] == 'A')].sort_values('res_id')
        res_ids = df_pdb['res_id'].values
        labels = df_pdb['epitope_label'].values

        fig, ax = plt.subplots(figsize=(112.5, 15))
        colors = [COLOR_EPITOPE if l == 1 else COLOR_NON_EPITOPE for l in labels]
        ax.bar(range(len(labels)), [1]*len(labels), color=colors, width=1, edgecolor='none')
        ax.set_xlim(0, len(labels))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Position in sequence (sorted by residue number)', fontsize=60)
        ax.set_title(f'{pdb_id} – Epitope annotation (red = epitope)', fontsize=72)

        ticks = np.arange(0, len(labels), 20)
        ticklabels = [str(res_ids[i]) for i in ticks if i < len(res_ids)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=45, ha='right', fontsize=40)
        plt.tight_layout()
        plt.savefig(f'{pdb_id}_sequence_epitopes.png', dpi=60)
        plt.show()
        print(f"Saved {pdb_id}_sequence_epitopes.png")
    except Exception as e:
        print(f"Failed to generate sequence plot: {e}")

def pymol_visualization(pdb_id, epitope_list):
    """
    Create a PyMOL script for coloring epitope residues.
    Saves a .pml file that can be executed in PyMOL.
    """
    epitope_set = set(epitope_list)
    pml_lines = [
        f"load {pdb_id}.pdb",
        "hide everything",
        "show cartoon",
        "color white",
        f"select epitope, (resi {','.join(map(str, epitope_list))})",
        "color red, epitope",
        "set cartoon_color, red, epitope",
        "show sticks, epitope",
        "zoom epitope",
        f"png {pdb_id}_pymol.png, width=800, height=600, dpi=150",
        "save",
    ]
    pml_file = f"{pdb_id}_view.pml"
    with open(pml_file, 'w') as f:
        f.write("\n".join(pml_lines))
    print(f"PyMOL script saved as {pml_file}")
    print(f"Run PyMOL and execute: @{pml_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize epitopes for the original dataset")
    parser.add_argument('--pdb', default='4FNK', help='PDB ID')
    parser.add_argument('--method', choices=['sequence', 'pymol', 'both'], default='both',
                        help='Visualization method')
    args = parser.parse_args()

    pdb_id = args.pdb.upper()
    if pdb_id not in H3N2_EPITOPES:
        print(f"Error: PDB {pdb_id} is not in the epitope dictionary.")
        print("Available entries:", list(H3N2_EPITOPES.keys()))
        sys.exit(1)

    epitope_list = H3N2_EPITOPES[pdb_id]
    print(f"Epitope residues for {pdb_id}: {epitope_list}")

    # Download PDB file if needed
    if args.method in ('pymol', 'both'):
        download_pdb(pdb_id)

    if args.method in ('sequence', 'both'):
        plot_sequence_epitopes(pdb_id, epitope_list)

    if args.method in ('pymol', 'both'):
        pymol_visualization(pdb_id, epitope_list)

if __name__ == "__main__":
    main()