#!/usr/bin/env python3
"""
Step 1: Generate dataset for H3N2 HA epitope prediction.

Purpose:
    Download PDB structures, build RIN graphs, compute features,
    and save tabular dataset and graphs for further experiments.

Input:
    - PDB IDs are hardcoded in H3N2_STRUCTURES dictionary.
    - Optional: h3n2_alignment.fasta for conservation scores.

Output:
    - h3n2_epitope_dataset.csv – tabular dataset (rows = residues, columns = features).
    - h3n2_gnn_graphs.pt – graphs for GNN (optional).

Dependencies:
    - numpy, pandas, networkx, scipy, torch, torch_geometric, requests.
"""

import numpy as np
import pandas as pd
import networkx as nx
import requests
import os
from scipy.spatial.distance import pdist, squareform
import torch
from torch_geometric.data import Data

# Dictionary of structures with known epitopes (H3 numbering)
H3N2_STRUCTURES = {
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
    # Negative examples (no epitopes)
    '6MZK': [],
    '6WXB': [],
    '9CXU': [],
}

AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

def download_pdb(pdb_id):
    """Download PDB file if not already present locally."""
    pdb_file = f"{pdb_id}.pdb"
    if os.path.exists(pdb_file):
        print(f"   {pdb_file} already exists")
        return pdb_file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"   Downloading {pdb_id}...")
    response = requests.get(url)
    with open(pdb_file, 'w') as f:
        f.write(response.text)
    return pdb_file

def parse_structure(pdb_file, chain_ids=['A', 'B']):
    """
    Extract Cα coordinates, sequence and residue IDs for the first chain that has data.
    Returns (coords, seq, res_ids).
    """
    for chain_id in chain_ids:
        coords, seq, res_ids = [], [], []
        seen_res = set()
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain_id:
                    res_num = int(line[22:26])
                    if res_num not in seen_res:
                        seen_res.add(res_num)
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                        resname = line[17:20].strip()
                        seq.append(resname)
                        res_ids.append(res_num)
        if coords:
            print(f"   {pdb_file}: {len(coords)} residues, chain {chain_id}")
            return np.array(coords), seq, res_ids
    print(f"   Warning: no CA atoms found in chains {chain_ids} for {pdb_file}. Skipping.")
    return np.array([]), [], []

def aa_to_onehot(aa):
    """One-hot encoding for standard 20 amino acids."""
    vec = [0]*20
    if aa in AA_ALPHABET:
        vec[AA_ALPHABET.index(aa)] = 1
    return vec

def get_region_label(res_num):
    """Return region label (A, B1, B2, D, E, none) based on H3 numbering."""
    regions = [
        (133, 145, 'A'), (155, 162, 'B1'), (172, 189, 'B2'),
        (194, 208, 'D'), (219, 230, 'E')
    ]
    for start, end, label in regions:
        if start <= res_num <= end:
            return label
    return 'none'

def get_hydro_score(aa):
    """Empirical hydrophilicity score."""
    scores = {
        'LYS': 4.0, 'ARG': 3.8, 'ASP': 3.5, 'GLU': 3.5, 'ASN': 3.2,
        'GLN': 3.0, 'SER': 2.8, 'THR': 2.6, 'TYR': 2.4, 'HIS': 2.2
    }
    return scores.get(aa, 1.5)

def approximate_sasa(G, node, max_degree=12):
    """Approximate SASA based on node degree."""
    degree = G.degree(node)
    return 1.0 - min(degree / max_degree, 1.0)

def compute_conservation(position, alignment_file='h3n2_alignment.fasta'):
    """Compute conservation score from multiple sequence alignment."""
    if not os.path.exists(alignment_file):
        return 0.5
    with open(alignment_file, 'r') as f:
        sequences = [line.strip() for line in f if not line.startswith('>')]
    if not sequences or position >= len(sequences[0]):
        return 0.5
    counts = {}
    for seq in sequences:
        aa = seq[position]
        if aa != '-':
            counts[aa] = counts.get(aa, 0) + 1
    if not counts:
        return 0.5
    return max(counts.values()) / len(sequences)

def build_rin(coords, seq, res_ids, distance_cutoff=8.0):
    """Build RIN graph based on Cα-Cα distances."""
    n = len(coords)
    if n < 2:
        return nx.Graph()
    dist = squareform(pdist(coords))
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, aa=seq[i], res_id=res_ids[i], coord=coords[i])
    for i in range(n):
        for j in range(i+1, n):
            if dist[i, j] < distance_cutoff:
                G.add_edge(i, j, dist=dist[i, j])
    for node in G.nodes:
        G.nodes[node]['degree'] = G.degree(node)
    return G

def extract_features_from_graph(G, epitope_set):
    """Extract features for all nodes in the graph."""
    features = []
    for node in G.nodes:
        aa = G.nodes[node]['aa']
        res_id = G.nodes[node]['res_id']
        feat = {
            'pdb_id': G.graph.get('pdb_id', 'unknown'),
            'node_idx': node,
            'res_id': res_id,
            'aa': aa,
            'degree': G.nodes[node]['degree'],
            'surface_score': max(0, 1.5 - G.nodes[node]['degree'] / 7.0),
            'hydro_score': get_hydro_score(aa),
            'region_label': get_region_label(res_id),
            'approx_sasa': approximate_sasa(G, node),
            'conservation': compute_conservation(res_id),
            'epitope_label': 1 if res_id in epitope_set else 0
        }
        aa_oh = aa_to_onehot(aa)
        for i, val in enumerate(aa_oh):
            feat[f'aa_oh_{i}'] = val
        features.append(feat)
    return features

def generate_dataset():
    """Main function: generate CSV dataset and save graphs."""
    all_features = []
    pyg_graphs = []

    for pdb_id, epitope_list in H3N2_STRUCTURES.items():
        print(f"\nProcessing {pdb_id}...")
        pdb_file = download_pdb(pdb_id)
        coords, seq, res_ids = parse_structure(pdb_file)
        if len(coords) == 0:
            print(f"Skipping {pdb_id}: no coordinates.")
            continue

        # Build RIN
        G = build_rin(coords, seq, res_ids)
        G.graph['pdb_id'] = pdb_id

        epitope_set = set(epitope_list)
        feats = extract_features_from_graph(G, epitope_set)
        all_features.extend(feats)

        # Create PyG Data object
        node_features = []
        labels = []
        for node in sorted(G.nodes):
            aa = G.nodes[node]['aa']
            aa_oh = aa_to_onehot(aa)
            region = get_region_label(G.nodes[node]['res_id'])
            region_map = {'A':0, 'B1':1, 'B2':2, 'D':3, 'E':4, 'none':5}
            region_onehot = [0]*6
            region_onehot[region_map[region]] = 1

            feat_vec = [
                G.nodes[node]['degree'],
                max(0, 4.0 - G.nodes[node]['degree'] / 3.5),  # surface_score
                get_hydro_score(aa),
                approximate_sasa(G, node),
                compute_conservation(G.nodes[node]['res_id']),
            ]
            feat_vec.extend(region_onehot)   # 6
            feat_vec.extend(aa_oh)           # 20
            node_features.append(feat_vec)
            labels.append(1 if G.nodes[node]['res_id'] in epitope_set else 0)

        # Edges
        edge_index = []
        for u, v in G.edges:
            edge_index.append([u, v])
            edge_index.append([v, u])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),
            pdb_id=pdb_id,
            num_nodes=len(node_features)
        )
        pyg_graphs.append(data)

    # Save tabular dataset
    df = pd.DataFrame(all_features)
    df.to_csv('h3n2_epitope_dataset.csv', index=False)
    print(f"\nTabular dataset saved: {len(df)} records -> h3n2_epitope_dataset.csv")

    # Save graphs
    torch.save(pyg_graphs, 'h3n2_gnn_graphs.pt')
    print(f"Saved {len(pyg_graphs)} graphs for GNN -> h3n2_gnn_graphs.pt")

    return df, pyg_graphs

if __name__ == "__main__":
    df, graphs = generate_dataset()
    print("\nDataset ready. You can now train models.")