#!/usr/bin/env python3
"""
Feature importance analysis for epitope prediction task.
Uses Random Forest (built-in importance) and permutation importance.
Results are saved to CSV and as a plot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Feature importance for epitope prediction")
    parser.add_argument("--csv", default="h3n2_epitope_dataset.csv", help="Path to dataset CSV")
    parser.add_argument("--old-only", action="store_true", help="Use only old structures (from original H3N2_STRUCTURES)")
    parser.add_argument("--importance-threshold", type=float, default=0.001,
                        help="Remove features with mean importance below this threshold")
    parser.add_argument("--output", default="feature_importance.csv", help="Output CSV for importance scores")
    return parser.parse_args()

def load_data(csv_path, old_only=False):
    df = pd.read_csv(csv_path)
    if old_only:
        # List of old structures (from H3N2_STRUCTURES)
        old_pdbs = [
            '7RS1', '6XPO', '3WHE', '5HMB', '4FNK', '3ZNZ', '2YPG', '3HMX',
            '1TI8', '2IBX', '5FTG', '6WXY', '6MZK', '6WXB', '9CXU'
        ]
        df = df[df['pdb_id'].isin(old_pdbs)]
        print(f"Using only old structures: {df['pdb_id'].nunique()} entries.")
    else:
        print(f"Using all structures: {df['pdb_id'].nunique()} entries.")
    return df

def prepare_features(df):
    # Features: all numeric columns except identifiers and the target label
    exclude = ['pdb_id', 'chain', 'node_idx', 'res_id', 'aa', 'epitope_label', 'region_label']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].values
    y = df['epitope_label'].values
    return X, y, feature_cols

def main():
    args = parse_args()
    df = load_data(args.csv, old_only=args.old_only)
    print(f"Total number of samples: {len(df)}")
    print(f"Epitope fraction: {df['epitope_label'].mean():.4f}")

    X, y, feature_names = prepare_features(df)
    print(f"Number of features: {len(feature_names)}")

    # Split into train/test for importance evaluation (not for final model)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Scaling is not strictly necessary for RF, but we do it for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    print(f"RF Accuracy on test: {rf.score(X_test_scaled, y_test):.4f}")

    # Built-in importance
    importances = rf.feature_importances_
    # Permutation importance (more robust but slower)
    perm_importance = permutation_importance(rf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Assemble results
    results = pd.DataFrame({
        'feature': feature_names,
        'importance_rf': importances,
        'importance_perm_mean': perm_importance.importances_mean,
        'importance_perm_std': perm_importance.importances_std
    }).sort_values('importance_rf', ascending=False)

    print("\nTop-15 most important features (RF importance):")
    print(results.head(15).to_string(index=False))

    print(f"\nFeatures with zero or very low importance (<{args.importance_threshold}):")
    low_importance = results[results['importance_rf'] < args.importance_threshold]
    if low_importance.empty:
        print("  No such features.")
    else:
        print(low_importance[['feature', 'importance_rf']].to_string(index=False))

    # Save results
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Visualization
    plt.figure(figsize=(12, 8))
    top_n = min(30, len(results))
    plt.barh(results['feature'][:top_n][::-1], results['importance_rf'][:top_n][::-1], color='steelblue')
    plt.xlabel('Random Forest Feature Importance')
    plt.title('Top {} Important Features'.format(top_n))
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("Plot saved as feature_importance.png")

    # Recommendation for feature removal
    if not low_importance.empty:
        print(f"\n=== RECOMMENDATION ===")
        print(f"Features with importance < {args.importance_threshold} can be removed. There are {len(low_importance)} such features.")
        print("Removing these features may simplify the model without loss of quality.")
        print("To verify, train a model without them and compare metrics (ROC-AUC, PR-AUC).")
    else:
        print("\nAll features have non-zero importance. Removal is not recommended.")

if __name__ == "__main__":
    main()