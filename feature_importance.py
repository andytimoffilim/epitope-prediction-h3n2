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
        # Список старых структур (из H3N2_STRUCTURES)
        old_pdbs = [
            '7RS1', '6XPO', '3WHE', '5HMB', '4FNK', '3ZNZ', '2YPG', '3HMX',
            '1TI8', '2IBX', '5FTG', '6WXY', '6MZK', '6WXB', '9CXU'
        ]
        df = df[df['pdb_id'].isin(old_pdbs)]
        print(f"Используются только старые структуры: {df['pdb_id'].nunique()} шт.")
    else:
        print(f"Используются все структуры: {df['pdb_id'].nunique()} шт.")
    return df

def prepare_features(df):
    # Признаки: все числовые колонки, кроме идентификаторов и метки
    exclude = ['pdb_id', 'chain', 'node_idx', 'res_id', 'aa', 'epitope_label', 'region_label']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].values
    y = df['epitope_label'].values
    return X, y, feature_cols

def main():
    args = parse_args()
    df = load_data(args.csv, old_only=args.old_only)
    print(f"Общее количество образцов: {len(df)}")
    print(f"Доля эпитопов: {df['epitope_label'].mean():.4f}")

    X, y, feature_names = prepare_features(df)
    print(f"Количество признаков: {len(feature_names)}")

    # Разделение на train/test для оценки важности (не для финальной модели)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Масштабирование не обязательно для RF, но для единообразия
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    print(f"RF Accuracy on test: {rf.score(X_test_scaled, y_test):.4f}")

    # Встроенная важность
    importances = rf.feature_importances_
    # Permutation importance (более надёжная, но медленнее)
    perm_importance = permutation_importance(rf, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    # Собираем результаты
    results = pd.DataFrame({
        'feature': feature_names,
        'importance_rf': importances,
        'importance_perm_mean': perm_importance.importances_mean,
        'importance_perm_std': perm_importance.importances_std
    }).sort_values('importance_rf', ascending=False)

    print("\nТоп-15 наиболее важных признаков (RF importance):")
    print(results.head(15).to_string(index=False))

    print("\nПризнаки с нулевой или очень низкой важностью (<0.001):")
    low_importance = results[results['importance_rf'] < args.importance_threshold]
    if low_importance.empty:
        print("  Нет таких признаков.")
    else:
        print(low_importance[['feature', 'importance_rf']].to_string(index=False))

    # Сохраняем результаты
    results.to_csv(args.output, index=False)
    print(f"\nРезультаты сохранены в {args.output}")

    # Визуализация
    plt.figure(figsize=(12, 8))
    top_n = min(30, len(results))
    plt.barh(results['feature'][:top_n][::-1], results['importance_rf'][:top_n][::-1], color='steelblue')
    plt.xlabel('Random Forest Feature Importance')
    plt.title('Top {} Important Features'.format(top_n))
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("График сохранён как feature_importance.png")

    # Рекомендация по удалению признаков
    if not low_importance.empty:
        print("\n=== РЕКОМЕНДАЦИЯ ===")
        print(f"Признаки с важностью < {args.importance_threshold} можно удалить. Их {len(low_importance)} шт.")
        print("Удаление этих признаков может упростить модель без потери качества.")
        print("Для проверки обучите модель без них и сравните метрики (ROC-AUC, PR-AUC).")
    else:
        print("\nВсе признаки имеют ненулевую важность. Удаление не рекомендуется.")

if __name__ == "__main__":
    main()