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

# Загрузка датасета
df = pd.read_csv("h3n2_epitope_dataset.csv")

print("=" * 80)
print("1. ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 80)
print(f"Количество строк (образцов): {len(df)}")
print(f"Количество колонок: {len(df.columns)}")
print(f"\nКолонки: {list(df.columns)}")
print(f"\nТипы данных:")
print(df.dtypes.value_counts())

print("\n" + "=" * 80)
print("2. ПЕРВЫЕ 10 СТРОК")
print("=" * 80)
# Выбираем только основные колонки для наглядности
main_cols = ['pdb_id', 'chain', 'res_id', 'aa', 'epitope_label',
             'degree', 'surface_score', 'hydro_score', 'approx_sasa',
             'conservation', 'region_label']
print(df[main_cols].head(10).to_string(index=False))

print("\n" + "=" * 80)
print("3. РАСПРЕДЕЛЕНИЕ МЕТОК (EPITOPE_LABEL)")
print("=" * 80)
label_counts = df['epitope_label'].value_counts()
print(label_counts)
print(f"Доля эпитопов: {label_counts.get(1, 0) / len(df) * 100:.2f}%")

print("\n" + "=" * 80)
print("4. ПРИМЕРЫ ПРИЗНАКОВ ДЛЯ ЭПИТОПНЫХ И НЕ-ЭПИТОПНЫХ ОСТАТКОВ")
print("=" * 80)
feature_cols = ['degree', 'clustering', 'avg_neighbor_degree', 'coreness',
                'surface_score', 'hydro_score', 'approx_sasa', 'conservation']
print("Средние значения признаков:")
print(df.groupby('epitope_label')[feature_cols].mean().round(3))

print("\n" + "=" * 80)
print("5. ПРОВЕРКА ONE-HOT КОДИРОВАНИЯ (первые 5 аминокислот)")
print("=" * 80)
aa_oh_cols = [f'aa_oh_{i}' for i in range(20)]
# Проверим, что для каждого остатка только одна единица
sample_aa = df[['aa'] + aa_oh_cols].head(5)
for idx, row in sample_aa.iterrows():
    aa = row['aa']
    oh_sum = row[aa_oh_cols].sum()
    print(f"Остаток {aa}: сумма one-hot = {oh_sum} (должна быть 1)")

print("\n" + "=" * 80)
print("6. УНИКАЛЬНЫЕ ЗНАЧЕНИЯ РЕГИОНОВ")
print("=" * 80)
print(df['region_label'].value_counts())

print("\n" + "=" * 80)
print("7. СТРУКТУРЫ В ДАТАСЕТЕ")
print("=" * 80)
structures = df.groupby('pdb_id')['chain'].unique()
for pdb, chains in structures.items():
    print(f"{pdb}: цепи {', '.join(chains)}")