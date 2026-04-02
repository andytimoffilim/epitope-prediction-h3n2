#!/usr/bin/env python3
"""
Визуализация разметки эпитопов для исходного датасета.
Для заданного PDB ID загружает структуру, окрашивает эпитопные остатки и
показывает их расположение на последовательности.

Требования:
    pip install matplotlib biopython pandas
    (для PyMOL нужен отдельный инсталлятор и лицензия)

Использование:
    python visualize_epitopes.py --pdb 4FNK
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
import sys

# Цвета для эпитопов и не-эпитопов
COLOR_EPITOPE = 'red'
COLOR_NON_EPITOPE = 'lightgray'

# Словарь эпитопов (копия из H3N2_STRUCTURES)
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
    """Скачать PDB файл, если его нет локально."""
    pdb_file = Path(out_dir) / f"{pdb_id}.pdb"
    if pdb_file.exists():
        print(f"Используется локальный файл: {pdb_file}")
        return pdb_file
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Скачивание {pdb_id}...")
    r = requests.get(url)
    with open(pdb_file, 'wb') as f:
        f.write(r.content)
    return pdb_file

def plot_sequence_epitopes(pdb_id, epitope_list, seq=None, title=None):
    """
    Построить линейную диаграмму последовательности.
    Если seq не задана, пытается извлечь последовательность из PDB файла.
    """
    if seq is None:
        # Простой парсинг PDB для извлечения последовательности (только цепь A)
        pdb_file = download_pdb(pdb_id)
        seq = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == 'A':
                    resname = line[17:20].strip()
                    seq.append(resname)
        # Убираем дубликаты по номерам остатков (но для визуализации можно оставить)
        # Здесь мы просто строим по порядку следования CA атомов
    if not seq:
        print("Не удалось извлечь последовательность. Визуализация последовательности невозможна.")
        return

    epitope_set = set(epitope_list)
    # Создаём массив меток для каждого остатка (по номеру, а не по индексу)
    # Но поскольку номера остатков могут быть не непрерывными, лучше использовать res_id из датасета.
    # Однако для простоты предположим, что последовательность идёт в порядке возрастания номеров.
    # Более надёжно: загрузить датасет и отфильтровать по pdb_id и chain 'A'
    try:
        df = pd.read_csv('h3n2_epitope_dataset.csv')
        df_pdb = df[(df['pdb_id'] == pdb_id) & (df['chain'] == 'A')].sort_values('res_id')
        res_ids = df_pdb['res_id'].values
        labels = df_pdb['epitope_label'].values
        # Построим цветную полоску
        fig, ax = plt.subplots(figsize=(15, 2))
        # Отрисовка как последовательности цветных прямоугольников
        colors = [COLOR_EPITOPE if l == 1 else COLOR_NON_EPITOPE for l in labels]
        ax.bar(range(len(labels)), [1]*len(labels), color=colors, width=1, edgecolor='none')
        ax.set_xlim(0, len(labels))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Position in sequence (sorted by residue number)')
        ax.set_title(f'{pdb_id} – Epitope annotation (red = epitope)')
        # Добавляем отметки номеров остатков (каждый 20-й)
        ticks = np.arange(0, len(labels), 20)
        ticklabels = [str(res_ids[i]) for i in ticks if i < len(res_ids)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{pdb_id}_sequence_epitopes.png', dpi=150)
        plt.show()
        print(f"Сохранено {pdb_id}_sequence_epitopes.png")
    except Exception as e:
        print(f"Не удалось построить график последовательности: {e}")

def pymol_visualization(pdb_id, epitope_list):
    """
    Создать PyMOL скрипт для окраски эпитопов.
    Сохраняет файл .pml, который можно выполнить в PyMOL.
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
    print(f"PyMOL скрипт сохранён как {pml_file}")
    print(f"Запустите PyMOL и выполните: @{pml_file}")

def main():
    parser = argparse.ArgumentParser(description="Визуализация эпитопов для исходного датасета")
    parser.add_argument('--pdb', default='5FTG', help='PDB ID , ...)')
    parser.add_argument('--method', choices=['sequence', 'pymol', 'both'], default='both',
                        help='Метод визуализации')
    args = parser.parse_args()

    pdb_id = args.pdb.upper()
    if pdb_id not in H3N2_EPITOPES:
        print(f"Ошибка: PDB {pdb_id} отсутствует в словаре эпитопов.")
        print("Доступные: ", list(H3N2_EPITOPES.keys()))
        sys.exit(1)

    epitope_list = H3N2_EPITOPES[pdb_id]
    print(f"Эпитопные остатки для {pdb_id}: {epitope_list}")

    # Скачиваем PDB файл (если нужно для визуализации)
    if args.method in ('pymol', 'both'):
        download_pdb(pdb_id)

    if args.method in ('sequence', 'both'):
        plot_sequence_epitopes(pdb_id, epitope_list)

    if args.method in ('pymol', 'both'):
        pymol_visualization(pdb_id, epitope_list)

if __name__ == "__main__":
    main()