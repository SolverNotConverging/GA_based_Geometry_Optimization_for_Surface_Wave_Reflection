import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(BASE_DIR, 'mean_R_over_length_TM.png')


def parse_length_from_dir(dirname):
    match = re.fullmatch(r'(\d+)_(\d+)mm', dirname)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def load_best_record(results_dir):
    best_fitness = -float('inf')
    best_csv_path = None

    if not os.path.isdir(results_dir):
        return best_fitness, best_csv_path

    for name in os.listdir(results_dir):
        if not re.fullmatch(r'best_gen_\d+\.csv', name):
            continue

        csv_path = os.path.join(results_dir, name)
        fitness = None
        with open(csv_path, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2 and row[0].strip() == 'Best Fitness':
                    fitness = float(row[1].strip())
                    break

        if fitness is not None and fitness > best_fitness:
            best_fitness = fitness
            best_csv_path = csv_path

    return best_fitness, best_csv_path


def load_mean_r(csv_path):
    r_values = []
    spectral_section = False

    with open(csv_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            if row[:4] == ['Frequency', 'T', 'R', 'L']:
                spectral_section = True
                continue
            if not spectral_section or len(row) < 4:
                continue
            r_values.append(float(row[2]))

    if not r_values:
        raise ValueError(f'No R data found in {csv_path}')

    return float(np.mean(r_values))


def find_results_dir(length_dir):
    candidate_dirs = []
    for name in os.listdir(length_dir):
        full_path = os.path.join(length_dir, name)
        if os.path.isdir(full_path) and name.startswith('optimization_results_'):
            candidate_dirs.append(full_path)
    if not candidate_dirs:
        return None
    candidate_dirs.sort()
    return candidate_dirs[0]


def collect_mean_r_by_length():
    records = []

    for name in os.listdir(BASE_DIR):
        length = parse_length_from_dir(name)
        if length is None:
            continue

        length_dir = os.path.join(BASE_DIR, name)
        if not os.path.isdir(length_dir):
            continue

        results_dir = find_results_dir(length_dir)
        if results_dir is None:
            print(f'Skipping {name}: results directory not found')
            continue

        best_fitness, best_csv_path = load_best_record(results_dir)
        if best_csv_path is None:
            print(f'Skipping {name}: best_gen CSV not found')
            continue

        mean_r = load_mean_r(best_csv_path)
        records.append((length, mean_r, best_fitness, best_csv_path))

    records.sort(key=lambda item: item[0])
    return records


def plot_mean_r_over_length(records):
    lengths = [item[0] for item in records]
    mean_r_values = [item[1] for item in records]

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, mean_r_values, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Length (mm)')
    plt.ylabel('Mean R')
    plt.title('Best Mean R over Length (TM)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()


def main():
    records = collect_mean_r_by_length()
    if not records:
        raise SystemExit('No valid best_gen CSV files were found.')

    print('Length (mm) | Mean R | Best Fitness | CSV')
    for length, mean_r, best_fitness, csv_path in records:
        print(f'{length:>10.1f} | {mean_r:>6.4f} | {best_fitness:>12.6f} | {csv_path}')

    plot_mean_r_over_length(records)
    print(f'Saved plot to {OUTPUT_PNG}')


if __name__ == '__main__':
    main()
