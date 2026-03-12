import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "optimization_results"
OUTPUT_PNG = BASE_DIR / "mean_R_over_length_TE.png"
CSV_PATTERN = re.compile(r"^best_gen_(\d+)_(\d+)mm\.csv$")


def parse_length_from_csv_name(filename):
    match = CSV_PATTERN.fullmatch(filename)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def load_mean_r(csv_path):
    r_values = []
    spectral_section = False

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[:4] == ["Frequency", "T", "R", "L"]:
                spectral_section = True
                continue
            if spectral_section and len(row) >= 4:
                r_values.append(float(row[2]))

    if not r_values:
        raise ValueError(f"No R data found in {csv_path}")

    return float(np.mean(r_values))


def collect_records():
    records = []
    if not RESULTS_DIR.is_dir():
        return records

    for csv_path in sorted(RESULTS_DIR.glob("best_gen_*.csv")):
        length_mm = parse_length_from_csv_name(csv_path.name)
        if length_mm is None:
            continue
        mean_r = load_mean_r(csv_path)
        records.append((length_mm, mean_r, csv_path))

    records.sort(key=lambda item: item[0])
    return records


def plot_records(records):
    lengths = [item[0] for item in records]
    mean_r_values = [item[1] for item in records]

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, mean_r_values, "o-", linewidth=2, markersize=6)
    plt.xlabel("Length (mm)")
    plt.ylabel("Mean R")
    plt.title("Mean R over Length (TE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()


def main():
    records = collect_records()
    if not records:
        raise SystemExit(f"No TE CSV files found in {RESULTS_DIR}")

    print("Length (mm) | Mean R | CSV")
    for length_mm, mean_r, csv_path in records:
        print(f"{length_mm:>10.1f} | {mean_r:>6.4f} | {csv_path}")

    plot_records(records)
    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
