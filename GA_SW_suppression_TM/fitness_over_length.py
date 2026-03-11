import argparse
import ast
import csv
import re
from pathlib import Path


CSV_NAME_PATTERN = re.compile(r"best_gen_(\d+)\.csv$")
LENGTH_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)mm$")


def parse_csv_data(csv_path: Path):
    best_fitness = None
    genome = None

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()

            if key == "Best Fitness":
                try:
                    best_fitness = float(value)
                except ValueError:
                    best_fitness = None

            elif key == "Genome":
                try:
                    genome = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    genome = value

    return best_fitness, genome


def parse_length_from_dir_name(dir_name: str):
    match = LENGTH_DIR_PATTERN.match(dir_name)
    if not match:
        return None
    whole, decimal = match.groups()
    return float(f"{whole}.{decimal}")


def discover_length_directories(base_dir: Path):
    dirs = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        length_mm = parse_length_from_dir_name(child.name)
        if length_mm is None:
            continue
        dirs.append((length_mm, child))
    dirs.sort(key=lambda item: item[0])
    return dirs


def find_newest_csv_in_length_dir(folder: Path):
    if not folder.exists() or not folder.is_dir():
        return None

    optimization_dirs = [p for p in folder.glob("optimization_results*") if p.is_dir()]
    if not optimization_dirs:
        return None

    csv_candidates = []
    for opt_dir in optimization_dirs:
        csv_candidates.extend(opt_dir.glob("best_gen_*.csv"))

    if not csv_candidates:
        return None

    ranked = []
    for path in csv_candidates:
        match = CSV_NAME_PATTERN.search(path.name)
        generation_from_name = int(match.group(1)) if match else -1
        ranked.append((generation_from_name, path.stat().st_mtime, path))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def collect_newest_data(length_dirs):
    results = []

    for length_mm, folder in length_dirs:
        newest_csv = find_newest_csv_in_length_dir(folder)
        if newest_csv is None:
            print(f"[SKIP] {length_mm}mm: no CSV found")
            continue

        best_fitness, genome = parse_csv_data(newest_csv)
        if best_fitness is None:
            print(f"[SKIP] {length_mm}mm: missing Best Fitness/Best Loss in {newest_csv}")
            continue

        results.append(
            {
                "length_mm": length_mm,
                "csv_path": newest_csv,
                "best_fitness": best_fitness,
                "genome": genome,
            }
        )

        print(
            f"[OK] {length_mm}mm -> {newest_csv.name}, "
            f"best_fitness={best_fitness:.6f}, genome={genome}"
        )

    results.sort(key=lambda item: item["length_mm"])
    return results


def plot_fitness_over_length(results, output_path: Path):
    if not results:
        print("No valid data to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed. Data extraction finished, but plotting was skipped.")
        return

    x = [item["length_mm"] for item in results]
    y = [item["best_fitness"] for item in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    for item in results:
        plt.annotate(
            str(item["genome"]),
            (item["length_mm"], item["best_fitness"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.title("Newest Gene Best Fitness vs Length (TM)")
    plt.xlabel("Length (mm)")
    plt.ylabel("Best Fitness")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"\nSaved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find newest gene CSV for all length folders and plot Best Fitness vs length."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory containing length folders (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fitness_over_length_TM.png"),
        help="Output image path for the plot.",
    )
    args = parser.parse_args()

    length_dirs = discover_length_directories(args.base_dir)
    if not length_dirs:
        print(f"No length directories found in: {args.base_dir}")
        return

    results = collect_newest_data(length_dirs)
    plot_fitness_over_length(results, args.output)


if __name__ == "__main__":
    main()
