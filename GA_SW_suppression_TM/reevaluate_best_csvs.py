import argparse
import csv
from pathlib import Path

import numpy as np


LENGTH_DIR_SUFFIX = "mm"
BEST_CSV_NAME = "best_gen_1.csv"


def is_length_dir(path: Path) -> bool:
    return path.is_dir() and path.name.endswith(LENGTH_DIR_SUFFIX) and "_" in path.name


def find_results_dirs(base_dir: Path):
    for child in sorted(base_dir.iterdir()):
        if not is_length_dir(child):
            continue
        for results_dir in sorted(child.glob("optimization_results_*")):
            if results_dir.is_dir():
                yield results_dir


def parse_csv_sections(csv_path: Path):
    metadata = []
    spectra_header = None
    spectra_rows = []
    in_spectra = False

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[:4] == ["Frequency", "T", "R", "L"]:
                spectra_header = row[:4]
                in_spectra = True
                continue
            if in_spectra:
                if len(row) >= 4:
                    spectra_rows.append(row[:4])
            else:
                metadata.append(row)

    if spectra_header is None or not spectra_rows:
        raise ValueError(f"No spectral R data found in {csv_path}")

    return metadata, spectra_header, spectra_rows


def compute_mean_r(spectra_rows):
    r_values = [float(row[2]) for row in spectra_rows]
    return float(np.mean(r_values))


def update_metadata(metadata, fitness_value: float):
    updated = []
    generation_written = False
    fitness_written = False

    for row in metadata:
        key = row[0].strip() if row else ""
        if key == "Generation":
            updated.append(["Generation", "1"])
            generation_written = True
        elif key == "Best Fitness":
            updated.append(["Best Fitness", repr(fitness_value)])
            fitness_written = True
        else:
            updated.append(row)

    if not generation_written:
        updated.insert(1 if updated else 0, ["Generation", "1"])
    if not fitness_written:
        insert_at = 2 if len(updated) >= 2 else len(updated)
        updated.insert(insert_at, ["Best Fitness", repr(fitness_value)])

    return updated


def write_csv(csv_path: Path, metadata, spectra_header, spectra_rows):
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(metadata)
        writer.writerow([])
        writer.writerow(spectra_header)
        writer.writerows(spectra_rows)


def select_best_csv(results_dir: Path):
    candidates = sorted(results_dir.glob("best_gen_*.csv"))
    if not candidates:
        return None, None, None

    best_path = None
    best_fitness = -float("inf")
    best_payload = None

    for candidate in candidates:
        metadata, spectra_header, spectra_rows = parse_csv_sections(candidate)
        fitness_value = compute_mean_r(spectra_rows)
        payload = (metadata, spectra_header, spectra_rows)

        if fitness_value > best_fitness:
            best_path = candidate
            best_fitness = fitness_value
            best_payload = payload

    return best_path, best_fitness, best_payload


def reevaluate_results_dir(results_dir: Path):
    best_path, best_fitness, payload = select_best_csv(results_dir)
    if best_path is None:
        return None

    metadata, spectra_header, spectra_rows = payload
    target_path = results_dir / BEST_CSV_NAME
    updated_metadata = update_metadata(metadata, best_fitness)
    write_csv(target_path, updated_metadata, spectra_header, spectra_rows)

    for candidate in results_dir.glob("best_gen_*.csv"):
        if candidate != target_path:
            candidate.unlink()

    return {
        "results_dir": results_dir,
        "source_csv": best_path.name,
        "target_csv": target_path.name,
        "fitness": best_fitness,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute CSV fitness as mean(R), keep the best CSV per length, and save it as best_gen_1.csv."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory containing the per-length folders.",
    )
    args = parser.parse_args()

    summaries = []
    for results_dir in find_results_dirs(args.base_dir):
        summary = reevaluate_results_dir(results_dir)
        if summary is None:
            print(f"[SKIP] {results_dir}")
            continue
        summaries.append(summary)
        print(
            f"[OK] {results_dir.parent.name}: kept {summary['source_csv']} as "
            f"{summary['target_csv']} with fitness={summary['fitness']:.6f}"
        )

    if not summaries:
        raise SystemExit("No best_gen CSV files were found to reevaluate.")


if __name__ == "__main__":
    main()
