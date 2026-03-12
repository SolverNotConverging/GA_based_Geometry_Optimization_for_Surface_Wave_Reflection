import ast
import csv
import re
from pathlib import Path

import meep as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TM_BASE_DIR = BASE_DIR.parent / "GA_SW_suppression_TM"
OUTPUT_DIR = BASE_DIR / "optimization_results"
OUTPUT_DIR.mkdir(exist_ok=True)

LENGTH_DIR_PATTERN = re.compile(r"^(\d+)_(\d+)mm$")
CSV_NAME_PATTERN = re.compile(r"^best_gen_(\d+)\.csv$")

RESOLUTION = 20
CELL_SIZE = mp.Vector3(30, 12, 0)
PML_LAYERS = [mp.PML(1.0)]

EPSILON_SUB = 10.2
H_SUB = 1.27
T_PEC = 0.035

FCEN = 0.08
DF = 0.05
NFREQ = 100

OPT_X_START = 0.0


def parse_length_dir_name(dirname):
    match = LENGTH_DIR_PATTERN.fullmatch(dirname)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def format_length_label(length_mm):
    return f"{length_mm:.1f}".replace(".", "_") + "mm"


def parse_genome(text):
    genome = ast.literal_eval(text)
    if not isinstance(genome, list):
        raise ValueError("Genome row did not contain a list.")
    return [int(value) for value in genome]


def discover_tm_lengths(base_dir):
    records = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        length_mm = parse_length_dir_name(child.name)
        if length_mm is None:
            continue
        results_dirs = sorted(path for path in child.glob("optimization_results_*") if path.is_dir())
        if not results_dirs:
            continue
        records.append((length_mm, child, results_dirs[0]))
    return records


def parse_tm_csv(csv_path):
    metadata = {}
    spectrum_rows = []
    in_spectrum = False

    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[:4] == ["Frequency", "T", "R", "L"]:
                in_spectrum = True
                continue
            if in_spectrum:
                if len(row) >= 4:
                    spectrum_rows.append(row[:4])
                continue
            if len(row) >= 2:
                metadata[row[0].strip()] = row[1].strip()

    generation = int(float(metadata["Generation"]))
    fitness = float(metadata["Best Fitness"])
    genome = parse_genome(metadata["Genome"])
    return {
        "generation": generation,
        "fitness": fitness,
        "genome": genome,
        "spectrum_rows": spectrum_rows,
    }


def find_newest_tm_csv(results_dir):
    newest = None

    for candidate in results_dir.glob("best_gen_*.csv"):
        match = CSV_NAME_PATTERN.fullmatch(candidate.name)
        if not match:
            continue
        generation_from_name = int(match.group(1))
        parsed = parse_tm_csv(candidate)
        sort_key = (parsed["generation"], generation_from_name, candidate.stat().st_mtime)
        if newest is None or sort_key > newest["sort_key"]:
            newest = {
                "path": candidate,
                "generation_from_name": generation_from_name,
                "sort_key": sort_key,
                **parsed,
            }

    return newest


def parse_cached_te_metadata(csv_path):
    metadata = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[:4] == ["Frequency", "T", "R", "L"]:
                break
            if len(row) >= 2:
                metadata[row[0].strip()] = row[1].strip()
    return metadata


def cached_result_is_current(output_csv, source_csv):
    if not output_csv.exists():
        return False

    metadata = parse_cached_te_metadata(output_csv)
    expected_mtime = f"{source_csv.stat().st_mtime:.6f}"

    return (
        metadata.get("Source TM CSV") == str(source_csv)
        and metadata.get("Source TM MTime") == expected_mtime
    )


def get_base_geometry():
    waveguide = mp.Block(
        mp.Vector3(mp.inf, H_SUB, mp.inf),
        center=mp.Vector3(0, H_SUB / 2, 0),
        material=mp.Medium(epsilon=EPSILON_SUB),
    )
    base_ground = mp.Block(
        mp.Vector3(mp.inf, T_PEC, mp.inf),
        center=mp.Vector3(0, -T_PEC / 2, 0),
        material=mp.perfect_electric_conductor,
    )
    return [waveguide, base_ground]


def get_optimization_geometry(genome, opt_x_end):
    geometry_additions = []
    feature_size = (opt_x_end - OPT_X_START) / len(genome)
    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * feature_size
        center_x = OPT_X_START + (start_idx * feature_size) + (length / 2)

        if state in (1, 3):
            geometry_additions.append(
                mp.Block(
                    mp.Vector3(length, T_PEC, mp.inf),
                    center=mp.Vector3(center_x, -T_PEC / 2, 0),
                    material=mp.Medium(epsilon=1.0),
                )
            )

        if state in (2, 3):
            geometry_additions.append(
                mp.Block(
                    mp.Vector3(length, T_PEC, mp.inf),
                    center=mp.Vector3(center_x, H_SUB + T_PEC / 2, 0),
                    material=mp.perfect_electric_conductor,
                )
            )

    for index in range(1, len(genome)):
        if genome[index] != current_state:
            add_block_from_state(current_state, segment_start_idx, index)
            current_state = genome[index]
            segment_start_idx = index
    add_block_from_state(current_state, segment_start_idx, len(genome))
    return geometry_additions


def run_te_simulation(genome, opt_x_end):
    src_pt = mp.Vector3(-7.5, H_SUB / 2, 0)
    refl_pt = mp.Vector3(-13, H_SUB / 2, 0)
    trans_pt = mp.Vector3(13, H_SUB / 2, 0)

    sources = [
        mp.Source(
            mp.GaussianSource(FCEN, fwidth=DF),
            component=mp.Ez,
            center=src_pt,
            size=mp.Vector3(0, H_SUB, 0),
        )
    ]

    sim_norm = mp.Simulation(
        cell_size=CELL_SIZE,
        boundary_layers=PML_LAYERS,
        geometry=get_base_geometry(),
        sources=sources,
        resolution=RESOLUTION,
    )

    refl_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
    trans_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

    sim_norm.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-9))

    straight_refl_data = sim_norm.get_flux_data(refl_mon)
    straight_trans_flux = np.array(mp.get_fluxes(trans_mon))
    flux_freqs = np.array(mp.get_flux_freqs(trans_mon))
    sim_norm.reset_meep()

    full_geometry = get_base_geometry() + get_optimization_geometry(genome, opt_x_end)
    sim_opt = mp.Simulation(
        cell_size=CELL_SIZE,
        boundary_layers=PML_LAYERS,
        geometry=full_geometry,
        sources=sources,
        resolution=RESOLUTION,
    )

    refl = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
    trans = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))
    sim_opt.load_minus_flux_data(refl, straight_refl_data)
    sim_opt.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    trans_flux = np.array(mp.get_fluxes(trans))
    refl_flux = np.array(mp.get_fluxes(refl))

    T = np.abs(
        np.divide(trans_flux, straight_trans_flux, out=np.zeros_like(trans_flux), where=straight_trans_flux != 0)
    )
    R = np.abs(
        np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)
    )
    L = 1 - T - R
    fitness = float(np.mean(R))

    sim_opt.reset_meep()
    return flux_freqs, T, R, L, fitness


def write_te_csv(output_csv, length_mm, source_record, flux_freqs, t_values, r_values, l_values, fitness):
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Length (mm)", length_mm])
        writer.writerow(["Generation", source_record["generation"]])
        writer.writerow(["Best Fitness", fitness])
        writer.writerow(["Genome", str(source_record["genome"])])
        writer.writerow(["Source TM CSV", str(source_record["path"])])
        writer.writerow(["Source TM Generation", source_record["generation"]])
        writer.writerow(["Source TM Fitness", source_record["fitness"]])
        writer.writerow(["Source TM MTime", f"{source_record['path'].stat().st_mtime:.6f}"])
        writer.writerow([])
        writer.writerow(["Frequency", "T", "R", "L"])
        for freq, t_value, r_value, l_value in zip(flux_freqs, t_values, r_values, l_values):
            writer.writerow([freq, t_value, r_value, l_value])


def process_length(length_mm, results_dir):
    source_record = find_newest_tm_csv(results_dir)
    if source_record is None:
        print(f"[SKIP] {results_dir.parent.name}: no TM best_gen CSV found")
        return None

    output_csv = OUTPUT_DIR / f"best_gen_{format_length_label(length_mm)}.csv"

    if cached_result_is_current(output_csv, source_record["path"]):
        print(
            f"[CACHE] {results_dir.parent.name}: using {output_csv.name} "
            f"(TM generation {source_record['generation']})"
        )
        return {
            "length_mm": length_mm,
            "output_csv": output_csv,
            "source_csv": source_record["path"],
            "generation": source_record["generation"],
            "cached": True,
        }

    print(
        f"[RUN] {results_dir.parent.name}: simulating TE from {source_record['path'].name} "
        f"(TM generation {source_record['generation']})"
    )
    flux_freqs, t_values, r_values, l_values, fitness = run_te_simulation(source_record["genome"], length_mm)
    write_te_csv(output_csv, length_mm, source_record, flux_freqs, t_values, r_values, l_values, fitness)
    print(f"[SAVE] {output_csv.name}: mean(R)={fitness:.6f}")

    return {
        "length_mm": length_mm,
        "output_csv": output_csv,
        "source_csv": source_record["path"],
        "generation": source_record["generation"],
        "cached": False,
    }


def main():
    tm_lengths = discover_tm_lengths(TM_BASE_DIR)
    if not tm_lengths:
        raise SystemExit(f"No TM length folders found in {TM_BASE_DIR}")

    summaries = []
    for length_mm, _, results_dir in tm_lengths:
        summary = process_length(length_mm, results_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No TE CSV files were produced.")

    print("\nCompleted TE batch processing:")
    for summary in summaries:
        status = "cached" if summary["cached"] else "updated"
        print(
            f"  {summary['length_mm']:>4.1f} mm -> {summary['output_csv'].name} "
            f"[{status}, source={summary['source_csv'].name}, gen={summary['generation']}]"
        )


if __name__ == "__main__":
    main()
