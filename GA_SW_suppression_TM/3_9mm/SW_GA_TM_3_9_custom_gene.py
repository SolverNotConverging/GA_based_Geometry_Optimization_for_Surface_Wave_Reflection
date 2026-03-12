import csv
import os
import re

import meep as mp
import numpy as np

# =======================================================================
# 1. CONFIGURATION
# =======================================================================

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
OPT_X_END = 3.9
MIN_FEATURE_SIZE = 0.3
NUM_SEGMENTS = int(round((OPT_X_END - OPT_X_START) / MIN_FEATURE_SIZE))

OUTPUT_DIR = 'optimization_results_3_9mm'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, OUTPUT_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Edit this list directly in PyCharm.
# Examples:
# USER_GENOMES = [[1, 3, 3]]
# USER_GENOMES = [[1, 3, 3], [3, 1, 3]]
USER_GENOMES = []


# =======================================================================
# 2. CSV / GENOME HELPERS
# =======================================================================

def parse_genome_string(genome_text):
    return [int(value) for value in re.findall(r'-?\d+', genome_text)]


def load_record_metadata(csv_path):
    generation = None
    fitness = None
    genome = None

    with open(csv_path, newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if key == "Generation":
                generation = int(float(value))
            elif key == "Best Fitness":
                fitness = float(value)
            elif key == "Genome":
                genome = parse_genome_string(value)

    return generation, fitness, genome


def load_best_record(results_dir):
    best_fitness = -float('inf')
    best_genome = None
    max_generation = 0

    if not os.path.isdir(results_dir):
        return best_fitness, best_genome, max_generation

    for name in os.listdir(results_dir):
        if not re.fullmatch(r'best_gen_\d+\.csv', name):
            continue

        csv_path = os.path.join(results_dir, name)
        generation, fitness, genome = load_record_metadata(csv_path)
        if generation is not None:
            max_generation = max(max_generation, generation)
        match = re.search(r'(\d+)', name)
        if match:
            max_generation = max(max_generation, int(match.group(1)))
        if fitness is not None and genome is not None and fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome

    return best_fitness, best_genome, max_generation


def save_generation_results(gen_num, fitness, genome, T, R, L, flux_freqs):
    filename = os.path.join(RESULTS_DIR, f"best_gen_{gen_num}.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Generation", gen_num])
        writer.writerow(["Best Fitness", fitness])
        writer.writerow(["Genome", str(genome)])
        writer.writerow([])
        writer.writerow(["Frequency", "T", "R", "L"])
        for k in range(len(flux_freqs)):
            writer.writerow([flux_freqs[k], T[k], R[k], L[k]])
    print(f"  -> Exported: {filename}")


def validate_genome(genome):
    if not isinstance(genome, list):
        raise ValueError(f"Genome must be a list, got {type(genome).__name__}")
    if len(genome) != NUM_SEGMENTS:
        raise ValueError(f"Genome length must be {NUM_SEGMENTS}, got {len(genome)}")
    if any(not isinstance(gene, int) for gene in genome):
        raise ValueError("Each gene must be an integer")
    if any(gene not in [0, 1, 2, 3] for gene in genome):
        raise ValueError("Each gene must be one of 0, 1, 2, 3")
    if genome[0] == 0 or genome[-1] == 0:
        raise ValueError("The first and last genes cannot be 0")
    return genome


def get_user_genomes():
    if not isinstance(USER_GENOMES, list):
        raise ValueError("USER_GENOMES must be a list")
    if not USER_GENOMES:
        raise ValueError("USER_GENOMES is empty. Add at least one genome before running.")

    validated_genomes = []
    for genome in USER_GENOMES:
        validated_genomes.append(validate_genome(genome))
    return validated_genomes


# =======================================================================
# 3. GEOMETRY GENERATION
# =======================================================================

def get_base_geometry():
    waveguide = mp.Block(mp.Vector3(mp.inf, H_SUB, mp.inf),
                         center=mp.Vector3(0, H_SUB / 2, 0),
                         material=mp.Medium(epsilon=EPSILON_SUB))

    base_ground = mp.Block(mp.Vector3(mp.inf, T_PEC, mp.inf),
                           center=mp.Vector3(0, -T_PEC / 2),
                           material=mp.perfect_electric_conductor)
    return [waveguide, base_ground]


def get_optimization_geometry(genome):
    geometry_additions = []
    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * MIN_FEATURE_SIZE
        center_x = OPT_X_START + (start_idx * MIN_FEATURE_SIZE) + (length / 2)

        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0)))

        if state == 2 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, H_SUB + T_PEC / 2, 0),
                         material=mp.perfect_electric_conductor))

    for i in range(1, len(genome)):
        if genome[i] != current_state:
            add_block_from_state(current_state, segment_start_idx, i)
            current_state = genome[i]
            segment_start_idx = i
    add_block_from_state(current_state, segment_start_idx, len(genome))

    return geometry_additions


# =======================================================================
# 4. NORMALIZATION (Straight Waveguide)
# =======================================================================

print("--- Running Normalization ---")
src_pt = mp.Vector3(-7.5, H_SUB / 2, 0)
refl_pt = mp.Vector3(-13, H_SUB / 2, 0)
trans_pt = mp.Vector3(13, H_SUB / 2, 0)

sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                     component=mp.Hz,
                     center=src_pt,
                     size=mp.Vector3(0, H_SUB, 0)),
           mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                     component=mp.Ez,
                     center=src_pt,
                     size=mp.Vector3(0, H_SUB, 0))]

sim = mp.Simulation(cell_size=CELL_SIZE,
                    boundary_layers=PML_LAYERS,
                    geometry=get_base_geometry(),
                    sources=sources,
                    resolution=RESOLUTION)

refl_mon = sim.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans_mon = sim.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-9))

straight_refl_data = sim.get_flux_data(refl_mon)
straight_trans_flux = np.array(mp.get_fluxes(trans_mon))
flux_freqs = np.array(mp.get_flux_freqs(trans_mon))

sim.reset_meep()
print("--- Normalization Complete ---")


# =======================================================================
# 5. FITNESS FUNCTION
# =======================================================================

def calculate_fitness(genome):
    sim.reset_meep()
    full_geometry = get_base_geometry() + get_optimization_geometry(genome)

    sim_opt = mp.Simulation(cell_size=CELL_SIZE,
                            boundary_layers=PML_LAYERS,
                            geometry=full_geometry,
                            sources=sources,
                            resolution=RESOLUTION)

    refl = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
    trans = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

    sim_opt.load_minus_flux_data(refl, straight_refl_data)
    sim_opt.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    trans_flux = np.array(mp.get_fluxes(trans))
    refl_flux = np.array(mp.get_fluxes(refl))

    T = np.divide(trans_flux, straight_trans_flux, out=np.zeros_like(trans_flux), where=straight_trans_flux != 0)
    R = np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)

    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)
    L = 1 - T - R
    fitness = np.mean(R)

    return fitness, T, R, L


# =======================================================================
# 6. CUSTOM GENE EVALUATION
# =======================================================================

genomes = get_user_genomes()
print(f"Target length: {OPT_X_END} mm")
print(f"Expected genome length: {NUM_SEGMENTS}")
print(f"User genomes: {genomes}")

stored_best_fitness, stored_best_genome, last_saved_generation = load_best_record(RESULTS_DIR)
if stored_best_fitness == -float('inf'):
    print("No stored best genome found yet in this directory.")
else:
    print(f"Stored best fitness: {stored_best_fitness:.6f} for genome {stored_best_genome}")

next_generation_number = last_saved_generation

for index, genome in enumerate(genomes, start=1):
    print(f"--- Evaluating genome {index}/{len(genomes)}: {genome} ---")
    fitness, T, R, L = calculate_fitness(genome)
    print(f"Fitness: {fitness:.6f}")

    if fitness > stored_best_fitness:
        next_generation_number += 1
        save_generation_results(next_generation_number, fitness, genome, T, R, L, flux_freqs)
        stored_best_fitness = fitness
        stored_best_genome = genome
        print("  -> This genome is better than the stored best. CSV appended.")
    else:
        print("  -> This genome is not better than the stored best. No CSV written.")
