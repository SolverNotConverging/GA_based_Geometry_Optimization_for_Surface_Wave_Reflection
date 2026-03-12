import csv
import os
import random
import re

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

# =======================================================================
# 1. CONFIGURATION
# =======================================================================

# Simulation Constants
RESOLUTION = 20
CELL_SIZE = mp.Vector3(30, 12, 0)
PML_LAYERS = [mp.PML(1.0)]

# Material Constants
EPSILON_SUB = 10.2
H_SUB = 1.27
T_PEC = 0.035

# Source / Frequency
FCEN = 0.08
DF = 0.05
NFREQ = 100

# Optimization Region
OPT_X_START = 0.0
OPT_X_END = 5.1
MIN_FEATURE_SIZE = 0.3
NUM_SEGMENTS = int(round((OPT_X_END - OPT_X_START) / MIN_FEATURE_SIZE))

# Genetic Algorithm Settings
POPULATION_SIZE = 30
GENERATIONS = 40
MUTATION_RATE = 0.1
ELITISM_COUNT = 2

# Output Directory
OUTPUT_DIR = 'optimization_results_5_1mm'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, OUTPUT_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =======================================================================
# 2. CSV / GENOME HELPERS
# =======================================================================

def format_length_dir(length_mm):
    return f"{length_mm:.1f}".replace('.', '_') + "mm"


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


def find_results_dir_for_length(length_mm):
    length_dir = os.path.join(os.path.dirname(CURRENT_DIR), format_length_dir(length_mm))
    if not os.path.isdir(length_dir):
        return None

    candidate_dirs = []
    for name in os.listdir(length_dir):
        full_path = os.path.join(length_dir, name)
        if os.path.isdir(full_path) and name.startswith("optimization_results_"):
            candidate_dirs.append(full_path)

    if not candidate_dirs:
        return None

    candidate_dirs.sort()
    return candidate_dirs[0]


def sanitize_genome(genome):
    sanitized = genome[:NUM_SEGMENTS]
    if len(sanitized) < NUM_SEGMENTS:
        sanitized.extend([1] * (NUM_SEGMENTS - len(sanitized)))
    if sanitized:
        if sanitized[0] == 0:
            sanitized[0] = random.choice([1, 2, 3])
        if sanitized[-1] == 0:
            sanitized[-1] = random.choice([1, 2, 3])
    return sanitized


def build_previous_length_seeds(previous_best_genome):
    seeds = []
    if previous_best_genome is None or len(previous_best_genome) != NUM_SEGMENTS - 1:
        return seeds

    for gene in [1, 2, 3]:
        seeds.append(sanitize_genome(previous_best_genome + [gene]))
    return seeds


def build_current_length_seed(current_best_genome):
    if current_best_genome is None or len(current_best_genome) != NUM_SEGMENTS:
        return []
    return [sanitize_genome(current_best_genome)]


def build_next_length_seed(next_best_genome):
    if next_best_genome is None or len(next_best_genome) != NUM_SEGMENTS + 1:
        return []
    return [sanitize_genome(next_best_genome[:-1])]


def unique_genomes(genomes):
    unique = []
    seen = set()
    for genome in genomes:
        key = tuple(genome)
        if key in seen:
            continue
        seen.add(key)
        unique.append(genome)
    return unique


def get_v1_seed_genomes():
    half_segments = (NUM_SEGMENTS + 1) // 2
    seed_gene_1 = sanitize_genome([2] * half_segments + [1] * (NUM_SEGMENTS - half_segments))
    seed_gene_2 = sanitize_genome([1] * half_segments + [2] * (NUM_SEGMENTS - half_segments))
    return [seed_gene_1, seed_gene_2]


def create_v3_seed_population():
    seed_population = []
    current_length = round(OPT_X_END, 10)
    previous_length = round(current_length - MIN_FEATURE_SIZE, 10)
    next_length = round(current_length + MIN_FEATURE_SIZE, 10)

    previous_results_dir = find_results_dir_for_length(previous_length)
    current_results_dir = RESULTS_DIR
    next_results_dir = find_results_dir_for_length(next_length)

    previous_best_fitness, previous_best_genome, _ = load_best_record(previous_results_dir) if previous_results_dir else (-float('inf'), None, 0)
    current_best_fitness, current_best_genome, _ = load_best_record(current_results_dir)
    next_best_fitness, next_best_genome, _ = load_best_record(next_results_dir) if next_results_dir else (-float('inf'), None, 0)

    current_seeds = build_current_length_seed(current_best_genome)
    previous_seeds = build_previous_length_seeds(previous_best_genome)
    next_seeds = build_next_length_seed(next_best_genome)
    v1_seeds = get_v1_seed_genomes()

    seed_population.extend(current_seeds)
    seed_population.extend(previous_seeds)
    seed_population.extend(next_seeds)
    seed_population.extend(v1_seeds)
    seed_population = unique_genomes(seed_population)

    if current_best_genome is not None:
        print(f"  -> Added current-length best seed {current_best_genome} (fitness: {current_best_fitness:.6f})")
    else:
        print("  -> No current-length stored best genome found.")

    if previous_seeds:
        print(f"  -> Added {len(previous_seeds)} previous-length seeds from {previous_best_genome} (fitness: {previous_best_fitness:.6f})")
    else:
        print("  -> No valid previous-length seed genomes found.")

    if next_seeds:
        print(f"  -> Added next-length trimmed seed from {next_best_genome} (fitness: {next_best_fitness:.6f})")
    else:
        print("  -> No valid next-length seed genome found.")

    print(f"  -> Added {len(v1_seeds)} v1 structured seeds.")
    print(f"  -> Total unique seeded genomes: {len(seed_population)}")

    return seed_population


# =======================================================================
# 3. GEOMETRY GENERATION
# =======================================================================

def get_base_geometry():
    """Defines the static waveguide and base ground plane."""
    waveguide = mp.Block(mp.Vector3(mp.inf, H_SUB, mp.inf),
                         center=mp.Vector3(0, H_SUB / 2, 0),
                         material=mp.Medium(epsilon=EPSILON_SUB))

    base_ground = mp.Block(mp.Vector3(mp.inf, T_PEC, mp.inf),
                           center=mp.Vector3(0, -T_PEC / 2),
                           material=mp.perfect_electric_conductor)
    return [waveguide, base_ground]


def get_optimization_geometry(genome):
    """
    Decodes the genome into physical blocks.
    States:
    0: Base (GDS) - No change
    1: Cut Ground - Adds Vacuum block at bottom
    2: Top PEC - Adds PEC block at top
    3: Both - Adds Vacuum at bottom AND PEC at top
    """
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
                     size=mp.Vector3(0, H_SUB, 0))
           ]

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
# 5. FITNESS FUNCTION (Calculates T, R, L)
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
# 6. LIVE PLOTTING SETUP
# =======================================================================

plt.ion()
fig, (ax_fitness, ax_trl) = plt.subplots(1, 2, figsize=(12, 5))

line_fitness, = ax_fitness.plot([], [], 'r-o', label='Best Fitness')
ax_fitness.set_title("Optimization Progress")
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")
ax_fitness.grid(True)
ax_fitness.legend()

line_T, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'b-', label='T', linewidth=2)
line_R, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'r--', label='R', alpha=0.7)
line_L, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'g:', label='L', alpha=0.7)
ax_trl.set_title("Best Device Performance")
ax_trl.set_xlabel("Frequency (MEEP units)")
ax_trl.set_ylabel("Power Fraction")
ax_trl.set_ylim(0, 1.05)
ax_trl.grid(True)
ax_trl.legend()

plt.tight_layout()


# =======================================================================
# 7. GENETIC ALGORITHM LOOP
# =======================================================================

def create_random_genome():
    genome = [random.choice([0, 1, 2, 3]) for _ in range(NUM_SEGMENTS)]
    if NUM_SEGMENTS > 0:
        genome[0] = random.choice([1, 2, 3])
        genome[-1] = random.choice([1, 2, 3])
    return genome


def mutate(genome):
    new_genome = genome[:]
    for i in range(len(new_genome)):
        if random.random() < MUTATION_RATE:
            if i == 0 or i == len(new_genome) - 1:
                new_genome[i] = random.choice([1, 2, 3])
            else:
                new_genome[i] = random.choice([0, 1, 2, 3])
    return new_genome


def crossover(parent1, parent2):
    point = random.randint(1, NUM_SEGMENTS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def save_generation_results(gen_num, fitness, genome, T, R, L):
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


print(f"--- Starting Optimization V3 ({GENERATIONS} Gens) ---")

stored_best_fitness, _, last_saved_generation = load_best_record(RESULTS_DIR)
if stored_best_fitness == -float('inf'):
    print("  -> No stored best genome found for this length yet.")
else:
    print(f"  -> Stored best fitness for this length: {stored_best_fitness:.6f}")

population = create_v3_seed_population()
while len(population) < POPULATION_SIZE:
    population.append(create_random_genome())
population = population[:POPULATION_SIZE]

history_fitness = []
next_saved_generation = last_saved_generation

best_global_fitness = -float('inf')
best_global_genome = None
best_global_T = None
best_global_R = None
best_global_L = None

for gen in range(GENERATIONS):
    ranked_population = []

    for genome in population:
        fitness, T, R, L = calculate_fitness(genome)
        ranked_population.append((fitness, genome, T, R, L))

        if fitness > best_global_fitness:
            best_global_fitness = fitness
            best_global_genome = genome
            best_global_T = T
            best_global_R = R
            best_global_L = L

            line_T.set_ydata(T)
            line_R.set_ydata(R)
            line_L.set_ydata(L)
            ax_trl.set_title(f"Best Result (Gen {gen + 1}) - Fitness: {fitness:.4f}")
            fig.canvas.flush_events()
            plt.draw()

    ranked_population.sort(key=lambda x: x[0], reverse=True)
    current_best_fitness, current_best_genome, current_best_T, current_best_R, current_best_L = ranked_population[0]
    history_fitness.append(current_best_fitness)

    line_fitness.set_data(range(1, len(history_fitness) + 1), history_fitness)
    ax_fitness.relim()
    ax_fitness.autoscale_view()
    fig.canvas.flush_events()
    plt.draw()
    plt.pause(0.01)

    print(f"Gen {gen + 1}/{GENERATIONS} | Best Fitness: {current_best_fitness:.4f}")

    if current_best_fitness > stored_best_fitness:
        next_saved_generation += 1
        save_generation_results(next_saved_generation, current_best_fitness, current_best_genome,
                                current_best_T, current_best_R, current_best_L)
        stored_best_fitness = current_best_fitness
    else:
        print("  -> No CSV appended. Stored best is still better.")

    next_generation = [entry[1][:] for entry in ranked_population[:ELITISM_COUNT]]
    while len(next_generation) < POPULATION_SIZE:
        c1 = random.sample(ranked_population, 3)
        c2 = random.sample(ranked_population, 3)
        p1 = max(c1, key=lambda x: x[0])[1]
        p2 = max(c2, key=lambda x: x[0])[1]

        child1, child2 = crossover(p1, p2)
        next_generation.append(mutate(child1))
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(mutate(child2))

    population = next_generation


# =======================================================================
# 8. FINAL VISUALIZATION (GEOMETRY PLOT)
# =======================================================================
plt.ioff()

print("--- Optimization Finished ---")
print(f"Final Best Fitness This Run: {best_global_fitness}")
print(f"Best Genome This Run: {best_global_genome}")

fig_geo = plt.figure(figsize=(10, 4))

final_geometry = get_base_geometry() + get_optimization_geometry(best_global_genome)
sim_final = mp.Simulation(cell_size=CELL_SIZE,
                          boundary_layers=PML_LAYERS,
                          geometry=final_geometry,
                          sources=sources,
                          resolution=RESOLUTION)

sim_final.plot2D()
plt.title(f"Optimized Topology (Fitness: {best_global_fitness:.3f})")
plt.show()
