import csv
import os
import random

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
OPT_X_END = 3.9
MIN_FEATURE_SIZE = 0.3
NUM_SEGMENTS = int(round((OPT_X_END - OPT_X_START) / MIN_FEATURE_SIZE))

# Genetic Algorithm Settings
POPULATION_SIZE = 40
GENERATIONS = 40
MUTATION_RATE = 0.1
ELITISM_COUNT = 2

# Output Directory
OUTPUT_DIR = "optimization_results_3_9mm"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# =======================================================================
# 2. GEOMETRY GENERATION
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

        # Logic for Cutting Ground (States 1 and 3)
        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0)))  # Vacuum to 'erase' PEC

        # Logic for Adding Top PEC (States 2 and 3)
        if state == 2 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, H_SUB + T_PEC / 2, 0),
                         material=mp.perfect_electric_conductor))

    # Iterate genome to merge adjacent identical segments (optimization for meshing)
    for i in range(1, len(genome)):
        if genome[i] != current_state:
            add_block_from_state(current_state, segment_start_idx, i)
            current_state = genome[i]
            segment_start_idx = i
    add_block_from_state(current_state, segment_start_idx, len(genome))

    return geometry_additions


# =======================================================================
# 3. NORMALIZATION (Straight Waveguide)
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

# Add fluxes
refl_mon = sim.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans_mon = sim.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-9))

# Save Normalization Data
straight_refl_data = sim.get_flux_data(refl_mon)
straight_trans_flux = np.array(mp.get_fluxes(trans_mon))
flux_freqs = np.array(mp.get_flux_freqs(trans_mon))

sim.reset_meep()
print("--- Normalization Complete ---")


# =======================================================================
# 4. FITNESS FUNCTION (Calculates T, R, L)
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

    # Load negative flux for reflection calculation
    sim_opt.load_minus_flux_data(refl, straight_refl_data)

    sim_opt.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    trans_flux = np.array(mp.get_fluxes(trans))
    refl_flux = np.array(mp.get_fluxes(refl))

    # Calculate S-Parameters
    T = np.divide(trans_flux, straight_trans_flux, out=np.zeros_like(trans_flux), where=straight_trans_flux != 0)
    R = np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)

    # Clip to physical bounds
    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)
    L = 1 - T - R

    # Fitness Function: maximize mean reflection
    fitness = np.mean(R)

    return fitness, T, R, L


# =======================================================================
# 5. LIVE PLOTTING SETUP
# =======================================================================

plt.ion()  # Enable Interactive Mode
fig, (ax_fitness, ax_trl) = plt.subplots(1, 2, figsize=(12, 5))

# Setup Loss Plot
line_fitness, = ax_fitness.plot([], [], 'r-o', label='Best Fitness')
ax_fitness.set_title("Optimization Progress")
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")
ax_fitness.grid(True)
ax_fitness.legend()

# Setup TRL Plot
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
# 6. GENETIC ALGORITHM LOOP
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
    """Exports the best results of the current generation to a CSV."""
    filename = f"{OUTPUT_DIR}/best_gen_{gen_num}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Header Info
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Generation", gen_num])
        writer.writerow(["Best Fitness", fitness])
        writer.writerow(["Genome", str(genome)])
        writer.writerow([])

        # Spectral Data
        writer.writerow(["Frequency", "T", "R", "L"])
        for k in range(len(flux_freqs)):
            writer.writerow([flux_freqs[k], T[k], R[k], L[k]])
    print(f"  -> Exported: {filename}")


print(f"--- Starting Optimization ({GENERATIONS} Gens) ---")

half_segments = (NUM_SEGMENTS + 1) // 2
seed_gene_1 = [2] * half_segments + [1] * (NUM_SEGMENTS - half_segments)
seed_gene_2 = [1] * half_segments + [2] * (NUM_SEGMENTS - half_segments)
population = [seed_gene_1, seed_gene_2]
while len(population) < POPULATION_SIZE:
    population.append(create_random_genome())
history_fitness = []

# Global best tracking
best_global_fitness = -float('inf')
best_global_genome = None
best_global_T = None
best_global_R = None
best_global_L = None

for gen in range(GENERATIONS):
    ranked_population = []

    # Evaluate Population
    for i, genome in enumerate(population):
        fitness, T, R, L = calculate_fitness(genome)
        ranked_population.append((fitness, genome))

        if fitness > best_global_fitness:
            best_global_fitness = fitness
            best_global_genome = genome
            # Capture the spectral data for the global best
            best_global_T = T
            best_global_R = R
            best_global_L = L

            # --- LIVE UPDATE PLOT ---
            line_T.set_ydata(T)
            line_R.set_ydata(R)
            line_L.set_ydata(L)
            ax_trl.set_title(f"Best Result (Gen {gen + 1}) - Fitness: {fitness:.4f}")
            fig.canvas.flush_events()
            plt.draw()

    # Sort to find best in this batch
    ranked_population.sort(key=lambda x: x[0], reverse=True)
    current_best_fitness = ranked_population[0][0]
    history_fitness.append(current_best_fitness)

    # --- UPDATE LOSS PLOT ---
    line_fitness.set_data(range(1, len(history_fitness) + 1), history_fitness)
    ax_fitness.relim()
    ax_fitness.autoscale_view()
    fig.canvas.flush_events()
    plt.draw()
    plt.pause(0.01)

    print(f"Gen {gen + 1}/{GENERATIONS} | Best Fitness: {current_best_fitness:.4f}")

    # --- EXPORT DATA FOR THIS GENERATION ---
    # We export the GLOBAL best found so far (safest bet),
    # but strictly associated with the current generation count.
    save_generation_results(gen + 1, best_global_fitness, best_global_genome,
                            best_global_T, best_global_R, best_global_L)

    # Next Gen Setup
    next_generation = [x[1] for x in ranked_population[:ELITISM_COUNT]]
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
# 7. FINAL VISUALIZATION (GEOMETRY PLOT)
# =======================================================================
plt.ioff()  # Turn off interactive mode

print(f"--- Optimization Finished ---")
print(f"Final Best Fitness: {best_global_fitness}")
print(f"Best Genome: {best_global_genome}")

# Create a new figure for the Geometry
fig_geo = plt.figure(figsize=(10, 4))

# Rebuild the final simulation object
final_geometry = get_base_geometry() + get_optimization_geometry(best_global_genome)
sim_final = mp.Simulation(cell_size=CELL_SIZE,
                          boundary_layers=PML_LAYERS,
                          geometry=final_geometry,
                          sources=sources,
                          resolution=RESOLUTION)

# Plot the 2D Geometry
sim_final.plot2D()
plt.title(f"Optimized Topology (Fitness: {best_global_fitness:.3f})")
plt.show()

# (Optional: The loop already exported the final generation, so Step 8 is redundant,
# but kept here if you want a specifically named 'final' file)



