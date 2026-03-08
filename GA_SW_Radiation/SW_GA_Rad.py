import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os

# =======================================================================
# 1. CONFIGURATION
# =======================================================================

# Simulation Constants
RESOLUTION = 20
CELL_SIZE = mp.Vector3(30, 16, 0)
PML_THICKNESS = 1.0
PML_LAYERS = [mp.PML(PML_THICKNESS)]

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
OPT_X_END = 10.0
MIN_FEATURE_SIZE = 0.5
NUM_SEGMENTS = int(round((OPT_X_END - OPT_X_START) / MIN_FEATURE_SIZE))

# Genetic Algorithm Settings
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.1
ELITISM_COUNT = 2
TOURNAMENT_K = 3

# Far-field fitness settings
FF_NANGLES = 360                 # samples over 0..2pi (full 360 deg)
TARGET_THETA = np.pi / 2         # +y direction = 90 deg
MAINLOBE_EXCLUDE_DEG = 30        # exclude +/- around main beam peak for sidelobe search
SLL_TARGET_DB = -15              # want sidelobes <= -15 dB (more negative is better)

# Objective weights (tune these)
W_LMAX = 1.0
W_ANGLE = 2.0
W_SLL = 1.0

# Output Directory
OUTPUT_DIR = "optimization_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    States: 0: Base, 1: Cut Ground, 2: Top PEC, 3: Both
    """
    geometry_additions = []
    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * MIN_FEATURE_SIZE
        if length <= 0:
            return
        center_x = OPT_X_START + (start_idx * MIN_FEATURE_SIZE) + (length / 2)

        # Cutting ground (States 1 and 3)
        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0))
            )

        # Adding top PEC (States 2 and 3)
        if state == 2 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, H_SUB + T_PEC / 2, 0),
                         material=mp.perfect_electric_conductor)
            )

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
                     size=mp.Vector3(0, H_SUB, 0))]

sim_norm = mp.Simulation(cell_size=CELL_SIZE,
                         boundary_layers=PML_LAYERS,
                         geometry=get_base_geometry(),
                         sources=sources,
                         resolution=RESOLUTION)

refl_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

sim_norm.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-9))

straight_refl_data = sim_norm.get_flux_data(refl_mon)
straight_trans_flux = np.array(mp.get_fluxes(trans_mon))
flux_freqs = np.array(mp.get_flux_freqs(trans_mon))

sim_norm.reset_meep()
print("--- Normalization Complete ---")


# =======================================================================
# 4. FITNESS & ANALYSIS FUNCTIONS
# =======================================================================

def circ_dist(a, b):
    """Circular distance on [0, 2pi). Works with numpy arrays."""
    d = np.abs(a - b)
    return np.minimum(d, 2*np.pi - d)

def wrap_angle_err(theta, target):
    """Smallest absolute angular difference (rad) on a 2pi circle."""
    theta = theta % (2*np.pi)
    target = target % (2*np.pi)
    d = abs(theta - target)
    return min(d, 2*np.pi - d)


def far_field_metrics(genome, target_freq):
    """
    Far field at target_freq using ONLY top/bottom N2F regions.
    Returns: peak_theta(rad in [0,2pi)), sll_db(dB),
             angles(rad in [0,2pi)), pattern(|Hz|)
    Angle convention:
      theta=0     -> +x
      theta=pi/2  -> +y (desired)
      theta=pi    -> -x
      theta=3pi/2 -> -y
    """

    full_geometry = get_base_geometry() + get_optimization_geometry(genome)

    sim_ff = mp.Simulation(cell_size=CELL_SIZE,
                           boundary_layers=PML_LAYERS,
                           geometry=full_geometry,
                           sources=sources,
                           resolution=RESOLUTION)

    # Top & bottom only (as you requested)
    n2f_width = CELL_SIZE.x
    n2f_y_pos = (CELL_SIZE.y / 2) - PML_THICKNESS

    p1 = mp.Near2FarRegion(center=mp.Vector3(0,  n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=+1)
    p2 = mp.Near2FarRegion(center=mp.Vector3(0, -n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=-1)

    n2f_obj = sim_ff.add_near2far(target_freq, 0, 1, p1, p2)

    sim_ff.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    ff_distance = 100 * (1 / target_freq)

    # FULL 360° (0..2pi)
    angles = np.linspace(0, 2*np.pi, FF_NANGLES, endpoint=False)
    pat = np.zeros_like(angles)

    for i, theta in enumerate(angles):
        ff_pt = mp.Vector3(ff_distance * np.cos(theta), ff_distance * np.sin(theta))
        ff = sim_ff.get_farfield(n2f_obj, ff_pt)
        pat[i] = np.abs(ff[5])  # Hz component

    main_peak = float(np.max(pat))
    if main_peak <= 0:
        peak_theta = float(TARGET_THETA % (2*np.pi))
        sll_db = 0.0
        return peak_theta, sll_db, angles, pat

    peak_idx = int(np.argmax(pat))
    peak_theta = float(angles[peak_idx])

    # Sidelobe: max outside +/- exclude around main beam peak (circular)
    exclude = np.deg2rad(MAINLOBE_EXCLUDE_DEG)
    mask = circ_dist(angles, peak_theta) > exclude

    sidelobe_peak = float(np.max(pat[mask])) if np.any(mask) else 0.0
    sll = sidelobe_peak / main_peak
    sll_db = 20 * np.log10(max(sll, 1e-12))  # negative is good

    return peak_theta, sll_db, angles, pat


# Cache fitness to avoid repeating expensive sims
fitness_cache = {}  # key: tuple(genome) -> dict of results


def calculate_fitness(genome):
    """
    Fitness to MINIMIZE:
      - maximize Lmax (at some freq f* in band)
      - at that same f*, main beam aims at +y (pi/2)
      - and sidelobes <= SLL_TARGET_DB
    Returns: loss, T, R, L, ff_info(dict)
    """
    key = tuple(genome)
    if key in fitness_cache:
        c = fitness_cache[key]
        return c["loss"], c["T"], c["R"], c["L"], c["ff_info"]

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
    R = np.divide(-refl_flux,  straight_trans_flux, out=np.zeros_like(refl_flux),  where=straight_trans_flux != 0)

    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)
    L = 1 - T - R

    # Find f* where L is max
    max_L_idx = int(np.argmax(L))
    Lmax = float(L[max_L_idx])
    f_star = float(flux_freqs[max_L_idx])

    # Far-field at f* (FULL 360° now)
    peak_theta, sll_db, ff_angles, ff_pat = far_field_metrics(genome, f_star)

    # Beam pointing penalty
    angle_err = wrap_angle_err(peak_theta, TARGET_THETA)   # rad
    angle_pen = angle_err / (np.pi / 2)                    # normalize ~0..1

    # SLL penalty: penalize if sidelobes are above the target (less negative)
    sll_pen = max(0.0, (sll_db - SLL_TARGET_DB) / abs(SLL_TARGET_DB))

    # Combine
    loss = (W_LMAX  * (1.0 - Lmax)
            + W_ANGLE * angle_pen
            + W_SLL   * sll_pen)

    ff_info = {
        "f_star": f_star,
        "Lmax": Lmax,
        "peak_theta": peak_theta,
        "peak_theta_deg": np.degrees(peak_theta),
        "angle_err_deg": np.degrees(angle_err),
        "sll_db": sll_db,
        "ff_angles": ff_angles,
        "ff_pat": ff_pat,
    }

    fitness_cache[key] = {"loss": loss, "T": T, "R": R, "L": L, "ff_info": ff_info}
    return loss, T, R, L, ff_info


# =======================================================================
# 5. LIVE PLOTTING SETUP
# =======================================================================

plt.ion()
fig = plt.figure(figsize=(16, 5))
ax_loss = plt.subplot(131)
ax_trl = plt.subplot(132)
ax_ff = plt.subplot(133, polar=True)

# Loss plot
line_loss, = ax_loss.plot([], [], 'r-o', label='Best Loss')
ax_loss.set_title("Optimization Progress")
ax_loss.set_xlabel("Generation")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend()

# TRL plot
line_T, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'b-', label='T')
line_R, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'r--', label='R')
line_L, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'g:', label='L')
ax_trl.set_title("S-Parameters (Best so far)")
ax_trl.set_ylim(0, 1.05)
ax_trl.grid(True)
ax_trl.legend()

# Far field plot
line_ff, = ax_ff.plot([], [], color='m', linewidth=2)
ax_ff.set_title("Far Field ($|H_z|$) @ f* where L is max (0–360°)")
ax_ff.grid(True)

plt.tight_layout()


# =======================================================================
# 6. GENETIC ALGORITHM LOOP
# =======================================================================

def create_random_genome():
    return [random.choice([0, 1, 2, 3]) for _ in range(NUM_SEGMENTS)]

def mutate(genome):
    new_genome = genome[:]
    for i in range(len(new_genome)):
        if random.random() < MUTATION_RATE:
            new_genome[i] = random.choice([0, 1, 2, 3])
    return new_genome

def crossover(parent1, parent2):
    point = random.randint(1, NUM_SEGMENTS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def tournament_select(ranked_population, k=TOURNAMENT_K):
    contestants = random.sample(ranked_population, k)
    return min(contestants, key=lambda x: x[0])[1]

print(f"--- Starting Optimization ({GENERATIONS} Gens) ---")
print(f"Data will be saved to: {OUTPUT_DIR}/")

population = [create_random_genome() for _ in range(POPULATION_SIZE)]
history_loss = []

best_global_loss = float('inf')
best_global_genome = None

# Best data for export/plotting
best_ff_angles = np.array([])
best_ff_data = np.array([])
best_ff_freq = 0.0
best_T = np.zeros_like(flux_freqs)
best_R = np.zeros_like(flux_freqs)
best_L = np.zeros_like(flux_freqs)
best_ff_info = None

for gen in range(GENERATIONS):
    ranked_population = []
    new_best_found_this_gen = False

    for i, genome in enumerate(population):
        loss, T, R, L, ff_info = calculate_fitness(genome)
        ranked_population.append((loss, genome))

        if loss < best_global_loss:
            best_global_loss = loss
            best_global_genome = genome
            new_best_found_this_gen = True

            best_T = T
            best_R = R
            best_L = L

            best_ff_info = ff_info
            best_ff_freq = ff_info["f_star"]
            best_ff_angles = ff_info["ff_angles"]
            best_ff_data = ff_info["ff_pat"]

            # Update TRL plot
            line_T.set_ydata(T)
            line_R.set_ydata(R)
            line_L.set_ydata(L)
            ax_trl.set_title(
                f"Best so far (Gen {gen+1}) | Loss={loss:.4f} | "
                f"Lmax={ff_info['Lmax']:.3f} @ f*={ff_info['f_star']:.4f}"
            )

    ranked_population.sort(key=lambda x: x[0])
    current_best_loss = ranked_population[0][0]
    history_loss.append(current_best_loss)

    # Update loss plot
    line_loss.set_data(range(1, len(history_loss) + 1), history_loss)
    ax_loss.relim()
    ax_loss.autoscale_view()

    # Update far-field plot if new global best
    if new_best_found_this_gen and best_ff_data.size > 0:
        line_ff.set_data(best_ff_angles, best_ff_data)
        ax_ff.set_ylim(0, float(np.max(best_ff_data)) * 1.1 if np.max(best_ff_data) > 0 else 1.0)
        ax_ff.set_title(
            f"Far Field @ f*={best_ff_freq:.4f} (0–360°)\n"
            f"Peak={best_ff_info['peak_theta_deg']:.1f}° (err {best_ff_info['angle_err_deg']:.1f}°), "
            f"SLL={best_ff_info['sll_db']:.1f} dB"
        )

    fig.canvas.flush_events()
    plt.draw()
    plt.pause(0.01)

    print(
        f"Gen {gen+1}/{GENERATIONS} | "
        f"BestGen Loss={current_best_loss:.4f} | "
        f"BestAll Loss={best_global_loss:.4f} | "
        f"Peak={best_ff_info['peak_theta_deg']:.1f}° | "
        f"SLL={best_ff_info['sll_db']:.1f} dB | "
        f"f*={best_ff_freq:.4f}"
    )

    # ===================================================================
    # EXPORT DATA FOR THIS GENERATION
    # ===================================================================
    gen_filename = f"{OUTPUT_DIR}/gen_{gen + 1:03d}.csv"
    with open(gen_filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["Generation", gen + 1])
        writer.writerow(["Best Global Loss", best_global_loss])
        if best_ff_info:
            writer.writerow(["f* (where L is max)", best_ff_info["f_star"]])
            writer.writerow(["Lmax", best_ff_info["Lmax"]])
            writer.writerow(["Peak Theta (deg)", best_ff_info["peak_theta_deg"]])
            writer.writerow(["Angle Error (deg)", best_ff_info["angle_err_deg"]])
            writer.writerow(["SLL (dB)", best_ff_info["sll_db"]])
        writer.writerow([])

        writer.writerow(["Best Genome"])
        writer.writerow(best_global_genome if best_global_genome is not None else [])
        writer.writerow([])

        writer.writerow(["Frequency", "T", "R", "L"])
        for j in range(len(flux_freqs)):
            writer.writerow([flux_freqs[j], best_T[j], best_R[j], best_L[j]])
        writer.writerow([])

        writer.writerow(["Far Field Data @ f*"])
        writer.writerow(["Angle (rad)", "Magnitude"])
        if best_ff_angles.size > 0:
            for j in range(len(best_ff_angles)):
                writer.writerow([best_ff_angles[j], best_ff_data[j]])

    print(f"Saved generation data to {gen_filename}")
    # ===================================================================

    # Next generation
    next_generation = [x[1] for x in ranked_population[:ELITISM_COUNT]]
    while len(next_generation) < POPULATION_SIZE:
        p1 = tournament_select(ranked_population, TOURNAMENT_K)
        p2 = tournament_select(ranked_population, TOURNAMENT_K)
        c1, c2 = crossover(p1, p2)
        next_generation.append(mutate(c1))
        if len(next_generation) < POPULATION_SIZE:
            next_generation.append(mutate(c2))

    population = next_generation


# =======================================================================
# 7. FINAL VISUALIZATION
# =======================================================================
plt.ioff()
print("--- Optimization Finished ---")

plot_filename = f"{OUTPUT_DIR}/final_optimization_results.png"
plt.savefig(plot_filename)
print(f"Final plot saved to {plot_filename}")
plt.show()
