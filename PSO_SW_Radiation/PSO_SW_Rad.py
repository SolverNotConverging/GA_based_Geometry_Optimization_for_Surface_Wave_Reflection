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

# =======================
# PSO Settings
# =======================
SWARM_SIZE = 40
ITERATIONS = 100

# "Inertia / cognitive / social" (discrete PSO variant uses these as weights for probabilities)
W_INERTIA = 0.50
C1 = 1.25  # attraction to personal best
C2 = 1.75  # attraction to global best

# Mutation-like exploration (prevents early stagnation)
RANDOM_FLIP_PROB = 0.03

# Data saving
OUTPUT_DIR = "optimization_data_pso"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_EVERY = 1  # save every N iterations

# Far-field fitness settings
FF_NANGLES = 360
TARGET_THETA = np.pi / 2
MAINLOBE_EXCLUDE_DEG = 30
SLL_TARGET_DB = -15

# Objective weights (tune these)
W_LMAX = 2.0
W_ANGLE = 1.0
W_SLL = 0.5


# =======================================================================
# 2. GEOMETRY GENERATION
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
    """
    Genome states:
      0: Base
      1: Cut Ground
      2: Top PEC
      3: Both
    """
    geometry_additions = []
    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * MIN_FEATURE_SIZE
        if length <= 0:
            return
        center_x = OPT_X_START + (start_idx * MIN_FEATURE_SIZE) + (length / 2)

        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0))
            )

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
    d = np.abs(a - b)
    return np.minimum(d, 2 * np.pi - d)


def wrap_angle_err(theta, target):
    theta = theta % (2 * np.pi)
    target = target % (2 * np.pi)
    d = abs(theta - target)
    return min(d, 2 * np.pi - d)


def far_field_metrics(genome, target_freq):
    full_geometry = get_base_geometry() + get_optimization_geometry(genome)

    sim_ff = mp.Simulation(cell_size=CELL_SIZE,
                           boundary_layers=PML_LAYERS,
                           geometry=full_geometry,
                           sources=sources,
                           resolution=RESOLUTION)

    n2f_width = CELL_SIZE.x
    n2f_y_pos = (CELL_SIZE.y / 2) - PML_THICKNESS

    p1 = mp.Near2FarRegion(center=mp.Vector3(0, n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=+1)
    p2 = mp.Near2FarRegion(center=mp.Vector3(0, -n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=-1)

    n2f_obj = sim_ff.add_near2far(target_freq, 0, 1, p1, p2)

    sim_ff.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    ff_distance = 100 * (1 / target_freq)

    angles = np.linspace(0, 2 * np.pi, FF_NANGLES, endpoint=False)
    pat = np.zeros_like(angles)

    for i, theta in enumerate(angles):
        ff_pt = mp.Vector3(ff_distance * np.cos(theta), ff_distance * np.sin(theta))
        ff = sim_ff.get_farfield(n2f_obj, ff_pt)
        pat[i] = np.abs(ff[5])  # Hz component

    main_peak = float(np.max(pat))
    if main_peak <= 0:
        peak_theta = float(TARGET_THETA % (2 * np.pi))
        sll_db = 0.0
        return peak_theta, sll_db, angles, pat

    peak_idx = int(np.argmax(pat))
    peak_theta = float(angles[peak_idx])

    exclude = np.deg2rad(MAINLOBE_EXCLUDE_DEG)
    mask = circ_dist(angles, peak_theta) > exclude

    sidelobe_peak = float(np.max(pat[mask])) if np.any(mask) else 0.0
    sll = sidelobe_peak / main_peak
    sll_db = 20 * np.log10(max(sll, 1e-12))

    return peak_theta, sll_db, angles, pat


fitness_cache = {}  # tuple(genome) -> results dict


def calculate_fitness(genome):
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
    R = np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)

    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)
    L = 1 - T - R

    max_L_idx = int(np.argmax(L))
    Lmax = float(L[max_L_idx])
    f_star = float(flux_freqs[max_L_idx])

    peak_theta, sll_db, ff_angles, ff_pat = far_field_metrics(genome, f_star)

    angle_err = wrap_angle_err(peak_theta, TARGET_THETA)
    angle_pen = angle_err / (np.pi / 2)

    sll_pen = max(0.0, (sll_db - SLL_TARGET_DB) / abs(SLL_TARGET_DB))

    loss = (W_LMAX * (1.0 - Lmax)
            + W_ANGLE * angle_pen
            + W_SLL * sll_pen)

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

line_loss, = ax_loss.plot([], [], 'r-o', label='Best Loss')
ax_loss.set_title("PSO Optimization Progress")
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Loss")
ax_loss.grid(True)
ax_loss.legend()

line_T, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'b-', label='T')
line_R, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'r--', label='R')
line_L, = ax_trl.plot(flux_freqs, np.zeros_like(flux_freqs), 'g:', label='L')
ax_trl.set_title("S-Parameters (Best so far)")
ax_trl.set_ylim(0, 1.05)
ax_trl.grid(True)
ax_trl.legend()

line_ff, = ax_ff.plot([], [], color='m', linewidth=2)
ax_ff.set_title("Far Field ($|H_z|$) @ f* where L is max (0–360°)")
ax_ff.grid(True)

plt.tight_layout()

# =======================================================================
# 6. DISCRETE PSO (CATEGORICAL GENOME) LOOP
# =======================================================================

STATES = [0, 1, 2, 3]


def random_genome():
    return [random.choice(STATES) for _ in range(NUM_SEGMENTS)]


def majority_vote(a, b, c, weights):
    """
    Pick among {a,b,c} with weighted probabilities.
    weights = (wa, wb, wc)
    """
    choices = [a, b, c]
    w = np.array(weights, dtype=float)
    w = np.maximum(w, 0)
    s = w.sum()
    if s <= 0:
        return random.choice(choices)
    w /= s
    return random.choices(choices, weights=w, k=1)[0]


def pso_step(x, pbest, gbest):
    """
    Discrete PSO update:
      - with inertia, keep current state
      - with C1, move toward pbest at that dim
      - with C2, move toward gbest at that dim
    plus small random flips.
    """
    new = x[:]
    for d in range(NUM_SEGMENTS):
        if random.random() < RANDOM_FLIP_PROB:
            new[d] = random.choice(STATES)
            continue

        new[d] = majority_vote(
            a=x[d],
            b=pbest[d],
            c=gbest[d],
            weights=(W_INERTIA, C1, C2)
        )
    return new


print(f"--- Starting PSO Optimization ({ITERATIONS} iterations) ---")
print(f"Data will be saved to: {OUTPUT_DIR}/")

# Swarm init
swarm = [random_genome() for _ in range(SWARM_SIZE)]
pbest = [s[:] for s in swarm]
pbest_loss = [float('inf')] * SWARM_SIZE

gbest = None
gbest_loss = float('inf')

history_best = []

# Best data for export/plotting
best_T = np.zeros_like(flux_freqs)
best_R = np.zeros_like(flux_freqs)
best_L = np.zeros_like(flux_freqs)
best_ff_angles = np.array([])
best_ff_data = np.array([])
best_ff_info = None

for it in range(1, ITERATIONS + 1):
    new_best_found = False

    # Evaluate swarm
    for i in range(SWARM_SIZE):
        loss, T, R, L, ff_info = calculate_fitness(swarm[i])

        # personal best update
        if loss < pbest_loss[i]:
            pbest_loss[i] = loss
            pbest[i] = swarm[i][:]

        # global best update
        if loss < gbest_loss:
            gbest_loss = loss
            gbest = swarm[i][:]
            new_best_found = True

            best_T, best_R, best_L = T, R, L
            best_ff_info = ff_info
            best_ff_angles = ff_info["ff_angles"]
            best_ff_data = ff_info["ff_pat"]

    history_best.append(gbest_loss)

    # Update plots
    line_loss.set_data(range(1, len(history_best) + 1), history_best)
    ax_loss.relim()
    ax_loss.autoscale_view()

    if new_best_found and best_ff_data.size > 0:
        line_T.set_ydata(best_T)
        line_R.set_ydata(best_R)
        line_L.set_ydata(best_L)
        ax_trl.set_title(
            f"Best so far (Iter {it}) | Loss={gbest_loss:.4f} | "
            f"Lmax={best_ff_info['Lmax']:.3f} @ f*={best_ff_info['f_star']:.4f}"
        )

        line_ff.set_data(best_ff_angles, best_ff_data)
        ax_ff.set_ylim(0, float(np.max(best_ff_data)) * 1.1 if np.max(best_ff_data) > 0 else 1.0)
        ax_ff.set_title(
            f"Far Field @ f*={best_ff_info['f_star']:.4f} (0–360°)\n"
            f"Peak={best_ff_info['peak_theta_deg']:.1f}° (err {best_ff_info['angle_err_deg']:.1f}°), "
            f"SLL={best_ff_info['sll_db']:.1f} dB"
        )

    fig.canvas.flush_events()
    plt.draw()
    plt.pause(0.01)

    print(
        f"Iter {it}/{ITERATIONS} | "
        f"Best Loss={gbest_loss:.4f} | "
        f"Peak={best_ff_info['peak_theta_deg']:.1f}° | "
        f"SLL={best_ff_info['sll_db']:.1f} dB | "
        f"f*={best_ff_info['f_star']:.4f}"
    )

    # Save data every SAVE_EVERY
    if (it % SAVE_EVERY) == 0:
        it_filename = f"{OUTPUT_DIR}/iter_{it:03d}.csv"
        with open(it_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["Iteration", it])
            writer.writerow(["Best Global Loss", gbest_loss])
            if best_ff_info:
                writer.writerow(["f* (where L is max)", best_ff_info["f_star"]])
                writer.writerow(["Lmax", best_ff_info["Lmax"]])
                writer.writerow(["Peak Theta (deg)", best_ff_info["peak_theta_deg"]])
                writer.writerow(["Angle Error (deg)", best_ff_info["angle_err_deg"]])
                writer.writerow(["SLL (dB)", best_ff_info["sll_db"]])
            writer.writerow([])

            writer.writerow(["Best Genome"])
            writer.writerow(gbest if gbest is not None else [])
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

        print(f"Saved iteration data to {it_filename}")

    # Move swarm (after evaluation)
    # If gbest is None (shouldn't happen), skip update
    if gbest is not None:
        for i in range(SWARM_SIZE):
            swarm[i] = pso_step(swarm[i], pbest[i], gbest)

# =======================================================================
# 7. FINAL VISUALIZATION
# =======================================================================
plt.ioff()
print("--- PSO Optimization Finished ---")

plot_filename = f"{OUTPUT_DIR}/final_optimization_results.png"
plt.savefig(plot_filename, dpi=200)
print(f"Final plot saved to {plot_filename}")
plt.show()
