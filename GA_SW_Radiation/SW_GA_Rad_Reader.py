import ast
import csv
import os

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

# =======================================================================
# 1. CONFIGURATION (Must match Optimization Script)
# =======================================================================

CSV_INPUT_FILENAME = "optimization_data/gen_001.csv"

# Output files
OUT_DIR = "reader_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GEOM_LAYOUT_PNG = os.path.join(OUT_DIR, "geometry_layout.png")
ANIM_MP4 = os.path.join(OUT_DIR, "fields_animation.mp4")
ANIM_GIF = os.path.join(OUT_DIR, "fields_animation.gif")
FINAL_ANALYSIS_PNG = os.path.join(OUT_DIR, "reader_results_analysis.png")

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

# Animation settings (frames recorded during sim_opt run)
ANIM_DT = 1.0  # time between frames in simulation units
ANIM_FPS = 10  # output video fps


# =======================================================================
# 2. HELPER: DATA LOADER
# =======================================================================

def load_data_from_csv(filename):
    print(f"--- Loading data from {filename} ---")
    genome = None

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

        for i, row in enumerate(rows):
            if not row:
                continue

            if "Best Genome" in row[0]:
                target_row = rows[i + 1]

                # Method 1: saved as one string "[0, 1, 2]"
                if len(target_row) == 1 and '[' in target_row[0]:
                    try:
                        genome = ast.literal_eval(target_row[0])
                    except Exception:
                        genome = None

                # Method 2: saved as many cells
                if genome is None:
                    try:
                        genome = [int(x) for x in target_row if x.strip() != ""]
                    except Exception:
                        genome = None

                print(f"Genome Found ({len(genome)} segments): {genome}")
                break

    if genome is None or not isinstance(genome, list) or len(genome) == 0:
        raise ValueError("Could not find a valid list for 'Best Genome' in the CSV file.")

    return genome


# =======================================================================
# 3. GEOMETRY BUILDER
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

    if not isinstance(genome, list):
        raise TypeError(f"Genome must be a list, got {type(genome)}")

    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * MIN_FEATURE_SIZE
        if length <= 0:
            return
        center_x = OPT_X_START + (start_idx * MIN_FEATURE_SIZE) + (length / 2)

        # Cut Ground
        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0)))

        # Add Top PEC
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
# 4. GEOMETRY IMAGE SAVER (no simulation required)
# =======================================================================

def save_geometry_images(full_geom):
    """
    Uses one short Simulation object ONLY for plotting the geometry.
    (No time stepping; just plot2D).
    """
    sim_plot = mp.Simulation(cell_size=CELL_SIZE,
                             boundary_layers=PML_LAYERS,
                             geometry=full_geom,
                             sources=[],
                             resolution=RESOLUTION)

    # Epsilon / geometry map
    fig1 = plt.figure(figsize=(8, 4))
    sim_plot.plot2D(ax=plt.gca(),
                    output_plane=mp.Volume(center=mp.Vector3(), size=CELL_SIZE),
                    fields=None)
    plt.title("Geometry / Epsilon map")
    plt.tight_layout()
    fig1.savefig(GEOM_LAYOUT_PNG, dpi=200)
    plt.close(fig1)

    print(f"Saved geometry images:\n  {GEOM_LAYOUT_PNG}")


# =======================================================================
# 5. MAIN SIMULATION & ANALYSIS
# =======================================================================

if __name__ == "__main__":

    # Load Genome
    best_genome = load_data_from_csv(CSV_INPUT_FILENAME)

    # Define Source and Monitor locations
    src_pt = mp.Vector3(-7.5, H_SUB / 2, 0)
    refl_pt = mp.Vector3(-13, H_SUB / 2, 0)
    trans_pt = mp.Vector3(13, H_SUB / 2, 0)

    sources = [mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                         component=mp.Hz,
                         center=src_pt,
                         size=mp.Vector3(0, H_SUB, 0))]

    base_geom = get_base_geometry()
    full_geom = base_geom + get_optimization_geometry(best_genome)

    # Save geometry images (no time stepping)
    print("\n--- Saving geometry images ---")
    save_geometry_images(full_geom)

    # ---------------------------------------------------------
    # STEP A: Normalization (ONE simulation)
    # ---------------------------------------------------------
    print("\n--- Step A: Running Normalization (straight waveguide) ---")
    sim_norm = mp.Simulation(cell_size=CELL_SIZE,
                             boundary_layers=PML_LAYERS,
                             geometry=base_geom,
                             sources=sources,
                             resolution=RESOLUTION)

    refl_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
    trans_mon = sim_norm.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

    sim_norm.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-9))

    straight_refl_data = sim_norm.get_flux_data(refl_mon)
    straight_trans_flux = np.array(mp.get_fluxes(trans_mon))
    flux_freqs = np.array(mp.get_flux_freqs(trans_mon))

    # ---------------------------------------------------------
    # STEP B: Optimized run + animation (ONE simulation)
    # ---------------------------------------------------------
    print("\n--- Step B: Running optimized geometry (and recording animation) ---")

    sim_opt = mp.Simulation(cell_size=CELL_SIZE,
                            boundary_layers=PML_LAYERS,
                            geometry=full_geom,
                            sources=sources,
                            resolution=RESOLUTION)

    refl = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
    trans = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

    sim_opt.load_minus_flux_data(refl, straight_refl_data)

    # Attach Animate2D to THIS simulation
    anim = mp.Animate2D(sim_opt,
                        fields=mp.Hz,
                        realtime=False,
                        normalize=True,
                        output_plane=mp.Volume(center=mp.Vector3(), size=CELL_SIZE))

    # Run once: flux + animation frames captured here
    sim_opt.run(mp.at_every(ANIM_DT, anim),
                until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    # Compute T/R/L from same run
    trans_flux = np.array(mp.get_fluxes(trans))
    refl_flux = np.array(mp.get_fluxes(refl))

    T = np.divide(trans_flux, straight_trans_flux, out=np.zeros_like(trans_flux), where=straight_trans_flux != 0)
    R = np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)

    T = np.clip(T, 0, 1)
    R = np.clip(R, 0, 1)
    L = 1 - T - R

    max_L_idx = int(np.argmax(L))
    target_freq = float(flux_freqs[max_L_idx])
    max_L_val = float(L[max_L_idx])

    print("\nAnalysis Results:")
    print(f"Max Loss (L): {max_L_val:.4f} at Frequency: {target_freq:.4f}")

    # Save animation AFTER optimized sim finishes
    print("\n--- Saving animation ---")
    try:
        anim.to_mp4(ANIM_FPS, ANIM_MP4)
        print(f"Saved animation MP4: {ANIM_MP4}")
    except Exception as e:
        print(f"Could not save MP4 via Animate2D: {e}")
        try:
            anim.to_gif(ANIM_FPS, ANIM_GIF)
            print(f"Saved animation GIF: {ANIM_GIF}")
        except Exception as e2:
            print(f"Could not save GIF via Animate2D either: {e2}")

    # ---------------------------------------------------------
    # STEP C: Far Field at peak-L frequency (separate sim)
    # ---------------------------------------------------------
    print(f"\n--- Step C: Calculating Far Field at f={target_freq:.4f} ---")

    sim_ff = mp.Simulation(cell_size=CELL_SIZE,
                           boundary_layers=PML_LAYERS,
                           geometry=full_geom,
                           sources=sources,
                           resolution=RESOLUTION)

    n2f_width = CELL_SIZE.x
    n2f_y_pos = (CELL_SIZE.y / 2) - PML_THICKNESS

    p1 = mp.Near2FarRegion(center=mp.Vector3(0, n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=+1)
    p2 = mp.Near2FarRegion(center=mp.Vector3(0, -n2f_y_pos), size=mp.Vector3(n2f_width, 0), weight=-1)

    n2f_obj = sim_ff.add_near2far(target_freq, 0, 1, p1, p2)

    sim_ff.run(until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

    ff_distance = 100 * (1 / target_freq)
    angles = np.linspace(0, 2 * np.pi, 360)
    far_fields_abs = []

    for theta in angles:
        ff_pt = mp.Vector3(ff_distance * np.cos(theta), ff_distance * np.sin(theta))
        ff = sim_ff.get_farfield(n2f_obj, ff_pt)
        far_fields_abs.append(np.abs(ff[5]))  # Hz component

    # ---------------------------------------------------------
    # STEP D: Final Plotting
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(flux_freqs, T, 'b-', label='T')
    ax1.plot(flux_freqs, R, 'r--', label='R')
    ax1.plot(flux_freqs, L, 'g:', label='Loss (L)')
    ax1.axvline(target_freq, color='k', linestyle='--', alpha=0.5, label='Max L Freq')
    ax1.set_xlabel("Frequency (Meep units)")
    ax1.set_ylabel("Power Fraction")
    ax1.set_title("Re-simulated S-Parameters")
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(1, 2, 2, polar=True)
    ax2.plot(angles, far_fields_abs, color='m', linewidth=2)
    ax2.set_title(f"Far Field Pattern @ f={target_freq:.3f}")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(FINAL_ANALYSIS_PNG, dpi=200)

    print("\nDone. Saved:")
    print(f"  {FINAL_ANALYSIS_PNG}")
    print(f"  {GEOM_LAYOUT_PNG}")
    print(f"  {ANIM_MP4} (or {ANIM_GIF})")
