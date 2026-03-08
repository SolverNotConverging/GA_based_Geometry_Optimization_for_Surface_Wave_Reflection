import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
import ast  # Safer way to parse string lists "[1, 0, ...]"

# =======================================================================
# 1. CONFIGURATION
# =======================================================================

# ---------------------------------------------------------
# UPDATE THIS FILENAME to the file you want to read
CSV_FILENAME = "optimization_results_4_2mm/best_gen_5.csv"
# ---------------------------------------------------------

# Simulation Constants
RESOLUTION = 20
CELL_SIZE = mp.Vector3(30, 20, 0)
PML_LAYERS = [mp.PML(1.0)]

# Material Constants
EPSILON_SUB = 10.2
H_SUB = 1.27
T_PEC = 0.035

# Source / Frequency
FCEN = 0.08
DF = 0.05
NFREQ = 100

# Optimization Region (Must match the writer script)
OPT_X_START = 0.0
OPT_X_END = 4.2


# =======================================================================
# 2. HELPER: ROBUST CSV LOADER
# =======================================================================

def load_genome_from_csv(filename):
    print(f"--- Loading genome from {filename} ---")
    genome = None
    historical_data = {'freq': [], 'T': [], 'R': [], 'L': []}

    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            parsing_data = False

            for row in reader:
                if not row: continue

                # 1. Parse the Genome
                # Looks for row: ["Genome", "[0, 1, 2...]"]
                if row[0] == "Genome":
                    # ast.literal_eval safely converts string string list to actual list
                    try:
                        genome = ast.literal_eval(row[1])
                        print(f"Found Genome (Length {len(genome)}): {genome}")
                    except:
                        print("Error parsing genome string.")

                # 2. Parse the Spectral Data (optional, for comparison)
                if row[0] == "Frequency":
                    parsing_data = True
                    continue

                if parsing_data and len(row) >= 4:
                    try:
                        historical_data['freq'].append(float(row[0]))
                        historical_data['T'].append(float(row[1]))
                        historical_data['R'].append(float(row[2]))
                        historical_data['L'].append(float(row[3]))
                    except ValueError:
                        pass

        if genome is None:
            raise ValueError(f"Could not find a 'Genome' row in {filename}")

        return genome, historical_data

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit()


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

    # Auto-calculate feature size based on genome length and region size
    # This ensures it matches the original simulation regardless of specific constants
    num_segments = len(genome)
    region_length = OPT_X_END - OPT_X_START
    feature_size = region_length / num_segments

    # print(f"Reconstructing Geometry: Feature Size = {feature_size:.4f}")

    current_state = genome[0]
    segment_start_idx = 0

    def add_block_from_state(state, start_idx, end_idx):
        length = (end_idx - start_idx) * feature_size
        center_x = OPT_X_START + (start_idx * feature_size) + (length / 2)

        # Logic for Cutting Ground (States 1 and 3)
        if state == 1 or state == 3:
            geometry_additions.append(
                mp.Block(mp.Vector3(length, T_PEC, mp.inf),
                         center=mp.Vector3(center_x, -T_PEC / 2, 0),
                         material=mp.Medium(epsilon=1.0)))

        # Logic for Adding Top PEC (States 2 and 3)
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
# 4. SIMULATION EXECUTION
# =======================================================================

# Load the Design
loaded_genome, historical_data = load_genome_from_csv(CSV_FILENAME)

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

# --- A. Normalization Run (Straight Waveguide) ---
print("\n--- 1. Running Normalization (Baseline) ---")
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

# --- B. Verification Run (Optimized Design) ---
print("\n--- 2. Running Verification (Loaded Design) ---")

# Rebuild full geometry
full_geometry = get_base_geometry() + get_optimization_geometry(loaded_genome)

sim_opt = mp.Simulation(cell_size=CELL_SIZE,
                        boundary_layers=PML_LAYERS,
                        geometry=full_geometry,
                        sources=sources,
                        resolution=RESOLUTION)

refl = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans = sim_opt.add_flux(FCEN, DF, NFREQ, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

sim_opt.load_minus_flux_data(refl, straight_refl_data)

# Setup Animation
animate = mp.Animate2D(fields=mp.Ez,
                       realtime=True,
                       normalize=True,
                       eps_parameters={'contour': True, 'alpha': 1.0})

# Run Simulation
sim_opt.run(mp.at_every(1, animate),
            until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-7))

# =======================================================================
# 5. DATA PROCESSING & OUTPUT
# =======================================================================

# Calculate S-Parameters
trans_flux = np.array(mp.get_fluxes(trans))
refl_flux = np.array(mp.get_fluxes(refl))

T = np.divide(trans_flux, straight_trans_flux, out=np.zeros_like(trans_flux), where=straight_trans_flux != 0)
R = np.divide(-refl_flux, straight_trans_flux, out=np.zeros_like(refl_flux), where=straight_trans_flux != 0)

# Clip
T = np.abs(T)
R = np.abs(R)
L = 1 - T - R

# Save Video
print("\n--- Exporting Animation ---")
anim_filename = CSV_FILENAME.replace(".csv", "_field.mp4")
animate.to_mp4(10, anim_filename)
print(f"Saved: {anim_filename}")

# Plot Results
print("\n--- Exporting Plot ---")
plt.figure(figsize=(10, 6))

# Plot Calculated Data (Solid Lines)
plt.plot(flux_freqs, T, 'b-', linewidth=2, label='Transmission (Re-Sim)')
plt.plot(flux_freqs, R, 'r-', linewidth=1.5, label='Reflection (Re-Sim)')
plt.plot(flux_freqs, L, 'g-', linewidth=1.5, label='Loss (Re-Sim)')

# Plot Historical Data from CSV (Dashed Lines) - if available
if len(historical_data['freq']) > 0:
    plt.plot(historical_data['freq'], historical_data['T'], 'b--', alpha=0.5, label='CSV Data')
    plt.plot(historical_data['freq'], historical_data['R'], 'r--', alpha=0.5)

plt.xlabel("Frequency (Meep units)")
plt.ylabel("Power Fraction")
plt.title(f"Verification: {CSV_FILENAME}")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)

plot_filename = CSV_FILENAME.replace(".csv", "_plot.png")
plt.savefig(plot_filename)
print(f"Saved: {plot_filename}")

# Visualize Geometry Snapshot
plt.figure(figsize=(10, 4))
sim_opt.plot2D()
plt.title("Reconstructed Geometry")
plt.savefig(CSV_FILENAME.replace(".csv", "_geo.png"))
print(f"Saved: {CSV_FILENAME.replace('.csv', '_geo.png')}")

plt.show()