import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================
# 1. SETUP VARIABLES
# =======================================================================
resolution = 20
cell = mp.Vector3(30, 16, 0)
pml_layers = [mp.PML(1.0)]

fcen = 0.08
df = 0.05
nfreq = 200

c = 3e8
unit = 1e-3

refl_pt = mp.Vector3(-13, 0.5, 0)
trans_pt = mp.Vector3(13, 0.5, 0)

waveguide = mp.Block(mp.Vector3(30, 1.27, mp.inf),
                     center=mp.Vector3(0, 1.27 / 2, 0),
                     material=mp.Medium(epsilon=10.2))

pec_block_1 = mp.Block(mp.Vector3(30, 0.035, mp.inf),
                       center=mp.Vector3(0, -0.035 / 2),
                       material=mp.perfect_electric_conductor)

pec_block_2 = mp.Block(mp.Vector3(20, 0.035, mp.inf),
                       center=mp.Vector3(-5, -0.035 / 2),
                       material=mp.perfect_electric_conductor)

pec_block_3 = mp.Block(mp.Vector3(10, 0.035, mp.inf),
                       center=mp.Vector3(10, 1.27 + 0.035 / 2),
                       material=mp.perfect_electric_conductor)

sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Hz,
                     center=mp.Vector3(-7.5, 1 / 2),
                     size=mp.Vector3(0, 1, 0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[waveguide, pec_block_1],
                    sources=sources,
                    resolution=resolution)

refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))

pt = mp.Vector3(0, 10.5, 0)
print("--- Running Device Simulation ---")
sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=100, decay_by=1e-9))

straight_refl_flux = np.array(mp.get_fluxes(refl))
straight_trans_flux = np.array(mp.get_fluxes(trans))
straight_refl_data = sim.get_flux_data(refl)

sim.reset_meep()

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[waveguide, pec_block_2, pec_block_3],
                    sources=sources,
                    resolution=resolution)

refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 4)))
trans = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 4)))
sim.load_minus_flux_data(refl, straight_refl_data)

a = 30
b = 14
p1 = mp.Near2FarRegion(center=mp.Vector3(0, b/2), size=mp.Vector3(a, 0), weight=+1)  # Top (+y)
p2 = mp.Near2FarRegion(center=mp.Vector3(0, -b/2), size=mp.Vector3(a, 0), weight=-1)  # Bottom (-y)


nfreq = 1
n2f_obj = sim.add_near2far(fcen, 0, nfreq, p1, p2)

# Animation setup
animate = mp.Animate2D(fields=mp.Ey, realtime=True, normalize=True)
print("--- Running Device Simulation ---")
sim.run(mp.at_every(10, animate),
        until_after_sources=mp.stop_when_energy_decayed(dt=100, decay_by=1e-9))

flux_freqs = np.array(mp.get_flux_freqs(refl))

perturb_refl_flux = np.array(mp.get_fluxes(refl))
perturb_trans_flux = np.array(mp.get_fluxes(trans))

F = flux_freqs * c / unit / 1e9
T = perturb_trans_flux / straight_trans_flux
R = -perturb_refl_flux / straight_trans_flux
L = 1 - T - R

# Define the far-field distance (r) and angles
r = 100 * (1 / fcen)  # Far field distance (100 wavelengths)
n_angles = 100  # Resolution of the far field pattern
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
# Container for the results
far_fields_abs = []

for theta in angles:
    # Convert polar angle to Cartesian coordinates for the far-field point
    ff_pt = mp.Vector3(r * np.cos(theta), r * np.sin(theta))

    # get_farfield returns a complex number (Ex, Ey, Ez, Hx, Hy, Hz)
    # dependent on polarization.
    ff = sim.get_farfield(n2f_obj, ff_pt)

    # For Hz mode (TE), we typically look at the magnitude of Hz
    # ff is a numpy array of complex numbers corresponding to field components
    # Index 2 corresponds to Hz (0:Ex, 1:Ey, 2:Ez, 3:Hx, 4:Hy, 5:Hz)
    far_fields_abs.append(np.abs(ff[5]))

# 6. Plotting
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, far_fields_abs, color='b', linewidth=2)
ax.set_title("Far Field Pattern ($|H_z|$)")
ax.grid(True)
plt.show()

plt.figure()
plt.plot(F, R, 'r-', label='Reflection')
plt.plot(F, T, 'b-', label='Transmission')
plt.plot(F, L, 'g-', label='Loss')
plt.ylim(0, 1)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power Fraction")
plt.legend()
plt.grid(True)
plt.title("Device S-Parameters")
plt.show()
