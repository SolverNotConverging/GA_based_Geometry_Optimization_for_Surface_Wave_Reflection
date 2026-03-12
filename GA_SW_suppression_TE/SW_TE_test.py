import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# =======================================================================
# 1. SETUP VARIABLES
# =======================================================================
resolution = 20
cell = mp.Vector3(30, 12, 0)
pml_layers = [mp.PML(1.0)]

fcen = 0.08
df = 0.05
nfreq = 100

c = 3e8
unit = 1e-3

refl_pt = mp.Vector3(-13, 1.27 / 2, 0)
trans_pt = mp.Vector3(13, 1.27 / 2, 0)

waveguide = mp.Block(mp.Vector3(mp.inf, 1.27, mp.inf),
                     center=mp.Vector3(0, 1.27 / 2, 0),
                     material=mp.Medium(epsilon=10.2))

pec_block_1 = mp.Block(mp.Vector3(15, 0.035, mp.inf),
                       center=mp.Vector3(-7.5, -0.035 / 2),
                       material=mp.perfect_electric_conductor)

pec_block_2 = mp.Block(mp.Vector3(15, 0.035, mp.inf),
                       center=mp.Vector3(7.5, -0.035 / 2),
                       material=mp.perfect_electric_conductor)

pec_block_3 = mp.Block(mp.Vector3(15, 0.035, mp.inf),
                       center=mp.Vector3(-7.5, 1.27 + 0.035 / 2),
                       material=mp.perfect_electric_conductor)

pec_block_4 = mp.Block(mp.Vector3(15, 0.035, mp.inf),
                       center=mp.Vector3(7.5, 1.27 + 0.035 / 2),
                       material=mp.perfect_electric_conductor)

sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-7.5, 1.27 / 2),
                     size=mp.Vector3(0, 0, 0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[waveguide, pec_block_1, pec_block_2],
                    sources=sources,
                    resolution=resolution)

refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 6)))
trans = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 6)))

pt = mp.Vector3(0, 10.5, 0)
print("--- Running Device Simulation ---")
sim.run(until_after_sources=mp.stop_when_energy_decayed(dt=100, decay_by=1e-5))

straight_refl_flux = np.array(mp.get_fluxes(refl))
straight_trans_flux = np.array(mp.get_fluxes(trans))
straight_refl_data = sim.get_flux_data(refl)

sim.reset_meep()

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[waveguide, pec_block_1, pec_block_2, pec_block_4],
                    sources=sources,
                    resolution=resolution)

refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, 6)))
trans = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=trans_pt, size=mp.Vector3(0, 6)))
sim.load_minus_flux_data(refl, straight_refl_data)

# Animation setup
animate = mp.Animate2D(fields=mp.Ez, realtime=True, normalize=True)
print("--- Running Device Simulation ---")
sim.run(mp.at_every(1, animate),
        until_after_sources=mp.stop_when_energy_decayed(dt=100, decay_by=1e-5))
animate.to_mp4(10, "TE_animation.mp4")
print("Saved: TE_animation.mp4")

flux_freqs = np.array(mp.get_flux_freqs(refl))

perturb_refl_flux = np.array(mp.get_fluxes(refl))
perturb_trans_flux = np.array(mp.get_fluxes(trans))

F = flux_freqs * c / unit / 1e9
T = perturb_trans_flux / straight_trans_flux
R = -perturb_refl_flux / straight_trans_flux
L = 1 - T - R

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
