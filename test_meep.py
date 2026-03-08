import meep as mp

cell = mp.Vector3(4, 4, 0)
sim = mp.Simulation(cell_size=cell, resolution=10)
print("Simulation object created OK")
