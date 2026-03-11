# GA SW Suppression (TM) - Run Guide

This folder contains Genetic Algorithm (GA) simulations for TM surface-wave suppression across multiple optimization lengths.

## Share and Run via GitHub

Use GitHub as the single source of truth for code and generated CSV results.

1. Clone the repo
```bash
git clone <your-repo-url>
cd GA_based_Geometry_Optimization_for_Surface_Wave_Reflection
```

2. Create and activate your Python/Meep environment (WSL + Ubuntu recommended)
- Install WSL + Ubuntu on Windows.
- Follow Meep install instructions: https://meep.readthedocs.io/en/master/Installation/
- Create a Conda environment with `pymeep`.

3. Run from Linux/WSL path
- Keep this repository under a Linux path, for example:
  `/home/<user>/projects/GA_based_Geometry_Optimization_for_Surface_Wave_Reflection`

4. Open in your IDE
- If using PyCharm or VS Code on Windows, point the interpreter to your WSL Conda environment.

## Running Simulations

Each subfolder maps to one optimization length. Run the script inside your assigned length folder:

- `GA_SW_suppression_TM/0_6mm/SW_GA_TM_0_6.py`
- `GA_SW_suppression_TM/0_9mm/SW_GA_TM_0_9.py`
- ...
- `GA_SW_suppression_TM/9_0mm/SW_GA_TM_9_0.py`

Example:
```bash
cd GA_SW_suppression_TM/3_0mm
python SW_GA_TM_3_0.py
```

## Outputs

Each run writes results into that folder's optimization-results directory, including:
- `best_gen_*.csv` (main data to share)
- plot/geometry PNG files (optional to share)

## Collaboration Workflow (Branch + PR)

When contributing results, use a branch and pull request instead of pushing directly to `main`.

1. Sync and create a branch:
```bash
git pull
git checkout -b <your-branch-name>
```

2. Run your assigned simulation(s).

3. Commit the generated CSV outputs:
```bash
git add GA_SW_suppression_TM/**/optimization_results*/best_gen_*.csv
git commit -m "Add TM GA CSV results for <length/range>"
git push -u origin <your-branch-name>
```

4. Open a Pull Request on GitHub from your branch into `main`.

5. After PR approval and merge, everyone updates with:
```bash
git pull
```

## Operational Notes

- If loss does not improve for more than 5 generations, early stop is acceptable.
- If a WSL run is stuck and IDE stop does not work:
```bash
pkill -f python
```
Use carefully if multiple Python jobs are running.
