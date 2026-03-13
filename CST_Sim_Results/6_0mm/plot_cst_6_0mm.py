import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit("matplotlib is required to plot the CST results. Install it with: pip install matplotlib") from exc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(BASE_DIR, "cst_6_0mm_sparams.png")

STRUCTURES = [
    ("MS_SWL_no_block.s2p", "No block", "-"),
    ("MS_SWL_ground_strip.s2p", "Ground strip", "--"),
    ("MS_SWL_GA.s2p", "Genetic Algorithm Optimized", "-."),
    ("MS_SWL_uniplanar.s2p", "Uniplanar", ":"),
]

S11_COLORS = ['#F08080', '#FF6347', '#DC143C', '#8B0000']
S21_COLORS = ['#90EE90', '#32CD32', '#228B22', '#006400']


def load_touchstone_ri(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            rows.append([float(value) for value in parts[:5]])

    if not rows:
        raise ValueError(f"No numeric S-parameter data found in {file_path}")

    data = np.array(rows, dtype=float)
    freq_ghz = data[:, 0]
    s11 = data[:, 1] + 1j * data[:, 2]
    s21 = data[:, 3] + 1j * data[:, 4]
    return freq_ghz, s11, s21


def magnitude_db(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-12))


def plot_sparameters():
    plt.figure(figsize=(6, 4), dpi=300)

    for index, (filename, label, linestyle) in enumerate(STRUCTURES):
        file_path = os.path.join(BASE_DIR, filename)
        freq_ghz, s11, s21 = load_touchstone_ri(file_path)

        plt.plot(
            freq_ghz,
            magnitude_db(s11),
            color=S11_COLORS[index],
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label} S11",
        )
        plt.plot(
            freq_ghz,
            magnitude_db(s21),
            color=S21_COLORS[index],
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label} S21",
        )

    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(15, 30)
    plt.ylim(-40, 0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()


def main():
    plot_sparameters()
    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
