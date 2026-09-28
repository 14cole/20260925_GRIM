#!/usr/bin/env python3
"""Plot RCS versus azimuth for a list of .grim files.

Edit the settings block below, then run::

    python simple_azimuth_sweep.py

Each (frequency, elevation, polarization) combination gets one PNG with one
curve per file. The nearest stored frequency and elevation are used.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from GRIM_Backend.io.loaders import load_dataset


# =============================================================================
# EDIT THESE SETTINGS
# =============================================================================
GRIM_FILES = [
    r"C:\data\case_a.grim",
    r"C:\data\case_b.grim",
]
OUTPUT_FOLDER = Path(r"C:\data\azimuth_sweeps")

FREQUENCIES = [1.0, 3.0]    # GHz; nearest stored frequency is used
ELEVATIONS = [0.0]          # deg; nearest stored elevation is used
POLARIZATIONS = ["VV", "HH"]

X_MIN, X_MAX, X_STEP = 0.0, 180.0, 30.0    # azimuth axis (deg)
Y_MIN, Y_MAX, Y_STEP = -40.0, 20.0, 10.0   # RCS axis (dB)

FREQ_TOL = 0.01             # GHz; skip a file whose nearest frequency is further off
ANGLE_TOL = 0.5             # deg; skip a file whose nearest elevation is further off
FIG_SIZE = (10, 6)          # inches
DPI = 150
# =============================================================================


def to_deg(grid, axis_name, values):
    values = np.asarray(values, dtype=float)
    return np.rad2deg(values) if grid.units.get(axis_name) == "rad" else values


def to_ghz(grid):
    return grid._frequency_value_to_hz(grid.frequencies) / 1.0e9


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    grids = [(Path(p).stem, load_dataset(str(p))) for p in GRIM_FILES]

    for freq in FREQUENCIES:
        for elevation in ELEVATIONS:
            for pol in POLARIZATIONS:
                fig, ax = plt.subplots(figsize=FIG_SIZE)
                unit = "dB"
                for name, grid in grids:
                    pols = list(grid.polarizations)
                    if pol.upper() not in pols:
                        print(f"{name}: no {pol} polarization, skipped")
                        continue
                    p_idx = pols.index(pol.upper())

                    f_ghz = to_ghz(grid)
                    f_idx = int(np.argmin(np.abs(f_ghz - freq)))
                    if abs(f_ghz[f_idx] - freq) > FREQ_TOL:
                        print(f"{name}: no frequency near {freq} GHz, skipped")
                        continue

                    el_deg = to_deg(grid, "elevation", grid.elevations)
                    e_idx = int(np.argmin(np.abs(el_deg - elevation)))
                    if abs(el_deg[e_idx] - elevation) > ANGLE_TOL:
                        print(f"{name}: no elevation near {elevation} deg, skipped")
                        continue

                    power = grid.rcs_power[:, e_idx, f_idx, p_idx]
                    db = grid.linear_to_default_db(power, grid.frequencies[f_idx])
                    unit = grid.default_log_unit()
                    ax.plot(to_deg(grid, "azimuth", grid.azimuths), db, label=name)

                ax.set_title(f"Azimuth sweep, {pol}, {freq:g} GHz, El {elevation:g} deg")
                ax.set_xlabel("Azimuth (deg)")
                ax.set_ylabel(f"RCS ({unit})")
                ax.set_xlim(X_MIN, X_MAX)
                ax.set_ylim(Y_MIN, Y_MAX)
                ax.set_xticks(np.arange(X_MIN, X_MAX + X_STEP / 2, X_STEP))
                ax.set_yticks(np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP))
                ax.grid(True, alpha=0.4)
                if ax.lines:
                    ax.legend()
                fig.tight_layout()

                out = OUTPUT_FOLDER / f"az_sweep_{pol}_{freq:g}GHz_el{elevation:g}.png"
                fig.savefig(out, dpi=DPI)
                plt.close(fig)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
