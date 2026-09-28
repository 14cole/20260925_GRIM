#!/usr/bin/env python3
"""Plot RCS versus frequency for a list of .grim files.

Edit the settings block below, then run::

    python simple_frequency_sweep.py

Each (elevation, polarization) pair gets one PNG with one curve per file. Each
curve is the chosen percentile, taken in dB, over every stored azimuth inside
the azimuth region. Set AZ_MIN == AZ_MAX to plot a single azimuth cut instead.
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
OUTPUT_FOLDER = Path(r"C:\data\frequency_sweeps")

AZ_MIN = 0.0            # azimuth region (deg); AZ_MIN > AZ_MAX wraps through 360/-180
AZ_MAX = 30.0
PERCENTILE = 50.0       # 0-100, taken over the azimuth region in dB
ELEVATIONS = [0.0]      # deg; nearest stored elevation is used
POLARIZATIONS = ["VV", "HH"]

X_MIN, X_MAX, X_STEP = 1.0, 5.0, 0.5       # frequency axis (GHz)
Y_MIN, Y_MAX, Y_STEP = -40.0, 20.0, 10.0   # RCS axis (dB)

ANGLE_TOL = 0.5         # deg; skip a file whose nearest elevation is further off
FIG_SIZE = (10, 6)      # inches
DPI = 150
# =============================================================================


def to_deg(grid, axis_name, values):
    values = np.asarray(values, dtype=float)
    return np.rad2deg(values) if grid.units.get(axis_name) == "rad" else values


def to_ghz(grid):
    return grid._frequency_value_to_hz(grid.frequencies) / 1.0e9


def azimuth_mask(az_deg, az_min, az_max):
    if az_min == az_max:
        # Single cut: take the nearest stored azimuth.
        mask = np.zeros(az_deg.shape, dtype=bool)
        mask[np.argmin(np.abs(az_deg - az_min))] = True
        return mask
    if az_min < az_max:
        return (az_deg >= az_min) & (az_deg <= az_max)
    # Region crosses the seam, e.g. (350, 10) or (170, -170).
    return (az_deg >= az_min) | (az_deg <= az_max)


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    grids = [(Path(p).stem, load_dataset(str(p))) for p in GRIM_FILES]

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

                el_deg = to_deg(grid, "elevation", grid.elevations)
                e_idx = int(np.argmin(np.abs(el_deg - elevation)))
                if abs(el_deg[e_idx] - elevation) > ANGLE_TOL:
                    print(f"{name}: no elevation near {elevation} deg, skipped")
                    continue

                az_deg = to_deg(grid, "azimuth", grid.azimuths)
                mask = azimuth_mask(az_deg, AZ_MIN, AZ_MAX)
                if not mask.any():
                    print(f"{name}: no azimuths in [{AZ_MIN}, {AZ_MAX}], skipped")
                    continue

                power = grid.rcs_power[mask, e_idx, :, p_idx]      # (n_az, n_freq)
                db = grid.linear_to_default_db(power, grid.frequencies[None, :])
                curve = np.nanpercentile(db, PERCENTILE, axis=0)
                unit = grid.default_log_unit()
                ax.plot(to_ghz(grid), curve, label=name)

            if AZ_MIN == AZ_MAX:
                az_text = f"Az {AZ_MIN:g} deg"
            else:
                az_text = f"P{PERCENTILE:g} over Az {AZ_MIN:g} to {AZ_MAX:g} deg"
            ax.set_title(f"Frequency sweep, {pol}, El {elevation:g} deg, {az_text}")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(f"RCS ({unit})")
            ax.set_xlim(X_MIN, X_MAX)
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_xticks(np.arange(X_MIN, X_MAX + X_STEP / 2, X_STEP))
            ax.set_yticks(np.arange(Y_MIN, Y_MAX + Y_STEP / 2, Y_STEP))
            ax.grid(True, alpha=0.4)
            if ax.lines:
                ax.legend()
            fig.tight_layout()

            out = OUTPUT_FOLDER / f"freq_sweep_{pol}_el{elevation:g}.png"
            fig.savefig(out, dpi=DPI)
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
