# Solver upgrade experiments, 6 October 2026

Candidate speed-ups that the 6 October audit (`tools/GHOST/SOLVER_WORKFLOW_AUDIT_2026-10-06.md`)
and its first implementation pass (`tools/GHOST/AUDIT_FIXES_2026-10-06.md`) left open, implemented
as overlays over `tools/GHOST/ghost_backend` (monkeypatches applied at interpreter start through
`sitecustomize.py`), so each could be measured against the unmodified project before being ported.
Nothing in this folder edits the project; the accepted experiment was ported by hand afterwards
(`AUDIT_FIXES_2026-10-06.md`, section 2.6) and the overlays stay here as the record of what was tried.

## Layout

- `ghost_experiments/<name>.py`: one experiment each (`DESCRIPTION`, `CASES`, `apply()`).
- `run_experiments.py`: `reference` records the project's own fields and timings;
  `run NAME[,NAME]` runs the named experiments (their default cases, or `--cases`) in fresh
  processes and compares; `compare LABEL` re-prints a comparison; `list` names them.
- `results/<label>/`: recorded fields (`.npz`), metadata (`.json`) and logs of every run;
  `results/reference/` (+ `reference_r1`, `reference_r2`) is the project before the port, `final_r1`,
  `final_r2` and `reference_final` are three runs of the project after it.
- `calibration/`: `far_grading_calibration.py` (finer |k| L rows of the 2-D far-rule tables, the
  project's method) and `verify_far_grading_rows.py` (a denser check of those rows, kept for a future
  port; not needed once the rows proved unreachable, see below).

Accuracy is the peak-relative change of the complex co-polarized amplitudes against the reference
(`max |new - old| / max |old|`); the acceptance rule is 1e-8.  Timings are wall seconds of the solve
call in fresh processes; single runs are +-5-10%, so marginal candidates were repeated three times
and compared on the minimum.

Reference cases (`tools/GHOST/scripts/audit_2026-10-06/bench.py`): airfoil certified 1, 3, 10 GHz
(`air_c1`, `air_c3`, `air_c10`), survey 3 GHz dense and forced compressed (`air_s3`, `air_s3c`),
TYPE 4 coating 10 GHz (`coat_c10`); BoR cylinder survey 2 and 6 GHz, certified 2 GHz (`cyl_s2`,
`cyl_s6`, `cyl_c2`), direct PEC, dielectric and coated spheres (`sph_direct`, `sph_diel`,
`sph_coated`).

## Results

| Experiment | What it changes | Cases | Peak-relative change | Wall ratio new/reference | Verdict |
|---|---|---|---|---|---|
| `bor_sampled_check` + `bor_sampled_meridian` | BoR graded near rules and disjoint meridian pairs: the coarse level evaluated on every 4th point / pair, the chunk accepted at the fine level when every probe passes, else the complete check | cyl_s2, cyl_c2, cyl_s6, sph_direct, sph_diel, sph_coated | 0 (bitwise) | 0.86, 0.86, 0.86, 0.92, 0.82, 0.79; 0 fallbacks in 46 chunks / 2-3 batches per case | **accepted, ported** (`GHOST_BOR_NEAR_CHECK_STRIDE`) |
| `twod_sampled_check` | 2-D polynomial near rule: the 20-point check on every 4th task of a batch | air_c1, air_c3, air_c10, coat_c10 | 7.5e-15 | 1.00, 1.09, 1.05, 1.00; a probe failed in 27/42, 70/93, 30/48, 7/8 batches, each failure re-running the complete check | rejected: slower |
| `dense_condition_probes` | dense condition estimate with four probe columns (`onenormest(t=4)`) | air_c3, air_s3, air_c10 | 0 (diagnostic only) | 0.99, 0.97, 0.99; the estimate itself 0.51 -> 0.50 s | rejected: no gain |
| `tables_contract_matmul` | BoR dense-tables path: the nine left contractions grouped per weight as batched matmuls | sph_diel, sph_coated (+ all six BoR cases, three runs) | 0 (bitwise) | single run 0.99, 0.96; minimum of three over six cases 0.98-1.01 | rejected: within noise |
| `far_tile_glue` | BoR far build: the brackets' 2 pi folded into the transform tables instead of a pass over the outputs | cyl_s6, sph_direct, sph_coated (+ all six, three runs) | <= 1.9e-15 | single run 0.94, 0.97, 0.96; minimum of three 0.98-1.01 | rejected: within noise |
| `far_grading_extension` | 2-D far-rule grading: calibrated rows at \|k\| L = 1.0, 2.0, 2.5 between the project's 0.5, 1.5 and 3.0 rows (`calibration/`) | air_c1, air_c3, air_s3, coat_c10 | 0 (bitwise) | 1.02, 1.06, 1.15, 0.99 (noise); every tile of these meshes has \|k\| L <= 0.5 or is capped by its own order, so no order changed | rejected: unreachable on production meshes |
| `far_ratio_split` | 2-D far tiles: the pairs at centre-distance ratio >= 10 evaluated separately with the far column's lower order | air_c1, air_c3, air_s3, coat_c10 | 7.4e-13 | 1.06, 1.13, 1.12, 1.01 (18, 47, 188, 0 split tiles) | rejected: a second native call per tile costs more than the lower order saves |
| `compressed_tm_in_ram` | compressed 2-D backend: the TM partner operator kept in RAM when both fit in half the storage budget | air_s3c | 0 (bitwise) | 1.06, 1.04 (two runs; the partner was kept resident in both, `kept: 1`) | rejected: no gain (the spool write and read-back are not on the critical path) |
| `compressed_admission_threads` | compressed 2-D backend: admission samples compressed on 8 threads | air_s3c | 0 (bitwise) | 1.00 | rejected: no gain (the tile compression already runs native threads) |
| HODLR threshold (`results/hodlr_lu`) | `GHOST_HIERARCHICAL_MIN_UNKNOWNS` raised so the 10 GHz P3 system (11,700 unknowns) takes dense LU | air_c10 | n/a | LU 27.72 s against HODLR 21.11 s | threshold of 10,000 kept |
| lazy near bands | BoR near bands prepared to the sweep horizon only | - | - | not built: the initial horizon equals the cap for bandwidth >= 12 and a band extension costs a full rebuild | dropped |

### After the port

The project with the BoR sampled checks ported (and the dead-code sweep and driver items of
`AUDIT_FIXES_2026-10-06.md` sections 2.5 and 2.7), three fresh runs against three runs of the
reference, minimum walls (`results/final_r1`, `final_r2`, `reference_final`):

| Case | Before (s, min of runs) | After (s, min of 3) | Change | Peak-relative change |
|---|---:|---:|---:|---:|
| Airfoil certified 1 GHz | 2.32 (1) | 2.25 | -3% | 0.0e+00 |
| Airfoil certified 3 GHz | 4.44 (1) | 4.58 | +3% | 0.0e+00 |
| Airfoil certified 10 GHz | 21.11 (1) | 23.24 | +10% | 0.0e+00 |
| Airfoil survey 3 GHz (dense) | 12.64 (1) | 12.40 | -2% | 2.0e-12 |
| Airfoil survey 3 GHz (compressed) | 11.56 (1) | 11.92 | +3% | 0.0e+00 |
| TYPE 2 IBC certified 1 GHz | 0.33 (1) | 0.32 | -2% | 0.0e+00 |
| TYPE 5 two dielectrics certified 3 GHz | 0.92 (1) | 0.89 | -3% | 0.0e+00 |
| TYPE 1 thin dielectric certified 1 GHz | 0.53 (1) | 0.50 | -5% | 0.0e+00 |
| TYPE 4 coating certified 10 GHz | 1.35 (1) | 1.29 | -4% | 0.0e+00 |
| BoR cylinder survey 2 GHz | 0.96 (3) | 0.79 | -18% | 0.0e+00 |
| BoR cylinder certified 2 GHz | 2.30 (3) | 1.94 | -16% | 0.0e+00 |
| BoR cylinder survey 6 GHz | 2.68 (3) | 2.20 | -18% | 0.0e+00 |
| BoR PEC sphere, direct API | 1.57 (3) | 1.32 | -16% | 0.0e+00 |
| BoR dielectric sphere, direct API | 1.80 (3) | 1.40 | -22% | 0.0e+00 |
| BoR coated sphere, direct API | 2.63 (3) | 2.08 | -21% | 0.0e+00 |

Every BoR case is bitwise identical in all three runs (the fine level is the published value in both
the sampled and the complete check, and no probe failed).  No 2-D path changed: two of the three runs
are bitwise identical to the reference in every 2-D case; the third differs by 2.0e-12 in the dense
3 GHz survey (8,608 unknowns), the run-to-run variation of the threaded LU, which is what the table's
worst-of-three column shows for that case.  The 2-D timing differences are run noise (the 10 GHz case
ran 21.1 s in the reference and 23.2-23.8 s in all three later runs, including the pre-port
`reference_final`).
