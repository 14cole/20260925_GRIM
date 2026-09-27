# GHOST audit fixes, 27 September 2026

This release implements the correctness and failure fixes of the 27 September
audit of the 2D and BoR solvers (`audit_2026-09-27/AUDIT_REPORT.md`), merged
with the verified items of the separate Codex audit
(`audit_2026-09-27/REVIEW_OF_CODEX_AUDIT.md`, section 6, items 1-7). Every
defect was reproduced before it was changed, and every change was checked
against the previous code (revision `321ed6b`). The audit report, the review
and the scripts named below live in a local `audit_2026-09-27/` folder that is
not part of the repository, like the earlier audits' experiment folders.

## What changes when you rerun

- **BoR partial coatings, banded coatings and coating patches** give
  different, more accurate results (fix 1). Coatings that must be invisible now
  track the plain body: a vacuum coating over half a PEC sphere is 0.158 dB off
  at 6 elements per hemisphere (1.426 dB before; the uncoated mesh gives
  0.141 dB), and a vacuum patch on a coated sphere 0.98 dB (1.82 dB before;
  0.92 dB without the patch). Other BoR results are unchanged.
- **2D reactive TYPE 1 cards** get finer meshes where they guide a bound surface
  wave (fix 3). A closed inductive card of 2+600j / 2+1000j ohm now has
  0.14 % / 0.19 % field error on its automatic mesh (6.4 % / 45 % before), and
  its certified run passes instead of failing. Capacitive cards (TM waves) also
  get up to 4x the panels on those segments, which lowers their error from
  1-3e-3 to 1e-5-3e-5. Resistive cards are unchanged.
- **2D geometry that used to fail** now solves:
  - a short collinear primitive on an inclined straight side (fix 2);
  - a thin layer on a finely split polygon (fix 4).

  The 2D results of all other regression cases are bitwise identical.
- **Automatic 2D backend** (fix 5). Every backend that fits the RAM budget
  stays a retry, and a compressed solve whose storage cannot fit its cap is
  refused before assembly. The NACA 0012 (16 m, 10 GHz, certified, 9 GiB
  budget) completes in 120 s at 5.7 GiB peak; it used to fail after 196 s. On
  Windows, available memory now also counts the commit headroom.
- **Memory limits on HPC nodes and in containers** (fix 6):
  - the SLURM per-CPU allocation counts every CPU the job has on the node;
  - nested cgroup limits are found, and reclaimable page cache is not counted
    as used;
  - a zero reading is no longer overridden by an explicit budget.

  Plans can therefore admit more (or, under a tight nested cgroup, less) than
  before.
- **HPC provenance** (fix 7). The runtime fingerprint no longer includes the
  kernel release, the CPU model or the SIMD extensions found at run time, and
  the source fingerprint no longer includes the `run_*.py` drivers or `ui/`.
  Fingerprints of existing runs differ, so start new runs with this code, as
  any change of the backend source already requires.
- **Linux BoR spill** (fix 8) no longer issues a synchronous `msync` per
  released range (165 s against 0.7 s for a 1 GB store).
- **BoR planning** (fixes 9-10):
  - previews of conductor solves price the mirror-half and hierarchical mode
    factors the solve uses, so symmetric bodies are admitted with more mode
    workers (10 GHz ogive, certified, 16 GB: 15 instead of 3);
  - multi-surface planners price compressed self streams (surfaces of 1,000 or
    more nodes) as the all-mode stores they are.

## Verification

**Test suite.** `python ghost_backend/tests/run_suite.py`: see the results
recorded at the end of this file. New tests:
`tests/test_audit_fixes_2026_09_27.py` (29 tests). 26 of them fail on the
previous code; the other three guard behaviour that must not change. Two
existing tests now isolate themselves from the host's real cgroups (fix 6):
`test_memory_safety.py`, RSS fallback, and `test_audit_fixes_drivers.py`,
SLURM over psutil.

**Against the previous code.** 18 regression cases give bitwise identical
complex amplitudes:
- 2D, TM and TE: PEC circle, NACA 0012, lossy dielectric circle, PEC|75-20j
  square with junction grading, resistive strip;
- BoR: PEC, 75-20j IBC, coated and dielectric spheres.

**Per fix**, reproduced before and after:
- `audit_2026-09-27/scripts/verify/verify_junction_fix.py` (fix 1);
- `verify_collinear_sliver.py` (fix 2);
- `verify_sheet_bound_wave.py` (fix 3);
- the review scripts `e10_thin_layer_curvature.py` (fix 4) and
  `e10_coated_plan_end_to_end.py` (fix 10);
- `bench2d.py` with `ram_budget_gib = 9` (fix 5).

# Changes by fix

## 1. BoR junctions tie the through currents only (`bor/solver.py`)

At a coating termination the junction basis ties the t currents of the
surfaces that meet: that carries the charge across the junction circle. The phi
currents flow around the circle, so each surface now keeps its own end
half-basis instead of `J_2phi = -J_dphi` and `J_1phi = +J_dphi`. Those ties
forced both conductor pieces' phi current toward zero at every coating edge.
The multi-region solver does the same for J and M at junctions of three or more
surfaces (coating patches); a surface merely split into two pieces keeps its
phi tie. `M_t = 0` at conductor junctions is unchanged. Each junction gains one
or two unknowns. No conditioning change was seen: the formulation review ran
PEC, dielectric, magnetic and impedance variants, with a 10-25 % lower maximum
error on real coatings.

## 2. Collinear near pairs (`twod/operators.py`)

The separated-near adaptive rule judged the K'/D error of a pair against the
pair's own K'/D norm, with an absolute floor of `eps * max(1, L*L)`. Two
collinear panels have an exactly zero K'/D block; on an inclined line it
evaluates to rounding noise, which never converges. The rule then raised
"Separated-near Galerkin quadrature did not converge ... Refine the boundary
mesh" (refining cannot help), or searched for minutes first. The K'/D error is
now judged against the larger of its own norm and the S norm over the longer
panel length. S is evaluated whenever K'/D is, and returned only on request.
A 1 m square with one side drawn 0.4 m | 2 mm | 0.6 m and rotated 30 degrees
certifies in 0.4 s, and matches the unrotated body to 1e-14.

## 3. Bound waves on TYPE 1 cards (`twod/geometry.py`)

`_segment_surface_wave_index` and `_surface_wave_mesh_wavelength` sized only
opaque TYPE 2/4 conductors. A free card guides the bound wave of an opaque
surface of twice its impedance (`SURFACE_WAVE_SHEET_IMPEDANCE_FACTOR`): the
field that is even about the card sees 2Z on each side. Cards now use the same
sizing, the same 4x density cap and the same unresolved limit. Thin dielectric
layers keep their own rule. A 2+2000j card (index 10.7) stays under-resolved
at the cap (4.8 %), as an opaque surface of the same index does, and its
certification refuses it.

## 4. Thin-layer curvature of the drawing (`twod/formulations/thin_layer.py`)

The guard (`thickness/curvature radius <= 0.05`) divided the turn at a node by
the adjacent panel lengths, so splitting a drawn primitive into more panels
raised the curvature without limit. A 64-gon was refused with 5 panels per
primitive although 4 passed, and its certified run failed on the fine mesh.
`_drawing_curvature` measures the turn over the straight runs of panels on
either side (the drawn primitives). At a genuinely sharp corner this compares
`d * turn` with the drawn side lengths, as the old code did when every side was
one panel. A layer on a tight curve is still refused.

## 5. Automatic 2D backend admission (`execution/policy.py`, `compressed/memory.py`, `twod/solver.py`, `twod/formulations/thin_layer.py`, `execution/options.py`)

- `rank_candidates` keeps every backend that fits the budget in the order:
  backends inside the 20 % margin first, then the rest as retries. Before, a
  compressed build that failed late ended the run although dense fit.
- The compressed forecast counts the TM partner that a co-polarized TE solve
  reserves inside the same storage cap (`resources['compressed_partner']`),
  and reports `storage_required_bytes` and `storage_fits`. The partner is
  spooled, so it is not priced as resident RAM.
- `_compressed_storage_gate` refuses a sampled compressed solve whose storage
  cannot fit, before assembly, at the monostatic, bistatic, diagnostics and
  thin-layer gates; the automatic retry then tries the next backend. In the
  P-1 case the P3 compressed build (previously 30 s of assembly before
  failing) is refused at once and dense P3 certifies.
- On Windows the available memory is also bounded by the commit headroom
  (`ullAvailPageFile`). On the audit workstation, free commit ran 5 GiB below
  free physical memory, and a 31.6 MiB allocation failed while another run
  held ~10 GiB.
- The request metadata keeps a coherent `retry_order` after a retry or an
  adaptive reselection.

## 6. Memory detection (`twod/solver.py`, `hpc/scheduler.py`, new `execution/cgroup.py`)

- `_slurm_available_bytes`: with `SLURM_MEM_PER_CPU` and no
  `SLURM_CPUS_PER_TASK` (as under `--exclusive`), the capacity is the per-CPU
  memory times `SLURM_CPUS_ON_NODE`, shared by the job's local processes like
  `SLURM_MEM_PER_NODE`. One CPU's share gave a 3.35 GiB solve limit on a
  414 GiB node. With neither variable set the one-CPU default is kept.
- The solver and the scheduler find the process's own memory cgroup and its
  ancestors from `/proc/self/cgroup` (`execution/cgroup.py`). The mount root,
  which a SLURM job or systemd unit does not use, is now only the fallback
  (for a cgroup namespace).
  - The solver's headroom is the tightest over those groups, with reclaimable
    page cache (`inactive_file`) not counted as used.
  - The scheduler's `detect_memory_gb` takes the tighter of the SLURM
    declaration and the cgroup limit.
- A zero availability reading means an exhausted allocation when a probe
  answered it (`_available_memory_bounds`); an explicit budget then no longer
  overrides it. It still does when no probe answers.

## 7. Provenance fingerprints (`execution/provenance.py`)

- The runtime fingerprint keeps build and version facts and drops those of the
  host: `platform.release()`, `platform.processor()`, and NumPy's
  run-time-detected SIMD lists. It is taken on the login node and compared on
  every compute node, so a login node that differed from the compute nodes
  made every array task refuse to run.
- The source fingerprint no longer covers the root-level `run_*.py` drivers
  (a running driver is covered by its configured copy,
  `driver_configured.py`) or `ui/`. Editing the next run's CONFIG block no
  longer aborts every running unit of the previous run.

## 8. Linux spill release (`bor/streaming.py`)

`_release_spilled` no longer calls `mmap.flush` (a synchronous `msync`) before
`MADV_DONTNEED` on POSIX. The spill arrays are shared file mappings, whose
dirty pages stay in the page cache for the kernel to write back. Windows keeps
its asynchronous `FlushViewOfFile` before `VirtualUnlock`.

## 9. BoR previews price the solve's mode factors (`bor/dispatch.py`, `bor/solver.py`)

`estimate_bor_resources` and the direct-API plan now pass `hierarchical=True`
and `mirrored=...` to `plan_bor_mode_workers` for conductor and sheet solves,
as `solve_bor` does. The mirror test (`generatrix_mirror_symmetric`) is now
shared by `mirror_map` and the previews, and runs on the solve's own mesh and
impedances (`_conductor_mesh`). Material solvers do not use these factors and
are priced as before.

## 10. Compressed self streams in multi-surface planners (`bor/streaming.py`, `bor/dispatch.py`, `bor/solver.py`)

Self streams are marked (`self_stream_spec`). One on a surface of at least
`FAR_COMPRESSION_MIN_NODES` nodes is a `CompressedFarBlocks` store that holds
every mode and never spills, so it is:
- priced in the plan's peak by `estimate_compressed_far_gb`;
- added back after a spill decision (`compressed_self_store_gb`);
- left out of the streamed range, the budget of streamed blocks, the worker
  alignment and the per-mode spill cost.

The coated and junction planners and the dispatch previews use the same
helpers. With compression forced on small surfaces and a 12 MB stream budget,
the coated sphere of the review now plans 8-mode ranges with 2 cross builds
and 4 of 4 mode workers (10.9 s). The previous code planned 1-mode ranges with
11 cross builds and 1 worker (17.6 s on the same machine), and the results
agree to 8e-15.

## Not done

- The compressed-far storage estimate is still a model: conservative on
  ordinary bodies (2.0-2.4x the measured stores) but not an upper bound on
  crowded generatrices. Pricing from the actual block partition (Codex F2)
  needs the solver's near lists in every estimate and is left for a later
  change.
- Snapshotting the backend into the run directory at submit (the alternative
  fix of H-3) was not done; the drivers are excluded from the fingerprint
  instead.
- The scheduler's workstation budget (`detect_available_memory_gb`) does not
  count the Windows commit headroom; the solver's own admission does.

## Test suite results

`python ghost_backend/tests/run_suite.py` on Windows (Python 3.11, NumPy 2.4.2,
SciPy 1.16.3):

| Tree | Headless | Qt modules (15, isolated) | Total |
| --- | --- | --- | --- |
| Previous code (`321ed6b`) | 1,087 passed, 1 skipped | 120 passed | 1,207 |
| With these fixes | 1,116 passed, 1 skipped | 120 passed | 1,236 |

`test_package_layout.py` requires `tools/GHOST` to hold only `ghost_backend`
and `scripts` as directories. It fails in a checkout that also holds the
untracked `audit_2026-09-27` folder, and passes without it (the run above).
