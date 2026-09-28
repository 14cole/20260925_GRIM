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

# Second round: speed, threads, scheduling and grading

The second round implements the remaining items recommended after the first:
- T2-12 and T2-10;
- T2-1, with the A-5 check at junctions;
- P-6 with F4;
- TB-1;
- H-1;
- A-4 and A-2.

H-5 and H-6 change how array tasks claim, fail and exit, and are left for a
test on a real SLURM allocation (see "Not done"). Every item was measured
against the committed first round (`520382d`).

## What changes when you rerun (second round)

- **2D certified runs on hp-eligible bodies** take about half the time and
  memory (fix 13):
  - NACA 0012 (8 m, 10 GHz, 361 angles): 26.1-27.1 s to 13.8-13.9 s, peak
    working set 3.9 to 2.0 GiB. The published field moves by 3.3e-6 of its
    peak.
  - 1 m PEC circle (1024-gon, 10 GHz, 72 angles): 11.0 to 3.8 s, 1.9 to 0.6 GiB,
    still within 9.6e-7 of the exact series.

  At impedance junctions the accuracy check now grades two levels deeper. The
  reported change was 4-6x below the true error there; it is now above it, and
  the published result is 2-10x more accurate.
- **2D linear (P1) meshes** are graded at free strip and card ends, branch
  points and corners of 150 degrees or less (fix 18). The error falls 13-69x
  (strip ends 16x) for 8 panels per corner (4 per free end), where uniform
  refinement would need about as many times the panels. Small polygonal bodies
  get noticeably more panels: a 1-wavelength 12-gon has 228 instead of 132.

  Only primitives with automatic counts and at least 8 base panels are graded,
  so grading at most about doubles the shortest of them. Unchanged: explicit
  counts, corrugations, short facets and densely drawn outlines (an airfoil
  meshes exactly as before), drawn curves, hp meshes, dielectric-only
  interfaces and thin layers.
- **BoR conductor and sheet bodies** are graded at rims, corners and tips by
  the same rules (fix 19). A PEC cylinder (ka = 3, L = 2a) gains 16 elements
  (40 to 56), and its error falls from 0.060 to 0.0039 dB. Spheres, other
  smooth bodies and material bodies are unchanged.
- **BoR certification** refines each drawn chain 1.5x instead of doubling
  one-element primitives (fix 16). A sphere drawn with 512 primitives is
  certified on 768 fine elements instead of 1,024, and the fine mesh stays
  mirror-symmetric (ka = 10: 32.0 to 29.1 s, as accurate).
- **2D thin dielectric layers** are meshed for their guided mode (fix 11): 4x
  fewer panels at the same accuracy.
- **Dense 2D LU** factors a fixed symmetric permutation (fix 12). Closed P1
  bodies no longer need a refinement pass, and results change at rounding
  level.
- **BoR driver units** (HPC and local) run BLAS on their CPU reservation
  instead of one thread (fix 14). The new default is `BLAS_THREADS_PER_WORKER
  = None`; an integer keeps the old pin.
- **Compressed products** use at most the solve's CPUs, one BLAS thread per
  product thread (fix 15).
- **BoR array tasks** run their own planned share before taking a peer's
  (fix 17).

## Verification (second round)

**Tests.** `tests/test_audit_fixes_2026_09_27.py` gains 19 tests for these
items (48 in the module). Tests of other features that pinned panel counts
now either isolate those features from edge grading, or use `HP_COARSENING`
instead of a fixed 4:
- bound-wave and explicit-floor sizing on squares;
- the deep-null certification gate on a plate;
- the first round's inductive card;
- two hp eligibility counts.

The BoR certification regression test expects the 1.5x chain count (18 fine
elements for 12, 24 before). The H-1 test fails on the previous driver: pools
of 6, 6 and 6 for planned shares of 2, 2 and 2.

**Against the first round** (`520382d`), the 18 regression cases of the first
round:

| Cases | Change |
| --- | --- |
| 2D PEC and dielectric circles | 4e-14 to 3e-13 of the peak (LU order, fix 12) |
| NACA 0012 (60 points per side) | 2e-15 (not graded: its trailing-edge primitives are short) |
| PEC\|75-20j square | 2.5e-4 (TM), 1.3e-3 (TE): graded corners, fix 18 |
| 200-ohm strip | 5.4e-6 (TM), 6.4e-3 (TE): graded ends, fix 18 |
| BoR PEC, impedance, coated and dielectric spheres | bitwise identical |

Other end-to-end checks:
- hp certified airfoil (fix 13): 3.3e-6.
- BoR unit with widened BLAS (fix 14): 1.3e-15 (dense), 1.3e-15 (compressed,
  no fault with 4 BLAS threads per mode worker).
- Measured accuracy gains of fixes 13, 18 and 19: see their sections.

# Changes by fix (second round)

## 11. Thin layers meshed for their guided mode (`twod/geometry.py`)

A TYPE 1 thin dielectric layer entered the mesh wavelength with the index of
its bulk material, so a 0.3 mm layer of eps = 10-1j got 4x the panels of the
free-space density (about 64x the LU work). Along the surface it carries its
guided mode, whose index is `sqrt(1 + (kappa/k0)^2)` with
`kappa/k0 = k0 d |eps mu - 1| / (2 min(|eps|, |mu|))`
(`_thin_slab_mode_index`). At most about 1.1 within the thin-layer limit. On a
64-gon at 20 panels per free-space wavelength the layer is within 1.6e-6 (TM)
and 1.8e-5 (TE) of a 12x finer mesh, closer than PEC on the same panels.

## 12. Stride-ordered dense LU (`linalg/dense.py`)

A closed P1 body in natural order grows LU pivots by 2.5e3-3e4, which pushed
the backward error of some right-hand sides over the 1e-12 gate: every batch
paid a refinement step (a second solve and residual product). `DenseFactor` now
factors `P A P^T`, with `p_i = i s mod N` and a stride `s` near 0.618 N coprime
to N, in memory and on the disk spool, and solves through the same
permutation. Only the columns above the gate are refined. The condition
estimate matches natural order to 8 digits.

On the audit's 0.5 x 0.15 m ellipse at 12 GHz (256 angles):
- 20 panels per wavelength, TE: no longer needs a refinement step (backward
  error below 1e-13);
- 30 panels per wavelength, TM (2,888 unknowns, 42 columns over the gate in
  natural order): `linear_solve` falls from 0.19 to 0.10 s;
- amplitudes move 1.2e-14 (TM) and 8.6e-12 (TE) of the peak.

The linear-algebra review measured 5.6 to 2.8-3.4 s on a 5,000-unknown
co-polarized solve with 512 angles.

## 13. hp coarsening 6, checked deeper at junctions (`twod/adaptive_geometry.py`, `twod/adaptivity.py`, `twod/geometry.py`)

`HP_COARSENING` is 6 instead of 4. The P2/P3 difference at coarsening 4 was
40-500x below the 2e-3 acceptance limit that the loop already enforces. The
refinement loop is unchanged.

The P3 accuracy check, the check meshes of the refinement loop and the
forecast meshes (`candidate_meshes`) grade impedance junctions
`HP_CHECK_SINGULAR_LEVELS` (two) levels deeper than the candidate they are
compared with (`_2d_hp_singular_levels`, added to `_junction_grading_levels`).
On one mesh, P2 and P3 miss the innermost element at an r^-1/2 junction.

On a 6.4-wavelength square with PEC|75-20j junctions (TM, against
cosine-graded references):

| Case | Shipped: true / reported | Now: true / reported |
| --- | --- | --- |
| One junction | 2.1e-4 / 5.1e-5 (4.1x under) | 2.0e-5 / 1.0e-4 |
| 16 junctions | 5.2e-4 / 9.9e-5 (5.2x under) | 2.3e-4 / 3.9e-4 |

The check costs two elements per junction side. On the 16-junction square
its P3 mesh has 288 elements against 256 shipped, while the P2 candidate
has 224. The plain square was already overestimated (1.1e-5 against 4.3e-4)
and is unchanged. Open strip ends were within 1.2x at coarsening 4 and 6.
Coarsening 6 alone left the ratio at 5.7x.

## 14. BoR driver units widen BLAS to their CPUs (`run_hpc_bor_monostatic.py`, `run_local_bor.py`, `hpc/scheduler.py`, `runs/config.py`, `hpc/bundle.py`)

Unit processes still start on one BLAS thread, so OpenBLAS does not reserve
buffers for every core at import. `hpc_scheduler.unit_blas_scope` then raises
the limit to the unit's `blas_core_budget` (its CPU reservation, at most the
physical cores) for the solve. The solvers only lower it from there:
- `bor.solver._bounded_blas_threads` splits it among mode workers;
- `single_thread_blas` holds thread pools at one thread.

`BLAS_THREADS_PER_WORKER = None` is the new default in both drivers, the
configuration files and portable bundles; an integer keeps the pin for the
whole run.

Measured on a unit of 8 CPUs with 2 mode workers (sphere, 1,024 elements,
10 GHz):
- factorization: 4.8 to 3.0 s;
- right-hand-side compression: 10.2 to 4.3 s;
- wall: 98.0 to 94.5 s (operator construction, already on 8 CPUs, dominates);
- results: agree to 1.3e-15.

A unit with more mode workers than CPUs keeps one thread per worker, as
before.

## 15. Compressed products follow the allocation (`compressed/operator.py`)

Multi-column products and the equilibration scan ran on a fixed four threads
(`MATMUL_WORKERS = min(4, os.cpu_count())`, Codex F4), each widening BLAS. They
now use `product_threads()`: at most four, and at most the solve's BLAS core
budget, read at call time. Each thread runs on one BLAS thread, as far tiles
and near threads do since the 24 September fix 12. A one-CPU unit runs the
products serially. Inside BoR mode workers, whose BLAS fix 14 widens, the
concurrent multithreaded OpenBLAS products that faulted in fix 12 do not
occur. The products sum in the same order, so results do not depend on the
thread count.

## 16. BoR certification refines chains 1.5x (`bor/dispatch.py`, `twod/geometry.py`)

The fine mesh gave each primitive `max(b + 1, ceil(1.5 b))` elements, so a
generatrix drawn with one element per primitive was doubled (about 8x the LU
work of the base instead of 3.4x). `_chain_mesh_counts` now applies the 2D
chain water-fill (`_certification_fine_counts`). Each chain gets
`max(B + 1, ceil(1.5 B))` elements, added where elements are longest.

Equal elements are refined in mirror pairs (`mirror_pairs=True`,
`_mirror_paired_spread`). The 2D spread would have refined every other
primitive of a sphere and broken the mirror symmetry that BoR's mirror-split
factors need. The counts are computed once per chain and wavelength and shared
by mesher, counts and previews. A 512-primitive sphere: 1,024 to 768 fine
elements, mirror-symmetric (also drawn as 511 primitives, or as two mirror
chains).

The audit's certified PEC sphere (ka = 10, 181 aspects) went from 32.0 to
29.1 s, within 0.0001 dB of the Mie series both times. The stages that follow
the fine mesh fell 45-60 % (stage-seconds summed over workers):

| Stage | Before | After |
| --- | --- | --- |
| RHS compression | 25.1 | 11.3 |
| Modal assembly | 16.0 | 6.1 |
| Factorization | 9.1 | 5.0 |
| RHS solve | 6.8 | 2.6 |

Operator construction (24.6 stage-seconds) did not change and now dominates.
The audit's estimate of 24.4 s, taken from survey timings, was optimistic
for this body. The saving grows where the LU dominates.

## 17. BoR array tasks run their own share first (`run_hpc_bor_monostatic.py`)

The worker sized its pool from every candidate unit and dispatched them in one
pass, so the first task to start filled its pool from the head of the list and
ran past its planned share into its peers'. As in the 2-D worker, the pool is
now sized from the task's own share, the per-unit CPU shares follow it, and a
peer's units are taken only after its own have finished. Three concurrent
tasks on six units now size their pools 2, 2, 2 instead of 6, 6, 6 (a new
end-to-end test runs the configured worker entry point three times).

## 18. 2D linear meshes graded at edges and corners (`twod/geometry.py`)

`_add_edge_vertices` adds three kinds of vertex of conductor and sheet
contours (TYPE 1, 2 and 4) to the vertices `_grade_toward_junctions` already
grades:
- free ends;
- branch points;
- corners that turn by `EDGE_GRADING_TURN_DEG` (30 degrees) or more.

Levels follow the density as at impedance junctions. The following are left
alone:
- vertices of thin layers, which keep their qualified discretization;
- dielectric-only interfaces;
- hp meshes, which cluster their elements toward these vertices themselves;
- explicit counts (N > 0), which keep the requested discretization;
- primitives of fewer than `EDGE_GRADING_MIN_PANELS` (8) base panels.

The last rule bounds the cost on corrugations, staircases and densely drawn
outlines. A corrugated body drawn with one panel per primitive would have
grown tenfold, and an airfoil drawn with 60-400 points per side meshes
exactly as before. The base mesh decides, so both meshes of a certification
pair grade the same vertices. A primitive too short to hold one level (the
2 nm notch of an existing test) is no longer split at its midpoint below the
node tolerance.

Four levels at 20 panels per wavelength, 1 GHz, error over peak amplitude
against cosine-graded references:

| Vertex | TM | TE | Extra panels |
| --- | --- | --- | --- |
| PEC strip end | 2.4e-3 to 1.5e-4 | (open TE contours unsupported) | 4 per end |
| 20 degree wedge tip | 2.0e-3 to 1.2e-4 | 6.7e-3 to 4.3e-4 | 8 per corner |
| Triangle (60 degrees) | 1.2e-3 to 5.2e-5 | 5.1e-3 to 2.6e-4 | 8 per corner |
| Square (90) | 6.4e-4 to 1.8e-5 | 2.6e-3 to 1.8e-4 | 8 per corner |
| Octagon (135) | 2.9e-4 to 4.1e-6 | 1.3e-3 to 8.7e-5 | 8 per corner |
| 12-gon (150) | 9.0e-5 to 4.0e-6 | 1.3e-3 to 9.5e-5 | 8 per corner |

## 19. BoR rims, corners and tips graded (`bor/dispatch.py`)

`_mark_edge_vertices` marks the vertices of a stitched conductor or sheet
generatrix that turn by `BOR_EDGE_GRADING_TURN_DEG` (30 degrees) or more, axial
tips (an end that meets the axis at least 15 degrees off a right angle) and
free sheet edges. It runs in `_prepare_bor_groups`, which previews and solves
share. `_primitive_mesh_plan` grades those vertices with the junction levels
and ratio, so mesher, counts and previews agree.

As in 2D, explicit counts and primitives of fewer than
`BOR_EDGE_GRADING_MIN_ELEMENTS` (8) base elements are not graded
(`_chain_mesh_counts` decides on the base mesh). An existing test's corrugated
body (201 one-element primitives) meshes as before.

PEC cylinder (ka = 3, L = 2a, CFIE) on the production path against the audit's
graded lambda/120 reference: 40 to 56 elements, 0.060 to 0.0039 dB.
Multi-surface material layouts share chains between runs and are not graded
yet.

## Not done (second round)

- **H-5** (array tasks exit only when every output exists) and **H-6**'s
  per-unit failure records change how tasks claim, retry and exit. A task
  that stops waiting must know that a peer's unit will still be finished, and
  a unit that fails for good must stop being waited for. Both need a test on a
  real SLURM allocation (requeue, preemption, a shared filesystem).
- **A-2** on dielectric, coated and multi-region BoR bodies: their runs share
  chains, which must be graded identically in every run.
- **A-5 on the P1 path** (reporting the fine error as `1/(1.5^p - 1)` of the
  change) is not done; grading (fixes 18 and 19) brings most singular features
  back to second order instead.

## Test suite results

`python ghost_backend/tests/run_suite.py` on Windows (Python 3.11, NumPy 2.4.2,
SciPy 1.16.3):

| Tree | Headless | Isolated (Qt-named) modules | Total |
| --- | --- | --- | --- |
| Previous code (`321ed6b`) | 1,087 passed, 1 skipped | 120 passed | 1,207 |
| With these fixes | 1,116 passed, 1 skipped | 120 passed | 1,236 |
| With the second round | 1,135 passed, 1 skipped | 120 passed (16 modules) | 1,255 |

`test_package_layout.py` requires `tools/GHOST` to hold only `ghost_backend`
and `scripts` as directories. It fails in a checkout that also holds the
untracked `audit_2026-09-27` folder, and passes without it (the run above).
