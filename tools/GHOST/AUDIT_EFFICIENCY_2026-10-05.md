# GHOST 2-D and BoR solver efficiency audit, 5 October 2026

Scope: time to completion and RAM of the 2-D (`ghost_backend/twod`) and body-of-revolution
(`ghost_backend/bor`) solvers as checked out at commit `042e6e3`, measured on a 4-core Linux
container (15 GB, Python 3.11, NumPy 2.4.6, SciPy 1.17.1, OpenBLAS 4 threads) with the native
kernels built for Linux (`bor_stream_kernel`, `libghost_far`, `libghost_table`). The three
earlier audits (22 and 24 September, 25 September architecture changes) are taken as given;
this audit measures what remains. Every change tested here is on the branch
`audit/solver-efficiency-20261005`; nothing was merged to `main`.

All numbers below were measured with nothing else running on the machine. Two early
measurements in this session were made while a background job ran and were discarded
(they inflated the 2-D QR and LU stages by 10-30x); run-to-run noise on this host is about
3-5 %.

## 1. Where the time goes

### BoR (`solve_bor`, PEC spheres, CFIE, 37 aspects)

| Case | Wall | Operators (near prep + far) | Modal assembly | LU + solves | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| ka = 10, 100 elements, tables, 1 worker | 16.9 s | 14.6 s | 0.8 s | 0.9 s | 447 MB |
| ka = 10, 100 elements, tables, 4 workers | 5.8 s | 4.9 s | 0.8 s | 0.2 s | 414 MB |
| ka = 30, 300 elements, streamed far blocks, 4 workers | 28.0 s | 26.3 s | 0.8 s | 1.5 s | 1,450 MB |
| ka = 6, 80 elements, lossy dielectric PMCHWT, 4 workers | 6.8 s | 6.0 s | 0.9 s | 0.2 s | 368 MB |

The operators stage is 85-93 % of every BoR solve measured, and inside it the near-pair
preparation (`_prepare_near_contractions`) dominates, not the far build:

* ka = 10, one worker: near preparation 12.1-13.3 s of 16.6 s. The native near-rule kernels
  (`near_green_rule`, `near_brackets_rule`) take 8.9-9.9 s in 1,100-1,150 calls over
  855,000 point pairs; the Python around them (`np.unique(axis=0)` grouping 0.5 s,
  `_near_check` 0.7 s, `_contract_near_chunk` 0.8 s, layouts and chunking 0.3 s) is 27 %.
  The far tables take 0.65-1.5 s and the whole mode sweep (assembly, LU, solves) 1.4 s.
* ka = 30, four workers, streamed: the far build is about 6 s of wall time on four tile
  threads (17 s of thread time in `banded_modal_kernels`: 9.2 s native azimuthal sampling,
  3.95 s projection GEMM/DCT, 3.5 s Python glue; plus 3.6 s `_efie_band` and 1.4 s
  `_bracket_band` contractions). The remaining 20 s of the 26.3 s operators stage is near
  preparation in four worker processes (4,176 pairs x 45 modes).

Why the near rule costs what it costs (micro-benchmark of the shipped kernel, one thread):

| Quantity | Green's function rule | Bracket (MFIE/IBC) rule |
| --- | --- | --- |
| Throughput | 11.9 M samples/s | 7.1 M samples/s |
| Fixed cost per sample (count = 1 order) | 62 ns | 82 ns |
| Cost per sample and order (recurrence) | 1.2 ns | 2.7 ns |
| Share of the recurrence at 20 orders | 27 % | 40 % |
| `-O3 -march=native` rebuild | no gain | 7 % slower |
| 4 OpenMP threads | 3.6x | 3.8x |

So 60-73 % of the kernel is the per-node transcendental work (`sin`, `cos`, `sqrt`,
`sincos` of k R), not the moment recurrence, and SIMD flags do not help because that work is
scalar libm calls. Every point also pays the coarse and the fine level of the graded rule
(fine orders are 1.5x the coarse ones), so the certified check is 40 % of the kernel work;
the 24 September audit kept it deliberately and this audit does not change it.

### 2-D (closed PEC rectangle 0.6 x 0.25 m, 10 GHz, 181 angles, 4 assembly threads)

| Case | Wall | Operators | LU | Solves + RHS compression | Far field | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| plain, P1, 2,270 unknowns per polarization | 7.7 s | 3.7 s | 0.33 s | 1.5 s | 0.34 s | 435 MB |
| certified (automatic hp: P2 1,140 + P3 1,710 unknowns) | 4.7 s | 1.8 s | 0.23 s | 0.8 s | 0.03 s | 375 MB |
| dielectric rectangle, 6 GHz, 2 x 2,362 unknowns | 15.2 s | 7.5 s | 2.6 s | 2.4 s | 0.36 s | 949 MB |

The 2-D time is spread out; no single stage dominates. Clean profile of the plain case (8.8 s
under cProfile):

| Part | Seconds | Notes |
| --- | --- | --- |
| Native far-block quadrature (`ghost_far_block`) | 8.9 thread-s, 2.45 s wall | 136 tiles on 4 threads; about 37-66 ns per kernel-table evaluation; compute bound |
| Serial tile commit (`ghost_scatter_tile`) | 1.4 s (overlapped) | 82 M updates at 15.7 ns each in the solve against 3.2 ns in isolation: the committing main thread competes with the four tile threads |
| Near pairs (`_near_pair_blocks`) | 1.55 s | NumPy batches; 0.31 s of it rebuilds element arrays from Python attribute lists |
| System preparation (`_prepare_system`, two polarizations) | 0.92 s | 0.53 s in 2-D `np.add.at` of the mass/jump terms |
| Backend forecast (`select_backend`) | 0.83 s | per solve; 0.36 s in `geometric_near_pair_count`, a Python loop of KD-tree queries |
| LU (2 x 2,270) | 0.34 s | 134 GFLOP/s on 4 threads |
| Residual products, LU solves, pivoted QR | 0.8 s | the QR of 2,270 x 181 takes 0.02-0.1 s; RHS compression falls back on this case |

Three theories were tested and rejected by measurement: (a) that the serial tile commit
bounds the tile phase (it overlaps the compute; 3 and 2 assembly threads were slower, 7.5 and
7.9 s); (b) that strided accumulator traffic slows the native far block (a bitwise-identical
restructure with pair-local accumulators changed nothing: 2.68 s against 2.66 s on 24 captured
tiles); (c) that pivoted QR or LU run with a pathological BLAS pool (they do not; the earlier
slow readings came from a concurrent job).

## 2. Findings ranked by expected payoff

### BoR

1. **Near-rule kernel recomputes the trigonometry of the shared angular nodes for every
   pair.** In `near_green_rule` and `near_brackets_rule` the shared nodes (geometric panels and
   tail) have the same `xi` for every pair of a call, yet `cos`, `sin` and `sin(xi/2)` were
   evaluated per pair; `cos(kR)` and `sin(kR)` were two calls where glibc's `sincos` gives both
   from one reduction. Fixed on the branch (bitwise identical over 3,074 captured calls, real
   and lossy k): kernel 1.27x faster. See section 3.
2. **A serial near preparation runs the native kernels on one thread.** When the planner
   admits one near worker (explicit `workers=1`, memory-limited plans, tests and API users) the
   OpenMP team inside the kernels was fixed at one thread although the whole CPU allocation is
   idle. Fixed on the branch: the serial path hands the kernels `blas_core_budget()` threads
   (pairs are independent, values identical). ka = 10, one worker: 16.9 s to 8.9 s.
3. **Python glue in the near rule.** `np.unique(axis=0)` to group points by rule layout cost
   0.5 s per solve at ka = 10; a stable lexsort with break detection gives the same partition.
   Fixed on the branch. The remaining glue (`_near_check` 0.7 s, `_contract_near_chunk` 0.8 s
   for 3,152 tiny matrix products) is left; batching the chunk contractions across chunks of
   equal shape would remove most of it.
4. **The fixed 1 GB far-tile budget drives peak RAM of mid-size streamed solves.**
   `_plan_banded_tiles` sizes every concurrent tile to fill `BOR_STREAM_TILE_BUDGET_GB / threads`,
   so a 300-element body whose retained far blocks are 0.4 GB still allocates about 1 GB of
   tile scratch, and the planner prices that gigabyte in every admission decision (fewer mode
   workers on small nodes). Measured on the ka = 30 streamed sphere (separate processes):

   | Tile budget | Wall | Peak RSS |
   | --- | --- | --- |
   | 1.0 GB (current) | 27.7 s | 1,432 MB |
   | 0.25 GB | 28.3 s (+2.5 %) | 764 MB (-47 %) |
   | 0.125 GB | 32.1 s (+16 %) | 739 MB |

   Recommendation: 0.25 GB, or a tile cap relative to the store (a tile of 2-4 MB of samples
   already saturates the GEMMs). Not changed on the branch because the constant is priced
   throughout the planning model and its documentation; see section 4.
5. **The far build's Python glue is 20 % of the far build** (`banded_modal_kernels`: fancy
   indexed writes of each group's projected rows into the [pairs, modes] outputs, per-chunk
   index gathers, `_parity_outputs`): 3.5 s of 17 thread-s at ka = 30, 0.6 s of 1.0 s for the
   ka = 10 tables. Writing each group's rows into a contiguous buffer in group order and
   un-permuting once would remove most of it. Not changed.
6. **Process workers pay off even for small near preparations**: ka = 10 processes 5.0 s
   against threads 6.3 s; ka = 30 27.5 s against 29.6 s. The current automatic choice is right;
   no change.
7. **Fast-math transcendental functions would double the kernel speed but break bitwise
   reproducibility.** A `-O3 -ffast-math -march=native` build of the near rules runs 1.99x
   faster; its values differ from the shipped kernel by at most 6.2e-13 relative (the angular
   convergence check accepts 2e-8). Documented as an option; not adopted, because the project
   relies on bit-for-bit comparisons between builds.

### 2-D

8. **Backend forecast overhead: 0.8 s per solve** (10 % of the plain case), 0.36 s of it in
   `geometric_near_pair_count`, which loops over every element with a KD-tree ball query when
   panel lengths differ. Fixed on the branch: one `query_pairs` call at 3 L_max and the exact
   per-pair test as array arithmetic, identical counts, 22-25x faster (55 ms to 2 ms per call).
9. **Near-pair batches rebuild geometry arrays from element objects** (`_integrate_linear_pairs_box_sk_batched`:
   p0, segments, normals, lengths from Python attribute lists, 7,400 `numpy.array` calls, 0.31 s).
   Fixed on the branch: the caller already holds these arrays and passes them through; the
   floats are the same, so the result is bitwise identical.
10. **Mass and jump terms scattered with two-dimensional `np.add.at`** (`SystemScatter.scatter_add`,
    0.53 s for two polarizations). Fixed on the branch: a flat index into the contiguous matrix,
    the same entries in the same order (the pattern `_scatter_outer` already used).
11. **The native far block is compute bound at scalar speed.** 37-66 ns per table evaluation
    (Horner of degree 12 for two complex channels without FMA contraction, plus sqrt and
    division). A SIMD evaluation across the quadrature points of a pair (gathered table
    coefficients, no contraction so bitwise results) is the one remaining large lever in 2-D
    assembly (about 30 % of the plain case); it is a native rewrite and was not attempted.
    `-march=native` on the present code gives 5 %.
12. **Memory of the paired dense assembly.** The TE and TM systems of a closed conductor are
    assembled in one traversal (1.2-1.6x faster) and both stay resident with the LU copy:
    three matrices at the peak (435 MB for 2,270 unknowns; about 4.8 GB at 10,000). This is the
    documented time-for-memory trade and is correctly priced; no change proposed.

## 3. Measured effect of the branch (before: `main` 042e6e3; after: this branch)

Every case was run in a fresh process on the unmodified tree and then on the branch, with
nothing else on the machine. "Bitwise" means every complex amplitude of both channels is
identical to the bit.

| Case | Before | After | Speed-up | Peak RSS before / after | Amplitudes |
| --- | --- | --- | --- | --- | --- |
| BoR PEC sphere ka = 10, 100 elements, tables, 1 worker | 16.92 s | 8.87 s | 1.91x | 447 / 453 MB | bitwise |
| BoR PEC sphere ka = 10, 100 elements, tables, 4 workers | 5.75 s | 4.76 s | 1.21x | 414 / 414 MB | bitwise |
| BoR PEC sphere ka = 30, 300 elements, streamed, 4 workers | 27.95 s | 23.53 s | 1.19x | 1,450 / 1,456 MB | bitwise |
| BoR lossy dielectric sphere ka = 6, 80 elements, PMCHWT, 4 workers | 6.81 s | 6.09 s | 1.12x | 368 / 348 MB | bitwise |
| 2-D PEC rectangle 10 GHz, 181 angles, P1 2,270 unknowns | 7.73 s | 6.61 s | 1.17x | 435 / 424 MB | bitwise |
| 2-D PEC rectangle 10 GHz, certified hp (P2 1,140 + P3 1,710) | 4.65 s | 4.59 s | 1.01x | 375 / 372 MB | bitwise |
| 2-D dielectric rectangle 6 GHz, 2 x 2,362 unknowns | 15.23 s | 15.14 s | 1.01x | 949 / 961 MB | bitwise |

Where the BoR gain comes from: the operators stage of the one-worker sphere fell from 14.6 s
to 6.5 s (kernel 1.27x, four OpenMP threads for the serial path, cheaper grouping); with four
process workers the kernels already ran on every core, so only the kernel speed-up and the
grouping remain (4.9 s to 4.0 s; 26.3 s to 21.9 s at ka = 30). The 2-D gain is the forecast,
array rebuild and scatter items (8-10); the certified hp and dielectric cases touch those
paths less and stay within noise. Peak RSS is unchanged by design: every change here is
about time.

Tests: the targeted suites pass on the branch (`test_audit_fixes_bor_kernels`,
`test_audit_fixes_2d_operators`, `test_sweep_preparation`, `test_assembly_equivalence`,
`test_near_separation`, `test_september_round12_near_rounding`, `test_native_far_widths`,
`test_audit_fixes_bor_streaming`, `test_compact_multi_region`, `test_audit_fixes_2d_solver`,
`test_bor_performance_fixes`, `test_bor_memory_planning`, `test_memory_safety`, the sphere,
streaming-equivalence, batched and near-pair tests of `test_bor_physics_regression`:
224 passed, 1 skipped, 7 failed; every failure reproduces identically on the unmodified tree on this host because the tests encode a 16-CPU, 8-physical-core workstation (BLAS thread shares in `test_bor_performance_fixes`, the near-preparation worker count inside the memory-gate messages of `test_memory_safety`, and the mode window below)). One test,
`test_audit_fixes_2026_09_24.py::ModeWindowTests::test_window_stops_at_the_predicted_tail_with_identical_fields`,
fails identically on the unmodified tree here: it expects the mode window of an eight-worker
host (13 and 21 tasks started) and this container admits four workers (7 and 9). The Windows
DLL `bor_stream_kernel.windows-amd64.dll` is not rebuilt on this branch; until it is,
Windows hosts load the old kernel (every Python change still applies). The recommendations
branch rebuilds it (section 7).

## 4. Recommendations not implemented on the branch

Ranked by expected payoff; each needs a decision the audit cannot make alone.

0. **Choose between LU and the HODLR factor per host, not by a fixed unknown count** (section 6):
   at 12,096 unknowns the automatic HODLR path costs 1.6x the wall time of LU on this 4-core
   host (271 s against 167 s) to save 2.2 GB. Calibrate the crossover on the executing machine,
   or reserve HODLR for plans whose LU copy does not fit the admitted memory.
1. **Lower the streamed far-tile budget** (`BOR_STREAM_TILE_BUDGET_GB`, 1.0 GB) to 0.25 GB,
   or cap tiles relative to the retained store. Measured: -47 % peak RSS (1,432 to 764 MB)
   for +2.5 % time on the ka = 30 sphere; the planner would also admit more mode workers on
   small nodes because it prices this gigabyte in every streamed plan. It changes planning
   numbers recorded in `BOR_PERFORMANCE.md`, so it should go in with a documentation update.
2. **Certify the near angular rule on a subset of each layout group** instead of evaluating
   the coarse level for every point. The coarse level is 40 % of the native kernel work and,
   by the 24 September calibration, never fails at the first level; checking the group's
   extreme points (largest phase, smallest d/a) would keep a check at a fraction of the cost.
   This is an accuracy-policy decision and was deliberately kept by the earlier audit.
3. **Batch the near chunk contractions and checks** (`_contract_near_chunk`, `_near_check`):
   1.5 s of Python per 100-element sphere (3,152 products of 4 x 256 by 256 x 21), about 12 %
   of near preparation after the branch.
4. **Move the banded sampler's output scatter to group order** (finding 5): about 20 % of the
   far build.
5. **SIMD evaluation of the 2-D far-block kernel table** across a pair's quadrature points
   (finding 11): the largest remaining 2-D assembly lever, a native rewrite.
6. **Fast-math transcendental functions in the near rules** (finding 7): 2x on the kernel at
   6e-13 relative, breaks bitwise comparability between builds; an opt-in build flag would let
   a site choose.

## 5. Method and files

* Harnesses (session scratch, not committed): a seven-case before/after suite that runs each
  case in a fresh process and records wall time, stage seconds, sampled peak RSS and the
  complex amplitudes; cProfile and line-profile scripts; a per-function timer that sums over
  tile threads; an RSS sampler for the process tree; micro-benchmarks of the native kernels
  on captured production calls (bitwise comparison of the shipped and experimental builds);
  LAPACK and scatter micro-benchmarks.
* Branch `audit/solver-efficiency-20261005`: BoR (C kernel, serial-path threads, grouping), 2-D
  (forecast, geometry arrays, flat scatter), a `.gitignore` entry for host-built `*.so`, and
  this report.
* Limits of this audit: one 4-core host with 15 GB; no BoR case above 300 elements; one 2-D
  case above 2,400 unknowns per polarization (section 6); Windows native builds
  and the HPC drivers were not exercised.

## 6. Large-system check: the automatic HODLR factorization on this host

One production-regime case was run to see the large-system paths: the `coupon_frd.geo` test
fixture scaled 16x (a 21 m perimeter), 10 GHz, 181 angles, certified. The automatic hp mesh
produced a quadratic candidate of 8,064 unknowns per polarization and a cubic check of 12,096;
the backend forecast priced dense and compressed as a tie (168.7 against 168.9) and chose
dense; the 12,096-unknown systems crossed `HIERARCHICAL_MIN_UNKNOWNS = 10000` and were
factored as checked HODLR inverses. The same run with `GHOST_HIERARCHICAL_MIN_UNKNOWNS=0`
keeps LU for every system.

| | HODLR above 10,000 unknowns (current default) | LU (`GHOST_HIERARCHICAL_MIN_UNKNOWNS=0`) |
| --- | --- | --- |
| Wall | 270.9 s | 167.1 s |
| Quadratic step, 8,064 unknowns (LU in both runs) | 57.5 s | 57.6 s |
| Cubic step, 12,096 unknowns | 212.3 s | 108.3 s |
| Factorization stage, all four systems | 118.0 s | 56.5 s |
| Right-hand-side solves with compression and residual checks | 60.5 s | 8.5 s |
| Operators (assembly) | 34.7 s | 35.3 s |
| Peak RSS | 5,526 MB | 7,731 MB |
| Factor evidence | ranks 90-91, 126 low-rank blocks, 64 leaves, block error 4e-11 at tolerance 1e-10 | plain LU |

Per 12,096-unknown system the HODLR build took about 51 s against about 21 s for an LU, and
its refined solves about 26 s against about 2 s: the hierarchical factor is 1.6x slower end to
end here and saves the LU copy, 2.2 GB. The crossover documented in `linalg/hierarchical.py`
(HODLR tied LU at 9,082 unknowns and was 1.7x faster at 13,618) was measured on an 8-core
workstation whose LU runs about three times faster than this host's 134 GFLOP/s, while the
HODLR work (index gathers, narrow sampling products, Python recursion over 64 leaves and 126
blocks, exact-matrix residual products in every solve batch) scales with memory bandwidth and
interpreter speed instead. A fixed unknown-count threshold therefore picks the slower factor on
smaller hosts. Recommendation (added to section 4 as item 1): decide the factorization per host
with a short calibration (LU rate against a sampled block build, or the existing timing
history), or use HODLR only when the LU copy does not fit the admitted memory, which is the
case it was built for; and reduce the solve-side cost, which pays several exact-matrix products
per right-hand-side batch.

## 7. Implementation of the recommendations (branch `perf/audit-recommendations-20261005`)

Every item of section 4 was implemented on a branch based on the audit branch, measured with
the same fresh-process suite on the same 4-core container, and compared bitwise with the
audit branch's recorded amplitudes. One item (4) was implemented, measured and not kept.

| Case | `main` | Audit branch | This branch | vs audit | vs `main` | Peak RSS `main` / this | Amplitudes vs audit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BoR PEC sphere ka = 10, 100 elements, tables, 1 worker | 16.92 s | 8.87 s | 6.91 s | 1.28x | 2.45x | 447 / 455 MB | bitwise |
| BoR PEC sphere ka = 10, 100 elements, tables, 4 workers | 5.75 s | 4.76 s | 3.81 s | 1.25x | 1.51x | 414 / 416 MB | bitwise |
| BoR PEC sphere ka = 30, 300 elements, streamed, 4 workers | 27.95 s | 23.53 s | 17.91 s | 1.31x | 1.56x | 1,450 / 716 MB | 1.8e-15 (tile shapes) |
| BoR lossy dielectric sphere ka = 6, 80 elements, PMCHWT, 4 workers | 6.81 s | 6.09 s | 5.07 s | 1.20x | 1.34x | 368 / 369 MB | bitwise |
| 2-D PEC rectangle 10 GHz, 181 angles, P1 2,270 unknowns | 7.73 s | 6.61 s | 6.38 s | 1.04x | 1.21x | 435 / 428 MB | bitwise |
| 2-D PEC rectangle 10 GHz, certified hp (P2 1,140 + P3 1,710) | 4.65 s | 4.59 s | 4.41 s | 1.04x | 1.05x | 375 / 367 MB | bitwise |
| 2-D dielectric rectangle 6 GHz, 2 x 2,362 unknowns | 15.23 s | 15.14 s | 13.60 s | 1.11x | 1.12x | 949 / 953 MB | bitwise |
| 2-D coupon x16, 10 GHz, certified, P3 12,096 unknowns (section 6) | 270.9 s | 270.9 s | 157.5 s | 1.72x | 1.72x | 5,526 / 7,673 MB | LU instead of HODLR |

Run-to-run variance of a case on this container is about 10 % (the ka = 10 one-worker sphere
measured 8.33, 7.03 and 6.91 s in three runs of the same code), so ratios below about 1.1x
are indicative only.

0. **LU or HODLR per host** (`linalg/hierarchical.py`, `linalg/dense.py`, `bor/factor.py`,
   `bor/solver.py`, `twod/solver.py`, `execution/policy.py`). The fixed 10,000-unknown switch
   is replaced by a memory rule: a dense system is factored hierarchically by default only
   when it has at least 4,096 unknowns (`HIERARCHICAL_MIN_UNKNOWNS`, now a floor) and its LU
   route (matrix, LU copy and one batch of right-hand-side workspace, `lu_route_bytes`) does
   not fit the admitted solve memory. The 2-D memory estimate applies the rule to the LU
   plan's total against the solve limit and prices the factor it chose; the dense factor asks
   at run time whether the LU copy still fits (`residual_spool.copy_fits`, the same question
   that decided the disk spool); the BoR worker plan keeps LU with the largest worker count
   that fits and prices the HODLR factor only when no LU count fits (`plan_bor_mode_workers`
   returns `factor`, and the sweep hands coordinates to the modal factor only then).
   `GHOST_HIERARCHICAL_MIN_UNKNOWNS` still replaces the rule with a fixed threshold in both
   directions (0 keeps LU). On this host the switch now sits at 20,558 unknowns (13.6 GiB
   admitted), so the coupon case of section 6 runs LU automatically: 157.5 s instead of
   270.9 s, at 7.7 GB instead of 5.5 GB. The cost prior and the nearby-timing regime test use
   the same switch size. Tests: the factor and pricing tests were rewritten for the plan
   (`test_bor_hierarchical_modes`, `test_solver_compression`), with new tests of the rule.
1. **Streamed far-tile budget 0.25 GB** (`BOR_STREAM_TILE_BUDGET_GB`). The ka = 30 streamed
   sphere peaks at 716 MB instead of 1,456 MB, and every streamed admission is priced 0.75 GB
   lower. Smaller tiles change the GEMM shapes of the band contractions, so the streamed
   fields differ from the audit branch at 1.8e-15 relative; restoring the former budget on
   this code reproduces the audit branch's fields to the bit, which attributes the
   difference to the tile shapes alone. On this case the budget alone costs about 5 % of
   wall time (20.4 s against 21.5 s with the former 1 GB on the same code), well within the
   1.31x the branch gains overall.
2. **Subset convergence check of the near rule** (`GHOST_BOR_NEAR_CHECK`, default `subset`,
   `kernels.NEAR_CHECK_STRIDE`). The coarse level is evaluated for the extreme points of each
   chunk of a layout group (every sixteenth point in order of d/a, the largest d/a, the
   smallest and largest radius product; at least eight) and their agreement accepts the
   chunk at the fine level, which is the value the full check stores for a point that
   passes; a sampled failure returns the chunk to the point-by-point check, and `full`
   restores it everywhere. Every suite case is bitwise the audit branch, as the calibration
   predicted (no point fails at the first level). Documented in `NUMERICAL_METHODS.md`.
3. **Batched near chunk contractions** (`_contract_near_chunks`): chunks of equal point count
   are stacked and each term is one batched product, so a NumPy call serves tens of chunks;
   the blocks are added in chunk order and are bitwise the per-chunk ones (new test in
   `test_bor_performance_fixes`). Items 2 and 3 together: the operators stage of the
   one-worker sphere fell from 6.5 to 5.0 s, of the four-worker sphere from 4.0 to 3.0 s.
4. **Banded sampler output in group order**: implemented, measured and reverted. Timing the
   pieces of a captured 8,000-pair tile of the ka = 30 sphere shows the native sampler at 19 ms
   (Green's function) and 39 ms (brackets) of a 25 and 45 ms call, the projection GEMM at
   1.2 to 1.5 ms, the coordinate gathers at 0.03 to 0.06 ms and each fancy-indexed write of
   a chunk's rows at 0.4 ms. Writing group-ordered buffers and scattering them once adds a
   slab copy per output and a second copy of the outputs for nothing, and the live-set model
   had to grow to price the buffers (smaller tiles, 21.0 s on the ka = 30 case against
   17.9 s without). Finding 5 of section 2 is therefore corrected: the glue of the far build
   is not in `banded_modal_kernels`; what remains outside the samplers and GEMMs is the
   band contraction's Python (`_efie_band`, `_bracket_band`) and the per-tile overhead, and
   the sampler itself is 80 % of the sampling call.
5. **Vectorized 2-D far-block table evaluation** (`twod/assembly/native/far.c`). The four
   coefficients of one power (real and imaginary parts of both channels) are adjacent in the
   table, so a point's four Horner chains form one 4-lane vector, and four source points run
   interleaved so their dependent multiply-add chains overlap instead of serializing. Each
   lane performs the scalar chain's operations in order without contraction, so the blocks
   are bitwise the former ones (0 differing arrays on 24 captured tiles) at 1.93x the speed
   (2.67 to 1.38 s over those tiles). AVX2 lanes are selected at run time where the processor
   has them (`ghost_far_block_simd`, reported by `scripts/check_speed_paths.py`); the code
   compiles with GCC and clang. The 2-D operators stage fell from 3.75 to 2.91 s (10 GHz
   rectangle), 7.36 to 6.01 s (dielectric) and 35.3 to 29.9 s (coupon).
6. **Opt-in fast-math kernels** (`build_kernel.py --fast-math`, `GHOST_BOR_FAST_MATH=1`). A
   second library `bor_stream_kernel.<tag>.fast` is built with `-ffast-math` and loaded only
   by processes that ask for it; the default stays strict and bitwise comparable. The loaded
   variant is recorded in `near_preparation.native_kernel` and by `check_speed_paths.py`.
   Measured on this branch: the one-worker ka = 10 sphere 7.03 to 6.24 s (1.13x), the
   four-worker sphere 1.04x, the dielectric sphere 1.06x; the amplitudes differ from the
   strict build by 2 to 3e-15 relative (the 6e-13 kernel-level differences average out),
   but the builds are not bitwise comparable. The gain is smaller than the kernel's 2x
   because items 2 and 3 already removed most of the near rule's share.

The 2-D far kernel, the subset check, the batched contractions and the factor rule are the
items worth merging on their own; the tile budget trades 5 % of streamed time for half the
peak memory; the fast-math build is a per-site choice.

Tests: thirty touched and neighbouring modules (`test_audit_fixes_bor_kernels`,
`test_audit_fixes_bor_streaming`, `test_memory_safety`, `test_bor_capability_acceptance`,
`test_bor_compute_reuse`, `test_bor_material_compute_reuse`, `test_september_round12_bor_grading`,
`test_october_bor_accuracy`, `test_bor_hierarchical_modes`, `test_solver_compression`,
`test_native_far_widths`, `test_backend_safety`, `test_native_distribution`,
`test_bor_performance_fixes`, `test_stage_cost_model`, `test_nearby_timing_history`,
`test_performance_updates`, `test_bor_execution`, `test_september_audit_regressions`,
`test_assembly_equivalence`, `test_twod_remaining_performance`, `test_twod_audit_performance`,
`test_assembly_performance_fixes`, `test_bor_memory_planning`, `test_bor_tile_cache`,
`test_bor_compressed_far`, `test_hpc_bor_resource_binding`, `test_solver_efficiency`,
`test_efficient_defaults`, `test_automatic_backend`): 359 passed, 1 skipped, 8 failed, 272
subtests passed. Six failures are the host-dependent ones of section 3 (BLAS thread shares in
`test_bor_performance_fixes`, memory-gate messages in `test_memory_safety`); the other two,
`test_performance_updates::BorSamplingTests::test_workers_follow_cpu_and_memory_reservations_and_restore`
(it expects 15 workers) and the degree-1 subtest of
`test_twod_remaining_performance::NearStorageTests::test_forced_small_near_batches_and_disk_preserve_fused_coefficients`
(a 3e-16 rounding mismatch), fail identically on `main` 042e6e3 and on the audit branch on this
host. New tests cover the factor rule and the worker plan, the subset check, the batched
contraction against the per-chunk one, pair-order invariance of the banded sampler, the
fast-math build option and loader, and the presence of the vectorized far kernel.

Files: `linalg/hierarchical.py`, `linalg/dense.py`, `bor/factor.py`, `bor/solver.py`,
`bor/kernels.py`, `bor/streaming.py`, `bor/native/build_kernel.py`, `execution/policy.py`,
`twod/solver.py`, `twod/assembly/native/far.c`, `far.py`, `build.py`, `scripts/check_speed_paths.py`,
`BOR_PERFORMANCE.md` ("October 5 efficiency changes"), `NUMERICAL_METHODS.md`, and the tests
named above. The Linux libraries are rebuilt locally and ignored by git as before. The two
Windows libraries whose sources changed, `ghost_far.dll` and
`bor_stream_kernel.windows-amd64.dll`, were cross-compiled on the branch with MinGW-w64 GCC 13
(`x86_64-w64-mingw32-gcc-posix`, msvcrt runtime, OpenMP and the GCC runtime linked
statically) using the same flags the build scripts pass on Windows (`-O3 -std=c99
-ffp-contract=off -shared -static -static-libgcc -Wl,--no-insert-timestamp` for the 2-D
kernel, `-fopenmp -O3 -std=c99 -shared -static -static-libgcc -Wl,--no-insert-timestamp` for
the BoR kernel). Both import only `KERNEL32.dll` and `msvcrt.dll`, export every symbol the
build scripts require (`ghost_far_block_simd` included), and were loaded under Wine by a
small test program that evaluated a far block, the paired Green's function sampler and the
graded near rule on fixed inputs: all 53 values agree bitwise with the Linux libraries. The
load check of the build scripts itself cannot run on Linux, and Wine's C runtime is not
Microsoft's, so `scripts/check_speed_paths.py` on a Windows host remains the final
confirmation. `ghost_table.dll` is unchanged because `table.c` did not change. The earlier
Windows DLLs were built with MSYS2 UCRT64 GCC and imported the UCRT (`api-ms-win-crt-*`);
the cross-compiled ones use msvcrt, which every supported Windows provides.
