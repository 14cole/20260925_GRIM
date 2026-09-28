# GHOST 0.1.1 distribution and testing

This release contains the 2-D and body-of-revolution solvers, the round-11/12
accuracy and storage changes, and mesh-allocation guards added during independent
review. Both fixed and adaptive 2-D meshing and the BoR resource preview reject
oversized meshes before allocating their panel coordinates. BoR counting and
meshing share an allocation-free grading plan.

## Install

Use Python 3.10 or later in a virtual environment. From the distribution folder:

```console
python -m pip install ghost_em2d-0.1.1-py3-none-any.whl
```

NumPy, SciPy and psutil are required. Their supported version ranges are declared
in the wheel. Qt and matplotlib are optional desktop dependencies; install the
`gui` extra when using the desktop interface. Configured solves use the CPU.

The wheel is platform-independent and contains no prebuilt native binaries.
Both solvers have NumPy fallbacks. The optional BoR and 2-D C sources and build
scripts are included, together with example geometries and placement templates.
Build native acceleration on the destination machine using the README's compiler
instructions. Existing binaries in a development checkout are not redistributed.

## Source and tests

Extract `ghost-em2d-0.1.1-source.zip` for the complete clean `GHOST` folder,
including tests, fixtures, documentation and release tools. It excludes audit
archives, bytecode, caches, generated solver outputs and native binaries.
The standard `.tar.gz` source distribution is also provided for package builders.

From the extracted `GHOST` folder:

```console
python -m pip install ".[test,gui]"
python ghost_backend/tests/run_suite.py
```

The test runner isolates Qt module lifetimes from numerical workers. For the
qualified headless solver subset, install `.[test]` and run
`python scripts/check_headless.py`. Run `scripts/check_installation.py` with the
installed environment's Python from an unrelated working directory to check
wheel resources and 2-D/BoR analytic reference cases. It deliberately refuses
imports from a source checkout. `scripts/check_speed_paths.py` checks the
checkout it belongs to instead: whether the native BoR samplers, the 2-D native
libraries and an optimized BLAS work on this machine, and whether a setting
switches a fast path off; it exits with status 1 when anything would fall back.

Tests that exercise the separate GRIM or FREDDY projects explicitly skip when
those companion checkouts are absent. GHOST's own solver and file-format tests
remain available. Companion integration was also tested in the development
workspace where those projects are present.

Rebuild the artifacts with an environment containing `setuptools>=68` and `wheel`:

```console
python scripts/build_distribution.py --output ../distributions
```

The build emits a wheel, standard source distribution, source/test ZIP, and a
SHA-256 manifest. It checks required wheel resources and excludes platform
binaries and generated data. The ZIP has stable member metadata for reproducible
source checksums.

## Numerical scope

Mesh certification and resource admission remain necessary for production inputs.
The quadrature rescue retains its convergence check; unresolved cases still raise
an error. BoR impedance grading currently covers closed conductor snapshots,
with four fixed refinement levels. Partial/banded material junctions and sheets
retain their existing discretization. The 2-D contrast threshold and added levels
improve the tested junctions. Since 27 September the same levels also grade:
- 2-D free ends, branch points and corners of 150 degrees or less on linear
  meshes;
- BoR conductor rims, corners and tips.

Explicit counts and primitives of fewer than eight automatic panels are left
as drawn.

This lowered the tested errors 13-69x. It does not establish accuracy for
arbitrary corners, open ends or triple junctions. See `NUMERICAL_METHODS.md` and
`GEOMETRY_INPUT_CHEATSHEET.md` for the supported inputs and limitations.

GPU support is limited to low-level diagnostics. No real CuPy hardware execution
is claimed by this release. Cross-platform wheel tagging reflects the portable
Python/NumPy implementation; validation for this release was performed on Windows.
