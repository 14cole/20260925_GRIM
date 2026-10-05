#!/usr/bin/env python3
"""Build and load-check the optional native BoR streaming sampler.

The default build is strict IEEE arithmetic (-O3 without -ffast-math), so its
results are bitwise comparable between machines and builds.  ``--fast-math``
builds a second, separately named library (``bor_stream_kernel.<tag>.fast``)
with ``-ffast-math``: the near angular rules run about twice as fast and the
kernels differ from the strict build by up to about 1e-12 relative (the
angular convergence check accepts 2e-8), but results are then not bitwise
comparable with a strict build or another host.  A site opts in per process
with ``GHOST_BOR_FAST_MATH=1``; without that setting the strict library is
loaded even when the fast one is present.
"""

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


# Every entry point bor_stream_kernel.c defines.  The Python callers detect
# each one with hasattr and fall back to NumPy, so a library that silently
# lacks one (stale object, wrong source) would only show up as a slow run;
# the load check below therefore requires all of them.
REQUIRED_SYMBOLS = (
    # streamed outer-product samplers (real k)
    "sample_g", "sample_mfie", "sample_ibc",
    # paired near samplers
    "near_mfie", "near_brackets", "near_brackets_stable", "near_green",
    # grouped half-grid far samplers (real/complex k, OpenMP team)
    "sample_g_pairs", "sample_brackets_pairs",
    # near-rule angular projections, and the fused graded near rules
    "trig_moments", "parity_moments", "near_green_rule", "near_brackets_rule",
)

_LOAD_CHECK = (
    "import ctypes, sys\n"
    "if sys.platform == 'win32': ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)\n"
    "lib = ctypes.CDLL(sys.argv[1])\n"
    "missing = [s for s in sys.argv[2:] if not hasattr(lib, s)]\n"
    "if missing:\n"
    "    sys.stderr.write('missing exports: ' + ', '.join(missing) + '\\n')\n"
    "    sys.exit(3)\n"
)


def output_name(tag: 'str', extension: 'str', fast_math: 'bool' = False) -> 'str':
    """``bor_stream_kernel.<tag><extension>``, or the ``.fast`` variant of ``--fast-math``."""
    return f"bor_stream_kernel.{tag}{'.fast' if fast_math else ''}{extension}"


def compile_command(compiler: 'str', source, output, system_name: 'str',
                    openmp: 'bool' = True, fast_math: 'bool' = False) -> 'list[str]':
    """The compiler command of one build (strict by default; see the module notes)."""
    command = [compiler]
    if openmp:
        command.append("-fopenmp")
    command.extend(["-O3", "-std=c99", "-shared"])
    if fast_math:
        command.append("-ffast-math")
    if system_name == "windows":
        # A release must load in a fresh Python process without the compiler's
        # bin directory. Link GCC/OpenMP/pthread runtime archives into the DLL;
        # Windows system libraries remain normal system dependencies.
        command.extend(["-static", "-static-libgcc", "-Wl,--no-insert-timestamp"])
    else:
        command.append("-fPIC")
    command.extend(["-o", str(output), str(source), "-lm"])
    return command


def _find_compiler(requested: 'str | None') -> 'str | None':
    """Resolve an explicit compiler or common native Windows toolchains."""

    if requested:
        return shutil.which(requested) or (
            str(Path(requested).resolve())
            if Path(requested).is_file() else None
        )
    for name in ("cc", "gcc", "clang"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    if platform.system().lower() == "windows":
        for path in (
            Path("C:/msys64/ucrt64/bin/gcc.exe"),
            Path("C:/msys64/mingw64/bin/gcc.exe"),
        ):
            if path.is_file():
                return str(path)
    return None


def _compiler_environment(compiler: str, system_name: str) -> 'dict[str, str]':
    """Return an environment in which the compiler's helper tools can run.

    A native Windows MSYS2 GCC keeps cc1.exe's runtime DLLs beside gcc.exe.
    Finding gcc by its standard absolute path is therefore not sufficient:
    that directory must also be on PATH while GCC launches its subprocesses.
    """

    environment = os.environ.copy()
    if system_name == "windows":
        compiler_dir = str(Path(compiler).resolve().parent)
        current_path = environment.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if os.path.normcase(compiler_dir) not in {
            os.path.normcase(entry) for entry in path_entries
        }:
            environment["PATH"] = os.pathsep.join(
                [compiler_dir, *path_entries]
            )
    return environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory (default: bor/native beside the C source).",
    )
    parser.add_argument(
        "--compiler",
        default=os.environ.get("CC"),
        help=(
            "C compiler command. By default, use CC, PATH, or a standard "
            "MSYS2 UCRT64 installation."
        ),
    )
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="Build the native sampler without OpenMP parallel loops.",
    )
    parser.add_argument(
        "--fast-math",
        action="store_true",
        help=(
            "Build the opt-in -ffast-math variant bor_stream_kernel.<tag>.fast instead of "
            "the strict library: about twice the near-rule speed, kernels within about 1e-12 "
            "relative of the strict build, results not bitwise comparable between builds. "
            "GHOST_BOR_FAST_MATH=1 selects it at load time."
        ),
    )
    args = parser.parse_args()

    compiler = _find_compiler(args.compiler)
    if compiler is None:
        raise SystemExit(
            "No compatible C compiler was found. Install MSYS2 UCRT64 GCC, "
            "put gcc on PATH, set CC, or pass --compiler explicitly."
        )
    source = Path(__file__).resolve().with_name("bor_stream_kernel.c")
    system_name = platform.system().lower()
    tag = f"{system_name}-{platform.machine().lower()}"
    output_extension = ".dll" if system_name == "windows" else ".so"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / output_name(tag, output_extension, args.fast_math)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    command = compile_command(compiler, source, temporary, system_name,
                              openmp=False, fast_math=args.fast_math)
    commands = [command]
    if not args.no_openmp:
        commands.insert(0, compile_command(compiler, source, temporary, system_name,
                                           openmp=True, fast_math=args.fast_math))
    compiler_environment = _compiler_environment(compiler, system_name)
    try:
        completed = None
        failures: 'list[tuple[list[str], subprocess.CompletedProcess[str]]]' = []
        for candidate in commands:
            completed = subprocess.run(
                candidate,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=compiler_environment,
            )
            if completed.returncode == 0:
                command = candidate
                break
            failures.append((candidate, completed))
        else:
            diagnostics = []
            for failed_command, failed in failures:
                diagnostics.append(
                    f"Command: {subprocess.list2cmdline(failed_command)}\n"
                    f"Exit code: {failed.returncode}\n"
                    f"{failed.stdout or '(compiler produced no output)'}"
                )
            raise SystemExit(
                "Native BoR sampler compilation failed:\n"
                + "\n\n".join(diagnostics)
            )
        # Validate exactly the runtime environment a fresh worker will have.
        # Loading here with add_dll_directory(compiler/bin) concealed missing
        # redistributables, and retained a Windows mapping of the staged DLL.
        try:
            checked = subprocess.run(
                [sys.executable, "-I", "-c", _LOAD_CHECK, str(temporary), *REQUIRED_SYMBOLS],
                check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                f"Native BoR sampler load check timed out for {temporary.name}. "
                "The previous library was not replaced."
            ) from exc
        if checked.returncode != 0:
            raise SystemExit(
                "Native BoR sampler load check failed for "
                f"{temporary.name} (exit code {checked.returncode}):\n"
                f"{checked.stderr.strip() or checked.stdout.strip() or '(no output)'}\n"
                "The library was not installed. Every entry point of "
                "bor_stream_kernel.c must be exported: "
                + ", ".join(REQUIRED_SYMBOLS)
            )
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(f"Built and load-checked {output}")
    print("Exports: " + ", ".join(REQUIRED_SYMBOLS))
    print(
        "OpenMP sampling: "
        + ("enabled" if "-fopenmp" in command else "unavailable/disabled")
    )
    if args.fast_math:
        print("Fast-math variant: loaded only by processes with GHOST_BOR_FAST_MATH=1; "
              "its results are not bitwise comparable with the strict build.")
    print("Re-run Python workers so they load the native kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
