"""Build a portable wheel and a clean source/test archive without generated data.

Run from any directory with a Python that provides setuptools>=68 and wheel:
    python scripts/build_distribution.py --output /path/to/distributions
Native binaries are built by recipients for their own platform; C sources ship.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.git', 'build', 'dist', 'results', 'generated'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.dll', '.so', '.dylib', '.grim', '.lock'}


def release_files(root):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink()
                or any(part in SKIP_DIRS or part.endswith('.egg-info') for part in relative.parts)
                or path.suffix.lower() in SKIP_SUFFIXES or path.name == '.coverage'):
            continue
        yield relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise SystemExit('Use an output directory outside the GHOST source tree.')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ghost-release-') as temporary:
        stage = Path(temporary)/'GHOST'
        files = list(release_files(ROOT))
        for relative in files:
            target = stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
        command = ('import sys; from setuptools.build_meta import build_wheel, build_sdist; '
                   'destination = sys.argv[1]; build_wheel(destination); build_sdist(destination)')
        subprocess.run([sys.executable, '-B', '-c', command, str(output)], cwd=stage, check=True)
        # Obtain the version from the project's own built metadata, including on Python 3.10.
        import email
        wheels = sorted(output.glob('ghost_em2d-*.whl'), key=lambda path: path.stat().st_mtime_ns)
        wheel = wheels[-1]
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            metadata = email.message_from_bytes(archive.read(next(n for n in names if n.endswith('/METADATA'))))
            required = {'ghost_backend/bor/native/bor_stream_kernel.c',
                        'ghost_backend/twod/assembly/native/table.c',
                        'ghost_backend/twod/assembly/native/far.c',
                        'ghost_backend/geometry/templates/point_features_template.csv'}
            missing = required - set(names)
            if missing:
                raise RuntimeError('Incomplete wheel: '+', '.join(sorted(missing)))
            if any(Path(n).suffix.lower() in SKIP_SUFFIXES or '/__pycache__/' in n
                   or n.startswith('ghost_backend/tests/') for n in names):
                raise RuntimeError('Wheel contains generated outputs, tests, or native binaries.')
        version = metadata['Version']
        source = output/f'ghost-em2d-{version}-source.zip'
        inventory = {}
        with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in files:
                data = (stage/relative).read_bytes()
                # Stable metadata makes repeated source archives byte-identical.
                info = zipfile.ZipInfo('GHOST/'+relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
                inventory[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        sdists = [*output.glob(f'ghost_em2d-{version}.tar.gz'), *output.glob(f'ghost-em2d-{version}.tar.gz')]
        if not sdists:
            raise RuntimeError('Source distribution was not written to the output directory.')
        artifacts = [wheel, source, *sdists]
        manifest = dict(version=version, files=inventory, artifacts={
            p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            for p in artifacts})
        destination = output/f'ghost-em2d-{version}-manifest.json'
        destination.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(manifest=str(destination), artifacts=list(manifest['artifacts'])), indent=2))


if __name__ == '__main__':
    main()
