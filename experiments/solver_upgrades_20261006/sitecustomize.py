"""Apply the GHOST solver experiments named in GHOST_EXPERIMENTS to every Python process started with
this folder on PYTHONPATH (the bench subprocesses and the solver's spawned worker processes alike)."""
import os
import sys

if os.environ.get('GHOST_EXPERIMENTS'):
    _here = os.path.dirname(os.path.abspath(__file__))
    _ghost = os.path.normpath(os.path.join(_here, '..', '..', 'tools', 'GHOST'))
    if _ghost not in sys.path:
        sys.path.insert(0, _ghost)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import ghost_experiments
    ghost_experiments.apply_from_environment()
