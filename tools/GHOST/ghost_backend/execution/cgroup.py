"""Memory cgroups of this process: its own group and its ancestors (v2 or v1).

A SLURM job step, a systemd unit or a container without its own cgroup
namespace is a nested group, and the files at the mount root describe the
machine, not the job (cgroup v2 has no ``memory.max`` at the root at all).
Membership comes from ``/proc/self/cgroup``; inside a cgroup namespace the
group is the mount root itself, which callers keep as their fallback.

The solver (headroom) and the scheduler (limit) share these paths; each reads
the files with its own reader so that tests can stand in for the host.
"""

MOUNT = "/sys/fs/cgroup"
# (limit file, usage file, statistics file, reclaimable page-cache key)
_V2 = ("memory.max", "memory.current", "memory.stat", "inactive_file")
_V1 = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat", "total_inactive_file")


def _files(directory, names):
    return tuple(directory + "/" + name for name in names[:3]) + (names[3],)


def memory_groups(read_text):
    """File paths of this process's nested memory cgroups, innermost first.

    Each entry is ``(limit, usage, stat, reclaimable key)``.  ``read_text``
    reads a whole text file and raises OSError when it cannot.  The mount
    roots are not included (see :func:`root_groups`).
    """

    try:
        lines = read_text("/proc/self/cgroup").splitlines()
    except (OSError, ValueError):
        return []
    groups = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        if hierarchy == "0" and not controllers:
            root, names = MOUNT, _V2
        elif "memory" in controllers.split(","):
            root, names = MOUNT + "/memory", _V1
        else:
            continue
        relative = relative.strip().rstrip("/")
        while relative.startswith("/"):
            groups.append(_files(root + relative, names))
            relative = relative.rsplit("/", 1)[0]
    return groups


def root_groups():
    """File paths of the v2 and v1 mount roots, the group inside a namespace."""

    return [_files(MOUNT, _V2), _files(MOUNT + "/memory", _V1)]


def stat_value(read_text, path, key):
    """``key`` of a cgroup ``memory.stat`` file, in bytes, or None."""

    try:
        text = read_text(path)
    except (OSError, ValueError):
        return None
    for line in text.splitlines():
        name, _, value = line.partition(" ")
        if name == key and value.strip().isdigit():
            return int(value)
    return None
