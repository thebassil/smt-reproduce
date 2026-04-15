from __future__ import annotations

from smt.backends.base import Backend, Job, Result


def get_backend(name: str) -> Backend:
    if name == "local":
        from smt.backends.local import LocalBackend
        return LocalBackend()
    elif name == "slurm":
        from smt.backends.slurm import SlurmBackend
        return SlurmBackend()
    elif name == "docker":
        from smt.backends.docker import DockerBackend
        return DockerBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")
