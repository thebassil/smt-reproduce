from __future__ import annotations
"""Backend protocol for execution dispatch."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Job:
    """A unit of work: a command to run in a given directory."""

    name: str
    cmd: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Result:
    job: Job
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Backend(Protocol):
    def submit(self, jobs: list[Job]) -> list[Result]:
        """Submit jobs and return results when complete."""
        ...
