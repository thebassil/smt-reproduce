from __future__ import annotations
"""Local execution backend using subprocess."""

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.progress import Progress

from smt.backends.base import Job, Result
from smt.display import console


def _run_job(job: Job) -> Result:
    env = {**os.environ, **job.env}
    proc = subprocess.run(
        job.cmd,
        cwd=job.cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    return Result(
        job=job,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


class LocalBackend:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or 1

    def submit(self, jobs: list[Job]) -> list[Result]:
        results = []
        if len(jobs) == 1:
            console.print(f"[dim]Running: {jobs[0].name}[/dim]")
            r = _run_job(jobs[0])
            if not r.ok:
                console.print(f"[red]FAILED:[/red] {r.job.name}")
                if r.stderr:
                    console.print(r.stderr[:500])
            else:
                console.print(f"[green]OK:[/green] {r.job.name}")
            return [r]

        with Progress() as progress:
            task = progress.add_task("Running jobs...", total=len(jobs))
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_run_job, j): j for j in jobs}
                for fut in as_completed(futures):
                    r = fut.result()
                    results.append(r)
                    status = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
                    progress.console.print(f"  {status} {r.job.name}")
                    progress.advance(task)
        return results
