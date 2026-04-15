from __future__ import annotations
"""Docker backend: runs experiments inside a container."""

import subprocess
import sys
from pathlib import Path

from smt.backends.base import Job, Result
from smt.config import REPO_ROOT
from smt.display import console

IMAGE_NAME = "smt-reproduce:latest"


class DockerBackend:
    def __init__(self, image: str = IMAGE_NAME):
        self.image = image

    def submit(self, jobs: list[Job]) -> list[Result]:
        if not jobs:
            return []

        # Check docker is available
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, text=True
        )
        if proc.returncode != 0:
            console.print("[red]Docker not available.[/red]")
            console.print("Install Docker or use --backend local.")
            return [
                Result(job=j, returncode=1, stdout="", stderr="Docker not available")
                for j in jobs
            ]

        # Check image exists
        proc = subprocess.run(
            ["docker", "image", "inspect", self.image],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            console.print(f"[yellow]Image {self.image} not found. Building...[/yellow]")
            build = subprocess.run(
                ["docker", "build", "-t", self.image, str(REPO_ROOT)],
                capture_output=True, text=True,
            )
            if build.returncode != 0:
                console.print(f"[red]Docker build failed:[/red]\n{build.stderr[-500:]}")
                return [
                    Result(job=j, returncode=1, stdout="", stderr=build.stderr)
                    for j in jobs
                ]

        results = []
        for job in jobs:
            console.print(f"[dim]Docker run: {job.name}[/dim]")

            # Mount data and conf directories
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{REPO_ROOT / 'data'}:/app/data",
                "-v", f"{REPO_ROOT / 'conf'}:/app/conf",
                "-v", f"{REPO_ROOT / 'final_data'}:/app/final_data",
            ]

            # Pass env vars
            for k, v in job.env.items():
                docker_cmd.extend(["-e", f"{k}={v}"])

            docker_cmd.extend([self.image] + job.cmd)

            proc = subprocess.run(
                docker_cmd, capture_output=True, text=True,
            )

            r = Result(
                job=job,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
            results.append(r)

            status = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
            console.print(f"  {status} {job.name}")

        return results
