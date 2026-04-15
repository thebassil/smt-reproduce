from __future__ import annotations
"""SLURM backend: generates and submits sbatch array jobs."""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from smt.backends.base import Job, Result
from smt.display import console

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{max_idx}
#SBATCH --output={log_dir}/slurm_%A_%a.out
#SBATCH --error={log_dir}/slurm_%A_%a.err
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{partition_line}

# Load manifest
MANIFEST="{manifest_path}"
CMD=$(python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))
t = m[str($SLURM_ARRAY_TASK_ID)]
print(t['cmd'])
")

echo "Task $SLURM_ARRAY_TASK_ID: $CMD"
eval "$CMD"
"""


class SlurmBackend:
    def __init__(
        self,
        partition: Optional[str] = None,
        time_limit: str = "08:00:00",
        cpus: int = 4,
        mem: str = "32G",
    ):
        self.partition = partition
        self.time_limit = time_limit
        self.cpus = cpus
        self.mem = mem

    def submit(self, jobs: list[Job]) -> list[Result]:
        if not jobs:
            return []

        # Create working directory for this submission
        work_dir = Path("slurm_submissions")
        work_dir.mkdir(exist_ok=True)
        log_dir = work_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Write manifest: maps array index -> command
        manifest = {}
        for i, job in enumerate(jobs):
            env_prefix = " ".join(f"{k}={v}" for k, v in job.env.items())
            cmd_str = " ".join(job.cmd)
            if job.cwd:
                cmd_str = f"cd {job.cwd} && {env_prefix} {cmd_str}"
            elif env_prefix:
                cmd_str = f"{env_prefix} {cmd_str}"
            manifest[str(i)] = {"name": job.name, "cmd": cmd_str}

        manifest_path = work_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Generate sbatch script
        partition_line = f"#SBATCH --partition={self.partition}" if self.partition else ""
        job_name = f"smt-reproduce-{int(time.time())}"

        sbatch_content = SBATCH_TEMPLATE.format(
            job_name=job_name,
            max_idx=len(jobs) - 1,
            log_dir=str(log_dir.resolve()),
            time_limit=self.time_limit,
            cpus=self.cpus,
            mem=self.mem,
            partition_line=partition_line,
            manifest_path=str(manifest_path.resolve()),
        )

        sbatch_path = work_dir / f"{job_name}.sbatch"
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        console.print(f"[bold]SLURM submission:[/bold]")
        console.print(f"  Jobs: {len(jobs)}")
        console.print(f"  Manifest: {manifest_path}")
        console.print(f"  Script: {sbatch_path}")

        # Submit
        proc = subprocess.run(
            ["sbatch", "--parsable", str(sbatch_path)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            console.print(f"[red]sbatch failed:[/red] {proc.stderr}")
            return [
                Result(job=j, returncode=1, stdout="", stderr=proc.stderr)
                for j in jobs
            ]

        job_id = proc.stdout.strip()
        console.print(f"  [green]Submitted![/green] Job ID: {job_id}")
        console.print(f"  Monitor: squeue -j {job_id}")
        console.print(f"  Logs: {log_dir}/slurm_{job_id}_*.out")

        # Return placeholder results — SLURM is async
        return [
            Result(
                job=j,
                returncode=0,
                stdout=f"Submitted as SLURM array task {i} in job {job_id}",
                stderr="",
            )
            for i, j in enumerate(jobs)
        ]
