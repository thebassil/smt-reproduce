from __future__ import annotations
"""Layer 2: Reproduce per-system ablation sweeps.

Each system has its own ablation harness with a `run` subcommand
that takes a YAML template path. The CLI invokes these verbatim.
"""

import sys
from pathlib import Path
from typing import Optional

from smt.backends.base import Job
from smt.backends import get_backend
from smt.config import SYSTEMS, resolve_db_path, resolve_ablations_dir
from smt.display import console


# Per-system ablation configuration: entry point script, templates location
ABLATION_CONFIG = {
    "machsmt": {
        "script": "ablation_harness.py",
        "cwd_suffix": "",  # run from ablations/machsmt/
        "templates_glob": "ablations/templates/*.yaml",
        "run_args": lambda tmpl: ["run", str(tmpl), "--execute"],
    },
    "sibyl": {
        "script": "ablation/harness.py",
        "cwd_suffix": "",
        "templates_glob": "ablation/configs/*.yaml",
        "run_args": lambda tmpl: ["run", str(tmpl), "--execute"],
    },
    "grackle": {
        "script": "ablation/cli.py",
        "cwd_suffix": "",
        "templates_glob": "ablation/experiments/*.yaml",
        "run_args": lambda tmpl: ["run", str(tmpl)],
    },
    "medley": {
        "script": "ablation_harness.py",
        "cwd_suffix": "",
        "templates_glob": "ablations/*.yaml",
        "run_args": lambda tmpl: ["run", str(tmpl), "--execute"],
    },
    "smtgazer": {
        "script": "ablation/cli.py",
        "cwd_suffix": "",
        "templates_glob": "ablation/experiments/*.yaml",
        "run_args": lambda tmpl: ["run", str(tmpl)],
    },
}


def run_subsystem(
    system: Optional[str] = None,
    backend: str = "local",
    db_override: Optional[str] = None,
) -> None:
    db = str(resolve_db_path(db_override))
    ablations_dir = resolve_ablations_dir()
    targets = [system] if system else SYSTEMS

    jobs = []
    for sys_name in targets:
        sys_dir = ablations_dir / sys_name
        if not sys_dir.exists():
            console.print(f"[yellow]Ablation directory not found: {sys_dir}[/yellow]")
            continue

        cfg = ABLATION_CONFIG.get(sys_name)
        if not cfg:
            console.print(f"[yellow]No ablation config for {sys_name}[/yellow]")
            continue

        script = sys_dir / cfg["script"]
        if not script.exists():
            console.print(f"[yellow]Script not found: {script}[/yellow]")
            continue

        # Find all YAML templates
        yamls = sorted(sys_dir.glob(cfg["templates_glob"]))
        if not yamls:
            console.print(f"[yellow]No templates for {sys_name} ({cfg['templates_glob']})[/yellow]")
            continue

        for yaml_path in yamls:
            cmd = [
                sys.executable,
                cfg["script"],
            ] + cfg["run_args"](yaml_path)

            jobs.append(
                Job(
                    name=f"ablation-{sys_name}-{yaml_path.stem}",
                    cmd=cmd,
                    cwd=str(sys_dir),
                    env={"SMT_DB_PATH": db},
                )
            )

    if not jobs:
        console.print("[red]No ablation jobs to run.[/red]")
        return

    console.print(f"[bold]Running {len(jobs)} ablation experiments...[/bold]")
    for sys_name in targets:
        sys_jobs = [j for j in jobs if j.name.startswith(f"ablation-{sys_name}")]
        if sys_jobs:
            console.print(f"  {sys_name}: {len(sys_jobs)} experiments")

    executor = get_backend(backend)
    results = executor.submit(jobs)

    ok = sum(1 for r in results if r.ok)
    console.print(f"\n[bold]Completed: {ok}/{len(results)} ablations succeeded.[/bold]")
    for r in results:
        if not r.ok:
            console.print(f"  [red]FAIL[/red] {r.job.name}")
            if r.stderr:
                last = r.stderr.strip().split("\n")[-1]
                console.print(f"    {last[:150]}")
