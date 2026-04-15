from __future__ import annotations
"""Layer 1: Reproduce selector system evaluations.

Each system is invoked via subprocess using the exact same scripts
and CLI arguments that were used in the original worktrees.
"""

import sys
from typing import Optional

from smt.backends.base import Job
from smt.backends import get_backend
from smt.config import SYSTEMS, resolve_db_path, resolve_systems_dir
from smt.display import console


# Maps each system to the exact command used to train + evaluate it.
# These match the original eval-* worktree invocations.
SYSTEM_COMMANDS = {
    "machsmt": {
        "script": "machsmt_portfolio_trainer.py",
        "args_fn": lambda db: [
            "--db", db,
            "--portfolio-id", "7",
            "--output-dir", "machsmt_results",
            "--k-fold", "5",
        ],
    },
    "sibyl": {
        "script": "sibyl_portfolio_trainer.py",
        "args_fn": lambda db: [
            "--db", db,
            "--output-dir", "sibyl_results",
            "--stage", "all",
            "--cpu",
        ],
    },
    "grackle": {
        "script": "grackle_portfolio_trainer.py",
        "args_fn": lambda db: [
            "--db", db,
            "--portfolio-id", "7",
            "--output-dir", "grackle_results",
        ],
    },
    "medley": {
        "script": "medley_portfolio_trainer.py",
        "args_fn": lambda db: [
            "--db", db,
            "--portfolio-id", "7",
            "--output-dir", "medley_results",
        ],
    },
    "smtgazer": {
        # smtgazer uses positional args, not --db; it reads DB_PATH from hardcoded constant
        "script": "smtgazer_full_pipeline.py",
        "args_fn": lambda db: ["all"],
    },
}


def run_systems(
    system: Optional[str] = None,
    backend: str = "local",
    db_override: Optional[str] = None,
) -> None:
    db = str(resolve_db_path(db_override))
    systems_dir = resolve_systems_dir()
    targets = [system] if system else SYSTEMS

    jobs = []
    for sys_name in targets:
        sys_dir = systems_dir / sys_name
        if not sys_dir.exists():
            console.print(f"[yellow]System directory not found: {sys_dir}[/yellow]")
            console.print("Run the copy step first. See README.")
            continue

        info = SYSTEM_COMMANDS[sys_name]
        cmd = [sys.executable, info["script"]] + info["args_fn"](db)
        jobs.append(
            Job(
                name=f"system-{sys_name}",
                cmd=cmd,
                cwd=str(sys_dir),
                env={"SMT_DB_PATH": db},
            )
        )

    if not jobs:
        console.print("[red]No systems to run.[/red]")
        return

    console.print(f"[bold]Running {len(jobs)} system reproduction(s)...[/bold]")
    executor = get_backend(backend)
    results = executor.submit(jobs)

    ok = sum(1 for r in results if r.ok)
    console.print(f"\n[bold]Completed: {ok}/{len(results)} systems succeeded.[/bold]")
    for r in results:
        if not r.ok:
            console.print(f"  [red]FAIL[/red] {r.job.name}")
            if r.stderr:
                last = r.stderr.strip().split("\n")[-1]
                console.print(f"    {last[:150]}")
