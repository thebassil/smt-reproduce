"""Inspect pre-computed subsystem ablation results."""

import csv
from pathlib import Path
from typing import Optional

from rich.table import Table

from smt.config import SYSTEMS, resolve_final_data
from smt.display import console, print_csv_table, section


def inspect_subsystem(system: Optional[str] = None) -> None:
    base = resolve_final_data() / "02_subsystem_ablations"
    if not base.exists():
        console.print("[red]No subsystem ablation data found.[/red]")
        return

    targets = [system] if system else SYSTEMS

    for sys in targets:
        sys_dir = base / sys
        if not sys_dir.exists():
            console.print(f"[yellow]No ablation data for {sys}[/yellow]")
            continue

        section(f"{sys.upper()} - Subsystem Ablations")

        # Find all summary CSVs
        summaries = sorted(sys_dir.glob("*_summary.csv"))
        if summaries:
            for s in summaries:
                print_csv_table(s, f"{sys} - {s.stem}")
        else:
            # Some systems use directory-based results (grackle, smtgazer)
            subdirs = sorted(
                [d for d in sys_dir.iterdir() if d.is_dir()]
            )
            if subdirs:
                table = Table(
                    title=f"{sys} - Ablation Variants", show_lines=True
                )
                table.add_column("Variant")
                table.add_column("Files")
                for d in subdirs:
                    files = [f.name for f in d.iterdir() if f.is_file()]
                    table.add_row(d.name, ", ".join(files[:5]))
                console.print(table)

            # Also check for loose CSVs
            csvs = sorted(sys_dir.glob("*.csv"))
            for c in csvs:
                print_csv_table(c, f"{sys} - {c.stem}")
