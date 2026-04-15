"""Inspect pre-computed system reproduction results."""

import csv
from pathlib import Path
from typing import Optional

from rich.table import Table

from smt.config import SYSTEMS, resolve_final_data
from smt.display import console, print_csv_table, section


def inspect_systems(system: Optional[str] = None) -> None:
    base = resolve_final_data() / "01_system_reproductions"
    if not base.exists():
        console.print("[red]No system reproduction data found.[/red]")
        console.print("Run `smt download` first or check final_data/ directory.")
        return

    targets = [system] if system else SYSTEMS

    if not system:
        _print_comparison_table(base, targets)

    for sys in targets:
        sys_dir = base / sys
        if not sys_dir.exists():
            console.print(f"[yellow]No data for {sys}[/yellow]")
            continue

        section(f"{sys.upper()} - System Reproduction")

        summary = sys_dir / "kfold_k5_summary.csv"
        if summary.exists():
            print_csv_table(summary, f"{sys} - K-Fold Summary")

        for name in [
            "kfold_per_fold_results.csv",
            "kfold_k5_perfold.csv",
            "kfold_k5_perlogic.csv",
            "kfold_k5_meanstd.csv",
            "evaluation_summary.csv",
            "per_logic_summary.csv",
        ]:
            f = sys_dir / name
            if f.exists():
                print_csv_table(f, f"{sys} - {name}")


def _print_comparison_table(base: Path, systems: list[str]) -> None:
    section("ALL SYSTEMS - Comparison")

    # Baselines (same across all systems)
    baselines = ["VBS", "SBS", "COMBINED_SBS", "CSBS", "RANDOM"]

    table = Table(title="System Comparison", show_lines=True)
    table.add_column("System")
    table.add_column("Total PAR-2", justify="right")
    table.add_column("Avg PAR-2", justify="right")
    table.add_column("Solved", justify="right")
    table.add_column("Solved %", justify="right")

    for sys in systems:
        summary = base / sys / "kfold_k5_summary.csv"
        if not summary.exists():
            summary = base / sys / "evaluation_summary.csv"
        if not summary.exists():
            table.add_row(sys, "-", "-", "-", "-")
            continue
        with open(summary) as f:
            rows = list(csv.DictReader(f))
        # Find the system's own row (not a baseline)
        for row in rows:
            strategy = row.get("strategy", "")
            if strategy.upper() not in [b.upper() for b in baselines]:
                table.add_row(
                    f"[bold]{sys}[/bold]",
                    row.get("total_par2", "-"),
                    row.get("avg_par2", "-"),
                    row.get("solved", "-"),
                    row.get("solved_pct", "-"),
                )
                break

    # Add VBS and SBS as reference rows
    table.add_section()
    # Read from any system's CSV for baselines
    for sys in systems:
        summary = base / sys / "kfold_k5_summary.csv"
        if not summary.exists():
            summary = base / sys / "evaluation_summary.csv"
        if summary.exists():
            with open(summary) as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row.get("strategy", "").upper() == "VBS":
                    table.add_row(
                        "[dim]VBS (oracle)[/dim]",
                        row.get("total_par2", "-"),
                        row.get("avg_par2", "-"),
                        row.get("solved", "-"),
                        row.get("solved_pct", "-"),
                    )
                elif row.get("strategy", "").upper() == "SBS":
                    table.add_row(
                        "[dim]SBS (best single)[/dim]",
                        row.get("total_par2", "-"),
                        row.get("avg_par2", "-"),
                        row.get("solved", "-"),
                        row.get("solved_pct", "-"),
                    )
            break

    console.print(table)
