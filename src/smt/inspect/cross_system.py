from __future__ import annotations
"""Inspect pre-computed cross-system ablation results."""

import csv
from pathlib import Path
from typing import Optional

from rich.table import Table

from smt.config import resolve_final_data
from smt.display import console, print_csv_table, section


def inspect_cross_system(axis: Optional[str] = None) -> None:
    base = resolve_final_data() / "03_cross_system_ablations"
    ext = resolve_final_data() / "04_cross_system_ablations_extended"

    if not base.exists():
        console.print("[red]No cross-system ablation data found.[/red]")
        return

    core_csv = base / "core_cross_system_ablation.csv"
    if not core_csv.exists():
        console.print("[red]core_cross_system_ablation.csv not found.[/red]")
        return

    section("Cross-System Ablation Results")

    rows = _load_csv(core_csv)

    if axis:
        _show_axis_summary(rows, axis)
    else:
        _show_overview(rows)

    # Extended results
    ext_dir = ext / "extended_runs" if ext.exists() else None
    if ext_dir and ext_dir.exists():
        ext_csvs = sorted(ext_dir.glob("*.csv"))
        if ext_csvs:
            section("Extended Ablation Results")
            console.print(f"[dim]{len(ext_csvs)} extended experiment CSVs available in final_data/04_*/extended_runs/[/dim]")


def _load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _show_overview(rows: list[dict]) -> None:
    featurisers = set(r.get("featuriser", "") for r in rows)
    models = set(r.get("model", "") for r in rows)
    policies = set(r.get("policy", "") for r in rows)
    logics = set(r.get("logic", "") for r in rows)
    seeds = set(r.get("seed", "") for r in rows)

    table = Table(title="Cross-System Ablation Overview")
    table.add_column("Dimension")
    table.add_column("Count")
    table.add_column("Values")
    table.add_row("Total rows", str(len(rows)), "")
    table.add_row("Featurisers", str(len(featurisers)), ", ".join(sorted(featurisers)))
    table.add_row("Models", str(len(models)), ", ".join(sorted(models)))
    table.add_row("Policies", str(len(policies)), ", ".join(sorted(policies)))
    table.add_row("Logics", str(len(logics)), ", ".join(sorted(logics)))
    table.add_row("Seeds", str(len(seeds)), ", ".join(sorted(seeds)))
    console.print(table)


def _show_axis_summary(rows: list[dict], axis: str) -> None:
    # Map axis name to CSV column
    col = axis if axis in rows[0] else None
    if col is None:
        console.print(f"[red]Axis '{axis}' not found in CSV columns.[/red]")
        return

    # Group by axis value, compute mean PAR-2
    from collections import defaultdict

    stats: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            stats[r[col]].append(float(r.get("par2", 0)))
        except (ValueError, KeyError):
            continue

    table = Table(title=f"Cross-System Ablation - by {axis}", show_lines=True)
    table.add_column(axis.capitalize())
    table.add_column("Mean PAR-2", justify="right")
    table.add_column("N runs", justify="right")

    for card, values in sorted(stats.items(), key=lambda x: sum(x[1]) / len(x[1])):
        mean = sum(values) / len(values)
        table.add_row(card, f"{mean:.2f}", str(len(values)))

    console.print(table)
