"""Rich display utilities for inspect output."""

import csv
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


def print_csv_table(path: Path, title: str, max_rows: int = 50) -> None:
    if not path.exists():
        console.print(f"[red]Not found:[/red] {path}")
        return
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        console.print(f"[yellow]Empty:[/yellow] {path}")
        return

    table = Table(title=title, show_lines=True)
    for col in rows[0]:
        table.add_column(col, overflow="fold")
    for row in rows[:max_rows]:
        table.add_row(*[row[c] for c in rows[0]])
    if len(rows) > max_rows:
        table.caption = f"... showing {max_rows} of {len(rows)} rows"
    console.print(table)


def print_status(status: dict) -> None:
    table = Table(title="SMT Reproduce - Status")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")

    env = status["environment"]
    table.add_row("Environment", f"[cyan]{env}[/cyan]", "")

    ok = "[green]YES[/green]"
    no = "[red]NO[/red]"

    table.add_row(
        "Database",
        ok if status["db_present"] else no,
        f"{status['db_path']} ({status['db_size_mb']} MB)"
        if status["db_present"]
        else status["db_path"],
    )

    for sys, present in status["systems_present"].items():
        table.add_row(f"  System: {sys}", ok if present else no, "")

    table.add_row(
        "Cross-system CSV",
        ok if status["cross_system_present"] else no,
        "",
    )
    table.add_row(
        "Extended ablations",
        ok if status["extended_present"] else no,
        "",
    )
    console.print(table)


def section(title: str) -> None:
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print(f"[bold blue]  {title}[/bold blue]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]\n")
