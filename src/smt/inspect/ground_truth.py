from __future__ import annotations
"""Inspect ground-truth solver run data from the database."""

import sqlite3
from pathlib import Path

from rich.table import Table

from smt.config import resolve_db_path
from smt.display import console, section


def inspect_ground_truth(db_override: str | None = None) -> None:
    db = resolve_db_path(db_override)
    if not db.exists():
        console.print(f"[red]Database not found:[/red] {db}")
        console.print("Run `smt download` to fetch the database.")
        return

    section("Ground-Truth Solver Runs")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    # Overall stats
    total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    total_instances = conn.execute("SELECT COUNT(*) FROM instances").fetchone()[0]
    total_configs = conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0]

    table = Table(title="Database Overview")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total runs", f"{total_runs:,}")
    table.add_row("Total instances", f"{total_instances:,}")
    table.add_row("Total configs", f"{total_configs:,}")
    console.print(table)

    # Per-logic breakdown
    rows = conn.execute("""
        SELECT i.logic,
               COUNT(DISTINCT i.id) as n_instances,
               COUNT(r.id) as n_runs,
               SUM(CASE WHEN r.status = 'sat' OR r.status = 'unsat' THEN 1 ELSE 0 END) as solved,
               ROUND(AVG(r.runtime_ms), 1) as avg_runtime_ms
        FROM instances i
        JOIN runs r ON r.instance_id = i.id
        GROUP BY i.logic
        ORDER BY i.logic
    """).fetchall()

    if rows:
        table = Table(title="Per-Logic Breakdown", show_lines=True)
        table.add_column("Logic")
        table.add_column("Instances", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Solved", justify="right")
        table.add_column("Solve Rate", justify="right")
        table.add_column("Avg Runtime (ms)", justify="right")
        for r in rows:
            rate = f"{r['solved'] / r['n_runs'] * 100:.1f}%" if r["n_runs"] > 0 else "-"
            table.add_row(
                r["logic"],
                f"{r['n_instances']:,}",
                f"{r['n_runs']:,}",
                f"{r['solved']:,}",
                rate,
                str(r["avg_runtime_ms"]),
            )
        console.print(table)

    # Portfolio summary
    rows = conn.execute("""
        SELECT p.id, p.name, p.timeout_s,
               COUNT(pc.config_id) as n_configs
        FROM portfolios p
        LEFT JOIN portfolio_configs pc ON pc.portfolio_id = p.id
        GROUP BY p.id
        ORDER BY p.id
    """).fetchall()

    if rows:
        table = Table(title="Portfolios", show_lines=True)
        table.add_column("ID", justify="right")
        table.add_column("Name")
        table.add_column("Timeout (s)", justify="right")
        table.add_column("Configs", justify="right")
        for r in rows:
            table.add_row(str(r["id"]), r["name"], str(r["timeout_s"]), str(r["n_configs"]))
        console.print(table)

    conn.close()
