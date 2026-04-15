from __future__ import annotations
"""Layer 4: Re-run solvers on benchmarks to regenerate ground-truth."""

from typing import Optional

from smt.display import console


def run_ground_truth(
    backend: str = "local",
    db_override: Optional[str] = None,
) -> None:
    console.print("[bold yellow]Ground-truth reproduction[/bold yellow]")
    console.print()
    console.print(
        "This re-executes Z3 and CVC5 on all benchmark instances "
        "across 118 solver configurations."
    )
    console.print(
        "[red]WARNING:[/red] This requires ~150 CPU-hours "
        "(9,500 instances x 118 configs x 60s timeout)."
    )
    console.print()
    console.print("Prerequisites:")
    console.print("  - Z3 binary:  set SMT_Z3 env var or install z3")
    console.print("  - CVC5 binary: set SMT_CVC5 env var or install cvc5")
    console.print("  - Benchmark suite: run `smt download --benchmarks`")
    console.print()
    console.print("[dim]Not yet implemented. Use `--inspect` to view existing run data.[/dim]")
