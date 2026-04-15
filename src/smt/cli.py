"""SMT Reproduce CLI - Reproducibility tool for SMT Algorithm Selection Dissertation."""

import click

from smt.config import SYSTEMS, ABLATION_AXES, data_status, resolve_db_path
from smt.display import print_status, console


@click.group()
@click.option(
    "--backend",
    type=click.Choice(["local", "slurm", "docker"]),
    default="local",
    help="Execution backend for compute tasks.",
)
@click.option("--db", default=None, help="Override database path.")
@click.pass_context
def smt(ctx, backend, db):
    """SMT Algorithm Selection - Dissertation Reproducibility Tool."""
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend
    ctx.obj["db"] = db


@smt.command()
def status():
    """Show what data and results are available locally."""
    print_status(data_status())


@smt.command()
@click.option("--db-only", is_flag=True, help="Only download the database.")
@click.option(
    "--benchmarks", is_flag=True, help="Also download benchmark suite (large)."
)
def download(db_only, benchmarks):
    """Download pre-computed database and/or benchmarks from GitHub release."""
    from smt.download import download_data
    download_data(db_only=db_only, benchmarks=benchmarks)


@smt.group()
@click.pass_context
def reproduce(ctx):
    """Reproduce experimental results."""
    pass


@reproduce.command("systems")
@click.option("--inspect", "inspect_", is_flag=True, help="Show pre-computed results only.")
@click.option(
    "--system",
    type=click.Choice(SYSTEMS),
    default=None,
    help="Run/inspect a single system.",
)
@click.pass_context
def systems(ctx, inspect_, system):
    """Reproduce or inspect the 5 selector system evaluations."""
    if inspect_:
        from smt.inspect.systems import inspect_systems
        inspect_systems(system)
    else:
        from smt.layers.systems import run_systems
        run_systems(
            system=system,
            backend=ctx.obj["backend"],
            db_override=ctx.obj["db"],
        )


@reproduce.command("subsystem")
@click.option("--inspect", "inspect_", is_flag=True, help="Show pre-computed results only.")
@click.option(
    "--system",
    type=click.Choice(SYSTEMS),
    default=None,
    help="Run/inspect ablations for a single system.",
)
@click.pass_context
def subsystem(ctx, inspect_, system):
    """Reproduce or inspect per-system ablation sweeps."""
    if inspect_:
        from smt.inspect.subsystem import inspect_subsystem
        inspect_subsystem(system)
    else:
        from smt.layers.subsystem import run_subsystem
        run_subsystem(
            system=system,
            backend=ctx.obj["backend"],
            db_override=ctx.obj["db"],
        )


@reproduce.command("cross-system")
@click.option("--inspect", "inspect_", is_flag=True, help="Show pre-computed results only.")
@click.option(
    "--axis",
    type=click.Choice(ABLATION_AXES),
    default=None,
    help="Filter by ablation axis.",
)
@click.pass_context
def cross_system(ctx, inspect_, axis):
    """Reproduce or inspect the cross-system ablation sweep."""
    if inspect_:
        from smt.inspect.cross_system import inspect_cross_system
        inspect_cross_system(axis)
    else:
        from smt.layers.cross_system import run_cross_system
        run_cross_system(
            axis=axis,
            backend=ctx.obj["backend"],
            db_override=ctx.obj["db"],
        )


@reproduce.command("ground-truth")
@click.option("--inspect", "inspect_", is_flag=True, help="Show database statistics.")
@click.pass_context
def ground_truth(ctx, inspect_):
    """Re-run solvers on benchmarks (expensive) or inspect run data."""
    if inspect_:
        from smt.inspect.ground_truth import inspect_ground_truth
        inspect_ground_truth(ctx.obj["db"])
    else:
        from smt.layers.ground_truth import run_ground_truth
        run_ground_truth(
            backend=ctx.obj["backend"],
            db_override=ctx.obj["db"],
        )
