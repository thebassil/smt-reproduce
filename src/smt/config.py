from __future__ import annotations
"""Path resolution and environment detection."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DCS_DB = Path("/dcs/large/u5573765/db/results.sqlite")
DCS_DATA = Path("/dcs/large/u5573765/data")
DCS_BENCHMARKS = Path("/dcs/large/u5573765/data/benchmarks")

SYSTEMS = ["machsmt", "sibyl", "grackle", "medley", "smtgazer"]
ABLATION_AXES = ["model", "featuriser", "policy", "portfolio"]


def is_dcs() -> bool:
    return DCS_DB.exists()


def resolve_db_path(cli_override: str | None = None) -> Path:
    if cli_override:
        return Path(cli_override)
    env = os.environ.get("SMT_DB_PATH")
    if env:
        return Path(env)
    if is_dcs():
        return DCS_DB
    local = REPO_ROOT / "data" / "results.sqlite"
    if local.exists():
        return local
    return local  # will be created by `smt download`


def resolve_final_data() -> Path:
    return REPO_ROOT / "final_data"


def resolve_systems_dir() -> Path:
    return REPO_ROOT / "systems"


def resolve_ablations_dir() -> Path:
    return REPO_ROOT / "ablations"


def data_status() -> dict:
    db = resolve_db_path()
    final = resolve_final_data()
    return {
        "db_present": db.exists(),
        "db_path": str(db),
        "db_size_mb": round(db.stat().st_size / 1e6, 1) if db.exists() else 0,
        "final_data_present": final.exists(),
        "systems_present": {
            s: (final / "01_system_reproductions" / s).exists()
            for s in SYSTEMS
        },
        "subsystem_present": {
            s: (final / "02_subsystem_ablations" / s).exists()
            for s in SYSTEMS
        },
        "cross_system_present": (
            final / "03_cross_system_ablations" / "core_cross_system_ablation.csv"
        ).exists(),
        "extended_present": (
            final / "04_cross_system_ablations_extended" / "extended_runs"
        ).exists(),
        "environment": "DCS" if is_dcs() else "standalone",
    }
