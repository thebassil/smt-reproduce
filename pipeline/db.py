"""Data I/O layer for cross-system ablation experiments."""

import json
import sqlite3
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _connect_with_retry(db_path: str, max_retries: int = 10) -> sqlite3.Connection:
    """Connect to SQLite with retry on NFS locking errors."""
    for attempt in range(max_retries):
        try:
            con = sqlite3.connect(db_path, timeout=30)
            con.execute("PRAGMA journal_mode=WAL")
            return con
        except sqlite3.OperationalError:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 5))
            else:
                raise


def _execute_with_retry(fn, max_retries: int = 10):
    """Retry a DB operation on locking/IO errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1 and ("locked" in str(e) or "I/O" in str(e) or "locking" in str(e)):
                time.sleep(random.uniform(1, 5))
            else:
                raise


# ---------------------------------------------------------------------------
# Schema for shard DBs (written by each SLURM array task)
# ---------------------------------------------------------------------------

SHARD_SCHEMA = """
CREATE TABLE IF NOT EXISTS ablation_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_card TEXT NOT NULL,
    featuriser_card TEXT NOT NULL,
    model_card TEXT NOT NULL,
    policy_card TEXT NOT NULL,
    portfolio_id INTEGER NOT NULL,
    suite_name TEXT NOT NULL,
    timeout_s INTEGER NOT NULL DEFAULT 60,
    cv_folds INTEGER NOT NULL DEFAULT 5,
    seeds TEXT NOT NULL DEFAULT '[42, 123, 456]',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    logic TEXT NOT NULL DEFAULT '',
    description TEXT,
    created_utc TEXT
);

CREATE TABLE IF NOT EXISTS ml_selectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model_type TEXT,
    portfolio_id INTEGER,
    model_path TEXT,
    training_info TEXT,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    selector_id INTEGER NOT NULL,
    instance_id INTEGER NOT NULL,
    selected_config_id INTEGER NOT NULL,
    step_num INTEGER DEFAULT 1,
    confidence REAL,
    confidence_scores TEXT,
    ts_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ablation_run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    selector_id INTEGER,
    fold INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    wall_time_s REAL,
    error_message TEXT,
    slurm_job_id TEXT,
    updated_utc TEXT
);
"""


def init_shard_db(shard_path: str) -> None:
    """Create a shard DB with the ablation schema (idempotent)."""
    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(shard_path)
    con.executescript(SHARD_SCHEMA)
    con.close()


def load_training_data(
    db_path: str, portfolio_id: int, logic: str
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load PAR-2 cost matrix for a portfolio/logic pair.

    Returns:
        instance_ids: list of file_path strings (length n_instances)
        cost_matrix:  ndarray shape (n_instances, n_configs) of PAR-2 values
        config_names: list of config name strings (length n_configs)
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    timeout_s = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (portfolio_id,)
    ).fetchone()[0]

    # Get ordered config list for this portfolio
    configs = cur.execute(
        """SELECT c.id, c.name
           FROM configs c
           JOIN portfolio_configs pc ON pc.config_id = c.id
           WHERE pc.portfolio_id = ?
           ORDER BY c.name""",
        (portfolio_id,),
    ).fetchall()
    config_ids = [c[0] for c in configs]
    config_names = [c[1] for c in configs]
    config_idx = {cid: i for i, cid in enumerate(config_ids)}

    # Get all runs for this portfolio + logic
    rows = cur.execute(
        """SELECT i.file_path, r.config_id, r.status, r.runtime_ms
           FROM runs r
           JOIN instances i ON r.instance_id = i.id
           WHERE r.portfolio_id = ?
             AND i.logic = ?
           ORDER BY i.file_path""",
        (portfolio_id, logic),
    ).fetchall()
    con.close()

    # Build instance list (preserving order)
    seen = {}
    instance_ids = []
    for file_path, _, _, _ in rows:
        if file_path not in seen:
            seen[file_path] = len(instance_ids)
            instance_ids.append(file_path)

    n_inst = len(instance_ids)
    n_cfg = len(config_names)
    cost_matrix = np.full((n_inst, n_cfg), 2.0 * timeout_s)

    for file_path, config_id, status, runtime_ms in rows:
        if config_id not in config_idx:
            continue
        i = seen[file_path]
        j = config_idx[config_id]
        if status in ("sat", "unsat"):
            cost_matrix[i, j] = runtime_ms / 1000.0
        # else: stays at 2*timeout (PAR-2 penalty)

    return instance_ids, cost_matrix, config_names


def register_selector(
    db_path: str,
    experiment_id: int,
    fold: int,
    seed: int,
    model_path: Optional[str] = None,
) -> int:
    """Register a new ML selector and ablation_run_log entry for an experiment fold/seed.

    Returns:
        selector_id: the new ml_selectors.id
    """
    con = _connect_with_retry(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    # Look up experiment details
    exp = cur.execute(
        """SELECT portfolio_card, featuriser_card, model_card, policy_card, portfolio_id
           FROM ablation_experiments WHERE id = ?""",
        (experiment_id,),
    ).fetchone()
    if exp is None:
        con.close()
        raise ValueError(f"Experiment {experiment_id} not found")

    portfolio_card, featuriser_card, model_card, policy_card, portfolio_id = exp

    experiment_code = f"exp{experiment_id}"
    selector_name = f"{experiment_code}_fold{fold}_seed{seed}"

    training_info = json.dumps({
        "portfolio_card": portfolio_card,
        "featuriser_card": featuriser_card,
        "model_card": model_card,
        "policy_card": policy_card,
        "fold": fold,
        "seed": seed,
    })

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cur.execute(
        """INSERT INTO ml_selectors (name, model_type, portfolio_id, model_path, training_info, created_utc)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (selector_name, model_card, portfolio_id, model_path, training_info, now_utc),
    )
    selector_id = cur.lastrowid

    cur.execute(
        """INSERT INTO ablation_run_log (experiment_id, selector_id, fold, seed, status)
           VALUES (?, ?, ?, ?, 'pending')""",
        (experiment_id, selector_id, fold, seed),
    )

    con.commit()
    con.close()
    return selector_id


def write_decisions(
    db_path: str,
    selector_id: int,
    instance_decisions: List[dict],
) -> None:
    """Batch-insert decisions for a selector.

    Each dict in instance_decisions must have:
        instance_id, selected_config_id, confidence, confidence_scores
    """
    def _do_write():
        con = _connect_with_retry(db_path)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        rows = []
        for d in instance_decisions:
            scores = d.get("confidence_scores")
            if isinstance(scores, dict):
                scores = json.dumps(scores)
            rows.append((
                selector_id,
                d["instance_id"],
                d["selected_config_id"],
                1,  # step_num
                d.get("confidence"),
                scores,
                now_utc,
            ))

        con.executemany(
            """INSERT INTO decisions
               (selector_id, instance_id, selected_config_id, step_num, confidence, confidence_scores, ts_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        con.commit()
        con.close()

    _execute_with_retry(_do_write)


def create_experiment(
    db_path: str,
    portfolio_card: str,
    featuriser_card: str,
    model_card: str,
    policy_card: str,
    portfolio_id: int,
    logic: str,
    suite_name: str,
    description: Optional[str] = None,
) -> int:
    """Create an ablation_experiments row and return its id."""
    con = sqlite3.connect(db_path)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    cur = con.execute(
        """INSERT INTO ablation_experiments
           (portfolio_card, featuriser_card, model_card, policy_card,
            portfolio_id, logic, suite_name, description, created_utc,
            timeout_s, cv_folds, seeds, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 60, 5, '[42, 123, 456]', 'pending')""",
        (portfolio_card, featuriser_card, model_card, policy_card,
         portfolio_id, logic, suite_name, description, now_utc),
    )
    experiment_id = cur.lastrowid
    con.commit()
    con.close()
    return experiment_id


def update_run_log(
    db_path: str,
    experiment_id: int,
    fold: int,
    seed: int,
    status: str,
    wall_time_s: Optional[float] = None,
    error_message: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
) -> None:
    """Update the status of an ablation_run_log entry."""
    def _do_update():
        con = _connect_with_retry(db_path)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        con.execute(
            """UPDATE ablation_run_log
               SET status = ?, wall_time_s = ?, error_message = ?,
                   slurm_job_id = ?, updated_utc = ?
               WHERE experiment_id = ? AND fold = ? AND seed = ?""",
            (status, wall_time_s, error_message, slurm_job_id, now_utc,
             experiment_id, fold, seed),
        )
        con.commit()
        con.close()

    _execute_with_retry(_do_update)


def compute_par2_score(
    runtimes: np.ndarray, statuses: List[str], timeout: float
) -> float:
    """Compute aggregate PAR-2 score.

    Args:
        runtimes: array of runtime values (seconds)
        statuses: list of status strings ('sat', 'unsat', 'timeout', etc.)
        timeout: timeout in seconds

    Returns:
        Sum of PAR-2 penalised runtimes.
    """
    solved_mask = np.array([s in ("sat", "unsat") for s in statuses])
    return float(np.where(solved_mask, runtimes, 2.0 * timeout).sum())
