#!/usr/bin/env python3
"""
Shared DB helper utilities for ML selector dual-write (DB + CSV).

Used by sibyl_portfolio_trainer.py and sibyl_evaluate_decisions.py.
Batch-loads instance and config mappings upfront to avoid per-row queries.
"""

import json
import sqlite3
from datetime import datetime, timezone


def get_utc_timestamp():
    """Return current UTC time as ISO format string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_instance_map(db_path):
    """Return {file_path: instance_id} for all instances."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, file_path FROM instances").fetchall()
    conn.close()
    return {file_path: id_ for id_, file_path in rows}


def get_config_map(db_path):
    """Return {config_name: config_id} for all configs."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, name FROM configs").fetchall()
    conn.close()
    return {name: id_ for id_, name in rows}


def get_solver_config_map(db_path):
    """Return {"solver::config_name": config_id} matching CSV format."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, solver, name FROM configs").fetchall()
    conn.close()
    return {f"{solver}::{name}": id_ for id_, solver, name in rows}


def register_ml_selector(db_path, name, model_type, portfolio_id,
                         model_path=None, training_info=None):
    """
    Register or retrieve an ML selector (idempotent).

    Returns the selector_id. If a selector with the same name exists,
    returns its existing id without modification.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    row = cur.execute(
        "SELECT id FROM ml_selectors WHERE name = ?", (name,)
    ).fetchone()

    if row:
        conn.close()
        return row[0]

    training_json = json.dumps(training_info) if training_info else None
    cur.execute(
        """INSERT INTO ml_selectors (name, model_type, portfolio_id,
           model_path, training_info, created_utc)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (name, model_type, portfolio_id, model_path,
         training_json, get_utc_timestamp())
    )
    conn.commit()
    selector_id = cur.lastrowid
    conn.close()
    return selector_id


def write_decisions_to_db(db_path, selector_id, decisions, instance_map,
                          config_map):
    """
    Batch-write decisions to the DB.

    Args:
        decisions: list of dicts with keys:
            benchmark (file_path), predicted_config (config_name),
            confidence (float, optional), confidence_scores (dict, optional)

    Returns (written, skipped) counts.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = get_utc_timestamp()

    written = 0
    skipped = 0

    for d in decisions:
        instance_id = instance_map.get(d['benchmark'])
        if instance_id is None:
            skipped += 1
            continue

        config_id = config_map.get(d['predicted_config'])
        if config_id is None:
            skipped += 1
            continue

        confidence = d.get('confidence')
        conf_scores = d.get('confidence_scores')
        conf_json = json.dumps(conf_scores) if conf_scores else None

        cur.execute(
            """INSERT OR REPLACE INTO decisions
               (selector_id, instance_id, selected_config_id, step_num,
                confidence, confidence_scores, ts_utc)
               VALUES (?, ?, ?, 1, ?, ?, ?)""",
            (selector_id, instance_id, config_id, confidence, conf_json, ts)
        )
        written += 1

    conn.commit()
    conn.close()
    return written, skipped


def load_decisions_from_db(db_path, selector_name):
    """
    Load decisions from DB for a given selector name.

    Returns {file_path: config_name} dict.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """SELECT i.file_path, c.name
           FROM decisions d
           JOIN ml_selectors s ON d.selector_id = s.id
           JOIN instances i ON d.instance_id = i.id
           JOIN configs c ON d.selected_config_id = c.id
           WHERE s.name = ?""",
        (selector_name,)
    ).fetchall()
    conn.close()
    return {file_path: config_name for file_path, config_name in rows}
