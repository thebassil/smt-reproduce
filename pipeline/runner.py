"""Hydra entry point for cross-system ablation experiments.

Usage:
    # Single reference system run
    python -m pipeline.runner reference=ref_vector seed=42

    # Sweep models
    python -m pipeline.runner -m model=knn,rf,xgboost seed=42,123

    # Override logic
    python -m pipeline.runner data.logic=QF_BV reference=ref_vector
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from pipeline.db import (
    create_experiment,
    init_shard_db,
    load_training_data,
    register_selector,
    update_run_log,
    write_decisions,
)
from pipeline.evaluate import aggregate_folds, cross_validate
from pipeline.types import FoldResult
from pipeline.validate import validate_pipeline

log = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _resolve_config_id_map(db_path: str, portfolio_id: int, config_names: list[str]) -> dict[str, int]:
    """Build config_name -> config_id map from DB for writing decisions."""
    import sqlite3
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """SELECT c.name, c.id
           FROM configs c
           JOIN portfolio_configs pc ON pc.config_id = c.id
           WHERE pc.portfolio_id = ?""",
        (portfolio_id,),
    ).fetchall()
    con.close()
    return {name: cid for name, cid in rows}


def _resolve_instance_id_map(db_path: str, file_paths: list[str]) -> dict[str, int]:
    """Build file_path -> instance_id map from DB for writing decisions."""
    import sqlite3
    con = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in file_paths)
    rows = con.execute(
        f"SELECT file_path, id FROM instances WHERE file_path IN ({placeholders})",
        file_paths,
    ).fetchall()
    con.close()
    return {fp: iid for fp, iid in rows}


def _decisions_to_db_rows(
    decisions: list,
    config_id_map: dict[str, int],
    instance_id_map: dict[str, int],
    config_names: list[str],
) -> list[dict]:
    """Convert Decision objects to dicts for write_decisions()."""
    rows = []
    for d in decisions:
        inst_db_id = instance_id_map.get(d.instance_id)
        if inst_db_id is None:
            continue

        # Determine selected config
        if d.decision_type == "select":
            selected = d.selected_config
        elif d.decision_type == "schedule" and d.schedule:
            selected = d.schedule[0][0]
        elif d.decision_type == "rank" and d.ranking:
            selected = d.ranking[0]
        else:
            continue

        config_db_id = config_id_map.get(selected)
        if config_db_id is None:
            continue

        rows.append({
            "instance_id": inst_db_id,
            "selected_config_id": config_db_id,
            "confidence": d.confidence,
            "confidence_scores": None,
        })
    return rows


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    """Main entry point: load data, build pipeline, run k-fold CV, write results."""
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed = cfg.seed
    _seed_everything(seed)

    # --- 1. Load data from DB ---
    db_path = cfg.data.db_path
    portfolio_id = cfg.data.portfolio_id
    logic = cfg.data.logic
    timeout_s = cfg.data.timeout_s

    log.info("Loading data: portfolio=%d, logic=%s", portfolio_id, logic)
    instance_ids, cost_matrix, config_names = load_training_data(db_path, portfolio_id, logic)
    log.info("Loaded %d instances, %d configs", len(instance_ids), len(config_names))

    if len(instance_ids) == 0:
        log.error("No instances found for portfolio=%d, logic=%s", portfolio_id, logic)
        return float("inf")

    # --- 2. Instantiate cards via Hydra ---
    # Cards may be at top level (CLI override) or nested under reference group
    feat_cfg = cfg.get("featuriser", None) or cfg.reference.featuriser
    model_cfg = cfg.get("model", None) or cfg.reference.model
    policy_cfg = cfg.get("policy", None) or cfg.reference.policy

    log.info("Instantiating pipeline: featuriser=%s, model=%s, policy=%s",
             feat_cfg._target_, model_cfg._target_, policy_cfg._target_)

    featuriser = instantiate(feat_cfg)
    model = instantiate(model_cfg)
    policy = instantiate(policy_cfg)

    # --- 3. Validate compatibility ---
    validate_pipeline(featuriser, model, policy)

    # Build SystemPipeline
    from pipeline.compose import SystemPipeline
    pipeline = SystemPipeline(featuriser, model, policy)

    # --- 4. K-fold cross-validation ---
    n_folds = cfg.get("k_folds", 5)
    log.info("Starting %d-fold CV with seed=%d", n_folds, seed)

    t0 = time.perf_counter()
    fold_results = cross_validate(
        pipeline=pipeline,
        instance_ids=instance_ids,
        cost_matrix=cost_matrix,
        config_names=config_names,
        n_folds=n_folds,
        seed=seed,
        timeout_s=timeout_s,
    )
    total_time = time.perf_counter() - t0

    # --- 5. Aggregate and report ---
    agg = aggregate_folds(fold_results)
    agg["total_time_s"] = total_time
    agg["seed"] = seed
    agg["logic"] = logic

    log.info("=== Results ===")
    log.info("  Mean PAR-2: %.2f (std %.2f)", agg["mean_par2"], agg["std_par2"])
    log.info("  Mean solved: %.1f%% (std %.1f%%)", agg["mean_solved_pct"], agg["std_solved_pct"])
    log.info("  Mean VBS closeness: %.1f%%", agg["mean_closeness_to_vbs_pct"])
    log.info("  Total wall time: %.1fs", total_time)

    # --- 5b. Write CSV results (always, independent of DB) ---
    # Use env vars from sbatch (config names) if available, else fall back to class names
    feat_name = os.environ.get("CARD_FEAT") or feat_cfg._target_.rsplit(".", 1)[-1]
    model_name = os.environ.get("CARD_MODEL") or model_cfg._target_.rsplit(".", 1)[-1]
    policy_name = os.environ.get("CARD_POLICY") or policy_cfg._target_.rsplit(".", 1)[-1]
    ref_code = cfg.reference.get("ref_code", "unknown")
    sweep_tier = os.environ.get("SWEEP_TIER", "unknown")
    cluster = os.environ.get("CLUSTER", "dcs")
    slurm_job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID", "0")

    csv_dir = os.path.join(os.getcwd(), "results_csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{sweep_tier}__{feat_name}__{model_name}__{policy_name}__{logic}__seed{seed}.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "ref", "cluster", "featuriser", "model", "policy", "logic", "seed", "fold",
                     "par2", "solved_pct", "vbs_gap", "train_time_s", "predict_time_s",
                     "n_train", "n_test", "slurm_job", "slurm_task"])
        for fr in fold_results:
            w.writerow([
                sweep_tier, ref_code, cluster,
                feat_name, model_name, policy_name, logic, seed, fr.fold_id,
                f"{fr.metrics.get('par2', 0):.2f}",
                f"{fr.metrics.get('solved_pct', 0):.1f}",
                f"{fr.metrics.get('vbs_gap', 0):.4f}",
                f"{fr.metrics.get('train_time_s', 0):.1f}",
                f"{fr.metrics.get('predict_time_s', 0):.1f}",
                fr.metrics.get("n_train", 0),
                fr.metrics.get("n_test", 0),
                slurm_job, slurm_task,
            ])
        # Summary row
        w.writerow([
            sweep_tier, ref_code, cluster,
            feat_name, model_name, policy_name, logic, seed, "MEAN",
            f"{agg['mean_par2']:.2f}", f"{agg['mean_solved_pct']:.1f}",
            f"{agg.get('mean_closeness_to_vbs_pct', 0):.4f}",
            f"{total_time:.1f}", "", "", "", slurm_job, slurm_task,
        ])
    log.info("  CSV written: %s", csv_path)

    # --- 6. Write to DB (if experiment_id provided) ---
    experiment_id = cfg.get("experiment_id")
    if experiment_id is not None:
        # Shard mode: write to separate shard DB, read from main DB
        write_db = cfg.data.get("shard_db") or cfg.data.db_path
        if write_db and write_db != cfg.data.db_path:
            init_shard_db(write_db)
            log.info("Shard mode: reading from %s, writing to %s", cfg.data.db_path, write_db)
            # Copy the experiment row into the shard so register_selector can find it
            _copy_experiment_to_shard(cfg.data.db_path, write_db, experiment_id)
        _write_to_db(cfg, experiment_id, fold_results, config_names, agg, write_db=write_db)
    else:
        log.info("No experiment_id — skipping DB write. Pass +experiment_id=N to persist.")

    # Return mean PAR-2 as Hydra objective (for sweeps)
    return agg["mean_par2"]


def _copy_experiment_to_shard(read_db: str, shard_db: str, experiment_id: int) -> None:
    """Copy a single experiment row from main DB into shard DB."""
    import sqlite3
    src = sqlite3.connect(read_db)
    cur = src.execute(
        "SELECT * FROM ablation_experiments WHERE id = ?", (experiment_id,)
    )
    row = cur.fetchone()
    if row is None:
        src.close()
        return
    cols = [d[0] for d in cur.description]
    src.close()
    placeholders = ",".join("?" for _ in cols)
    col_names = ",".join(cols)
    dst = sqlite3.connect(shard_db)
    dst.execute(f"INSERT OR IGNORE INTO ablation_experiments ({col_names}) VALUES ({placeholders})", row)
    dst.commit()
    dst.close()


def _write_to_db(
    cfg: DictConfig,
    experiment_id: int,
    fold_results: list[FoldResult],
    config_names: list[str],
    agg: dict,
    write_db: str | None = None,
) -> None:
    """Persist fold results to the ablation DB tables.

    Reads (config/instance lookups) always come from cfg.data.db_path.
    Writes go to write_db (defaults to cfg.data.db_path).
    """
    read_db = cfg.data.db_path
    write_db = write_db or read_db
    portfolio_id = cfg.data.portfolio_id
    seed = cfg.seed
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    config_id_map = _resolve_config_id_map(read_db, portfolio_id, config_names)

    for fr in fold_results:
        test_paths = [d.instance_id for d in fr.decisions]
        instance_id_map = _resolve_instance_id_map(read_db, test_paths)

        try:
            selector_id = register_selector(write_db, experiment_id, fr.fold_id, seed)

            db_rows = _decisions_to_db_rows(
                fr.decisions, config_id_map, instance_id_map, config_names
            )
            write_decisions(write_db, selector_id, db_rows)

            update_run_log(
                write_db,
                experiment_id=experiment_id,
                fold=fr.fold_id,
                seed=seed,
                status="completed",
                wall_time_s=fr.metrics.get("train_time_s", 0) + fr.metrics.get("predict_time_s", 0),
                slurm_job_id=slurm_job_id,
            )
            log.info("  Wrote fold %d: selector=%d, %d decisions", fr.fold_id, selector_id, len(db_rows))

        except Exception as e:
            log.error("  Failed to write fold %d: %s", fr.fold_id, e)
            update_run_log(
                write_db,
                experiment_id=experiment_id,
                fold=fr.fold_id,
                seed=seed,
                status="failed",
                error_message=str(e),
                slurm_job_id=slurm_job_id,
            )


if __name__ == "__main__":
    main()
