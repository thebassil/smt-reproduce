#!/usr/bin/env python3
"""
Ablation experiment runner.

Orchestrates per-value: create dirs → freeze config → generate sbatch → submit.
Skips already-completed experiments (checks for kfold_k5_summary.csv).
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ablation.config import (
    experiment_code,
    get_wall_time,
    load_reference,
    load_template,
    merge_config,
    validate_one_at_a_time,
)
from ablation.sbatch_template import generate_sbatch


def freeze_config(config, exp_code, knob, value, results_dir):
    """Write a frozen config.yaml for one experiment."""
    frozen = {
        "experiment_code": exp_code,
        "ablation_knob": knob,
        "ablation_value": value,
        "algorithmic": {
            "cluster_num": config["cluster_num"],
            "portfolio_size": config["portfolio_size"],
            "seed": config["seed"],
            "smac_n_trials": config["smac_n_trials"],
            "smac_w1": config["smac_w1"],
            "kmeans_n_init": config["kmeans_n_init"],
            "smac_internal_cv_splits": config["smac_internal_cv_splits"],
        },
        "hardware": config["hardware"],
        "convention": config["convention"],
        "paths": config["paths"],
    }
    config_path = results_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(frozen, f, default_flow_style=False, sort_keys=False)
    return config_path


def write_experiment_meta(results_dir, exp_code, job_id=None, status="submitted"):
    """Write experiment_meta.json."""
    meta = {
        "experiment_code": exp_code,
        "status": status,
        "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job_id": job_id,
    }
    meta_path = results_dir / "experiment_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def is_completed(results_dir, k_folds=5):
    """Check if an experiment has already completed (pipeline_complete marker exists)."""
    marker = results_dir / "results" / "pipeline_complete"
    return marker.exists()


def cmd_run(template_path, dry_run=False, values=None, force=False):
    """Run ablation experiments for a template.

    Args:
        template_path: Path to experiment YAML template
        dry_run: If True, validate and print but don't submit
        values: Optional list of specific values to run (None = all)
        force: If True, rerun even if completed
    """
    reference = load_reference()
    template = load_template(template_path)
    knob = template["ablation_knob"]
    ref_value = template["reference_value"]
    all_values = template["values"]

    if values is not None:
        # Parse and filter to requested values
        run_values = []
        for v in values:
            # Try to match type with template values
            matched = False
            for tv in all_values:
                if str(tv) == str(v):
                    run_values.append(tv)
                    matched = True
                    break
            if not matched:
                print(f"WARNING: Value {v} not in template values {all_values}, skipping")
        if not run_values:
            print("ERROR: No valid values to run")
            return 1
    else:
        run_values = all_values

    results_base = Path(reference["paths"]["results_base"])
    k_folds = reference["convention"]["k_folds"]

    print(f"Ablation: {knob}")
    print(f"  Reference value: {ref_value}")
    print(f"  Values to run: {run_values}")
    print(f"  Results base: {results_base}")
    print()

    submitted = 0
    skipped = 0
    errors = 0

    for value in run_values:
        exp_code = experiment_code(knob, value)
        results_dir = results_base / exp_code

        print(f"--- {exp_code} ---")

        # Check if already completed
        if not force and is_completed(results_dir, k_folds):
            print(f"  SKIP: Already completed (use --force to rerun)")
            skipped += 1
            continue

        # Merge config
        config = merge_config(reference, knob, value)

        # Validate one-at-a-time invariant
        diff = validate_one_at_a_time(config, reference)
        if diff is None:
            print(f"  Control point (reference config)")
        else:
            print(f"  Ablating: {diff[0]} = {diff[1]}")

        # Get wall time
        wall_time = get_wall_time(template, value, reference)
        print(f"  Wall time: {wall_time}")

        if dry_run:
            print(f"  DRY RUN: Would create {results_dir}")
            print(f"  DRY RUN: Would submit sbatch with wall_time={wall_time}")
            submitted += 1
            continue

        # Create directories
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "workdir").mkdir(exist_ok=True)
        (results_dir / "logs").mkdir(exist_ok=True)
        (results_dir / "results").mkdir(exist_ok=True)

        # Freeze config
        config_path = freeze_config(config, exp_code, knob, value, results_dir)
        print(f"  Config: {config_path}")

        # Generate sbatch
        sbatch_path = generate_sbatch(exp_code, config, wall_time, results_dir)
        print(f"  Sbatch: {sbatch_path}")

        # Submit
        try:
            result = subprocess.run(
                ["sbatch", str(sbatch_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                # Parse job ID from "Submitted batch job 12345"
                job_id = result.stdout.strip().split()[-1]
                write_experiment_meta(results_dir, exp_code, job_id=job_id)
                print(f"  Submitted: job {job_id}")
                submitted += 1
            else:
                print(f"  ERROR: sbatch failed: {result.stderr.strip()}")
                write_experiment_meta(results_dir, exp_code, status="submit_failed")
                errors += 1
        except FileNotFoundError:
            print(f"  ERROR: sbatch command not found (not on a Slurm node?)")
            write_experiment_meta(results_dir, exp_code, status="submit_failed")
            errors += 1

        print()

    print(f"\nSummary: {submitted} submitted, {skipped} skipped, {errors} errors")
    return 0 if errors == 0 else 1
