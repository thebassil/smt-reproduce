#!/usr/bin/env python3
"""Aggregate ablation results for SMTGazer experiments."""

import csv
import json
import os
import sqlite3
import glob
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Paths
ABLATION_RESULTS = Path("/dcs/large/u5573765/ablation_results/smtgazer")
ABLATION_META = Path("/dcs/23/u5573765/cs351/_system_design/project-documents/scratch-pad/final-report/data/02_subsystem_ablations/smtgazer")
DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")
REFERENCE_YAML = ABLATION_META / "reference.yaml"

TIMEOUT_S = 60
PAR2_PENALTY = 120.0
PORTFOLIO_ID = 7
SUITE_NAME = "suite_9k"
CONFIG_IDS = [109, 112, 114, 115, 116, 118]
SOLVER_LIST = [
    "cvc5::cvc5_qfnra_05_nl_cov",
    "cvc5::cvc5_qfnra_08_decision_just",
    "z3::z3_baseline_default",
    "cvc5::cvc5_baseline_default",
    "z3::z3_baseline_qflia_case_split",
    "z3::z3_baseline_qfbv_sat_euf_sat",
]
LOGICS = ["QF_BV", "QF_LIA", "QF_NRA"]
INCOMPLETE = {"portfolio_size_5", "portfolio_size_6", "smac_n_trials_200"}


def load_db_runs():
    """Load all run data from DB for PAR-2 computation."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT i.family, i.file_name, i.logic, r.config_id, r.status, r.runtime_ms
        FROM runs r JOIN instances i ON r.instance_id = i.id
        WHERE r.portfolio_id = ? AND i.suite_name = ?
          AND r.config_id IN ({})
    """.format(",".join("?" * len(CONFIG_IDS))),
    (PORTFOLIO_ID, SUITE_NAME, *CONFIG_IDS)).fetchall()
    conn.close()

    data = {}
    for family, file_name, logic, cid, status, runtime_ms in rows:
        key = f"{family}/{file_name}"
        if key not in data:
            data[key] = {"logic": logic, "runs": {}}
        data[key]["runs"][cid] = (status, runtime_ms)
    return data


def compute_par2(status, runtime_ms):
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return PAR2_PENALTY


def evaluate_predictions(predictions, db_data):
    """Compute avg_par2 and solved_pct from test predictions."""
    par2_values = []
    for key, pred in predictions.items():
        if key not in db_data:
            continue
        bdata = db_data[key]
        if len(bdata["runs"]) < len(CONFIG_IDS):
            continue

        pred_solvers, pred_times = pred
        instance_par2 = PAR2_PENALTY
        cumulative_time = 0

        for solver_name, time_alloc in zip(pred_solvers, pred_times):
            if solver_name in SOLVER_LIST:
                solver_idx = SOLVER_LIST.index(solver_name)
                cid = CONFIG_IDS[solver_idx]
                status, runtime_ms = bdata["runs"][cid]
                runtime_s = runtime_ms / 1000.0
                if status in ('sat', 'unsat') and runtime_s <= time_alloc:
                    instance_par2 = cumulative_time + runtime_s
                    break
            cumulative_time += time_alloc

        par2_values.append(instance_par2)

    if not par2_values:
        return None, None

    avg_par2 = sum(par2_values) / len(par2_values)
    solved_pct = sum(1 for p in par2_values if p < PAR2_PENALTY) / len(par2_values) * 100
    return avg_par2, solved_pct


def process_experiment(exp_code, db_data):
    """Process a single completed experiment."""
    exp_results_dir = ABLATION_RESULTS / exp_code / "workdir" / "output"

    # Find test result files
    test_files = sorted(glob.glob(str(exp_results_dir / "test_result_*_fold0_*.json")))
    if not test_files:
        print(f"  WARNING: No test results for {exp_code}")
        return None, None

    # Merge predictions from all logics
    all_predictions = {}
    for tf in test_files:
        with open(tf) as f:
            preds = json.load(f)
        all_predictions.update(preds)

    return evaluate_predictions(all_predictions, db_data)


def main():
    print("Loading DB data...")
    db_data = load_db_runs()
    print(f"  {len(db_data)} instances loaded")

    # Load reference config
    with open(REFERENCE_YAML) as f:
        ref = yaml.safe_load(f)

    # Identify reference experiment
    ref_values = ref["algorithmic"]

    # Get all experiment dirs
    experiments = sorted([d.name for d in ABLATION_META.iterdir()
                         if d.is_dir() and d.name != "experiment_configs"])

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []
    reference_par2 = None

    # First pass: compute all results
    for exp_code in experiments:
        meta_dir = ABLATION_META / exp_code
        config_path = meta_dir / "config.yaml"
        meta_path = meta_dir / "experiment_meta.json"

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = yaml.safe_load(f)

        ablation_knob = config.get("ablation_knob", "")
        ablation_value = config.get("ablation_value", "")

        if exp_code in INCOMPLETE:
            results.append({
                "experiment_code": exp_code,
                "ablation_knob": ablation_knob,
                "ablation_value": ablation_value,
                "avg_par2": "",
                "solved_pct": "",
                "delta_vs_reference": "",
                "status": "incomplete",
            })
            # Update meta
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["status"] = "incomplete"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            continue

        avg_par2, solved_pct = process_experiment(exp_code, db_data)
        if avg_par2 is None:
            print(f"  TRAIN_ONLY: {exp_code} (no test predictions)")
            results.append({
                "experiment_code": exp_code,
                "ablation_knob": ablation_knob,
                "ablation_value": ablation_value,
                "avg_par2": "",
                "solved_pct": "",
                "delta_vs_reference": "",
                "status": "train_only",
            })
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["status"] = "train_only"
                meta["completed_utc"] = timestamp
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            continue

        print(f"  {exp_code}: avg_par2={avg_par2:.3f}, solved={solved_pct:.1f}%")

        # Check if this is the reference
        is_ref = (
            config["algorithmic"]["cluster_num"] == ref_values["cluster_num"] and
            config["algorithmic"]["portfolio_size"] == ref_values["portfolio_size"] and
            config["algorithmic"]["seed"] == ref_values["seed"] and
            config["algorithmic"]["smac_n_trials"] == ref_values["smac_n_trials"] and
            str(config["algorithmic"]["smac_w1"]) == str(ref_values["smac_w1"]) and
            config["algorithmic"]["kmeans_n_init"] == ref_values["kmeans_n_init"] and
            config["algorithmic"]["smac_internal_cv_splits"] == ref_values["smac_internal_cv_splits"]
        )

        results.append({
            "experiment_code": exp_code,
            "ablation_knob": ablation_knob,
            "ablation_value": ablation_value,
            "avg_par2": avg_par2,
            "solved_pct": solved_pct,
            "delta_vs_reference": "",  # fill later
            "status": "completed",
            "_is_ref": is_ref,
        })

        # Update experiment_meta.json
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["status"] = "completed"
            meta["completed_utc"] = timestamp
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Write results_summary.json
        summary = {
            "experiment_code": exp_code,
            "ablation_knob": ablation_knob,
            "ablation_value": ablation_value,
            "avg_par2": avg_par2,
            "solved_pct": solved_pct,
            "status": "completed",
        }
        with open(meta_dir / "results_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Find reference PAR-2
    for r in results:
        if r.get("_is_ref"):
            reference_par2 = r["avg_par2"]
            print(f"\nReference experiment PAR-2: {reference_par2:.3f}")
            break

    if reference_par2 is None:
        # Try multiple candidates - reference matches multiple experiments
        # (cluster_num_10, seed_42, smac_n_trials_50, etc.)
        # Use cluster_num_10 as canonical reference
        for r in results:
            if r["experiment_code"] == "cluster_num_10":
                reference_par2 = r["avg_par2"]
                print(f"\nUsing cluster_num_10 as reference: {reference_par2:.3f}")
                break

    # Compute deltas
    for r in results:
        if r.get("_is_ref") is not None:
            del r["_is_ref"]
        if r["status"] == "completed" and reference_par2 is not None:
            r["delta_vs_reference"] = r["avg_par2"] - reference_par2

    # Write master CSV
    csv_path = ABLATION_META / "ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_code", "ablation_knob", "ablation_value",
            "avg_par2", "solved_pct", "delta_vs_reference", "status"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")
    print(f"Total: {sum(1 for r in results if r['status']=='completed')} completed, "
          f"{sum(1 for r in results if r['status']=='incomplete')} incomplete")


if __name__ == "__main__":
    main()
