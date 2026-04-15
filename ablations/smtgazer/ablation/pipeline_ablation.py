#!/usr/bin/env python3
"""
Ablation pipeline for SMTGazer experiments.

Adapted from smtgazer_full_pipeline.py (944 lines) with these changes:
  1. Reads ALL params from a frozen YAML config (no hardcoded constants)
  2. Isolated WORKDIR per experiment
  3. Patches copied artifact scripts for non-CLI params
  4. Symlinks pre-extracted features from main workdir (saves ~13 min)
  5. Runs: setup → patch → train → kfold → CSV output
  6. NO phase_register (no DB writes)

Usage:
  python3 -m ablation.pipeline_ablation /path/to/experiment/config.yaml
"""

import csv
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# =====================================================================
# Config loading
# =====================================================================

def load_experiment_config(config_path):
    """Load the frozen experiment config.yaml and return a flat namespace."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    class Config:
        pass

    c = Config()

    # Algorithmic
    alg = raw["algorithmic"]
    c.cluster_num = alg["cluster_num"]
    c.portfolio_size = alg["portfolio_size"]
    c.seed = alg["seed"]
    c.smac_n_trials = alg["smac_n_trials"]
    c.smac_w1 = alg["smac_w1"]
    c.kmeans_n_init = alg["kmeans_n_init"]
    c.smac_internal_cv_splits = alg["smac_internal_cv_splits"]

    # Hardware
    hw = raw["hardware"]
    c.smac_workers = hw["smac_workers"]
    c.feature_workers = hw["feature_workers"]
    c.cpus_per_task = hw.get("cpus_per_task", 64)

    # Convention
    conv = raw["convention"]
    c.timeout_s = conv["timeout_s"]
    c.par2_penalty = conv["par2_penalty"]
    c.k_folds = conv["k_folds"]
    c.portfolio_id = conv["portfolio_id"]
    c.suite_name = conv["suite_name"]
    c.logics = conv["logics"]
    c.config_ids = conv["config_ids"]
    c.solver_list = conv["solver_list"]

    # Paths
    paths = raw["paths"]
    c.artifact_dir = Path(paths["artifact_dir"])
    c.results_base = Path(paths["results_base"])
    c.main_workdir = Path(paths["main_workdir"])
    c.db_path = Path(paths["db_path"])
    c.folds_path = Path(paths["folds_path"])

    # Experiment metadata
    c.experiment_code = raw.get("experiment_code", "unknown")
    c.ablation_knob = raw.get("ablation_knob", "unknown")
    c.ablation_value = raw.get("ablation_value")

    # Derived paths
    c.experiment_dir = c.results_base / c.experiment_code
    c.workdir = c.experiment_dir / "workdir"
    c.results_dir = c.experiment_dir / "results"

    return c


# =====================================================================
# Utility
# =====================================================================

def compute_par2(status, runtime_ms, par2_penalty):
    if status in ("sat", "unsat"):
        return runtime_ms / 1000.0
    return par2_penalty


def run_command(args, workdir, env, label="command", timeout=7200):
    """Run a command in the workdir with logging."""
    print(f"  [{label}] {' '.join(str(a) for a in args)}")
    t0 = time.time()
    result = subprocess.run(
        [str(a) for a in args],
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    print(f"  [{label}] Finished in {elapsed:.1f}s (rc={result.returncode})")
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-15:]:
            print(f"    {line}")
    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split("\n")[-15:]:
            print(f"    STDERR: {line}")
    return result.returncode == 0


# =====================================================================
# Setup & Patching
# =====================================================================

def setup_workdir(cfg):
    """Create isolated workdir with patched artifact copies and symlinked features."""
    workdir = cfg.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(exist_ok=True)
    (workdir / "machfea" / "infer_result").mkdir(parents=True, exist_ok=True)
    (workdir / "tmp").mkdir(exist_ok=True)
    (workdir / "output").mkdir(exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifact scripts (fresh copies for patching)
    for fname in ["SMTportfolio.py", "portfolio_smac3.py"]:
        dst = workdir / fname
        shutil.copy2(cfg.artifact_dir / fname, dst)

    # Symlink smac directory
    smac_link = workdir / "smac"
    if not smac_link.exists():
        smac_link.symlink_to(cfg.artifact_dir / "smac")

    # Symlink pre-extracted features from main workdir (saves ~13 min)
    for logic in cfg.logics:
        feat_name = f"{logic}_train_feature.json"
        src = cfg.main_workdir / "machfea" / "infer_result" / feat_name
        dst = workdir / "machfea" / "infer_result" / feat_name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    # Copy data files (Labels.json, solver.json, metadata) from main workdir
    for logic in cfg.logics:
        labels_src = cfg.main_workdir / "data" / f"{logic}Labels.json"
        labels_dst = workdir / "data" / f"{logic}Labels.json"
        if labels_src.exists() and not labels_dst.exists():
            shutil.copy2(labels_src, labels_dst)

        solver_src = cfg.main_workdir / "machfea" / f"{logic}_solver.json"
        solver_dst = workdir / "machfea" / f"{logic}_solver.json"
        if solver_src.exists() and not solver_dst.exists():
            shutil.copy2(solver_src, solver_dst)

    meta_src = cfg.main_workdir / "instance_metadata.json"
    meta_dst = workdir / "instance_metadata.json"
    if meta_src.exists() and not meta_dst.exists():
        shutil.copy2(meta_src, meta_dst)

    print(f"  Workdir ready: {workdir}")


def patch_artifacts(cfg):
    """Patch copied artifact scripts for non-CLI parameters.

    Patches:
      SMTportfolio.py:
        - portfolioSize=N (line 383 area, in get_portfolio_3 call)
        - n_init=N (KMeans constructor)
        - range(0,N) for smac_internal_cv_splits

      portfolio_smac3.py:
        - n_trials=N (Scenario constructor)
        - w1 = N (module-level variable)
    """
    workdir = cfg.workdir

    # --- Patch SMTportfolio.py ---
    smt_path = workdir / "SMTportfolio.py"
    smt_code = smt_path.read_text()

    # portfolioSize=4 → portfolioSize=N in the get_portfolio_3 call
    smt_code = re.sub(
        r"(get_portfolio_3\(.*?portfolioSize\s*=\s*)\d+",
        rf"\g<1>{cfg.portfolio_size}",
        smt_code,
    )

    # KMeans n_init=10 → n_init=N
    smt_code = re.sub(
        r"(KMeans\([^)]*n_init\s*=\s*)\d+",
        rf"\g<1>{cfg.kmeans_n_init}",
        smt_code,
    )

    # range(0,5) → range(0,N) for internal CV splits
    # This appears in the greedy solver selection loop
    smt_code = re.sub(
        r"for si in range\(0\s*,\s*\d+\)",
        f"for si in range(0,{cfg.smac_internal_cv_splits})",
        smt_code,
    )

    # Pool(processes=10) → Pool(processes=N) to use available CPUs
    # 3 logics run in parallel, so pool_size = cpus // 3
    # Only first occurrence (line 212 training pool; line 307 inference must stay 1)
    pool_size = max(10, cfg.cpus_per_task // 3)
    smt_code = re.sub(
        r"Pool\(processes\s*=\s*\d+\)",
        f"Pool(processes={pool_size})",
        smt_code,
        count=1,
    )

    smt_path.write_text(smt_code)
    print(f"  Patched SMTportfolio.py: portfolio_size={cfg.portfolio_size}, "
          f"kmeans_n_init={cfg.kmeans_n_init}, cv_splits={cfg.smac_internal_cv_splits}, "
          f"pool={pool_size}")

    # --- Patch portfolio_smac3.py ---
    smac_path = workdir / "portfolio_smac3.py"
    smac_code = smac_path.read_text()

    # n_trials=200 → n_trials=N
    smac_code = re.sub(
        r"(Scenario\([^)]*n_trials\s*=\s*)\d+",
        rf"\g<1>{cfg.smac_n_trials}",
        smac_code,
    )

    # w1 = 0.5 → w1 = N (module-level assignment)
    smac_code = re.sub(
        r"^(w1\s*=\s*)[\d.]+",
        rf"\g<1>{cfg.smac_w1}",
        smac_code,
        flags=re.MULTILINE,
    )

    smac_path.write_text(smac_code)
    print(f"  Patched portfolio_smac3.py: smac_n_trials={cfg.smac_n_trials}, "
          f"smac_w1={cfg.smac_w1}")


# =====================================================================
# Training
# =====================================================================

def train_single_logic(cfg, logic, dataset_name=None, timeout=162000):
    """Train a single logic. Returns (logic, success, output_file_path)."""
    if dataset_name is None:
        dataset_name = logic

    labels_path = cfg.workdir / "data" / f"{dataset_name}Labels.json"
    features_path = cfg.workdir / "machfea" / "infer_result" / f"{dataset_name}_train_feature.json"
    solver_path = cfg.workdir / "machfea" / f"{logic}_solver.json"

    for p in [labels_path, features_path, solver_path]:
        if not p.exists():
            print(f"  [{logic}] ERROR: Missing {p}")
            return (logic, False, None)

    env = os.environ.copy()
    env["SMAC_WORKERS"] = str(cfg.smac_workers)

    cmd = [
        sys.executable, "SMTportfolio.py", "train",
        "-dataset", dataset_name,
        "-solverdict", f"machfea/{logic}_solver.json",
        "-seed", str(cfg.seed),
        "-cluster_num", str(cfg.cluster_num),
    ]

    log_path = cfg.workdir / f"train_{dataset_name}.log"
    print(f"  [{dataset_name}] Starting training (log: {log_path.name})")

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, cwd=str(cfg.workdir), env=env,
            stdout=log_f, stderr=subprocess.STDOUT, timeout=timeout,
        )

    output_file = (
        cfg.workdir / "output"
        / f"train_result_{dataset_name}_{cfg.portfolio_size}_{cfg.cluster_num}_{cfg.seed}.json"
    )
    success = result.returncode == 0 and output_file.exists()
    print(f"  [{dataset_name}] rc={result.returncode}, output={'exists' if output_file.exists() else 'MISSING'}")
    return (logic, success, str(output_file) if success else None)


def run_logics_parallel(fn, args_list, label="task"):
    """Run a function for each logic in parallel using threads."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=len(args_list)) as executor:
        futures = {executor.submit(fn, *args): args for args in args_list}
        for future in as_completed(futures):
            result = future.result()
            logic = result[0]
            results[logic] = result
            print(f"  [{label}] {logic} done: success={result[1]}")
    return results


def phase_train(cfg):
    """Train SMTGazer with ablated config for each logic."""
    print("=" * 70)
    print(f"TRAINING (cluster_num={cfg.cluster_num}, portfolio_size={cfg.portfolio_size}, "
          f"seed={cfg.seed})")
    print("=" * 70)

    args = [(cfg, logic) for logic in cfg.logics]
    results = run_logics_parallel(train_single_logic, args, label="train")

    all_ok = True
    for logic in cfg.logics:
        _, success, output_path = results.get(logic, (logic, False, None))
        if not success:
            print(f"  FAILED: Training {logic}")
            all_ok = False
            continue

        with open(output_path) as f:
            result = json.load(f)
        print(f"  {logic}: {len(result['portfolio'])} clusters")

    if all_ok:
        print("\n  All training complete!")
    return all_ok


# =====================================================================
# K-Fold Cross-Validation
# =====================================================================

def load_run_data(cfg):
    """Load run data from DB. Returns {family/filename: {..., runs: {cid: (status, ms)}}}."""
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT i.id, i.logic, i.family, i.file_name, i.file_path,
               r.config_id, r.status, r.runtime_ms
        FROM runs r JOIN instances i ON r.instance_id = i.id
        WHERE r.portfolio_id = ? AND i.suite_name = ?
          AND r.config_id IN ({})
        """.format(",".join("?" * len(cfg.config_ids))),
        (cfg.portfolio_id, cfg.suite_name, *cfg.config_ids),
    ).fetchall()
    conn.close()

    data = {}
    for iid, logic, family, file_name, file_path, cid, status, runtime_ms in rows:
        key = f"{family}/{file_name}"
        if key not in data:
            data[key] = {
                "instance_id": iid, "logic": logic, "family": family,
                "file_name": file_name, "file_path": file_path, "runs": {},
            }
        data[key]["runs"][cid] = (status, runtime_ms)

    complete = {k: v for k, v in data.items() if len(v["runs"]) == len(cfg.config_ids)}
    return complete


def load_shared_folds(folds_path, data):
    """Load shared fold definitions from JSON."""
    with open(folds_path) as f:
        fold_data = json.load(f)

    fp_to_key = {}
    for key, bdata in data.items():
        fp_to_key[bdata["file_path"]] = key

    folds = []
    for fold_def in fold_data["folds"]:
        train_keys = [fp_to_key[p] for p in fold_def["train_paths"] if p in fp_to_key]
        test_keys = [fp_to_key[p] for p in fold_def["test_paths"] if p in fp_to_key]
        folds.append((train_keys, test_keys))

    return folds


def write_fold_data(cfg, data, train_keys, test_keys, logic, fold_idx):
    """Write Labels.json, train features, and test features for one fold+logic."""
    dataset_name = f"{logic}_fold{fold_idx}"

    logic_train = [k for k in train_keys if data[k]["logic"] == logic]
    logic_test = [k for k in test_keys if data[k]["logic"] == logic]

    if not logic_train or not logic_test:
        return None, 0, 0

    # Labels (train only)
    train_labels = {}
    for key in logic_train:
        times = []
        for cid in cfg.config_ids:
            status, runtime_ms = data[key]["runs"][cid]
            times.append(compute_par2(status, runtime_ms, cfg.par2_penalty))
        train_labels[key] = times

    labels = {"train": train_labels}
    labels_path = cfg.workdir / "data" / f"{dataset_name}Labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f)

    # Full features
    full_feat_path = cfg.workdir / "machfea" / "infer_result" / f"{logic}_train_feature.json"
    with open(full_feat_path) as f:
        all_features = json.load(f)

    # Train features
    train_features = {k: all_features[k] for k in logic_train if k in all_features}
    train_feat_path = cfg.workdir / "machfea" / "infer_result" / f"{dataset_name}_train_feature.json"
    with open(train_feat_path, "w") as f:
        json.dump(train_features, f)

    # Test features
    test_features = {k: all_features[k] for k in logic_test if k in all_features}
    test_feat_path = cfg.workdir / "machfea" / "infer_result" / f"{dataset_name}_test_feature.json"
    with open(test_feat_path, "w") as f:
        json.dump(test_features, f)

    # Solver.json
    solver_src = cfg.workdir / "machfea" / f"{logic}_solver.json"
    solver_dst = cfg.workdir / "machfea" / f"{dataset_name}_solver.json"
    shutil.copy2(solver_src, solver_dst)

    return dataset_name, len(logic_train), len(logic_test)


def parse_infer_output(cfg, data, test_keys, logic, fold_idx):
    """Parse SMTGazer infer output and compute per-instance PAR-2."""
    dataset_name = f"{logic}_fold{fold_idx}"
    output_file = (
        cfg.workdir / "output"
        / f"test_result_{dataset_name}_{cfg.seed}_{cfg.cluster_num}.json"
    )

    if not output_file.exists():
        print(f"  WARNING: Infer output not found: {output_file}")
        return []

    with open(output_file) as f:
        predictions = json.load(f)

    logic_test = [k for k in test_keys if data[k]["logic"] == logic]
    results = []

    for key in logic_test:
        if key not in predictions:
            continue
        bdata = data[key]
        pred_solvers, pred_times = predictions[key]

        # Simulate sequential portfolio execution
        smtgazer_par2 = cfg.par2_penalty
        smtgazer_cid = None
        cumulative_time = 0

        for solver_name, time_alloc in zip(pred_solvers, pred_times):
            if solver_name in cfg.solver_list:
                solver_idx = cfg.solver_list.index(solver_name)
                cid = cfg.config_ids[solver_idx]
                status, runtime_ms = bdata["runs"][cid]
                runtime_s = runtime_ms / 1000.0

                if status in ("sat", "unsat") and runtime_s <= time_alloc:
                    smtgazer_par2 = cumulative_time + runtime_s
                    smtgazer_cid = cid
                    break

            cumulative_time += time_alloc

        if smtgazer_cid is None:
            smtgazer_par2 = cfg.par2_penalty

        # VBS
        vbs_par2 = cfg.par2_penalty
        vbs_cid = None
        for cid in cfg.config_ids:
            status, runtime_ms = bdata["runs"][cid]
            p2 = compute_par2(status, runtime_ms, cfg.par2_penalty)
            if p2 < vbs_par2:
                vbs_par2 = p2
                vbs_cid = cid

        results.append({
            "key": key,
            "logic": logic,
            "smtgazer_par2": smtgazer_par2,
            "smtgazer_solved": smtgazer_par2 < cfg.par2_penalty,
            "smtgazer_cid": smtgazer_cid,
            "vbs_par2": vbs_par2,
            "vbs_solved": vbs_par2 < cfg.par2_penalty,
            "vbs_cid": vbs_cid,
            "smtgazer_matches_vbs": smtgazer_cid == vbs_cid,
        })

    return results


def compute_baselines(cfg, data, train_keys, test_keys, logic, rng):
    """Compute SBS, CSBS, Random baselines from training data, evaluate on test."""
    logic_train = [k for k in train_keys if data[k]["logic"] == logic]
    logic_test = [k for k in test_keys if data[k]["logic"] == logic]

    totals = defaultdict(float)
    for key in logic_train:
        for cid in cfg.config_ids:
            status, runtime_ms = data[key]["runs"][cid]
            totals[cid] += compute_par2(status, runtime_ms, cfg.par2_penalty)
    sbs_cid = min(totals, key=totals.get) if totals else cfg.config_ids[0]

    results = {}
    for key in logic_test:
        bdata = data[key]
        status, runtime_ms = bdata["runs"][sbs_cid]
        sbs_par2 = compute_par2(status, runtime_ms, cfg.par2_penalty)

        rand_cid = rng.choice(cfg.config_ids)
        status, runtime_ms = bdata["runs"][rand_cid]
        rand_par2 = compute_par2(status, runtime_ms, cfg.par2_penalty)

        results[key] = {
            "sbs_par2": sbs_par2,
            "sbs_solved": sbs_par2 < cfg.par2_penalty,
            "csbs_par2": sbs_par2,
            "csbs_solved": sbs_par2 < cfg.par2_penalty,
            "random_par2": rand_par2,
            "random_solved": rand_par2 < cfg.par2_penalty,
        }

    return results


def compute_fold_metrics(instances, fold_idx, cfg):
    """Compute key metrics for a single fold."""
    strategies = ["vbs", "smtgazer", "csbs", "sbs", "random"]
    logics = sorted(set(i["logic"] for i in instances))
    m = {"fold": fold_idx + 1}

    for strat in strategies:
        par2s = [i.get(f"{strat}_par2", cfg.par2_penalty) for i in instances]
        solveds = [i.get(f"{strat}_solved", False) for i in instances]
        n = len(instances)
        m[f"{strat}_avg_par2"] = float(np.mean(par2s))
        m[f"{strat}_solved_pct"] = sum(solveds) / n * 100 if n > 0 else 0

        for logic in logics:
            li = [i for i in instances if i["logic"] == logic]
            lp = [i.get(f"{strat}_par2", cfg.par2_penalty) for i in li]
            ls = [i.get(f"{strat}_solved", False) for i in li]
            ln = len(li)
            m[f"{strat}_{logic}_avg_par2"] = float(np.mean(lp)) if lp else 0
            m[f"{strat}_{logic}_solved_pct"] = sum(ls) / ln * 100 if ln > 0 else 0

    smtgazer_total = sum(i.get("smtgazer_par2", cfg.par2_penalty) for i in instances)
    csbs_total = sum(i.get("csbs_par2", cfg.par2_penalty) for i in instances)
    m["delta_par2_vs_csbs"] = csbs_total - smtgazer_total

    disagree = sum(1 for i in instances if not i.get("smtgazer_matches_vbs", False))
    m["vbs_disagreement_pct"] = disagree / len(instances) * 100 if instances else 0

    return m


def phase_kfold(cfg):
    """Run k-fold family-aware CV using the artifact."""
    print("=" * 70)
    print(f"K-FOLD CV (k={cfg.k_folds})")
    print("=" * 70)

    print("Loading run data from DB...")
    data = load_run_data(cfg)
    print(f"  {len(data)} complete instances")

    if cfg.folds_path.exists():
        print(f"\nLoading shared folds from {cfg.folds_path}...")
        folds = load_shared_folds(cfg.folds_path, data)
        print(f"  Loaded {len(folds)} folds")
    else:
        print(f"\nERROR: Shared folds file not found: {cfg.folds_path}")
        return False

    for i, (train, test) in enumerate(folds):
        train_fam = set((data[k]["logic"], data[k]["family"]) for k in train)
        test_fam = set((data[k]["logic"], data[k]["family"]) for k in test)
        overlap = train_fam & test_fam
        assert not overlap, f"Fold {i}: family overlap"
        print(f"  Fold {i}: train={len(train)}, test={len(test)}")

    all_instances = []
    per_fold_metrics = []
    env = os.environ.copy()
    env["SMAC_WORKERS"] = str(cfg.smac_workers)

    for fold_idx, (train_keys, test_keys) in enumerate(folds):
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx + 1}/{cfg.k_folds}")
        print(f"{'=' * 50}")
        rng = random.Random(cfg.seed + fold_idx)

        fold_results = []

        # Write fold data
        fold_datasets = {}
        for logic in cfg.logics:
            dataset_name, n_train, n_test = write_fold_data(
                cfg, data, train_keys, test_keys, logic, fold_idx
            )
            if dataset_name is not None:
                fold_datasets[logic] = (dataset_name, n_train, n_test)
                print(f"  {logic}: train={n_train}, test={n_test}")

        # Train all logics in parallel
        print(f"\n  Training {len(fold_datasets)} logics in parallel...")
        train_args = [(cfg, logic, fold_datasets[logic][0]) for logic in fold_datasets]
        train_results = run_logics_parallel(train_single_logic, train_args, label=f"fold{fold_idx}")

        # Infer sequentially
        for logic in cfg.logics:
            if logic not in fold_datasets:
                continue
            dataset_name = fold_datasets[logic][0]

            _, success, _ = train_results.get(logic, (logic, False, None))
            if not success:
                print(f"  WARNING: Training failed for {logic} fold {fold_idx}")
                continue

            train_result = f"output/train_result_{dataset_name}_{cfg.portfolio_size}_{cfg.cluster_num}_{cfg.seed}.json"
            ok = run_command(
                [sys.executable, "SMTportfolio.py", "infer",
                 "-clusterPortfolio", train_result,
                 "-dataset", dataset_name,
                 "-solverdict", f"machfea/{logic}_solver.json",
                 "-seed", str(cfg.seed)],
                workdir=cfg.workdir, env=env,
                label=f"infer-{logic}-f{fold_idx}",
                timeout=600,
            )
            if not ok:
                print(f"  WARNING: Inference failed for {logic} fold {fold_idx}")
                continue

            logic_results = parse_infer_output(cfg, data, test_keys, logic, fold_idx)
            baselines = compute_baselines(cfg, data, train_keys, test_keys, logic, rng)

            for inst in logic_results:
                key = inst["key"]
                if key in baselines:
                    inst.update(baselines[key])
                fold_results.append(inst)

        if fold_results:
            n = len(fold_results)
            smtgazer_solved = sum(1 for i in fold_results if i["smtgazer_solved"])
            smtgazer_avg = np.mean([i["smtgazer_par2"] for i in fold_results])
            print(f"\n  Fold {fold_idx + 1} summary:")
            print(f"    SMTGAZER: {smtgazer_solved}/{n} solved ({smtgazer_solved / n * 100:.1f}%), avg PAR-2={smtgazer_avg:.3f}")

            fm = compute_fold_metrics(fold_results, fold_idx, cfg)
            per_fold_metrics.append(fm)

        all_instances.extend(fold_results)

    if all_instances:
        save_kfold_results(cfg, all_instances, per_fold_metrics)

    return True


def save_kfold_results(cfg, all_instances, per_fold_metrics):
    """Save all k-fold CSV results."""
    strategies = ["vbs", "smtgazer", "csbs", "sbs", "random"]
    logics = sorted(set(i["logic"] for i in all_instances))
    n_total = len(all_instances)

    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    agg = {}
    for strat in strategies:
        par2s = [i.get(f"{strat}_par2", cfg.par2_penalty) for i in all_instances]
        solveds = [i.get(f"{strat}_solved", False) for i in all_instances]
        agg[strat] = {
            "total_par2": sum(par2s),
            "avg_par2": float(np.mean(par2s)),
            "solved": sum(solveds),
            "solved_pct": sum(solveds) / n_total * 100,
            "total": n_total,
        }

    print(f"\n{'Strategy':<15} {'Total PAR-2':>12} {'Avg PAR-2':>12} {'Solved':>10} {'%':>8}")
    print("-" * 70)
    for strat in strategies:
        s = agg[strat]
        print(
            f"{strat.upper():<15} {s['total_par2']:>12.2f} {s['avg_par2']:>12.3f} "
            f"{s['solved']:>10} {s['solved_pct']:>7.1f}%"
        )

    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    k = cfg.k_folds

    # 1. Summary
    summary_path = out / f"kfold_k{k}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "total_par2", "avg_par2", "solved", "solved_pct"])
        writer.writeheader()
        for strat in strategies:
            writer.writerow({"strategy": strat.upper(), **agg[strat]})
    print(f"\nSaved: {summary_path}")

    # 2. Per-fold
    perfold_path = out / f"kfold_k{k}_perfold.csv"
    if per_fold_metrics:
        fieldnames = sorted(per_fold_metrics[0].keys())
        with open(perfold_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_fold_metrics)
    print(f"Saved: {perfold_path}")

    # 3. Per-logic
    perlogic_path = out / f"kfold_k{k}_perlogic.csv"
    rows = []
    for logic in logics:
        li = [i for i in all_instances if i["logic"] == logic]
        for strat in strategies:
            par2s = [i.get(f"{strat}_par2", cfg.par2_penalty) for i in li]
            solveds = [i.get(f"{strat}_solved", False) for i in li]
            rows.append({
                "logic": logic, "strategy": strat.upper(),
                "total_par2": sum(par2s), "avg_par2": float(np.mean(par2s)),
                "solved": sum(solveds), "solved_pct": sum(solveds) / len(li) * 100,
                "total": len(li),
            })
    with open(perlogic_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["logic", "strategy", "total_par2", "avg_par2", "solved", "solved_pct", "total"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {perlogic_path}")

    # 4. Mean ± Std
    key_metrics = [
        ("smtgazer_avg_par2", "SMTGazer avg PAR-2"),
        ("smtgazer_solved_pct", "SMTGazer solved%"),
        ("csbs_avg_par2", "CSBS avg PAR-2"),
        ("csbs_solved_pct", "CSBS solved%"),
        ("delta_par2_vs_csbs", "Delta PAR-2 vs CSBS"),
        ("vbs_disagreement_pct", "VBS disagreement%"),
    ]
    meanstd_path = out / f"kfold_k{k}_meanstd.csv"
    meanstd_rows = []
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        meanstd_rows.append({"metric": label, "mean": float(np.mean(vals)), "std": float(np.std(vals))})
    for logic in logics:
        for strat in ["smtgazer", "csbs", "vbs"]:
            for metric in ["avg_par2", "solved_pct"]:
                mkey = f"{strat}_{logic}_{metric}"
                vals = [fm[mkey] for fm in per_fold_metrics]
                meanstd_rows.append({
                    "metric": f"{strat.upper()} {logic} {metric}",
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                })
    with open(meanstd_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        writer.writerows(meanstd_rows)
    print(f"Saved: {meanstd_path}")

    # 5. Tail 10%
    tail_path = out / f"kfold_k{k}_tail10.csv"
    sorted_by_vbs = sorted(all_instances, key=lambda i: i.get("vbs_par2", 0), reverse=True)
    tail_n = max(1, len(sorted_by_vbs) // 10)
    tail = sorted_by_vbs[:tail_n]
    tail_rows = []
    for strat in strategies:
        par2s = [i.get(f"{strat}_par2", cfg.par2_penalty) for i in tail]
        tail_rows.append({
            "strategy": strat.upper(),
            "tail10_total_par2": sum(par2s),
            "tail10_avg_par2": float(np.mean(par2s)),
        })
    with open(tail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "tail10_total_par2", "tail10_avg_par2"])
        writer.writeheader()
        writer.writerows(tail_rows)
    print(f"Saved: {tail_path}")

    print(f"\nAll results in: {out}")


# =====================================================================
# Main
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m ablation.pipeline_ablation <config.yaml>")
        return 1

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    cfg = load_experiment_config(config_path)

    print("=" * 70)
    print(f"SMTGazer Ablation Pipeline — {cfg.experiment_code}")
    print("=" * 70)
    print(f"  Knob: {cfg.ablation_knob} = {cfg.ablation_value}")
    print(f"  Workdir: {cfg.workdir}")
    print(f"  Results: {cfg.results_dir}")
    print(f"  SMAC workers: {cfg.smac_workers}")
    print()

    # Phase 1: Setup & Patch
    print("=== Phase 1: Setup & Patch ===")
    setup_workdir(cfg)
    patch_artifacts(cfg)

    # Phase 2: Train
    print("\n=== Phase 2: Training ===")
    if not phase_train(cfg):
        print("\nTRAINING FAILED")
        return 1

    # Phase 3: K-Fold CV (skip when k_folds == 0)
    if cfg.k_folds > 0:
        print("\n=== Phase 3: K-Fold CV ===")
        if not phase_kfold(cfg):
            print("\nK-FOLD FAILED")
            return 1
    else:
        print("\n=== Phase 3: K-Fold CV — SKIPPED (k_folds=0) ===")

    # Write completion marker
    marker = cfg.results_dir / "pipeline_complete"
    marker.write_text(f"completed\n")
    print(f"  Wrote completion marker: {marker}")

    print("\n" + "=" * 70)
    print("ABLATION PIPELINE COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
