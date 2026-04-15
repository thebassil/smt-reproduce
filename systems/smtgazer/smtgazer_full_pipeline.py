#!/usr/bin/env python3
"""
SMTGazer Full Pipeline for suite_9k / Portfolio 7

Phases:
  train      — Run SMTportfolio.py train for each logic (SMAC3 optimization)
  kfold      — 5-fold family-aware CV using the actual artifact
  register   — Write decisions to DB from full-data model
  aggregate  — Combine per-fold results into summary CSVs

Usage:
  python smtgazer_full_pipeline.py train
  python smtgazer_full_pipeline.py kfold [--fold N]    # N = 0..4
  python smtgazer_full_pipeline.py aggregate
  python smtgazer_full_pipeline.py register
  python smtgazer_full_pipeline.py all
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# === Configuration ===
WORKDIR = Path("/dcs/large/u5573765/smtgazer_workdir")
ARTIFACT_DIR = Path("/dcs/23/u5573765/cs351/smt-small-sbs-smtgazer/artifacts/smtgazer/SMTGazer")
DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")
RESULTS_DIR = Path("/dcs/23/u5573765/cs351/eval-smtgazer/smtgazer_results")

PORTFOLIO_ID = 7
SUITE_NAME = "suite_9k"
TIMEOUT_S = 60
PAR2_PENALTY = TIMEOUT_S * 2
LOGICS = ["QF_BV", "QF_LIA", "QF_NRA"]
CONFIG_IDS = [109, 112, 114, 115, 116, 118]
SOLVER_LIST = [
    "cvc5::cvc5_qfnra_05_nl_cov",
    "cvc5::cvc5_qfnra_08_decision_just",
    "z3::z3_baseline_default",
    "cvc5::cvc5_baseline_default",
    "z3::z3_baseline_qflia_case_split",
    "z3::z3_baseline_qfbv_sat_euf_sat",
]
CLUSTER_NUM = 10
SEED = 42
K_FOLDS = 5
SMAC_WORKERS = int(os.environ.get("SMAC_WORKERS", "1"))
SHARED_FOLDS_PATH = Path("/dcs/23/u5573765/cs351/eval-smtgazer/shared_folds_k5.json")


# =====================================================================
# Utility
# =====================================================================

def get_utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_par2(status, runtime_ms):
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return PAR2_PENALTY


def run_artifact_command(args, label="command", timeout=162000):
    """Run a command in the workdir with logging."""
    print(f"  [{label}] {' '.join(args)}")
    env = os.environ.copy()
    env["SMAC_WORKERS"] = str(SMAC_WORKERS)
    t0 = time.time()
    result = subprocess.run(
        args, cwd=str(WORKDIR), env=env,
        capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - t0
    print(f"  [{label}] Finished in {elapsed:.1f}s (rc={result.returncode})")
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        for line in lines[-15:]:
            print(f"    {line}")
    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split('\n')[-15:]:
            print(f"    STDERR: {line}")
    return result.returncode == 0


def setup_workdir():
    """Ensure workdir has artifact copies and directory structure."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "data").mkdir(exist_ok=True)
    (WORKDIR / "machfea" / "infer_result").mkdir(parents=True, exist_ok=True)
    (WORKDIR / "tmp").mkdir(exist_ok=True)
    (WORKDIR / "output").mkdir(exist_ok=True)

    # Copy artifact scripts (always fresh for patching)
    for fname in ["SMTportfolio.py", "portfolio_smac3.py"]:
        dst = WORKDIR / fname
        shutil.copy2(ARTIFACT_DIR / fname, dst)

    # Symlink smac — portfolio_smac3.py does sys.path.insert(0, CWD+'/smac')
    # then "from smac import ...", so we need workdir/smac/smac/__init__.py to exist
    smac_link = WORKDIR / "smac"
    if not smac_link.exists():
        smac_link.symlink_to(ARTIFACT_DIR / "smac")

    # Patch artifact scripts for feasible config
    _patch_artifacts()


# Feasible training config — matches ablation reference.yaml
_SMAC_N_TRIALS = 50
_SMAC_CV_SPLITS = 3
_CPUS_PER_TASK = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))


def _patch_artifacts():
    """Patch copied artifact scripts for feasible training config.

    Patches SMTportfolio.py: cv_splits, Pool(processes=N)
    Patches portfolio_smac3.py: n_trials
    """
    # --- SMTportfolio.py ---
    smt_path = WORKDIR / "SMTportfolio.py"
    smt_code = smt_path.read_text()

    # range(0,5) → range(0,N) for internal CV splits
    smt_code = re.sub(
        r"for si in range\(0\s*,\s*\d+\)",
        f"for si in range(0,{_SMAC_CV_SPLITS})",
        smt_code,
    )

    # Pool(processes=10) → Pool(processes=N) — only the first occurrence
    # (line 212 is the training pool; line 307 is inference and must stay 1)
    pool_size = max(10, _CPUS_PER_TASK // 3)
    smt_code = re.sub(
        r"Pool\(processes\s*=\s*\d+\)",
        f"Pool(processes={pool_size})",
        smt_code,
        count=1,
    )

    smt_path.write_text(smt_code)
    print(f"  Patched SMTportfolio.py: cv_splits={_SMAC_CV_SPLITS}, pool={pool_size}")

    # --- portfolio_smac3.py ---
    smac_path = WORKDIR / "portfolio_smac3.py"
    smac_code = smac_path.read_text()

    # n_trials=200 → n_trials=N
    smac_code = re.sub(
        r"(Scenario\([^)]*n_trials\s*=\s*)\d+",
        rf"\g<1>{_SMAC_N_TRIALS}",
        smac_code,
    )

    smac_path.write_text(smac_code)
    print(f"  Patched portfolio_smac3.py: n_trials={_SMAC_N_TRIALS}")


def load_run_data():
    """Load all run data from DB. Returns {file_path: {instance_id, logic, family, file_name, runs: {cid: (status, runtime_ms)}}}"""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT
            i.id, i.logic, i.family, i.file_name, i.file_path,
            r.config_id, r.status, r.runtime_ms
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        WHERE r.portfolio_id = ? AND i.suite_name = ?
          AND r.config_id IN ({})
    """.format(",".join("?" * len(CONFIG_IDS))),
    (PORTFOLIO_ID, SUITE_NAME, *CONFIG_IDS)).fetchall()
    conn.close()

    data = {}
    for iid, logic, family, file_name, file_path, cid, status, runtime_ms in rows:
        key = f"{family}/{file_name}"
        if key not in data:
            data[key] = {
                'instance_id': iid, 'logic': logic, 'family': family,
                'file_name': file_name, 'file_path': file_path, 'runs': {}
            }
        data[key]['runs'][cid] = (status, runtime_ms)

    # Filter to complete
    complete = {k: v for k, v in data.items() if len(v['runs']) == len(CONFIG_IDS)}
    return complete


# =====================================================================
# Phase 3: Full SMAC3 Training
# =====================================================================

def train_single_logic(logic, dataset_name=None, timeout=162000):
    """Train a single logic. Returns (logic, success, output_file_path)."""
    if dataset_name is None:
        dataset_name = logic

    labels_path = WORKDIR / "data" / f"{dataset_name}Labels.json"
    features_path = WORKDIR / "machfea" / "infer_result" / f"{dataset_name}_train_feature.json"
    solver_path = WORKDIR / "machfea" / f"{logic}_solver.json"

    for p in [labels_path, features_path, solver_path]:
        if not p.exists():
            print(f"  [{logic}] ERROR: Missing {p}")
            return (logic, False, None)

    env = os.environ.copy()
    env["SMAC_WORKERS"] = str(SMAC_WORKERS)
    cmd = [
        sys.executable, "SMTportfolio.py", "train",
        "-dataset", dataset_name,
        "-solverdict", f"machfea/{logic}_solver.json",
        "-seed", str(SEED),
        "-cluster_num", str(CLUSTER_NUM),
    ]

    log_path = WORKDIR / f"train_{dataset_name}.log"
    print(f"  [{dataset_name}] Starting training (log: {log_path.name})")
    with open(log_path, 'w') as log_f:
        result = subprocess.run(
            cmd, cwd=str(WORKDIR), env=env,
            stdout=log_f, stderr=subprocess.STDOUT, timeout=timeout
        )

    output_file = WORKDIR / "output" / f"train_result_{dataset_name}_4_{CLUSTER_NUM}_{SEED}.json"
    success = result.returncode == 0 and output_file.exists()
    print(f"  [{dataset_name}] rc={result.returncode}, output={'exists' if output_file.exists() else 'MISSING'}")
    return (logic, success, str(output_file) if success else None)


def run_logics_parallel(fn, args_per_logic, label="task"):
    """Run a function for each logic in parallel using subprocess threads.
    fn must accept a single arg tuple and return (logic, success, ...).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=len(args_per_logic)) as executor:
        futures = {executor.submit(fn, *args): args for args in args_per_logic}
        for future in as_completed(futures):
            result = future.result()
            logic = result[0]
            results[logic] = result
            print(f"  [{label}] {logic} done: success={result[1]}")
    return results


def phase_train():
    """Train SMTGazer on full dataset for each logic (3 logics in parallel)."""
    print("=" * 70)
    print("PHASE 3: Full SMAC3 Training (parallel across logics)")
    print("=" * 70)

    setup_workdir()

    # Run all 3 logics in parallel — each uses Pool(10) internally
    args = [(logic,) for logic in LOGICS]
    results = run_logics_parallel(train_single_logic, args, label="train")

    all_ok = True
    for logic in LOGICS:
        logic_name, success, output_path = results.get(logic, (logic, False, None))
        if not success:
            print(f"  FAILED: Training {logic}")
            all_ok = False
            continue

        with open(output_path) as f:
            result = json.load(f)
        print(f"  {logic}: {len(result['portfolio'])} clusters")
        for cid, entry in result['portfolio'].items():
            solvers, times = entry
            print(f"    Cluster {cid}: solvers={solvers} times=[{', '.join(f'{t:.1f}' for t in times)}] sum={sum(times):.1f}")

    if all_ok:
        print("\n  All training complete!")
    return all_ok


# =====================================================================
# Phase 4: K-Fold Cross-Validation
# =====================================================================

def load_shared_folds(folds_path, data):
    """Load shared fold definitions from JSON. Maps file_path → family/filename keys."""
    with open(folds_path) as f:
        fold_data = json.load(f)

    # Build file_path → family/filename mapping
    fp_to_key = {}
    for key, bdata in data.items():
        fp_to_key[bdata['file_path']] = key

    folds = []
    for fold_def in fold_data["folds"]:
        train_keys = [fp_to_key[p] for p in fold_def["train_paths"] if p in fp_to_key]
        test_keys = [fp_to_key[p] for p in fold_def["test_paths"] if p in fp_to_key]
        folds.append((train_keys, test_keys))

    return folds


def make_family_aware_folds(data, k, rng_seed):
    """Fallback: Split benchmarks into k folds. Only used if shared folds unavailable."""
    rng = random.Random(rng_seed)

    logic_families = defaultdict(lambda: defaultdict(list))
    for key, bdata in data.items():
        logic_families[bdata['logic']][bdata['family']].append(key)

    family_fold = {}
    for logic in sorted(logic_families):
        families = sorted(logic_families[logic].keys())
        rng.shuffle(families)
        for i, fam in enumerate(families):
            family_fold[(logic, fam)] = i % k

    fold_paths = [[] for _ in range(k)]
    for key, bdata in data.items():
        fold_idx = family_fold[(bdata['logic'], bdata['family'])]
        fold_paths[fold_idx].append(key)

    folds = []
    for test_fold in range(k):
        test_keys = fold_paths[test_fold]
        train_keys = []
        for i in range(k):
            if i != test_fold:
                train_keys.extend(fold_paths[i])
        folds.append((train_keys, test_keys))

    return folds


def write_fold_data(data, train_keys, test_keys, logic, fold_idx):
    """Write Labels.json, train features, and test features for one fold+logic."""
    dataset_name = f"{logic}_fold{fold_idx}"

    # Filter to this logic
    logic_train = [k for k in train_keys if data[k]['logic'] == logic]
    logic_test = [k for k in test_keys if data[k]['logic'] == logic]

    if not logic_train or not logic_test:
        return None, 0, 0

    # Labels (train only — the artifact only reads "train" key)
    train_labels = {}
    for key in logic_train:
        times = []
        for cid in CONFIG_IDS:
            status, runtime_ms = data[key]['runs'][cid]
            times.append(compute_par2(status, runtime_ms))
        train_labels[key] = times

    labels = {"train": train_labels}
    labels_path = WORKDIR / "data" / f"{dataset_name}Labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f)

    # Load full features
    full_feat_path = WORKDIR / "machfea" / "infer_result" / f"{logic}_train_feature.json"
    with open(full_feat_path) as f:
        all_features = json.load(f)

    # Train features
    train_features = {k: all_features[k] for k in logic_train if k in all_features}
    train_feat_path = WORKDIR / "machfea" / "infer_result" / f"{dataset_name}_train_feature.json"
    with open(train_feat_path, 'w') as f:
        json.dump(train_features, f)

    # Test features
    test_features = {k: all_features[k] for k in logic_test if k in all_features}
    test_feat_path = WORKDIR / "machfea" / "infer_result" / f"{dataset_name}_test_feature.json"
    with open(test_feat_path, 'w') as f:
        json.dump(test_features, f)

    # Solver.json (copy from main)
    solver_src = WORKDIR / "machfea" / f"{logic}_solver.json"
    solver_dst = WORKDIR / "machfea" / f"{dataset_name}_solver.json"
    shutil.copy2(solver_src, solver_dst)

    return dataset_name, len(logic_train), len(logic_test)


def parse_infer_output(data, test_keys, logic, fold_idx):
    """Parse SMTGazer infer output and compute per-instance PAR-2."""
    dataset_name = f"{logic}_fold{fold_idx}"
    output_file = WORKDIR / "output" / f"test_result_{dataset_name}_{SEED}_{CLUSTER_NUM}.json"

    if not output_file.exists():
        print(f"  WARNING: Infer output not found: {output_file}")
        return []

    with open(output_file) as f:
        predictions = json.load(f)

    logic_test = [k for k in test_keys if data[k]['logic'] == logic]
    results = []

    for key in logic_test:
        if key not in predictions:
            continue
        bdata = data[key]

        # SMTGazer prediction: [[solver_names], [time_allocations]]
        pred_solvers, pred_times = predictions[key]

        # Simulate sequential portfolio execution
        smtgazer_par2 = PAR2_PENALTY
        smtgazer_cid = None
        cumulative_time = 0

        for solver_name, time_alloc in zip(pred_solvers, pred_times):
            # Map solver name to config_id
            if solver_name in SOLVER_LIST:
                solver_idx = SOLVER_LIST.index(solver_name)
                cid = CONFIG_IDS[solver_idx]
                status, runtime_ms = bdata['runs'][cid]
                runtime_s = runtime_ms / 1000.0

                if status in ('sat', 'unsat') and runtime_s <= time_alloc:
                    smtgazer_par2 = cumulative_time + runtime_s
                    smtgazer_cid = cid
                    break

            cumulative_time += time_alloc

        if smtgazer_cid is None:
            smtgazer_par2 = PAR2_PENALTY

        # VBS
        vbs_par2 = PAR2_PENALTY
        vbs_cid = None
        for cid in CONFIG_IDS:
            status, runtime_ms = bdata['runs'][cid]
            p2 = compute_par2(status, runtime_ms)
            if p2 < vbs_par2:
                vbs_par2 = p2
                vbs_cid = cid

        results.append({
            'key': key,
            'logic': logic,
            'smtgazer_par2': smtgazer_par2,
            'smtgazer_solved': smtgazer_par2 < PAR2_PENALTY,
            'smtgazer_cid': smtgazer_cid,
            'vbs_par2': vbs_par2,
            'vbs_solved': vbs_par2 < PAR2_PENALTY,
            'vbs_cid': vbs_cid,
            'smtgazer_matches_vbs': smtgazer_cid == vbs_cid,
        })

    return results


def compute_baselines(data, train_keys, test_keys, logic, rng):
    """Compute SBS, CSBS, Random baselines from training data, evaluate on test."""
    logic_train = [k for k in train_keys if data[k]['logic'] == logic]
    logic_test = [k for k in test_keys if data[k]['logic'] == logic]

    # SBS: overall best single config on training data
    totals = defaultdict(float)
    for key in logic_train:
        for cid in CONFIG_IDS:
            status, runtime_ms = data[key]['runs'][cid]
            totals[cid] += compute_par2(status, runtime_ms)
    sbs_cid = min(totals, key=totals.get) if totals else CONFIG_IDS[0]

    # CSBS: per-logic best single config (same as SBS when filtering by logic)
    csbs_cid = sbs_cid

    results = {}
    for key in logic_test:
        bdata = data[key]

        # SBS
        status, runtime_ms = bdata['runs'][sbs_cid]
        sbs_par2 = compute_par2(status, runtime_ms)

        # CSBS (same as SBS per-logic)
        csbs_par2 = sbs_par2

        # Random
        rand_cid = rng.choice(CONFIG_IDS)
        status, runtime_ms = bdata['runs'][rand_cid]
        rand_par2 = compute_par2(status, runtime_ms)

        results[key] = {
            'sbs_par2': sbs_par2,
            'sbs_solved': sbs_par2 < PAR2_PENALTY,
            'csbs_par2': csbs_par2,
            'csbs_solved': csbs_par2 < PAR2_PENALTY,
            'random_par2': rand_par2,
            'random_solved': rand_par2 < PAR2_PENALTY,
        }

    return results


def _load_folds_and_data():
    """Load run data and fold definitions. Returns (data, folds)."""
    print("Loading run data from DB...")
    data = load_run_data()
    print(f"  {len(data)} complete instances")

    if SHARED_FOLDS_PATH.exists():
        print(f"\nLoading shared folds from {SHARED_FOLDS_PATH}...")
        folds = load_shared_folds(SHARED_FOLDS_PATH, data)
        print(f"  Loaded {len(folds)} folds from shared definitions")
    else:
        print(f"\nCreating {K_FOLDS} family-aware folds (internal, seed={SEED})...")
        folds = make_family_aware_folds(data, K_FOLDS, SEED)

    for i, (train, test) in enumerate(folds):
        train_fam = set((data[k]['logic'], data[k]['family']) for k in train)
        test_fam = set((data[k]['logic'], data[k]['family']) for k in test)
        overlap = train_fam & test_fam
        assert not overlap, f"Fold {i}: family overlap"
        print(f"  Fold {i}: train={len(train)}, test={len(test)}")

    return data, folds


def _run_single_fold(data, folds, fold_idx):
    """Run a single fold of k-fold CV. Returns (fold_results, fold_metrics)."""
    train_keys, test_keys = folds[fold_idx]

    print(f"\n{'='*50}")
    print(f"FOLD {fold_idx + 1}/{K_FOLDS}")
    print(f"{'='*50}")
    rng = random.Random(SEED + fold_idx)

    fold_results = []

    # Write fold data for all logics first
    fold_datasets = {}
    for logic in LOGICS:
        dataset_name, n_train, n_test = write_fold_data(
            data, train_keys, test_keys, logic, fold_idx
        )
        if dataset_name is not None:
            fold_datasets[logic] = (dataset_name, n_train, n_test)
            print(f"  {logic}: train={n_train}, test={n_test}")

    # Train all 3 logics in parallel
    print(f"\n  Training {len(fold_datasets)} logics in parallel...")
    train_args = [(logic, fold_datasets[logic][0]) for logic in fold_datasets]
    train_results = run_logics_parallel(train_single_logic, train_args, label=f"fold{fold_idx}")

    # Infer sequentially (fast, <1 min each)
    for logic in LOGICS:
        if logic not in fold_datasets:
            continue
        dataset_name = fold_datasets[logic][0]

        logic_ok, success, _ = train_results.get(logic, (logic, False, None))
        if not success:
            print(f"  WARNING: Training failed for {logic} fold {fold_idx}")
            continue

        train_result = f"output/train_result_{dataset_name}_4_{CLUSTER_NUM}_{SEED}.json"
        ok = run_artifact_command(
            [sys.executable, "SMTportfolio.py", "infer",
             "-clusterPortfolio", train_result,
             "-dataset", dataset_name,
             "-solverdict", f"machfea/{logic}_solver.json",
             "-seed", str(SEED)],
            label=f"infer-{logic}-f{fold_idx}",
            timeout=600
        )
        if not ok:
            print(f"  WARNING: Inference failed for {logic} fold {fold_idx}")
            continue

        logic_results = parse_infer_output(data, test_keys, logic, fold_idx)
        baselines = compute_baselines(data, train_keys, test_keys, logic, rng)

        for inst in logic_results:
            key = inst['key']
            if key in baselines:
                inst.update(baselines[key])
            fold_results.append(inst)

    fold_metrics = None
    if fold_results:
        n = len(fold_results)
        smtgazer_solved = sum(1 for i in fold_results if i['smtgazer_solved'])
        csbs_solved = sum(1 for i in fold_results if i.get('csbs_solved', False))
        smtgazer_avg = np.mean([i['smtgazer_par2'] for i in fold_results])
        csbs_avg = np.mean([i.get('csbs_par2', PAR2_PENALTY) for i in fold_results])

        print(f"\n  Fold {fold_idx+1} summary:")
        print(f"    SMTGAZER: {smtgazer_solved}/{n} solved ({smtgazer_solved/n*100:.1f}%), avg PAR-2={smtgazer_avg:.3f}")
        print(f"    CSBS:     {csbs_solved}/{n} solved ({csbs_solved/n*100:.1f}%), avg PAR-2={csbs_avg:.3f}")

        fold_metrics = compute_fold_metrics(fold_results, fold_idx)

    return fold_results, fold_metrics


def phase_kfold(fold_num=None):
    """Run k-fold family-aware CV using the actual artifact.

    Args:
        fold_num: If not None, run only this single fold (0-indexed) and save
                  per-fold results to JSON for later aggregation.
    """
    print("=" * 70)
    if fold_num is not None:
        print(f"PHASE 4: K-Fold CV — Fold {fold_num} only")
    else:
        print("PHASE 4: 5-Fold Family-Aware Cross-Validation")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if fold_num is None:
        # All-folds mode: setup workdir fresh
        setup_workdir()
    # Single-fold mode: train job already set up the workdir

    data, folds = _load_folds_and_data()

    if fold_num is not None:
        # Single-fold mode: run one fold, save results for later aggregation
        fold_results, fold_metrics = _run_single_fold(data, folds, fold_num)

        # Save per-fold results as JSON
        fold_out = RESULTS_DIR / f"fold_{fold_num}_results.json"
        with open(fold_out, 'w') as f:
            json.dump({'instances': fold_results, 'metrics': fold_metrics}, f)
        print(f"\nSaved fold {fold_num} results: {fold_out}")
        return True

    # All-folds mode (original behavior)
    all_instances = []
    per_fold_metrics = []

    for fold_idx in range(len(folds)):
        fold_results, fold_metrics = _run_single_fold(data, folds, fold_idx)
        if fold_metrics:
            per_fold_metrics.append(fold_metrics)
        all_instances.extend(fold_results)

    if all_instances:
        save_kfold_results(all_instances, per_fold_metrics)

    return True


def phase_aggregate():
    """Aggregate per-fold results from individual fold jobs into summary CSVs."""
    print("=" * 70)
    print("PHASE: Aggregate K-Fold Results")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_instances = []
    per_fold_metrics = []

    for fold_idx in range(K_FOLDS):
        fold_path = RESULTS_DIR / f"fold_{fold_idx}_results.json"
        if not fold_path.exists():
            print(f"  ERROR: Missing fold {fold_idx} results: {fold_path}")
            return False

        with open(fold_path) as f:
            fold_data = json.load(f)

        instances = fold_data['instances']
        metrics = fold_data['metrics']

        all_instances.extend(instances)
        if metrics:
            per_fold_metrics.append(metrics)

        n = len(instances)
        solved = sum(1 for i in instances if i.get('smtgazer_solved', False))
        print(f"  Fold {fold_idx}: {n} instances, {solved} solved")

    if not all_instances:
        print("  ERROR: No instances found across folds")
        return False

    print(f"\n  Total: {len(all_instances)} instances across {len(per_fold_metrics)} folds")
    save_kfold_results(all_instances, per_fold_metrics)
    return True


def compute_fold_metrics(instances, fold_idx):
    """Compute key metrics for a single fold."""
    strategies = ['vbs', 'smtgazer', 'csbs', 'sbs', 'random']
    logics = sorted(set(i['logic'] for i in instances))
    m = {'fold': fold_idx + 1}

    for strat in strategies:
        par2s = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in instances]
        solveds = [i.get(f'{strat}_solved', False) for i in instances]
        n = len(instances)
        m[f'{strat}_avg_par2'] = float(np.mean(par2s))
        m[f'{strat}_solved_pct'] = sum(solveds) / n * 100 if n > 0 else 0

        for logic in logics:
            li = [i for i in instances if i['logic'] == logic]
            lp = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in li]
            ls = [i.get(f'{strat}_solved', False) for i in li]
            ln = len(li)
            m[f'{strat}_{logic}_avg_par2'] = float(np.mean(lp)) if lp else 0
            m[f'{strat}_{logic}_solved_pct'] = sum(ls) / ln * 100 if ln > 0 else 0

    # Delta
    smtgazer_total = sum(i.get('smtgazer_par2', PAR2_PENALTY) for i in instances)
    csbs_total = sum(i.get('csbs_par2', PAR2_PENALTY) for i in instances)
    m['delta_par2_vs_csbs'] = csbs_total - smtgazer_total

    # VBS disagreement
    disagree = sum(1 for i in instances if not i.get('smtgazer_matches_vbs', False))
    m['vbs_disagreement_pct'] = disagree / len(instances) * 100 if instances else 0

    return m


def save_kfold_results(all_instances, per_fold_metrics):
    """Save all k-fold CSV results."""
    strategies = ['vbs', 'smtgazer', 'csbs', 'sbs', 'random']
    logics = sorted(set(i['logic'] for i in all_instances))
    n_total = len(all_instances)

    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS (AGGREGATED)")
    print("=" * 70)

    # Overall metrics
    agg = {}
    for strat in strategies:
        par2s = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in all_instances]
        solveds = [i.get(f'{strat}_solved', False) for i in all_instances]
        agg[strat] = {
            'total_par2': sum(par2s),
            'avg_par2': float(np.mean(par2s)),
            'solved': sum(solveds),
            'solved_pct': sum(solveds) / n_total * 100,
            'total': n_total,
        }

    print(f"\n{'Strategy':<15} {'Total PAR-2':>12} {'Avg PAR-2':>12} {'Solved':>10} {'%':>8}")
    print("-" * 70)
    for strat in strategies:
        s = agg[strat]
        print(f"{strat.upper():<15} {s['total_par2']:>12.2f} {s['avg_par2']:>12.3f} "
              f"{s['solved']:>10} {s['solved_pct']:>7.1f}%")

    # Per-logic
    for logic in logics:
        li = [i for i in all_instances if i['logic'] == logic]
        print(f"\n  {logic} ({len(li)} instances):")
        for strat in strategies:
            par2s = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in li]
            solveds = [i.get(f'{strat}_solved', False) for i in li]
            print(f"    {strat.upper():<15} avg={np.mean(par2s):.3f} solved={sum(solveds)}/{len(li)} ({sum(solveds)/len(li)*100:.1f}%)")

    # Mean ± Std
    print(f"\nMean +/- Std across {K_FOLDS} folds:")
    key_metrics = [
        ('smtgazer_avg_par2', 'SMTGazer avg PAR-2'),
        ('smtgazer_solved_pct', 'SMTGazer solved%'),
        ('csbs_avg_par2', 'CSBS avg PAR-2'),
        ('csbs_solved_pct', 'CSBS solved%'),
        ('delta_par2_vs_csbs', 'Delta PAR-2 vs CSBS'),
        ('vbs_disagreement_pct', 'VBS disagreement%'),
    ]
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        print(f"  {label:<30} {np.mean(vals):>10.3f} +/- {np.std(vals):>8.3f}")

    # --- Save CSVs ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Summary
    summary_path = RESULTS_DIR / f"kfold_k{K_FOLDS}_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'total'])
        writer.writeheader()
        for strat in strategies:
            writer.writerow({'strategy': strat.upper(), **agg[strat]})
    print(f"\nSaved: {summary_path}")

    # 2. Per-fold
    perfold_path = RESULTS_DIR / f"kfold_k{K_FOLDS}_perfold.csv"
    if per_fold_metrics:
        fieldnames = sorted(per_fold_metrics[0].keys())
        with open(perfold_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_fold_metrics)
    print(f"Saved: {perfold_path}")

    # 3. Per-logic
    perlogic_path = RESULTS_DIR / f"kfold_k{K_FOLDS}_perlogic.csv"
    rows = []
    for logic in logics:
        li = [i for i in all_instances if i['logic'] == logic]
        for strat in strategies:
            par2s = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in li]
            solveds = [i.get(f'{strat}_solved', False) for i in li]
            rows.append({
                'logic': logic,
                'strategy': strat.upper(),
                'total_par2': sum(par2s),
                'avg_par2': float(np.mean(par2s)),
                'solved': sum(solveds),
                'solved_pct': sum(solveds) / len(li) * 100,
                'total': len(li),
            })
    with open(perlogic_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['logic', 'strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct', 'total'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {perlogic_path}")

    # 4. Mean ± Std
    meanstd_path = RESULTS_DIR / f"kfold_k{K_FOLDS}_meanstd.csv"
    meanstd_rows = []
    for key, label in key_metrics:
        vals = [fm[key] for fm in per_fold_metrics]
        meanstd_rows.append({'metric': label, 'mean': float(np.mean(vals)), 'std': float(np.std(vals))})
    for logic in logics:
        for strat in ['smtgazer', 'csbs', 'vbs']:
            for metric in ['avg_par2', 'solved_pct']:
                key = f'{strat}_{logic}_{metric}'
                vals = [fm.get(key, np.nan) for fm in per_fold_metrics]
                vals = [v for v in vals if not np.isnan(v)]
                if not vals:
                    continue
                meanstd_rows.append({
                    'metric': f'{strat.upper()} {logic} {metric}',
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                })
    with open(meanstd_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'mean', 'std'])
        writer.writeheader()
        writer.writerows(meanstd_rows)
    print(f"Saved: {meanstd_path}")

    # 5. Tail 10%
    tail_path = RESULTS_DIR / f"kfold_k{K_FOLDS}_tail10.csv"
    sorted_by_vbs = sorted(all_instances, key=lambda i: i.get('vbs_par2', 0), reverse=True)
    tail_n = max(1, len(sorted_by_vbs) // 10)
    tail = sorted_by_vbs[:tail_n]
    tail_rows = []
    for strat in strategies:
        par2s = [i.get(f'{strat}_par2', PAR2_PENALTY) for i in tail]
        tail_rows.append({
            'strategy': strat.upper(),
            'tail10_total_par2': sum(par2s),
            'tail10_avg_par2': float(np.mean(par2s)),
        })
    with open(tail_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy', 'tail10_total_par2', 'tail10_avg_par2'])
        writer.writeheader()
        writer.writerows(tail_rows)
    print(f"Saved: {tail_path}")

    print(f"\nAll results in: {RESULTS_DIR}")


# =====================================================================
# Phase 5: DB Registration
# =====================================================================

def phase_register():
    """Register selector + decisions in DB from full-data trained models."""
    print("=" * 70)
    print("PHASE 5: DB Registration")
    print("=" * 70)

    # Load data
    data = load_run_data()
    print(f"  {len(data)} instances loaded")

    # Load metadata for instance_id lookup
    meta_path = WORKDIR / "instance_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    # Run inference using full-data models (already trained in Phase 3)
    # We need normalized test features — use the getTestPortfoliio function via CLI
    setup_workdir()

    # For each logic, copy train features as test features (we're predicting on all data)
    for logic in LOGICS:
        src = WORKDIR / "machfea" / "infer_result" / f"{logic}_train_feature.json"
        dst = WORKDIR / "machfea" / "infer_result" / f"{logic}_test_feature.json"
        shutil.copy2(src, dst)

    # Run infer for each logic
    all_predictions = {}
    for logic in LOGICS:
        print(f"\n  --- Inferring {logic} ---")
        train_result = f"output/train_result_{logic}_4_{CLUSTER_NUM}_{SEED}.json"

        if not (WORKDIR / train_result).exists():
            print(f"  ERROR: Train result not found: {train_result}")
            return False

        ok = run_artifact_command(
            [sys.executable, "SMTportfolio.py", "infer",
             "-clusterPortfolio", train_result,
             "-dataset", logic,
             "-solverdict", f"machfea/{logic}_solver.json",
             "-seed", str(SEED)],
            label=f"infer-{logic}",
            timeout=600
        )
        if not ok:
            print(f"  ERROR: Inference failed for {logic}")
            return False

        output_file = WORKDIR / "output" / f"test_result_{logic}_{SEED}_{CLUSTER_NUM}.json"
        with open(output_file) as f:
            preds = json.load(f)
        all_predictions.update(preds)
        print(f"  {logic}: {len(preds)} predictions")

    # Write to DB
    print(f"\nRegistering selector in DB...")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    selector_name = "smtgazer_suite9k_portfolio7"
    ts = get_utc_timestamp()

    row = cur.execute("SELECT id FROM ml_selectors WHERE name = ?", (selector_name,)).fetchone()
    if row:
        selector_id = row[0]
        # Clear old decisions
        cur.execute("DELETE FROM decisions WHERE selector_id = ?", (selector_id,))
        print(f"  Using existing selector '{selector_name}' (id={selector_id}), cleared old decisions")
    else:
        cur.execute("""
            INSERT INTO ml_selectors (name, model_type, portfolio_id, model_path, training_info, created_utc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            selector_name, "SMTGazer_Artifact", PORTFOLIO_ID,
            str(WORKDIR / "output"),
            json.dumps({
                'cluster_num': CLUSTER_NUM, 'seed': SEED,
                'portfolio_size': 4, 'logics': LOGICS,
                'smac_n_trials': 200, 'smac_workers': SMAC_WORKERS,
                'suite': SUITE_NAME, 'k_fold_cv': K_FOLDS,
            }),
            ts
        ))
        selector_id = cur.lastrowid
        print(f"  Registered selector '{selector_name}' (id={selector_id})")

    conn.commit()

    # Write decisions
    written = 0
    skipped = 0

    for key, prediction in all_predictions.items():
        if key not in metadata:
            skipped += 1
            continue

        instance_id = metadata[key]['instance_id']
        pred_solvers, pred_times = prediction

        # Pick the solver with largest time allocation as the "selected" one
        # (SMTGazer uses sequential portfolio, but we record the primary choice)
        # For confidence: use inverse distance from cluster center
        if pred_solvers:
            # First solver in the portfolio gets the most time usually
            primary_solver = pred_solvers[0]
            if primary_solver in SOLVER_LIST:
                solver_idx = SOLVER_LIST.index(primary_solver)
                config_id = CONFIG_IDS[solver_idx]
            else:
                skipped += 1
                continue

            # Confidence: fraction of total time allocated to primary solver
            total_time = sum(pred_times) if pred_times else 1
            confidence = pred_times[0] / total_time if total_time > 0 else 0

            # Confidence scores: map each step's solver to its time fraction
            conf_scores = {}
            for solver_name, t in zip(pred_solvers, pred_times):
                if solver_name in SOLVER_LIST:
                    sidx = SOLVER_LIST.index(solver_name)
                    cid = CONFIG_IDS[sidx]
                    conf_scores[str(cid)] = round(t / total_time, 4) if total_time > 0 else 0

            cur.execute("""
                INSERT OR REPLACE INTO decisions
                (selector_id, instance_id, selected_config_id, step_num, confidence, confidence_scores, ts_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (selector_id, instance_id, config_id, 1, confidence,
                  json.dumps(conf_scores), ts))
            written += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()

    print(f"\n  Decisions written: {written}")
    print(f"  Skipped: {skipped}")
    print(f"  Selector ID: {selector_id}")

    return True


# =====================================================================
# Main
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python smtgazer_full_pipeline.py {train|kfold|register|aggregate|all} [--fold N]")
        return 1

    step = sys.argv[1]

    # Parse --fold N
    fold_num = None
    if "--fold" in sys.argv:
        fold_idx = sys.argv.index("--fold")
        if fold_idx + 1 < len(sys.argv):
            fold_num = int(sys.argv[fold_idx + 1])

    print("=" * 70)
    label = step
    if fold_num is not None:
        label = f"{step} --fold {fold_num}"
    print(f"SMTGazer Full Pipeline — {label}")
    print("=" * 70)
    print(f"Workdir: {WORKDIR}")
    print(f"DB: {DB_PATH}")
    print(f"SMAC workers: {SMAC_WORKERS}")
    print()

    if step in ("train", "all"):
        if not phase_train():
            print("\nTRAINING FAILED")
            return 1

    if step in ("kfold", "all"):
        if not phase_kfold(fold_num=fold_num):
            print("\nK-FOLD FAILED")
            return 1

    if step == "aggregate":
        if not phase_aggregate():
            print("\nAGGREGATE FAILED")
            return 1

    if step in ("register", "all"):
        if not phase_register():
            print("\nREGISTRATION FAILED")
            return 1

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
