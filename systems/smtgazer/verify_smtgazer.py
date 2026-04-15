#!/usr/bin/env python3
"""
SMTGazer Artifact Smoke Test

Verifies the actual SMTGazer artifact works end-to-end on tiger:
1. Feature extraction via machfea
2. SMTportfolio.py train (normalize → cluster → SMAC3 greedy portfolio)
3. SMTportfolio.py infer (assign test instances to clusters → pick portfolio)
4. Output format validation

Uses 10 QF_BV benchmarks with 2 clusters for a fast check (~5 min).
"""

import json
import os
import shutil
import sys
import sqlite3
import tempfile
from pathlib import Path

# === Paths ===
ARTIFACT_DIR = Path("/dcs/23/u5573765/cs351/smt-small-sbs-smtgazer/artifacts/smtgazer/SMTGazer")
MACHFEA_DIR = ARTIFACT_DIR / "machfea"
BENCHMARK_DIR = Path("/dcs/large/u5573765/data/benchmarks/suite_9k/QF_BV")
DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")
WORKDIR = Path("/dcs/large/u5573765/smtgazer_verify")

PORTFOLIO_ID = 7
TIMEOUT_S = 60
PAR2_PENALTY = TIMEOUT_S * 2
CONFIG_IDS = [109, 112, 114, 115, 116, 118]
SOLVER_LIST = [
    "cvc5::cvc5_qfnra_05_nl_cov",
    "cvc5::cvc5_qfnra_08_decision_just",
    "z3::z3_baseline_default",
    "cvc5::cvc5_baseline_default",
    "z3::z3_baseline_qflia_case_split",
    "z3::z3_baseline_qfbv_sat_euf_sat",
]
N_BENCHMARKS = 10
CLUSTER_NUM = 2
SEED = 42
DATASET = "QF_BV"


def setup_workdir():
    """Create working directory with artifact copies and symlinks."""
    print(f"Setting up workdir: {WORKDIR}")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)

    # Copy artifact scripts
    shutil.copy2(ARTIFACT_DIR / "SMTportfolio.py", WORKDIR / "SMTportfolio.py")
    shutil.copy2(ARTIFACT_DIR / "portfolio_smac3.py", WORKDIR / "portfolio_smac3.py")

    # Symlink custom SMAC3 — portfolio_smac3.py does sys.path.insert(0, CWD+'/smac')
    # then "from smac import ...", so we need workdir/smac/smac/__init__.py to exist
    (WORKDIR / "smac").symlink_to(ARTIFACT_DIR / "smac")

    # Create directory structure
    (WORKDIR / "data").mkdir()
    (WORKDIR / "machfea" / "infer_result").mkdir(parents=True)
    (WORKDIR / "tmp").mkdir()
    (WORKDIR / "output").mkdir()

    print("  Copied SMTportfolio.py, portfolio_smac3.py")
    print(f"  Symlinked smac/ -> {ARTIFACT_DIR / 'smac'}")


def pick_benchmarks():
    """Pick N_BENCHMARKS from QF_BV that have complete runs in DB."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT i.id, i.file_path, i.family, i.file_name
        FROM instances i
        WHERE i.suite_name = 'suite_9k' AND i.logic = ?
        ORDER BY i.id
        LIMIT ?
    """, (DATASET, N_BENCHMARKS * 2)).fetchall()

    selected = []
    for inst_id, file_path, family, file_name in rows:
        runs = cur.execute("""
            SELECT config_id, status, runtime_ms
            FROM runs
            WHERE instance_id = ? AND portfolio_id = ? AND config_id IN ({})
        """.format(",".join("?" * len(CONFIG_IDS))),
        (inst_id, PORTFOLIO_ID, *CONFIG_IDS)).fetchall()

        if len(runs) == len(CONFIG_IDS):
            selected.append((inst_id, file_path, family, file_name, runs))
            if len(selected) >= N_BENCHMARKS:
                break

    conn.close()
    print(f"  Selected {len(selected)} benchmarks with complete runs")
    return selected


def generate_labels(benchmarks):
    """Generate Labels.json from selected benchmarks."""
    train_data = {}
    for inst_id, file_path, family, file_name, runs in benchmarks:
        key = f"{family}/{file_name}"
        runs_by_config = {r[0]: r for r in runs}
        times = []
        for cid in CONFIG_IDS:
            _, status, runtime_ms = runs_by_config[cid]
            if status in ('sat', 'unsat'):
                times.append(runtime_ms / 1000.0)
            else:
                times.append(PAR2_PENALTY)
        train_data[key] = times

    labels = {"train": train_data}
    labels_path = WORKDIR / "data" / f"{DATASET}Labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f"  Labels: {labels_path} ({len(train_data)} entries)")
    return train_data


def generate_solver_json():
    """Generate solver.json."""
    solver_path = WORKDIR / "machfea" / f"{DATASET}_solver.json"
    with open(solver_path, 'w') as f:
        json.dump({"solver_list": SOLVER_LIST}, f)
    print(f"  Solver list: {solver_path}")


def extract_features(benchmarks):
    """Extract machfea features for selected benchmarks."""
    sys.path.insert(0, str(MACHFEA_DIR))
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]

    from machsmt.benchmark import Benchmark as MachBenchmark

    sys.argv = saved_argv

    features = {}
    base_dir = Path("/dcs/large/u5573765/data/benchmarks/suite_9k")
    for inst_id, file_path, family, file_name, runs in benchmarks:
        key = f"{family}/{file_name}"
        full_path = base_dir / DATASET / family / file_name
        try:
            b = MachBenchmark(str(full_path))
            b.parse()
            features[key] = b.get_features()
        except Exception as e:
            print(f"  WARNING: Failed to extract {key}: {e}")

    # Write as both train and test features (we'll use same set for smoke test)
    train_path = WORKDIR / "machfea" / "infer_result" / f"{DATASET}_train_feature.json"
    test_path = WORKDIR / "machfea" / "infer_result" / f"{DATASET}_test_feature.json"
    with open(train_path, 'w') as f:
        json.dump(features, f)
    with open(test_path, 'w') as f:
        json.dump(features, f)

    print(f"  Features: {len(features)} benchmarks × {len(next(iter(features.values())))} dims")
    return features


def run_train():
    """Run SMTportfolio.py train."""
    import subprocess
    cmd = [
        sys.executable, "SMTportfolio.py", "train",
        "-dataset", DATASET,
        "-solverdict", f"machfea/{DATASET}_solver.json",
        "-seed", str(SEED),
        "-cluster_num", str(CLUSTER_NUM),
    ]
    print(f"  Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["SMAC_WORKERS"] = "1"  # n_workers>1 deadlocks on tiger
    result = subprocess.run(cmd, cwd=str(WORKDIR), env=env,
                           capture_output=True, text=True, timeout=6000)
    print(f"  Return code: {result.returncode}")
    if result.stdout:
        # Print last 30 lines of stdout
        lines = result.stdout.strip().split('\n')
        for line in lines[-30:]:
            print(f"    {line}")
    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split('\n')[-20:]:
            print(f"    STDERR: {line}")
    return result.returncode == 0


def validate_train_output():
    """Validate the train output file."""
    output_file = WORKDIR / "output" / f"train_result_{DATASET}_{4}_{CLUSTER_NUM}_{SEED}.json"
    if not output_file.exists():
        print(f"  FAIL: Output file not found: {output_file}")
        return False

    with open(output_file) as f:
        data = json.load(f)

    # Check required keys
    for key in ['portfolio', 'lim', 'center']:
        if key not in data:
            print(f"  FAIL: Missing key '{key}' in output")
            return False

    portfolio = data['portfolio']
    print(f"  Portfolio has {len(portfolio)} clusters")

    for cluster_id, entry in portfolio.items():
        solver_indices, time_allocs = entry
        print(f"    Cluster {cluster_id}: solvers={solver_indices}, "
              f"times={[f'{t:.1f}' for t in time_allocs]}, "
              f"sum={sum(time_allocs):.1f}s")
        if len(solver_indices) != len(time_allocs):
            print(f"  FAIL: Mismatch in cluster {cluster_id}")
            return False

    # Check center and lim
    centers = data['center']['center']
    print(f"  Centers: {len(centers)} clusters × {len(centers[0])} dims")

    lim = data['lim']
    print(f"  Normalization: min({len(lim['min'])} dims), sub({len(lim['sub'])} dims)")

    print("  PASS: Train output is valid")
    return True


def run_infer():
    """Run SMTportfolio.py infer."""
    import subprocess
    output_file = f"output/train_result_{DATASET}_{4}_{CLUSTER_NUM}_{SEED}.json"
    cmd = [
        sys.executable, "SMTportfolio.py", "infer",
        "-clusterPortfolio", output_file,
        "-dataset", DATASET,
        "-solverdict", f"machfea/{DATASET}_solver.json",
        "-seed", str(SEED),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(WORKDIR),
                           capture_output=True, text=True, timeout=300)
    print(f"  Return code: {result.returncode}")
    if result.stdout:
        lines = result.stdout.strip().split('\n')
        for line in lines[-10:]:
            print(f"    {line}")
    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split('\n')[-10:]:
            print(f"    STDERR: {line}")
    return result.returncode == 0


def validate_infer_output():
    """Validate the infer output file."""
    output_file = WORKDIR / "output" / f"test_result_{DATASET}_{SEED}_{CLUSTER_NUM}.json"
    if not output_file.exists():
        print(f"  FAIL: Infer output not found: {output_file}")
        return False

    with open(output_file) as f:
        data = json.load(f)

    print(f"  Infer output: {len(data)} predictions")
    for key, value in list(data.items())[:3]:
        solvers, times = value
        print(f"    {key}: solvers={solvers[:2]}..., times={[f'{t:.1f}' for t in times[:2]]}...")

    print("  PASS: Infer output is valid")
    return True


def main():
    print("=" * 70)
    print("SMTGazer Artifact Smoke Test")
    print("=" * 70)

    results = {}

    # Step 1: Setup
    print("\n[1/7] Setting up working directory...")
    setup_workdir()
    results['setup'] = True

    # Step 2: Pick benchmarks
    print("\n[2/7] Selecting benchmarks from DB...")
    benchmarks = pick_benchmarks()
    results['benchmarks'] = len(benchmarks) >= N_BENCHMARKS

    # Step 3: Generate labels
    print("\n[3/7] Generating Labels.json...")
    generate_labels(benchmarks)
    results['labels'] = True

    # Step 4: Generate solver.json
    print("\n[4/7] Generating solver.json...")
    generate_solver_json()
    results['solver_json'] = True

    # Step 5: Extract features
    print("\n[5/7] Extracting machfea features...")
    features = extract_features(benchmarks)
    results['features'] = len(features) >= N_BENCHMARKS

    # Step 6: Train
    print("\n[6/7] Running SMTportfolio.py train...")
    train_ok = run_train()
    results['train'] = train_ok
    if train_ok:
        results['train_output'] = validate_train_output()
    else:
        results['train_output'] = False

    # Step 7: Infer
    print("\n[7/7] Running SMTportfolio.py infer...")
    if results['train_output']:
        infer_ok = run_infer()
        results['infer'] = infer_ok
        if infer_ok:
            results['infer_output'] = validate_infer_output()
        else:
            results['infer_output'] = False
    else:
        results['infer'] = False
        results['infer_output'] = False

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    all_pass = True
    for step, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {step:<20} {status}")

    print("=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED - Artifact is ready for full pipeline")
    else:
        print("SOME CHECKS FAILED - Investigate before running full pipeline")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
