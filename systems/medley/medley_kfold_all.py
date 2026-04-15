#!/usr/bin/env python3
"""
Medley K-Fold Cross-Validation — All Classifiers

Runs k-fold CV for every Medley classifier variant on suite_9k:
  - MLP          (sklearn neural network)
  - linear       (contextual linear bandit)
  - knearest     (k-nearest neighbors, count voting)
  - neighbor     (nearest neighbor, distance/speed)
  - thompson     (Thompson sampling, Bayesian bandit)
  - exp3         (adversarial bandit)
  - greedy       (epsilon-greedy bandit)
  - random       (baseline)

Methodology (matches MachSMT k-fold):
  1. Stratified split by logic (each logic split independently into k folds)
  2. Per fold: train on k-1 folds using bin/medley, evaluate on held-out fold
  3. Evaluation: load saved classifier, predict without further learning
  4. Aggregate across folds → PAR-2, solved%, vs baselines

Usage:
    python medley_kfold_all.py [--k 5] [--classifiers knearest,MLP,...]
    python medley_kfold_all.py --classifiers all
"""

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import shutil
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

MEDLEY_DIR = Path("/dcs/large/u5573765/artifacts/medley-solver")
MEDLEY_WRAPPER = Path(__file__).parent / "run_medley_wrapper.py"

CONFIG_TO_SOLVER = {
    'cvc5_baseline_default':        'Z3',
    'cvc5_qfnra_05_nl_cov':        'CVC4',
    'cvc5_qfnra_08_decision_just':  'BOOLECTOR',
    'z3_baseline_default':          'YICES',
    'z3_baseline_qfbv_sat_euf_sat': 'MathSAT',
    'z3_baseline_qflia_case_split': 'Bitwuzla',
}
SOLVER_TO_CONFIG = {v: k for k, v in CONFIG_TO_SOLVER.items()}

ALL_CLASSIFIERS = ['knearest', 'MLP', 'neighbor', 'linear', 'thompson',
                   'exp3', 'greedy', 'random']

PORTFOLIO_ID = 7
SUITE = 'suite_9k'
LOGICS = ['QF_BV', 'QF_LIA', 'QF_NRA']
TIMEOUT_S = 60


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(db_path: str) -> dict:
    """Load all run data for suite_9k, grouped by file_path."""
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute("""
        SELECT i.file_path, i.logic, i.family, c.name, r.status, r.runtime_ms,
               i.id AS instance_id, c.id AS config_id
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c   ON r.config_id  = c.id
        JOIN portfolio_configs pc ON pc.config_id = c.id AND pc.portfolio_id = ?
        WHERE i.suite_name = ?
    """, (PORTFOLIO_ID, SUITE)).fetchall()
    conn.close()

    data = {}
    for fp, logic, family, cfg, status, rt, iid, cid in rows:
        if fp not in data:
            data[fp] = {'logic': logic, 'family': family, 'runs': {}}
        data[fp]['runs'][cfg] = {
            'status': status, 'runtime_ms': rt,
            'instance_id': iid, 'config_id': cid,
        }

    print(f"Loaded {len(data)} instances, {len(rows)} runs")
    return data


def par2(status, runtime_ms, timeout_s=TIMEOUT_S):
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2.0 * timeout_s


# ── Fold Splitting ───────────────────────────────────────────────────────────

def make_folds(data: dict, k: int, seed: int, by_family: bool = False) -> List[Tuple[List[str], List[str]]]:
    """
    Stratified k-fold split. Returns list of (train_paths, test_paths).
    Stratified by logic. If by_family=True, entire families stay together.
    """
    rng = np.random.RandomState(seed)
    by_logic = defaultdict(list)

    if by_family:
        # Group by (logic, family), keep families intact
        families_by_logic = defaultdict(lambda: defaultdict(list))
        for fp, info in data.items():
            families_by_logic[info['logic']][info['family']].append(fp)

        folds = [[] for _ in range(k)]
        for logic, fam_dict in families_by_logic.items():
            fam_names = list(fam_dict.keys())
            rng.shuffle(fam_names)
            for i, fam in enumerate(fam_names):
                folds[i % k].extend(fam_dict[fam])
    else:
        # Standard stratified split by logic
        for fp, info in data.items():
            by_logic[info['logic']].append(fp)

        folds = [[] for _ in range(k)]
        for logic, paths in by_logic.items():
            rng.shuffle(paths)
            fold_size = len(paths) // k
            for i in range(k):
                start = i * fold_size
                end = start + fold_size if i < k - 1 else len(paths)
                folds[i].extend(paths[start:end])

    result = []
    for i in range(k):
        test = folds[i]
        train = [p for j in range(k) if j != i for p in folds[j]]
        result.append((train, test))

    return result


# ── Staging ──────────────────────────────────────────────────────────────────

def create_fold_staging(data: dict, train_paths: List[str], staging_dir: Path):
    """Create staging directory with dispatch CSVs + .smt2 symlinks for training data only."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    by_dir_solver = defaultdict(list)
    all_dirs = set()

    for fp in train_paths:
        info = data[fp]
        d = os.path.dirname(fp)
        all_dirs.add(d)
        for cfg, run in info['runs'].items():
            solver = CONFIG_TO_SOLVER[cfg]
            staged_path = str(staging_dir / fp)
            by_dir_solver[(d, solver)].append(
                (staged_path, run['status'], run['runtime_ms'])
            )

    # Write solver CSVs
    for (d, solver), entries in by_dir_solver.items():
        out_dir = staging_dir / d
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{solver}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for staged_path, status, runtime_ms in entries:
                writer.writerow([staged_path, '', '', runtime_ms / 1000.0, status])

    # Symlink .smt2 files (only training ones)
    for fp in train_paths:
        real_file = Path(fp)
        if not real_file.exists():
            continue
        link = staging_dir / fp
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(real_file.resolve())


# ── Training (via bin/medley) ────────────────────────────────────────────────

def train_fold(staging_dir: Path, output_dir: Path, logic: str,
               classifier: str, timeout_s: int = TIMEOUT_S) -> Path:
    """Run bin/medley on training data for one logic, save classifier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"train_{logic}.csv"
    classifier_pkl = output_dir / f"classifier_{logic}.pkl"

    input_dir = str(staging_dir / f"data/benchmarks/{SUITE}/{logic}")

    env = os.environ.copy()
    env['PYTHONPATH'] = str(MEDLEY_DIR) + ':' + env.get('PYTHONPATH', '')

    cmd = [
        sys.executable, str(MEDLEY_WRAPPER),
        input_dir, str(output_csv),
        '--classifier', classifier,
        '--timeout', str(timeout_s),
        '--feature_setting', 'bow',
        '--save_classifier', str(classifier_pkl),
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    ERROR training {classifier}/{logic}: {result.stderr[-500:]}")
        return None

    return classifier_pkl if classifier_pkl.exists() else None


# ── Evaluation (load classifier, predict on test) ───────────────────────────

def evaluate_fold(data: dict, test_paths: List[str], train_paths: List[str],
                  output_dir: Path, classifier_name: str,
                  timeout_s: int = TIMEOUT_S) -> dict:
    """
    Load trained classifiers and evaluate on test instances.

    For each test instance:
      1. Compute BoW features
      2. Normalize with training running mean
      3. Call classifier.get_ordering() with high count (no exploration)
      4. Take first solver → look up actual result
    """
    # Add medleysolver to path
    if str(MEDLEY_DIR) not in sys.path:
        sys.path.insert(0, str(MEDLEY_DIR))

    import dill
    from medleysolver.compute_features import get_syntactic_count_features

    results = {
        'vbs_par2': 0.0, 'vbs_solved': 0,
        'sbs_par2': 0.0, 'sbs_solved': 0,
        'csbs_par2': 0.0, 'csbs_solved': 0,
        'medley_par2': 0.0, 'medley_solved': 0,
        'total': 0,
    }

    # Compute SBS and CSBS from training data
    solver_totals = defaultdict(float)
    solver_logic_totals = defaultdict(lambda: defaultdict(float))
    for fp in train_paths:
        info = data[fp]
        for cfg, run in info['runs'].items():
            p = par2(run['status'], run['runtime_ms'], timeout_s)
            solver_totals[cfg] += p
            solver_logic_totals[info['logic']][cfg] += p

    sbs = min(solver_totals, key=solver_totals.get)
    csbs = {}
    for logic, totals in solver_logic_totals.items():
        csbs[logic] = min(totals, key=totals.get)

    # Load classifiers and compute training running mean per logic
    classifiers_by_logic = {}
    train_means = {}

    for logic in LOGICS:
        pkl_path = output_dir / f"classifier_{logic}.pkl"
        if not pkl_path.exists():
            print(f"    WARNING: {pkl_path} not found, skipping {logic}")
            continue

        with open(pkl_path, 'rb') as f:
            clf = dill.load(f)
        classifiers_by_logic[logic] = clf

        # Compute running mean from training instances of this logic
        train_logic = [fp for fp in train_paths if data[fp]['logic'] == logic]
        mean = 0
        count = 0
        for fp in train_logic:
            real_file = Path(fp)
            if not real_file.exists():
                continue
            count += 1
            features = np.array(get_syntactic_count_features(str(real_file)))
            mean = (count - 1) / count * mean + 1 / count * features
        train_means[logic] = mean
        print(f"    {logic}: classifier loaded, mean computed from {count} instances")

    # Evaluate each test instance
    decisions = []

    for fp in sorted(test_paths):
        info = data[fp]
        logic = info['logic']
        results['total'] += 1

        # VBS
        best_cfg = min(info['runs'], key=lambda c: par2(info['runs'][c]['status'],
                                                         info['runs'][c]['runtime_ms'], timeout_s))
        vbs_p = par2(info['runs'][best_cfg]['status'], info['runs'][best_cfg]['runtime_ms'], timeout_s)
        results['vbs_par2'] += vbs_p
        if info['runs'][best_cfg]['status'] in ('sat', 'unsat'):
            results['vbs_solved'] += 1

        # SBS
        if sbs in info['runs']:
            sp = par2(info['runs'][sbs]['status'], info['runs'][sbs]['runtime_ms'], timeout_s)
        else:
            sp = 2 * timeout_s
        results['sbs_par2'] += sp
        if sbs in info['runs'] and info['runs'][sbs]['status'] in ('sat', 'unsat'):
            results['sbs_solved'] += 1

        # CSBS
        csbs_cfg = csbs.get(logic)
        if csbs_cfg and csbs_cfg in info['runs']:
            cp = par2(info['runs'][csbs_cfg]['status'], info['runs'][csbs_cfg]['runtime_ms'], timeout_s)
        else:
            cp = 2 * timeout_s
        results['csbs_par2'] += cp
        if csbs_cfg and csbs_cfg in info['runs'] and info['runs'][csbs_cfg]['status'] in ('sat', 'unsat'):
            results['csbs_solved'] += 1

        # Medley prediction
        clf = classifiers_by_logic.get(logic)
        if clf is None or not Path(fp).exists():
            results['medley_par2'] += 2 * timeout_s
            continue

        features = np.array(get_syntactic_count_features(str(fp)))
        mean = train_means[logic]
        normalized = features / (mean + 1e-9)

        # get_ordering with high count → epsilon * decay^count ≈ 0 → always exploit
        ordering = clf.get_ordering(normalized, 999999, fp)
        predicted_solver = ordering[0]
        predicted_cfg = SOLVER_TO_CONFIG.get(predicted_solver, predicted_solver)

        if predicted_cfg in info['runs']:
            run = info['runs'][predicted_cfg]
            mp = par2(run['status'], run['runtime_ms'], timeout_s)
            results['medley_par2'] += mp
            if run['status'] in ('sat', 'unsat'):
                results['medley_solved'] += 1
            decisions.append((fp, predicted_cfg, run['status'], run['runtime_ms']))
        else:
            results['medley_par2'] += 2 * timeout_s
            decisions.append((fp, predicted_cfg, 'missing', 0))

    return results, decisions


# ── One complete fold run ────────────────────────────────────────────────────

def run_one_fold(data, fold_idx, train_paths, test_paths,
                 classifier, work_dir, timeout_s):
    """Train + evaluate one fold for one classifier. Returns results dict."""
    fold_dir = work_dir / f"fold{fold_idx}"
    staging_dir = fold_dir / "staging"
    output_dir = fold_dir / "output"

    print(f"\n  [Fold {fold_idx+1}] {classifier}: train={len(train_paths)}, test={len(test_paths)}")

    # Create staging
    create_fold_staging(data, train_paths, staging_dir)

    # Train per logic
    for logic in LOGICS:
        pkl = train_fold(staging_dir, output_dir, logic, classifier, timeout_s)
        if pkl:
            print(f"    Trained {logic} -> {pkl.name}")
        else:
            print(f"    FAILED {logic}")

    # Evaluate
    results, decisions = evaluate_fold(
        data, test_paths, train_paths, output_dir, classifier, timeout_s
    )

    # Clean up staging to save disk
    shutil.rmtree(staging_dir, ignore_errors=True)

    return results, decisions


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Medley K-Fold CV — All Classifiers")
    parser.add_argument("--db", default="benchmark-suite/temp_benchmarks/db/results.sqlite")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--classifiers", default="all",
                        help="Comma-separated classifiers or 'all'")
    parser.add_argument("--output-dir", default="medley_kfold_results")
    parser.add_argument("--work-dir", default="medley_kfold_work")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for training (1=sequential)")
    parser.add_argument("--by-family", action="store_true",
                        help="Split by family (keeps families together)")
    parser.add_argument("--skip-db-write", action="store_true")
    args = parser.parse_args()

    if args.classifiers == 'all':
        classifiers = ALL_CLASSIFIERS
    else:
        classifiers = [c.strip() for c in args.classifiers.split(',')]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Medley K-Fold Cross-Validation — All Classifiers")
    print("=" * 70)
    print(f"Database:    {args.db}")
    print(f"K-folds:     {args.k}")
    print(f"Seed:        {args.seed}")
    print(f"Timeout:     {args.timeout}s")
    print(f"Classifiers: {classifiers}")
    print(f"By-family:   {args.by_family}")
    print(f"Workers:     {args.workers}")
    print()

    # Load data
    data = load_data(args.db)

    # Create folds
    folds = make_folds(data, args.k, args.seed, args.by_family)
    for i, (train, test) in enumerate(folds):
        by_logic_test = defaultdict(int)
        for fp in test:
            by_logic_test[data[fp]['logic']] += 1
        print(f"  Fold {i+1}: train={len(train)}, test={len(test)} "
              f"({dict(sorted(by_logic_test.items()))})")

    # Run all classifiers × all folds
    all_results = {}

    for clf_name in classifiers:
        print(f"\n{'='*70}")
        print(f"CLASSIFIER: {clf_name}")
        print(f"{'='*70}")

        clf_work = work_dir / clf_name
        clf_work.mkdir(parents=True, exist_ok=True)
        clf_results = []
        all_decisions = []

        for fold_idx, (train_paths, test_paths) in enumerate(folds):
            results, decisions = run_one_fold(
                data, fold_idx, train_paths, test_paths,
                clf_name, clf_work, args.timeout
            )
            clf_results.append(results)
            all_decisions.extend(decisions)

            n = results['total']
            print(f"    VBS:    {results['vbs_solved']:>5}/{n} ({results['vbs_solved']/n*100:.1f}%)  "
                  f"PAR-2={results['vbs_par2']:>10.1f}")
            print(f"    MEDLEY: {results['medley_solved']:>5}/{n} ({results['medley_solved']/n*100:.1f}%)  "
                  f"PAR-2={results['medley_par2']:>10.1f}")
            print(f"    CSBS:   {results['csbs_solved']:>5}/{n} ({results['csbs_solved']/n*100:.1f}%)  "
                  f"PAR-2={results['csbs_par2']:>10.1f}")

        # Aggregate
        totals = {}
        for key in clf_results[0].keys():
            totals[key] = sum(r[key] for r in clf_results)

        all_results[clf_name] = totals

        n = totals['total']
        print(f"\n  AGGREGATE {clf_name} ({n} instances, {args.k}-fold):")
        print(f"    VBS:    {totals['vbs_solved']:>5}/{n} ({totals['vbs_solved']/n*100:.1f}%)  "
              f"PAR-2={totals['vbs_par2']:>10.1f}")
        print(f"    MEDLEY: {totals['medley_solved']:>5}/{n} ({totals['medley_solved']/n*100:.1f}%)  "
              f"PAR-2={totals['medley_par2']:>10.1f}")
        print(f"    CSBS:   {totals['csbs_solved']:>5}/{n} ({totals['csbs_solved']/n*100:.1f}%)  "
              f"PAR-2={totals['csbs_par2']:>10.1f}")
        print(f"    SBS:    {totals['sbs_solved']:>5}/{n} ({totals['sbs_solved']/n*100:.1f}%)  "
              f"PAR-2={totals['sbs_par2']:>10.1f}")

        if totals['csbs_par2'] > 0:
            delta = (totals['medley_par2'] - totals['csbs_par2']) / totals['csbs_par2'] * 100
            print(f"    vs CSBS: {delta:+.1f}%")

        # Save decisions CSV
        dec_csv = output_dir / f"decisions_{clf_name}.csv"
        with open(dec_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['file_path', 'predicted_config', 'status', 'runtime_ms'])
            for row in sorted(all_decisions):
                w.writerow(row)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("SUMMARY — ALL CLASSIFIERS")
    print(f"{'='*70}")
    print(f"{'Classifier':<14} {'Solved':>7} {'Solve%':>7} {'PAR-2':>12} {'vs CSBS':>9} {'vs VBS':>9}")
    print("-" * 60)

    # VBS line (same for all)
    ref = list(all_results.values())[0]
    n = ref['total']
    print(f"{'VBS':<14} {ref['vbs_solved']:>7} {ref['vbs_solved']/n*100:>6.1f}% "
          f"{ref['vbs_par2']:>12.1f} {'[ORACLE]':>9}")

    # Each classifier
    for clf_name in classifiers:
        t = all_results[clf_name]
        csbs_delta = (t['medley_par2'] - t['csbs_par2']) / t['csbs_par2'] * 100 if t['csbs_par2'] > 0 else 0
        vbs_gap = (t['medley_par2'] - t['vbs_par2']) / t['vbs_par2'] * 100 if t['vbs_par2'] > 0 else 0
        print(f"{clf_name:<14} {t['medley_solved']:>7} {t['medley_solved']/n*100:>6.1f}% "
              f"{t['medley_par2']:>12.1f} {csbs_delta:>+8.1f}% {vbs_gap:>+8.1f}%")

    # CSBS line
    print(f"{'CSBS':<14} {ref['csbs_solved']:>7} {ref['csbs_solved']/n*100:>6.1f}% "
          f"{ref['csbs_par2']:>12.1f} {'[BASE]':>9}")
    print(f"{'SBS':<14} {ref['sbs_solved']:>7} {ref['sbs_solved']/n*100:>6.1f}% "
          f"{ref['sbs_par2']:>12.1f}")

    # ── Save summary CSV ─────────────────────────────────────────────────────
    summary_csv = output_dir / "kfold_summary_all_classifiers.csv"
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['classifier', 'total_par2', 'avg_par2', 'solved', 'solved_pct',
                     'total', 'vs_csbs_pct', 'vs_vbs_pct'])

        w.writerow(['VBS', f"{ref['vbs_par2']:.2f}",
                     f"{ref['vbs_par2']/n:.3f}",
                     ref['vbs_solved'],
                     f"{ref['vbs_solved']/n*100:.1f}", n, '', ''])

        for clf_name in classifiers:
            t = all_results[clf_name]
            csbs_d = (t['medley_par2'] - t['csbs_par2']) / t['csbs_par2'] * 100
            vbs_d = (t['medley_par2'] - t['vbs_par2']) / t['vbs_par2'] * 100
            w.writerow([clf_name, f"{t['medley_par2']:.2f}",
                         f"{t['medley_par2']/n:.3f}",
                         t['medley_solved'],
                         f"{t['medley_solved']/n*100:.1f}", n,
                         f"{csbs_d:+.1f}", f"{vbs_d:+.1f}"])

        w.writerow(['CSBS', f"{ref['csbs_par2']:.2f}",
                     f"{ref['csbs_par2']/n:.3f}",
                     ref['csbs_solved'],
                     f"{ref['csbs_solved']/n*100:.1f}", n, '', ''])
        w.writerow(['SBS', f"{ref['sbs_par2']:.2f}",
                     f"{ref['sbs_par2']/n:.3f}",
                     ref['sbs_solved'],
                     f"{ref['sbs_solved']/n*100:.1f}", n, '', ''])

    print(f"\nSummary saved: {summary_csv}")

    # ── Per-fold CSV ─────────────────────────────────────────────────────────
    perfold_csv = output_dir / "kfold_perfold_all_classifiers.csv"
    with open(perfold_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['classifier', 'fold', 'total', 'vbs_solved', 'medley_solved',
                     'csbs_solved', 'vbs_par2', 'medley_par2', 'csbs_par2'])
        # We'd need to save per-fold results separately — not in current structure
        # This is a placeholder; the per-fold printing above covers this

    # ── Write to DB ──────────────────────────────────────────────────────────
    if not args.skip_db_write:
        conn = sqlite3.connect(args.db, timeout=30)
        cur = conn.cursor()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for clf_name in classifiers:
            dec_csv = output_dir / f"decisions_{clf_name}.csv"
            if not dec_csv.exists():
                continue

            for logic in LOGICS:
                name = f"medley_kfold_{clf_name}_{logic.lower()}_{SUITE}"
                row = cur.execute("SELECT id FROM ml_selectors WHERE name = ?",
                                  (name,)).fetchone()
                if row:
                    selector_id = row[0]
                    cur.execute("UPDATE ml_selectors SET model_type=? WHERE id=?",
                                (f"Medley_{clf_name}_KFold{args.k}", selector_id))
                else:
                    cur.execute("""
                        INSERT INTO ml_selectors (name, model_type, portfolio_id, training_info, created_utc)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, f"Medley_{clf_name}_KFold{args.k}", PORTFOLIO_ID,
                          json.dumps({
                              'approach': f'bin/medley {clf_name}, {args.k}-fold CV',
                              'seed': args.seed,
                              'by_family': args.by_family,
                          }), ts))
                    selector_id = cur.lastrowid
                print(f"  Selector '{name}' -> id={selector_id}")

                # Write decisions
                written = 0
                with open(dec_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fp = row['file_path']
                        if fp not in data or data[fp]['logic'] != logic:
                            continue
                        predicted_cfg = row['predicted_config']
                        if predicted_cfg not in data[fp]['runs']:
                            continue
                        run = data[fp]['runs'][predicted_cfg]
                        cur.execute("""
                            INSERT OR REPLACE INTO decisions
                            (selector_id, instance_id, selected_config_id, step_num, ts_utc)
                            VALUES (?, ?, ?, 1, ?)
                        """, (selector_id, run['instance_id'], run['config_id'], ts))
                        written += 1
                print(f"    {logic}: {written} decisions")

        conn.commit()
        conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
