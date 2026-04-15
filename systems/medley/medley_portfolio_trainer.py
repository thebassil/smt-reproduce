#!/usr/bin/env python3
"""
Medley Portfolio Trainer for suite_9k benchmarks (6-config portfolio)

Runs the REAL medleysolver pipeline end-to-end:
  /dcs/large/u5573765/artifacts/medley-solver/
  (https://github.com/uclid-org/medley-solver)

Approach:
  1. Export our pre-computed run data from the DB into per-directory CSV files
     in the format that Medley's dispatch.py expects
  2. Call the real medleysolver.runner.execute() with KNearest classifier and
     Exponential time manager — the same code path as `bin/medley`
  3. Parse the output CSV produced by execute()
  4. Record decisions to our DB and CSV files

This is a TRUE reproduction — the entire Medley code path is exercised:
  runner.execute() → compute_features.get_features() → classifiers.KNearest →
  timers.Exponential → dispatch.run_problem() → constants.Solved_Problem
"""

import sys
import os
import types

# ============================================================================
# Setup: import real medleysolver with patched SOLVERS
# ============================================================================

MEDLEY_ARTIFACT = '/dcs/large/u5573765/artifacts/medley-solver'
sys.path.insert(0, MEDLEY_ARTIFACT)

# Mock z3 — only needed for probe-based features, not bag-of-words
sys.modules['z3'] = types.ModuleType('z3')

# Patch SOLVERS *before* importing anything else from medleysolver
import medleysolver.constants
from collections import OrderedDict

CONFIG_NAMES = [
    'cvc5_baseline_default',        # config_id 115
    'cvc5_qfnra_05_nl_cov',        # config_id 109
    'cvc5_qfnra_08_decision_just',  # config_id 112
    'z3_baseline_default',          # config_id 114
    'z3_baseline_qfbv_sat_euf_sat', # config_id 118
    'z3_baseline_qflia_case_split', # config_id 116
]

medleysolver.constants.SOLVERS = OrderedDict({name: '' for name in CONFIG_NAMES})

# Now import real Medley components (they see our patched SOLVERS)
from medleysolver.runner import execute
from medleysolver.classifiers import KNearest
from medleysolver.timers import Exponential
from medleysolver.constants import SOLVERS, Solved_Problem, keyword_list

import argparse
import csv
import json
import sqlite3
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

LOGICS = ['QF_BV', 'QF_LIA', 'QF_NRA']

DEFAULT_PORTFOLIO_ID = 7
DEFAULT_SUITE = 'suite_9k'
DEFAULT_TIMEOUT_S = 60
DEFAULT_NFOLDS = 5

# KNearest defaults (from Medley CLI: bin/medley)
DEFAULT_K = 10
DEFAULT_EPSILON = 0.9       # bin/medley default
DEFAULT_DECAY = 0.9         # bin/medley default (--epsilon_decay)
DEFAULT_TIME_K = 40         # bin/medley default
DEFAULT_CONFIDENCE = 0.95   # bin/medley default
DEFAULT_REWARD = 'bump'     # bin/medley default
DEFAULT_EXTRA_TIME_FIRST = True  # bin/medley default
DEFAULT_SEED = hash("Go bears!") % 2**32  # bin/medley default seed


# ============================================================================
# Database helpers
# ============================================================================

def get_utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_run_data(db_path, portfolio_id, suite_name):
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute("""
        SELECT i.file_path, i.logic, c.name AS config_name, r.status, r.runtime_ms,
               i.id AS instance_id, c.id AS config_id
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c   ON r.config_id  = c.id
        JOIN portfolio_configs pc ON pc.config_id = c.id AND pc.portfolio_id = ?
        WHERE i.suite_name = ?
          AND r.config_id IN (SELECT config_id FROM portfolio_configs WHERE portfolio_id = ?)
        ORDER BY i.file_path, c.name
    """, (portfolio_id, suite_name, portfolio_id)).fetchall()
    conn.close()
    return [
        {'file_path': r[0], 'logic': r[1], 'config_name': r[2],
         'status': r[3], 'runtime_ms': r[4],
         'instance_id': r[5], 'config_id': r[6]}
        for r in rows
    ]


def register_ml_selector(db_path, name, model_type, portfolio_id, model_path=None, training_info=None):
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    row = cur.execute("SELECT id FROM ml_selectors WHERE name = ?", (name,)).fetchone()
    if row:
        selector_id = row[0]
        cur.execute("""
            UPDATE ml_selectors SET model_type=?, portfolio_id=?, model_path=?, training_info=?
            WHERE id=?
        """, (model_type, portfolio_id, model_path,
              json.dumps(training_info) if training_info else None, selector_id))
        print(f"    Updated selector '{name}' (id={selector_id})")
    else:
        cur.execute("""
            INSERT INTO ml_selectors (name, model_type, portfolio_id, model_path, training_info, created_utc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, model_type, portfolio_id, model_path,
              json.dumps(training_info) if training_info else None, get_utc_timestamp()))
        selector_id = cur.lastrowid
        print(f"    Registered selector '{name}' (id={selector_id})")
    conn.commit()
    conn.close()
    return selector_id


def write_decisions_to_db(db_path, selector_id, decisions):
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    ts = get_utc_timestamp()
    written = 0
    for d in decisions:
        try:
            cur.execute("""
                INSERT OR REPLACE INTO decisions
                (selector_id, instance_id, selected_config_id, step_num, confidence, confidence_scores, ts_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (selector_id, d['instance_id'], d['config_id'],
                  d.get('step_num', 1), d.get('confidence'),
                  json.dumps(d.get('confidence_scores')) if d.get('confidence_scores') else None, ts))
            written += 1
        except sqlite3.IntegrityError as e:
            print(f"      Warning: decision write failed for instance {d['instance_id']}: {e}")
    conn.commit()
    conn.close()
    return written


# ============================================================================
# PAR-2 / baselines
# ============================================================================

def par2(status, runtime_ms, timeout_s):
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2.0 * timeout_s


def group_by_benchmark(run_data):
    grouped = {}
    for r in run_data:
        fp = r['file_path']
        if fp not in grouped:
            grouped[fp] = {}
        grouped[fp][r['config_name']] = r
    return grouped


def compute_baselines(grouped, timeout_s):
    benchmarks = sorted(grouped.keys())
    n = len(benchmarks)

    # VBS
    vbs_par2_total = 0.0
    vbs_solved = 0
    vbs_decisions = {}
    for fp in benchmarks:
        best_cfg = min(grouped[fp], key=lambda c: par2(grouped[fp][c]['status'], grouped[fp][c]['runtime_ms'], timeout_s))
        score = par2(grouped[fp][best_cfg]['status'], grouped[fp][best_cfg]['runtime_ms'], timeout_s)
        vbs_par2_total += score
        vbs_solved += 1 if grouped[fp][best_cfg]['status'] in ('sat', 'unsat') else 0
        vbs_decisions[fp] = best_cfg

    # SBS
    cfg_totals = defaultdict(float)
    for fp in benchmarks:
        for cfg, row in grouped[fp].items():
            cfg_totals[cfg] += par2(row['status'], row['runtime_ms'], timeout_s)
    sbs_cfg = min(cfg_totals, key=cfg_totals.get)
    sbs_par2 = 0.0
    sbs_solved = 0
    for fp in benchmarks:
        row = grouped[fp].get(sbs_cfg)
        if row:
            s = par2(row['status'], row['runtime_ms'], timeout_s)
            sbs_par2 += s
            sbs_solved += 1 if row['status'] in ('sat', 'unsat') else 0
        else:
            sbs_par2 += 2.0 * timeout_s

    # Combined SBS
    logic_map = {}
    for fp in benchmarks:
        first_row = next(iter(grouped[fp].values()))
        logic = first_row['logic']
        if logic not in logic_map:
            logic_map[logic] = defaultdict(float)
        for cfg, row in grouped[fp].items():
            logic_map[logic][cfg] += par2(row['status'], row['runtime_ms'], timeout_s)
    combined_sbs_cfg = {logic: min(ct, key=ct.get) for logic, ct in logic_map.items()}
    csbs_par2 = 0.0
    csbs_solved = 0
    for fp in benchmarks:
        first_row = next(iter(grouped[fp].values()))
        logic = first_row['logic']
        csbs = combined_sbs_cfg[logic]
        row = grouped[fp].get(csbs)
        if row:
            s = par2(row['status'], row['runtime_ms'], timeout_s)
            csbs_par2 += s
            csbs_solved += 1 if row['status'] in ('sat', 'unsat') else 0
        else:
            csbs_par2 += 2.0 * timeout_s

    # Random (expected)
    random_par2 = 0.0
    random_solved = 0
    for fp in benchmarks:
        configs = grouped[fp]
        avg = sum(par2(r['status'], r['runtime_ms'], timeout_s) for r in configs.values()) / len(configs)
        random_par2 += avg
        random_solved += sum(1 for r in configs.values() if r['status'] in ('sat', 'unsat')) / len(configs)

    return {
        'VBS': {'total_par2': vbs_par2_total, 'solved': vbs_solved, 'decisions': vbs_decisions},
        'SBS': {'total_par2': sbs_par2, 'solved': sbs_solved, 'config': sbs_cfg},
        'COMBINED_SBS': {'total_par2': csbs_par2, 'solved': csbs_solved, 'per_logic_config': combined_sbs_cfg},
        'RANDOM': {'total_par2': random_par2, 'solved': int(round(random_solved))},
        'n': n,
    }


# ============================================================================
# Step 1: Export DB results to Medley's dispatch CSV format
# ============================================================================

def export_dispatch_csvs(run_data, staging_dir):
    """
    Export run results to per-directory CSV files in the format that
    medleysolver.dispatch.run_problem() expects:

        directory/{config_name}.csv

    Each row: benchmark_path, _, _, runtime_seconds, status_string

    run_problem() reads:
      - column 0: benchmark path (matched by basename)
      - column 3: elapsed time (float, seconds)
      - column 4: result string (parsed by output2result: sat/unsat/unknown/error)
    """
    # Group by (directory, config)
    by_dir_cfg = defaultdict(list)
    for r in run_data:
        d = os.path.dirname(r['file_path'])
        by_dir_cfg[(d, r['config_name'])].append(r)

    staging = Path(staging_dir)
    written_files = 0

    dirs_seen = set()
    for (d, cfg), rows in by_dir_cfg.items():
        # Create staging directory mirroring the benchmark layout
        out_dir = staging / d
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{cfg}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for r in rows:
                runtime_s = r['runtime_ms'] / 1000.0
                status = r['status']  # sat, unsat, timeout, error → output2result handles these
                writer.writerow([r['file_path'], '', '', runtime_s, status])
        written_files += 1
        dirs_seen.add(d)

    return written_files, len(dirs_seen)


def create_symlinks(staging_dir, benchmark_dirs):
    """
    Create symlinks from staging benchmark dirs to real .smt2 files,
    so Medley's get_syntactic_count_features() can read them.
    """
    staging = Path(staging_dir)
    for d in benchmark_dirs:
        staging_d = staging / d
        staging_d.mkdir(parents=True, exist_ok=True)
        real_d = Path(d)
        if not real_d.exists():
            continue
        for smt2 in real_d.glob("*.smt2"):
            link = staging_d / smt2.name
            if not link.exists():
                link.symlink_to(smt2.resolve())


# ============================================================================
# Step 2: Run real Medley execute()
# ============================================================================

def run_medley_execute(problems, output_csv, timeout, k, epsilon, decay, time_k,
                       confidence, reward, extra_time_first, seed):
    """
    Call the real medleysolver.runner.execute() — the same code path as bin/medley.
    """
    np.random.seed(seed)

    # Shuffle problems (same as bin/medley does)
    np.random.shuffle(problems)

    # Create classifier (same as bin/medley --classifier knearest)
    classifier = KNearest(k, epsilon, decay, time_k)

    # Create time manager (same as bin/medley --timeout_manager expo)
    init_lambda = 1 / (timeout / len(SOLVERS))
    time_manager = Exponential(init_lambda, confidence, timeout)

    # Call the real execute() — this is the complete Medley pipeline
    execute(
        problems=problems,
        output=output_csv,
        classifier=classifier,
        time_manager=time_manager,
        timeout=timeout,
        feature_file="bow",  # use bag-of-words features (default)
        extra_time_to_first=extra_time_first,
        reward=reward,
        features2use=None,
    )

    return classifier


def parse_medley_output(output_csv):
    """
    Parse the CSV output from medleysolver.runner.execute().

    Each row is a Solved_Problem namedtuple written as CSV:
      problem, datapoint, solve_method, time, result, order, time_spent
    """
    decisions = {}
    with open(output_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            problem = row[0]       # file path
            solve_method = row[2]  # selected solver config
            elapsed = float(row[3])
            result = row[4]
            decisions[problem] = {
                'solve_method': solve_method,
                'elapsed': elapsed,
                'result': result,
            }
    return decisions


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medley KNearest — REAL end-to-end medleysolver pipeline, suite_9k 6-config portfolio")
    parser.add_argument("--db", default="benchmark-suite/temp_benchmarks/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=DEFAULT_PORTFOLIO_ID)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", default="medley_results")
    parser.add_argument("--staging-dir", default="medley_staging",
                        help="Temp dir for Medley dispatch CSV files")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY)
    parser.add_argument("--time-k", type=int, default=DEFAULT_TIME_K)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--reward", default=DEFAULT_REWARD, choices=["binary", "bump", "exp"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-db-write", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Medley Portfolio Trainer (REAL end-to-end medleysolver pipeline)")
    print("=" * 70)
    print(f"  Medley artifact  : {MEDLEY_ARTIFACT}")
    print(f"  Database         : {args.db}")
    print(f"  Portfolio ID     : {args.portfolio_id}")
    print(f"  Suite            : {args.suite}")
    print(f"  Output dir       : {output_dir}")
    print(f"  Staging dir      : {args.staging_dir}")
    print(f"  KNearest params  : k={args.k}, epsilon={args.epsilon}, decay={args.decay}, time_k={args.time_k}")
    print(f"  Time manager     : Exponential (confidence={args.confidence})")
    print(f"  Reward           : {args.reward}")
    print(f"  Timeout          : {args.timeout}s")
    print(f"  Seed             : {args.seed}")
    print(f"  SOLVERS (patched): {list(SOLVERS.keys())}")

    # ------------------------------------------------------------------
    # 1. Extract run data
    # ------------------------------------------------------------------
    print("\n[1/5] Extracting run data from database ...")
    run_data = extract_run_data(args.db, args.portfolio_id, args.suite)
    print(f"  Total runs: {len(run_data)}")
    all_grouped = group_by_benchmark(run_data)
    print(f"  Unique benchmarks: {len(all_grouped)}")

    logic_runs = defaultdict(list)
    for r in run_data:
        logic_runs[r['logic']].append(r)

    logic_grouped = {}
    for logic in LOGICS:
        logic_grouped[logic] = group_by_benchmark(logic_runs[logic])
        print(f"  {logic}: {len(logic_grouped[logic])} benchmarks")

    # ------------------------------------------------------------------
    # 2. Export dispatch CSVs (Medley's expected format)
    # ------------------------------------------------------------------
    print("\n[2/5] Exporting dispatch CSV files for Medley ...")
    staging_dir = Path(args.staging_dir)

    n_files, n_dirs = export_dispatch_csvs(run_data, staging_dir)
    print(f"  Written {n_files} CSV files across {n_dirs} directories")

    # Symlink .smt2 files so Medley's feature extractor can read them
    benchmark_dirs = set(os.path.dirname(fp) for fp in all_grouped.keys())
    create_symlinks(staging_dir, benchmark_dirs)
    print(f"  Symlinked .smt2 files in {len(benchmark_dirs)} directories")

    # ------------------------------------------------------------------
    # 3. Run real Medley per logic
    # ------------------------------------------------------------------
    print("\n[3/5] Running real medleysolver.runner.execute() per logic ...")

    medley_decisions = {}  # file_path -> solve_method

    for logic in LOGICS:
        bms = sorted(logic_grouped[logic].keys())
        # Remap paths into staging dir so run_problem finds the CSVs
        staged_problems = [str(staging_dir / fp) for fp in bms]

        output_csv = str(output_dir / f"medley_raw_{logic}.csv")

        print(f"\n  --- {logic} ({len(bms)} benchmarks) ---")
        print(f"    Running execute() ...")

        classifier = run_medley_execute(
            problems=staged_problems,
            output_csv=output_csv,
            timeout=args.timeout,
            k=args.k,
            epsilon=args.epsilon,
            decay=args.decay,
            time_k=args.time_k,
            confidence=args.confidence,
            reward=args.reward,
            extra_time_first=DEFAULT_EXTRA_TIME_FIRST,
            seed=args.seed,
        )

        print(f"    Classifier has {len(classifier.solved)} solved problems")
        print(f"    Raw output: {output_csv}")

        # Save classifier
        model_path = output_dir / f"medley_{logic}.pkl"
        classifier.save(str(model_path))
        print(f"    Model saved: {model_path}")

        # Parse output
        raw_decisions = parse_medley_output(output_csv)
        print(f"    Parsed {len(raw_decisions)} decisions")

        # Map staged paths back to original paths
        for staged_path, info in raw_decisions.items():
            # Strip staging_dir prefix to recover original relative path
            orig_path = staged_path
            staging_prefix = str(staging_dir) + '/'
            if orig_path.startswith(staging_prefix):
                orig_path = orig_path[len(staging_prefix):]
            medley_decisions[orig_path] = info['solve_method']

        # Selection distribution
        dist = defaultdict(int)
        for info in raw_decisions.values():
            dist[info['solve_method']] += 1
        for cfg in CONFIG_NAMES:
            print(f"      {cfg}: {dist.get(cfg, 0)}")

    print(f"\n  Total Medley decisions: {len(medley_decisions)}")

    # ------------------------------------------------------------------
    # 4. Evaluate against baselines
    # ------------------------------------------------------------------
    print("\n[4/5] Computing evaluation metrics ...")
    baselines = compute_baselines(all_grouped, args.timeout)
    n = baselines['n']

    medley_par2_total = 0.0
    medley_solved = 0
    per_benchmark_rows = []

    for fp in sorted(all_grouped.keys()):
        row_data = all_grouped[fp]
        first_row = next(iter(row_data.values()))
        logic = first_row['logic']

        vbs_cfg = baselines['VBS']['decisions'][fp]
        vbs_score = par2(row_data[vbs_cfg]['status'], row_data[vbs_cfg]['runtime_ms'], args.timeout)

        med_cfg = medley_decisions.get(fp)
        if med_cfg and med_cfg in row_data:
            med_score = par2(row_data[med_cfg]['status'], row_data[med_cfg]['runtime_ms'], args.timeout)
            med_status = row_data[med_cfg]['status']
        else:
            med_cfg = med_cfg or 'N/A'
            med_score = 2.0 * args.timeout
            med_status = 'missing'

        medley_par2_total += med_score
        medley_solved += 1 if med_status in ('sat', 'unsat') else 0

        per_benchmark_rows.append({
            'file_path': fp, 'logic': logic,
            'vbs_config': vbs_cfg, 'vbs_par2': f"{vbs_score:.3f}",
            'medley_config': med_cfg, 'medley_par2': f"{med_score:.3f}",
            'matched_vbs': 'yes' if med_cfg == vbs_cfg else 'no',
            'regret': f"{med_score - vbs_score:.3f}",
        })

    strategies = [
        ('VBS',          baselines['VBS']['total_par2'],          baselines['VBS']['solved']),
        ('MEDLEY',       medley_par2_total,                       medley_solved),
        ('COMBINED_SBS', baselines['COMBINED_SBS']['total_par2'], baselines['COMBINED_SBS']['solved']),
        ('SBS',          baselines['SBS']['total_par2'],          baselines['SBS']['solved']),
        ('RANDOM',       baselines['RANDOM']['total_par2'],       baselines['RANDOM']['solved']),
    ]

    print("\n" + "=" * 70)
    print(f"  {'Strategy':<15} {'Total PAR-2':>14} {'Avg PAR-2':>12} {'Solved':>8} {'%':>8}")
    print("  " + "-" * 62)
    for name, total, solved in strategies:
        avg = total / n if n else 0
        pct = solved / n * 100 if n else 0
        print(f"  {name:<15} {total:>14.2f} {avg:>12.3f} {solved:>8} {pct:>7.1f}%")
    print("=" * 70)

    vbs_p = baselines['VBS']['total_par2']
    sbs_p = baselines['SBS']['total_par2']
    if sbs_p != vbs_p:
        closeness = (sbs_p - medley_par2_total) / (sbs_p - vbs_p) * 100
        print(f"  Medley closeness to VBS: {closeness:.1f}%")
    print(f"  SBS config: {baselines['SBS']['config']}")
    print(f"  Combined SBS: {baselines['COMBINED_SBS']['per_logic_config']}")

    # ------------------------------------------------------------------
    # 5. Save outputs + DB
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving CSV outputs and writing to DB ...")

    # Summary CSV
    summary_csv = output_dir / "kfold_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['strategy', 'total_par2', 'avg_par2', 'solved', 'solved_pct'])
        w.writeheader()
        for name, total, solved in strategies:
            w.writerow({
                'strategy': name,
                'total_par2': f"{total:.2f}",
                'avg_par2': f"{total / n:.3f}" if n else '',
                'solved': solved,
                'solved_pct': f"{solved / n * 100:.1f}%" if n else '',
            })
    print(f"  {summary_csv}")

    # Per-benchmark CSV
    per_bm_csv = output_dir / "kfold_per_benchmark.csv"
    if per_benchmark_rows:
        with open(per_bm_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=per_benchmark_rows[0].keys())
            w.writeheader()
            w.writerows(per_benchmark_rows)
    print(f"  {per_bm_csv}")

    # Decisions CSV
    decisions_csv = output_dir / "decisions_all.csv"
    with open(decisions_csv, 'w', newline='') as f:
        fieldnames = ['file_path', 'logic', 'predicted_config']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for fp in sorted(all_grouped.keys()):
            first_row = next(iter(all_grouped[fp].values()))
            w.writerow({
                'file_path': fp,
                'logic': first_row['logic'],
                'predicted_config': medley_decisions.get(fp, 'N/A'),
            })
    print(f"  {decisions_csv}")

    # Per-logic decisions
    for logic in LOGICS:
        logic_csv = output_dir / f"decisions_{logic}.csv"
        with open(logic_csv, 'w', newline='') as f:
            fieldnames = ['file_path', 'predicted_config']
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for fp in sorted(logic_grouped[logic].keys()):
                w.writerow({
                    'file_path': fp,
                    'predicted_config': medley_decisions.get(fp, 'N/A'),
                })
        print(f"  {logic_csv}")

    # Write to DB
    if not args.skip_db_write:
        all_db_decisions = []
        for logic in LOGICS:
            selector_name = f"medley_knearest_{logic.lower()}_{args.suite}"
            model_path = str(output_dir / f"medley_{logic}.pkl")
            selector_id = register_ml_selector(
                db_path=args.db,
                name=selector_name,
                model_type="Medley_KNearest_EndToEnd",
                portfolio_id=args.portfolio_id,
                model_path=model_path,
                training_info={
                    'medley_artifact': MEDLEY_ARTIFACT,
                    'classifier': 'KNearest',
                    'k': args.k, 'epsilon': args.epsilon, 'decay': args.decay,
                    'time_k': args.time_k, 'confidence': args.confidence,
                    'reward': args.reward, 'timeout': args.timeout,
                    'seed': args.seed, 'suite': args.suite,
                    'config_names': CONFIG_NAMES,
                    'approach': 'real_execute_end_to_end',
                },
            )

            logic_decisions = []
            lg = logic_grouped[logic]
            for fp in sorted(lg.keys()):
                med_cfg = medley_decisions.get(fp)
                if not med_cfg or med_cfg not in lg[fp]:
                    continue
                first_row = next(iter(lg[fp].values()))
                instance_id = first_row['instance_id']
                config_id = lg[fp][med_cfg]['config_id']
                logic_decisions.append({
                    'instance_id': instance_id,
                    'config_id': config_id,
                    'step_num': 1,
                })

            written = write_decisions_to_db(args.db, selector_id, logic_decisions)
            print(f"    {logic}: {written} decisions to DB (selector_id={selector_id})")
            all_db_decisions.extend(logic_decisions)

        print(f"  Total DB decisions: {len(all_db_decisions)}")

    print("\n" + "=" * 70)
    print("COMPLETE (REAL end-to-end medleysolver pipeline)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
