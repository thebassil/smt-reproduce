#!/usr/bin/env python3
"""
Export DB run data to CSV files in the format medleysolver.dispatch expects.

Medley's dispatch.py looks for: {benchmark_dir}/{SOLVER_NAME}.csv
where SOLVER_NAME is one of: Z3, CVC4, BOOLECTOR, YICES, MathSAT, Bitwuzla

We map our 6 configs to these names so bin/medley can run unmodified:
  Z3         <- cvc5_baseline_default
  CVC4       <- cvc5_qfnra_05_nl_cov
  BOOLECTOR  <- cvc5_qfnra_08_decision_just
  YICES      <- z3_baseline_default
  MathSAT    <- z3_baseline_qfbv_sat_euf_sat
  Bitwuzla   <- z3_baseline_qflia_case_split

Also symlinks .smt2 files so Medley's feature extractor can read them.
"""

import argparse
import csv
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

# Our configs -> Medley's SOLVERS keys
CONFIG_TO_SOLVER = {
    'cvc5_baseline_default':        'Z3',
    'cvc5_qfnra_05_nl_cov':        'CVC4',
    'cvc5_qfnra_08_decision_just':  'BOOLECTOR',
    'z3_baseline_default':          'YICES',
    'z3_baseline_qfbv_sat_euf_sat': 'MathSAT',
    'z3_baseline_qflia_case_split': 'Bitwuzla',
}

SOLVER_TO_CONFIG = {v: k for k, v in CONFIG_TO_SOLVER.items()}

PORTFOLIO_ID = 7
SUITE = 'suite_9k'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="benchmark-suite/temp_benchmarks/db/results.sqlite")
    parser.add_argument("--staging-dir", default="medley_staging_bin")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db, timeout=30)
    rows = conn.execute("""
        SELECT i.file_path, c.name AS config_name, r.status, r.runtime_ms
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c   ON r.config_id  = c.id
        JOIN portfolio_configs pc ON pc.config_id = c.id AND pc.portfolio_id = ?
        WHERE i.suite_name = ?
          AND r.config_id IN (SELECT config_id FROM portfolio_configs WHERE portfolio_id = ?)
    """, (PORTFOLIO_ID, SUITE, PORTFOLIO_ID)).fetchall()
    conn.close()

    print(f"  Loaded {len(rows)} runs from DB")

    staging = Path(args.staging_dir)

    # Group by (directory, config)
    by_dir_solver = defaultdict(list)
    all_dirs = set()
    for file_path, config_name, status, runtime_ms in rows:
        d = os.path.dirname(file_path)
        solver_name = CONFIG_TO_SOLVER[config_name]
        by_dir_solver[(d, solver_name)].append((file_path, status, runtime_ms))
        all_dirs.add(d)

    # Write dispatch CSVs
    for (d, solver_name), entries in by_dir_solver.items():
        out_dir = staging / d
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{solver_name}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for file_path, status, runtime_ms in entries:
                # Use staging path so basename matching works
                staged_path = str(staging / file_path)
                writer.writerow([staged_path, '', '', runtime_ms / 1000.0, status])

    print(f"  Written {len(by_dir_solver)} CSV files across {len(all_dirs)} directories")

    # Symlink .smt2 files
    for d in all_dirs:
        staging_d = staging / d
        staging_d.mkdir(parents=True, exist_ok=True)
        real_d = Path(d)
        if not real_d.exists():
            continue
        for smt2 in real_d.glob("*.smt2"):
            link = staging_d / smt2.name
            if not link.exists():
                link.symlink_to(smt2.resolve())

    print(f"  Symlinked .smt2 files in {len(all_dirs)} directories")

    # Save mapping for later
    mapping_path = staging / "solver_config_mapping.csv"
    with open(mapping_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['medley_solver', 'our_config'])
        for solver, config in sorted(SOLVER_TO_CONFIG.items()):
            w.writerow([solver, config])
    print(f"  Mapping saved: {mapping_path}")


if __name__ == "__main__":
    main()
