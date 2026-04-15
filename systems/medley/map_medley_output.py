#!/usr/bin/env python3
"""
Map bin/medley output (which uses Z3/CVC4/BOOLECTOR/YICES/MathSAT/Bitwuzla names)
back to our actual config names, evaluate against baselines, and write to DB.
"""

import argparse
import csv
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOLVER_TO_CONFIG = {
    'Z3':        'cvc5_baseline_default',
    'CVC4':      'cvc5_qfnra_05_nl_cov',
    'BOOLECTOR': 'cvc5_qfnra_08_decision_just',
    'YICES':     'z3_baseline_default',
    'MathSAT':   'z3_baseline_qfbv_sat_euf_sat',
    'Bitwuzla':  'z3_baseline_qflia_case_split',
}

CONFIG_NAMES = list(SOLVER_TO_CONFIG.values())
LOGICS = ['QF_BV', 'QF_LIA', 'QF_NRA']
PORTFOLIO_ID = 7
SUITE = 'suite_9k'


def par2(status, runtime_ms, timeout_s):
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2.0 * timeout_s


def get_utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="medley_results_bin")
    parser.add_argument("--db", default="benchmark-suite/temp_benchmarks/db/results.sqlite")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-db-write", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Parse raw Medley output and map solver names back to config names
    medley_decisions = {}  # original_file_path -> config_name

    for logic in LOGICS:
        raw_csv = input_dir / f"medley_raw_{logic}.csv"
        if not raw_csv.exists():
            print(f"  Warning: {raw_csv} not found, skipping {logic}")
            continue

        with open(raw_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                staged_path = row[0]      # staged path
                solve_method = row[2]     # Medley solver name (Z3, CVC4, etc.)

                # Convert solver name back to our config name
                config_name = SOLVER_TO_CONFIG.get(solve_method, solve_method)

                # Strip staging prefix to get original path
                # staged_path looks like: medley_staging_bin/data/benchmarks/...
                parts = staged_path.split('/')
                # Find 'data' in the path
                for i, p in enumerate(parts):
                    if p == 'data':
                        orig_path = '/'.join(parts[i:])
                        break
                else:
                    orig_path = staged_path

                medley_decisions[orig_path] = config_name

    print(f"  Mapped {len(medley_decisions)} decisions")

    # Write cleaned decisions CSV
    decisions_csv = input_dir / "decisions_all.csv"
    with open(decisions_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_path', 'predicted_config'])
        for fp in sorted(medley_decisions.keys()):
            w.writerow([fp, medley_decisions[fp]])
    print(f"  Written: {decisions_csv}")

    # Load DB data for evaluation
    conn = sqlite3.connect(args.db, timeout=30)
    rows = conn.execute("""
        SELECT i.file_path, i.logic, c.name, r.status, r.runtime_ms,
               i.id AS instance_id, c.id AS config_id
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c   ON r.config_id  = c.id
        JOIN portfolio_configs pc ON pc.config_id = c.id AND pc.portfolio_id = ?
        WHERE i.suite_name = ?
          AND r.config_id IN (SELECT config_id FROM portfolio_configs WHERE portfolio_id = ?)
    """, (PORTFOLIO_ID, SUITE, PORTFOLIO_ID)).fetchall()

    grouped = {}
    for fp, logic, cfg, status, rt, iid, cid in rows:
        if fp not in grouped:
            grouped[fp] = {'logic': logic, 'runs': {}}
        grouped[fp]['runs'][cfg] = {'status': status, 'runtime_ms': rt,
                                     'instance_id': iid, 'config_id': cid}

    # Evaluate
    n = len(grouped)
    medley_par2 = 0.0
    medley_solved = 0
    vbs_par2 = 0.0
    vbs_solved = 0

    for fp, data in sorted(grouped.items()):
        # VBS
        best_cfg = min(data['runs'], key=lambda c: par2(data['runs'][c]['status'], data['runs'][c]['runtime_ms'], args.timeout))
        vbs_par2 += par2(data['runs'][best_cfg]['status'], data['runs'][best_cfg]['runtime_ms'], args.timeout)
        vbs_solved += 1 if data['runs'][best_cfg]['status'] in ('sat', 'unsat') else 0

        # Medley
        med_cfg = medley_decisions.get(fp)
        if med_cfg and med_cfg in data['runs']:
            s = par2(data['runs'][med_cfg]['status'], data['runs'][med_cfg]['runtime_ms'], args.timeout)
            medley_par2 += s
            medley_solved += 1 if data['runs'][med_cfg]['status'] in ('sat', 'unsat') else 0
        else:
            medley_par2 += 2.0 * args.timeout

    print(f"\n  Results (n={n}):")
    print(f"    VBS:    PAR-2={vbs_par2:>12.2f}  solved={vbs_solved}/{n} ({vbs_solved/n*100:.1f}%)")
    print(f"    MEDLEY: PAR-2={medley_par2:>12.2f}  solved={medley_solved}/{n} ({medley_solved/n*100:.1f}%)")

    # Write to DB
    if not args.skip_db_write:
        cur = conn.cursor()
        ts = get_utc_timestamp()

        for logic in LOGICS:
            name = f"medley_bin_knearest_{logic.lower()}_{SUITE}"
            row = cur.execute("SELECT id FROM ml_selectors WHERE name = ?", (name,)).fetchone()
            if row:
                selector_id = row[0]
                cur.execute("UPDATE ml_selectors SET model_type=? WHERE id=?",
                            ("Medley_bin_medley_EndToEnd", selector_id))
            else:
                cur.execute("""
                    INSERT INTO ml_selectors (name, model_type, portfolio_id, training_info, created_utc)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, "Medley_bin_medley_EndToEnd", PORTFOLIO_ID,
                      json.dumps({'approach': 'bin/medley unmodified, SOLVERS name mapping'}),
                      ts))
                selector_id = cur.lastrowid
            print(f"    Selector '{name}' -> id={selector_id}")

            written = 0
            for fp, data in grouped.items():
                if data['logic'] != logic:
                    continue
                med_cfg = medley_decisions.get(fp)
                if not med_cfg or med_cfg not in data['runs']:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO decisions
                    (selector_id, instance_id, selected_config_id, step_num, ts_utc)
                    VALUES (?, ?, ?, 1, ?)
                """, (selector_id, data['runs'][med_cfg]['instance_id'],
                      data['runs'][med_cfg]['config_id'], ts))
                written += 1
            print(f"      {logic}: {written} decisions")

        conn.commit()

    conn.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
