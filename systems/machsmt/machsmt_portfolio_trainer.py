#!/usr/bin/env python3
"""
MachSMT Portfolio Trainer for suite_9k (6-config cross-logic portfolio)

Trains SEPARATE models per logic (QF_BV, QF_LIA, QF_NRA), each choosing
from 6 solver-config pairs that are run across all logics.

CSV Schema (per logic):
    benchmark,solver,score
    /path/to/file.smt2,z3::z3_baseline_default,0.523
    /path/to/file.smt2,cvc5::cvc5_baseline_default,1.204
    ...

Assumptions:
    - benchmark: absolute path to .smt2 file (must exist for feature extraction)
    - solver: "{solver}::{config_name}" format
    - score: PAR-2 in seconds (runtime for sat/unsat, 2*timeout otherwise)
    - Each benchmark has 6 runs (one per config in portfolio 7)
"""

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

# Add MachSMT to path
MACHSMT_PATH = Path(__file__).parent / "artifacts" / "machsmt" / "MachSMT"
sys.path.insert(0, str(MACHSMT_PATH))


def extract_training_data_by_logic(db_path: str, portfolio_id: int, output_dir: Path):
    """
    Extract run data from SQLite, split by logic into separate CSVs.
    
    Returns: dict mapping logic -> (csv_path, num_benchmarks, num_rows)
    """
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()

    # Get portfolio info
    row = cur.execute("SELECT name, timeout_s FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
    if not row:
        raise ValueError(f"Portfolio {portfolio_id} not found")
    portfolio_name, timeout_s = row
    
    print(f"Portfolio: {portfolio_name} (timeout={timeout_s}s)")
    
    # Get all runs grouped by logic
    runs = cur.execute("""
        SELECT 
            i.file_path,
            i.logic,
            c.name as config_name,
            r.solver,
            r.status,
            r.runtime_ms
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        JOIN configs c ON r.config_id = c.id
        WHERE r.portfolio_id = ?
        ORDER BY i.logic, i.file_path, c.name
    """, (portfolio_id,)).fetchall()
    
    conn.close()
    
    if not runs:
        print("WARNING: No runs found for this portfolio")
        return {}
    
    # Group by logic
    by_logic = {}
    for file_path, logic, config_name, solver, status, runtime_ms in runs:
        if logic not in by_logic:
            by_logic[logic] = []
        by_logic[logic].append((file_path, config_name, solver, status, runtime_ms))
    
    # Write separate CSVs
    par2_penalty = 2 * timeout_s
    result = {}
    
    for logic, logic_runs in by_logic.items():
        csv_path = output_dir / f"training_{logic}.csv"
        benchmarks_seen = set()
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['benchmark', 'solver', 'score'])
            
            for file_path, config_name, solver, status, runtime_ms in logic_runs:
                solver_config = f"{solver}::{config_name}"
                
                if status in ('sat', 'unsat'):
                    score = runtime_ms / 1000.0
                else:
                    score = par2_penalty
                
                writer.writerow([file_path, solver_config, score])
                benchmarks_seen.add(file_path)
        
        result[logic] = {
            'csv_path': csv_path,
            'num_benchmarks': len(benchmarks_seen),
            'num_rows': len(logic_runs),
        }
        print(f"  {logic}: {len(benchmarks_seen)} benchmarks × {len(logic_runs)//len(benchmarks_seen)} configs = {len(logic_runs)} rows -> {csv_path.name}")
    
    return result


def train_machsmt_for_logic(csv_path: str, model_path: str, logic: str, k_fold: int = 5):
    """Train a MachSMT model for a single logic."""
    # Save and clear sys.argv to prevent MachSMT from parsing our arguments
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    from machsmt import MachSMT
    from machsmt.config.config import CONFIG_OBJ
    sys.argv = saved_argv

    CONFIG_OBJ.k = k_fold
    CONFIG_OBJ.cores = min(os.cpu_count() or 1, 8)
    CONFIG_OBJ.min_datapoints = 5
    
    print(f"\n  Training {logic} model on {csv_path}...")
    mach = MachSMT(csv_path, train_on_init=True)
    
    solvers = [s.get_name() for s in mach.db.get_solvers()]
    print(f"    Benchmarks: {len(mach.db.get_benchmarks())}")
    print(f"    Solver-configs: {solvers}")
    
    mach.save(model_path)
    print(f"    Saved: {model_path}")
    
    return mach


def record_decisions_for_logic(mach, logic: str, output_path: str,
                               db_path=None, selector_id=None,
                               instance_map=None, solver_config_map=None):
    """Record decisions for a single logic model. CSV always; DB if args provided."""
    benchmarks = mach.db.get_benchmarks()
    solver_configs = [s.get_name() for s in mach.db.get_solvers()]

    predictions, confidences = mach.predict(
        benchmarks=benchmarks,
        include_predictions=True,
    )

    fieldnames = ['benchmark', 'logic', 'predicted_solver_config'] + \
                 [f'conf_{sc}' for sc in solver_configs]

    rows = []
    for benchmark, pred_solver, conf_dict in zip(benchmarks, predictions, confidences):
        row = {
            'benchmark': benchmark.get_path(),
            'logic': logic,
            'predicted_solver_config': pred_solver.get_name(),
        }
        conf_scores = {}
        for sc in solver_configs:
            solver_obj = mach.db.get_solver(sc)
            score = conf_dict.get(solver_obj, 0.0)
            row[f'conf_{sc}'] = f"{score:.4f}"
            conf_scores[sc] = score
        row['_confidence_scores'] = conf_scores
        rows.append(row)

    # CSV write (always)
    csv_fieldnames = [f for f in fieldnames]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print(f"    Decisions (CSV): {output_path}")

    # DB write (optional)
    if db_path and selector_id is not None and instance_map and solver_config_map:
        from ml_db_utils import write_decisions_to_db
        db_decisions = []
        for row in rows:
            pred_sc = row['predicted_solver_config']
            conf_scores = row['_confidence_scores']
            max_conf = max(conf_scores.values()) if conf_scores else None
            db_decisions.append({
                'benchmark': row['benchmark'],
                'predicted_solver_config': pred_sc,
                'confidence': max_conf,
                'confidence_scores': conf_scores,
            })
        written, skipped = write_decisions_to_db(
            db_path, selector_id, db_decisions, instance_map, solver_config_map
        )
        print(f"    Decisions (DB): {written} written, {skipped} skipped")

    return rows


def merge_decisions(decision_files: list, output_path: str):
    """Merge per-logic decision CSVs into one combined file."""
    all_rows = []
    all_fieldnames = set()
    
    for fpath in decision_files:
        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            all_fieldnames.update(reader.fieldnames)
            all_rows.extend(list(reader))
    
    # Sort fieldnames: fixed cols first, then confidence cols
    fixed = ['benchmark', 'logic', 'predicted_solver_config']
    conf_cols = sorted([c for c in all_fieldnames if c.startswith('conf_')])
    fieldnames = fixed + conf_cols
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\nMerged decisions: {output_path} ({len(all_rows)} rows)")


def generate_summary(decisions_csv: str, summary_path: str):
    """Generate selection distribution summary."""
    from collections import defaultdict
    
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    with open(decisions_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            logic = row['logic']
            selected = row['predicted_solver_config']
            counts[logic][selected] += 1
            totals[logic] += 1
    
    all_solver_configs = sorted(set(sc for lc in counts.values() for sc in lc))
    
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ['logic', 'total'] + all_solver_configs
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for logic in sorted(counts.keys()):
            row = {'logic': logic, 'total': totals[logic]}
            for sc in all_solver_configs:
                count = counts[logic].get(sc, 0)
                pct = (count / totals[logic] * 100) if totals[logic] > 0 else 0
                row[sc] = f"{count} ({pct:.1f}%)" if count > 0 else "-"
            writer.writerow(row)
    
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train per-logic MachSMT models for Portfolio Type 1"
    )
    parser.add_argument("--db", default="/dcs/large/u5573765/db/results.sqlite")
    parser.add_argument("--portfolio-id", type=int, default=7)
    parser.add_argument("--output-dir", default="machsmt_results")
    parser.add_argument("--k-fold", type=int, default=5)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--write-to-db", action="store_true",
                        help="Also write decisions to the live DB (ml_selectors + decisions tables)")
    parser.add_argument("--selector-prefix", default="machsmt_suite9k",
                        help="Prefix for DB selector names, e.g. 'machsmt_suite9k' -> 'machsmt_suite9k_qf_lia'")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MachSMT Portfolio Trainer (Per-Logic Models)")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Portfolio ID: {args.portfolio_id}")
    print(f"Output: {output_dir}/")
    print()
    
    # Step 1: Extract per-logic CSVs
    print("Step 1: Extracting training data by logic...")
    logic_data = extract_training_data_by_logic(args.db, args.portfolio_id, output_dir)
    
    if not logic_data:
        print("\nERROR: No training data. Run portfolio benchmarks first.")
        return 1
    
    # Load DB maps if dual-write enabled
    instance_map = None
    solver_config_map = None
    if args.write_to_db:
        from ml_db_utils import (get_instance_map, get_solver_config_map,
                                 register_ml_selector)
        print(f"\nDB dual-write enabled (prefix: {args.selector_prefix})")
        instance_map = get_instance_map(args.db)
        solver_config_map = get_solver_config_map(args.db)
        print(f"  Loaded {len(instance_map)} instances, {len(solver_config_map)} configs from DB")

    # Step 2: Train per-logic models
    print("\nStep 2: Training per-logic models...")
    models = {}
    decision_files = []

    for logic, info in logic_data.items():
        model_path = output_dir / f"machsmt_{logic}.pkl"
        decisions_path = output_dir / f"decisions_{logic}.csv"

        if args.skip_training and model_path.exists():
            saved_argv = sys.argv
            sys.argv = [sys.argv[0]]
            from machsmt import MachSMT
            sys.argv = saved_argv
            print(f"\n  Loading existing {logic} model...")
            mach = MachSMT.load(str(model_path))
        else:
            mach = train_machsmt_for_logic(
                str(info['csv_path']),
                str(model_path),
                logic,
                args.k_fold
            )

        models[logic] = mach

        # Register selector and write decisions
        selector_id = None
        if args.write_to_db:
            selector_name = f"{args.selector_prefix}_{logic.lower()}"
            selector_id = register_ml_selector(
                args.db,
                name=selector_name,
                model_type="MachSMT",
                portfolio_id=args.portfolio_id,
                model_path=str(model_path),
                training_info={"k_fold": args.k_fold, "logic": logic},
            )
            print(f"    Selector '{selector_name}' -> id={selector_id}")

        record_decisions_for_logic(
            mach, logic, str(decisions_path),
            db_path=args.db if args.write_to_db else None,
            selector_id=selector_id,
            instance_map=instance_map,
            solver_config_map=solver_config_map,
        )
        decision_files.append(decisions_path)
    
    # Step 3: Merge and summarize
    print("\nStep 3: Merging results...")
    combined_decisions = output_dir / "decisions_all.csv"
    summary_path = output_dir / "decisions_summary.csv"
    
    merge_decisions([str(f) for f in decision_files], str(combined_decisions))
    generate_summary(str(combined_decisions), str(summary_path))
    
    print("\n" + "=" * 60)
    print("Complete! Output files:")
    for logic in logic_data:
        print(f"  {logic}:")
        print(f"    - Training CSV: training_{logic}.csv")
        print(f"    - Model: machsmt_{logic}.pkl")
        print(f"    - Decisions: decisions_{logic}.csv")
    print(f"  Combined:")
    print(f"    - All decisions: decisions_all.csv")
    print(f"    - Summary: decisions_summary.csv")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
