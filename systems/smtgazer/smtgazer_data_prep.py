#!/usr/bin/env python3
"""
SMTGazer Data Preparation for suite_9k / Portfolio 7

Queries the DB and generates the input files SMTportfolio.py expects:
  data/{logic}Labels.json       — PAR-2 times per benchmark per config
  machfea/{logic}_solver.json   — solver list in config_id order
  instance_metadata.json        — maps family/filename → instance details

Run interactively or in sbatch (~10 seconds).
"""

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# === Configuration ===
DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")
WORKDIR = Path("/dcs/large/u5573765/smtgazer_workdir")
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


def prepare_data():
    print("=" * 70)
    print("SMTGazer Data Preparation")
    print("=" * 70)
    print(f"DB: {DB_PATH}")
    print(f"Workdir: {WORKDIR}")
    print(f"Portfolio: {PORTFOLIO_ID}, Suite: {SUITE_NAME}")
    print(f"Logics: {LOGICS}")
    print(f"Config IDs: {CONFIG_IDS}")
    print()

    # Create directories
    (WORKDIR / "data").mkdir(parents=True, exist_ok=True)
    (WORKDIR / "machfea" / "infer_result").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()

    # Verify portfolio
    row = cur.execute(
        "SELECT timeout_s FROM portfolios WHERE id = ?", (PORTFOLIO_ID,)
    ).fetchone()
    assert row and row[0] == TIMEOUT_S, f"Portfolio {PORTFOLIO_ID} timeout mismatch"

    # Load all runs at once for efficiency
    print("Loading runs from DB...")
    runs = cur.execute("""
        SELECT
            i.id AS instance_id, i.logic, i.family, i.file_name, i.file_path,
            r.config_id, r.status, r.runtime_ms
        FROM runs r
        JOIN instances i ON r.instance_id = i.id
        WHERE r.portfolio_id = ?
          AND i.suite_name = ?
          AND r.config_id IN ({})
        ORDER BY i.logic, i.id, r.config_id
    """.format(",".join("?" * len(CONFIG_IDS))),
    (PORTFOLIO_ID, SUITE_NAME, *CONFIG_IDS)).fetchall()
    conn.close()

    print(f"  Loaded {len(runs)} run records")

    # Group by instance
    instances = defaultdict(lambda: {
        'logic': None, 'family': None, 'file_name': None,
        'file_path': None, 'runs': {}
    })
    for inst_id, logic, family, file_name, file_path, config_id, status, runtime_ms in runs:
        inst = instances[inst_id]
        inst['logic'] = logic
        inst['family'] = family
        inst['file_name'] = file_name
        inst['file_path'] = file_path
        inst['runs'][config_id] = (status, runtime_ms)

    # Filter to instances with complete runs
    complete = {iid: inst for iid, inst in instances.items()
                if len(inst['runs']) == len(CONFIG_IDS)}
    print(f"  Complete instances (all {len(CONFIG_IDS)} configs): {len(complete)}")

    # Group by logic
    by_logic = defaultdict(dict)
    for iid, inst in complete.items():
        by_logic[inst['logic']][iid] = inst

    # Generate files per logic
    metadata = {}
    for logic in LOGICS:
        logic_insts = by_logic.get(logic, {})
        print(f"\n{logic}: {len(logic_insts)} instances")

        if not logic_insts:
            print(f"  WARNING: No instances for {logic}")
            continue

        # Labels.json
        train_data = {}
        for iid, inst in logic_insts.items():
            key = f"{inst['family']}/{inst['file_name']}"
            times = []
            for cid in CONFIG_IDS:
                status, runtime_ms = inst['runs'][cid]
                if status in ('sat', 'unsat'):
                    times.append(runtime_ms / 1000.0)
                else:
                    times.append(PAR2_PENALTY)
            train_data[key] = times

            metadata[key] = {
                'instance_id': iid,
                'logic': logic,
                'family': inst['family'],
                'file_name': inst['file_name'],
                'file_path': inst['file_path'],
            }

        labels = {"train": train_data}
        labels_path = WORKDIR / "data" / f"{logic}Labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels, f)
        print(f"  Labels: {labels_path.name} ({len(train_data)} entries)")

        # Verify a sample
        sample_key = next(iter(train_data))
        sample_times = train_data[sample_key]
        print(f"  Sample: {sample_key} -> {[f'{t:.2f}' for t in sample_times]}")

        # solver.json (same for all logics — portfolio 7 has all 6 configs)
        solver_path = WORKDIR / "machfea" / f"{logic}_solver.json"
        with open(solver_path, 'w') as f:
            json.dump({"solver_list": SOLVER_LIST}, f)
        print(f"  Solver: {solver_path.name}")

    # Instance metadata
    meta_path = WORKDIR / "instance_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f)
    print(f"\nMetadata: {meta_path.name} ({len(metadata)} entries)")

    # Summary
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    for logic in LOGICS:
        n = len(by_logic.get(logic, {}))
        print(f"  {logic}: {n} instances")
    print(f"  Total: {len(complete)} instances")
    print(f"  Output: {WORKDIR}")


if __name__ == "__main__":
    prepare_data()
