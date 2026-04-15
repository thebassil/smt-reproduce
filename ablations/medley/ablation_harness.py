#!/usr/bin/env python3
"""
Medley CLI-Space Ablation Harness

Config-driven ablation harness that systematically tests every CLI-configurable
knob one-at-a-time, holding all others at their production reference values.
Results go to CSV (no DB writes).

Usage:
    python3 ablation_harness.py run <yaml> [--dry-run]
    python3 ablation_harness.py _execute <yaml>
    python3 ablation_harness.py report <code>
    python3 ablation_harness.py list
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
ABLATIONS_DIR = BASE_DIR / "ablations"
RESULTS_DIR = BASE_DIR / "results"
SBATCH_DIR = BASE_DIR / "sbatch"

MEDLEY_DIR = Path("/dcs/large/u5573765/artifacts/medley-solver")
MEDLEY_WRAPPER = Path("/dcs/23/u5573765/cs351/eval-medley/run_medley_wrapper.py")
REPRO_JSON = Path("/dcs/23/u5573765/cs351/eval-medley/medley_reproducibility.json")
FOLDS_DIR = Path("/dcs/23/u5573765/cs351/eval-medley")
DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")

# ── Constants (from production run) ───────────────────────────────────────────

CONFIG_TO_SOLVER = {
    'cvc5_baseline_default':        'Z3',
    'cvc5_qfnra_05_nl_cov':        'CVC4',
    'cvc5_qfnra_08_decision_just':  'BOOLECTOR',
    'z3_baseline_default':          'YICES',
    'z3_baseline_qfbv_sat_euf_sat': 'MathSAT',
    'z3_baseline_qflia_case_split': 'Bitwuzla',
}
SOLVER_TO_CONFIG = {v: k for k, v in CONFIG_TO_SOLVER.items()}

LOGICS = ['QF_BV', 'QF_LIA', 'QF_NRA']
SUITE = 'suite_9k'
PORTFOLIO_ID = 7
TIMEOUT_S = 60

# Valid choices for categorical flags
VALID_CLASSIFIERS = ['neighbor', 'knearest', 'MLP', 'linear', 'thompson',
                     'exp3', 'greedy', 'random']
VALID_TIMEOUT_MANAGERS = ['expo', 'const', 'nearest', 'sgd']
VALID_REWARDS = ['binary', 'bump', 'exp']
VALID_KINDS = ['full', 'single', 'greedy']

# Flag mapping: config key → CLI flag
FLAG_MAP = {
    'classifier':      '--classifier',
    'timeout_manager':  '--timeout_manager',
    'k':               '--k',
    'time_k':          '--time_k',
    'epsilon':         '--epsilon',
    'epsilon_decay':   '--epsilon_decay',
    'gamma':           '--gamma',
    'confidence':      '--confidence',
    'reward':          '--reward',
    'kind':            '--kind',
    'timeout':         '--timeout',
    'seed':            '--seed',
    'set_lambda':      '--set_lambda',
    'set_const':       '--set_const',
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Parsed YAML ablation configuration."""
    code: str
    name: str
    hypothesis: str
    reference: Dict[str, Any]
    ablated_flag: str
    ablation_values: list
    k_fold: int
    folds_file: str
    suite_name: str
    portfolio_id: int
    logics: List[str]
    yaml_path: Optional[Path] = None


# ── ConfigLoader ──────────────────────────────────────────────────────────────

class ConfigLoader:
    """Parse and validate YAML ablation configs."""

    CATEGORICAL_CHOICES = {
        'classifier': VALID_CLASSIFIERS,
        'timeout_manager': VALID_TIMEOUT_MANAGERS,
        'reward': VALID_REWARDS,
        'kind': VALID_KINDS,
        'extra': [True, False, 'true', 'false'],
    }

    NUMERIC_RANGES = {
        'k':             (1, 200, int),
        'time_k':        (1, 500, int),
        'epsilon':       (0.0, 1.0, float),
        'epsilon_decay':  (0.0, 1.0, float),
        'gamma':         (0.001, 1.0, float),
        'confidence':    (0.0, 1.0, float),
        'set_lambda':    (0, 100, int),
        'set_const':     (0, 120, int),
    }

    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        with open(yaml_path) as f:
            self.raw = yaml.safe_load(f)

    def load(self) -> AblationConfig:
        exp = self.raw['experiment']
        ref = self.raw['reference']
        abl = self.raw['ablation']
        exe = self.raw['execution']

        config = AblationConfig(
            code=exp['code'],
            name=exp['name'],
            hypothesis=exp['hypothesis'],
            reference=ref,
            ablated_flag=abl['flag'],
            ablation_values=abl['values'],
            k_fold=exe['k_fold'],
            folds_file=exe['folds_file'],
            suite_name=exe['suite_name'],
            portfolio_id=exe['portfolio_id'],
            logics=exe['logics'],
            yaml_path=self.yaml_path,
        )
        return config

    def validate(self, config: AblationConfig) -> List[str]:
        """Validate config against reproducibility JSON and constraints.
        Returns list of error messages (empty = valid)."""
        errors = []

        # Load reproducibility JSON
        if not REPRO_JSON.exists():
            errors.append(f"Reproducibility JSON not found: {REPRO_JSON}")
            return errors

        with open(REPRO_JSON) as f:
            repro = json.load(f)

        # Check classifier is known
        known_classifiers = [c['name'] for c in repro['classifiers_evaluated']]
        if config.reference.get('classifier') not in known_classifiers:
            errors.append(
                f"Reference classifier '{config.reference['classifier']}' "
                f"not in known classifiers: {known_classifiers}"
            )

        # Check timeout matches
        repro_timeout = repro['training_parameters']['timeout_per_instance']
        if config.reference.get('timeout') != repro_timeout:
            errors.append(
                f"Reference timeout={config.reference['timeout']} != "
                f"repro timeout={repro_timeout}"
            )

        # Check feature_setting matches
        repro_fs = repro['training_parameters']['feature_setting']
        if not repro_fs.startswith(str(config.reference.get('feature_setting', ''))):
            errors.append(
                f"Reference feature_setting='{config.reference['feature_setting']}' "
                f"doesn't match repro: '{repro_fs}'"
            )

        # Validate ablation values
        flag = config.ablated_flag
        if flag in self.CATEGORICAL_CHOICES:
            valid = self.CATEGORICAL_CHOICES[flag]
            for v in config.ablation_values:
                if v not in valid:
                    errors.append(
                        f"Ablation value '{v}' for --{flag} not in "
                        f"valid choices: {valid}"
                    )
        elif flag in self.NUMERIC_RANGES:
            lo, hi, typ = self.NUMERIC_RANGES[flag]
            for v in config.ablation_values:
                try:
                    val = typ(v)
                except (ValueError, TypeError):
                    errors.append(f"Cannot convert '{v}' to {typ.__name__} for --{flag}")
                    continue
                if val < lo or val > hi:
                    errors.append(
                        f"Ablation value {val} for --{flag} outside range [{lo}, {hi}]"
                    )

        # Check reference value in ablation values
        ref_val = config.reference.get(flag)
        if ref_val is not None:
            # Normalize types for comparison
            abl_vals_str = [str(v) for v in config.ablation_values]
            if str(ref_val) not in abl_vals_str:
                errors.append(
                    f"Reference value '{ref_val}' for --{flag} not in "
                    f"ablation values: {config.ablation_values}"
                )

        # Check folds file
        folds_path = FOLDS_DIR / config.folds_file
        if not folds_path.exists():
            errors.append(f"Folds file not found: {folds_path}")

        # Read-only DB check
        if DB_PATH.exists():
            try:
                conn = sqlite3.connect(str(DB_PATH), timeout=10)
                # Check suite has instances
                count = conn.execute(
                    "SELECT COUNT(*) FROM instances WHERE suite_name = ?",
                    (config.suite_name,)
                ).fetchone()[0]
                if count == 0:
                    errors.append(f"Suite '{config.suite_name}' has 0 instances in DB")
                # Check portfolio exists
                pcount = conn.execute(
                    "SELECT COUNT(*) FROM portfolio_configs WHERE portfolio_id = ?",
                    (config.portfolio_id,)
                ).fetchone()[0]
                if pcount == 0:
                    errors.append(f"Portfolio {config.portfolio_id} not found in DB")
                conn.close()
            except sqlite3.Error as e:
                errors.append(f"DB check failed: {e}")
        else:
            errors.append(f"Database not found: {DB_PATH}")

        return errors


# ── Data Loading (from medley_kfold_all.py) ───────────────────────────────────

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
    """PAR-2 scoring: actual runtime if solved, else 2 * timeout."""
    if status in ('sat', 'unsat'):
        return runtime_ms / 1000.0
    return 2.0 * timeout_s


def load_shared_folds(folds_file: Path, data: dict, k: int) -> List[Tuple[List[str], List[str]]]:
    """Load pre-computed folds from shared JSON file."""
    with open(folds_file) as f:
        folds_data = json.load(f)

    # Unwrap {"folds": [...], "meta": ...} wrapper if present
    if isinstance(folds_data, dict) and 'folds' in folds_data:
        folds_data = folds_data['folds']

    # Expected format: list of k lists (each list = file paths in that fold)
    if isinstance(folds_data, dict):
        # Handle dict format: {"0": [...], "1": [...], ...}
        fold_lists = [folds_data[str(i)] for i in range(k)]
    elif isinstance(folds_data, list) and folds_data and isinstance(folds_data[0], dict):
        # Handle list-of-dicts format: [{"fold": 0, "test_paths": [...], ...}, ...]
        folds_data_sorted = sorted(folds_data, key=lambda x: x['fold'])
        fold_lists = [entry['test_paths'] for entry in folds_data_sorted]
    else:
        fold_lists = folds_data

    # Filter to only paths present in our data
    fold_lists_filtered = []
    for fold_paths in fold_lists:
        filtered = [fp for fp in fold_paths if fp in data]
        fold_lists_filtered.append(filtered)

    result = []
    for i in range(k):
        test = fold_lists_filtered[i]
        train = [p for j in range(k) if j != i for p in fold_lists_filtered[j]]
        result.append((train, test))

    return result


# ── Staging (from medley_kfold_all.py) ────────────────────────────────────────

def create_fold_staging(data: dict, train_paths: List[str], staging_dir: Path):
    """Create staging directory with dispatch CSVs + .smt2 symlinks for training data."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    by_dir_solver = defaultdict(list)

    for fp in train_paths:
        info = data[fp]
        d = os.path.dirname(fp)
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


# ── AblationRunner ────────────────────────────────────────────────────────────

class AblationRunner:
    """Execute k-fold CV for each ablation variant."""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.data = None
        self.folds = None

    def run(self):
        """Full execution: load data, run all variants, write CSVs."""
        print("=" * 70)
        print(f"Ablation: {self.config.code} — {self.config.name}")
        print(f"Flag: --{self.config.ablated_flag}")
        print(f"Values: {self.config.ablation_values}")
        print(f"Hypothesis: {self.config.hypothesis}")
        print("=" * 70)

        # Load data
        self.data = load_data(str(DB_PATH))

        # Load folds
        folds_path = FOLDS_DIR / self.config.folds_file
        self.folds = load_shared_folds(folds_path, self.data, self.config.k_fold)
        for i, (train, test) in enumerate(self.folds):
            print(f"  Fold {i+1}: train={len(train)}, test={len(test)}")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        perfold_csv = RESULTS_DIR / f"{self.config.code}_perfold.csv"
        summary_csv = RESULTS_DIR / f"{self.config.code}_summary.csv"

        perfold_rows = []
        summary_rows = []

        for vidx, value in enumerate(self.config.ablation_values):
            print(f"\n{'─'*70}")
            print(f"VARIANT {vidx}: --{self.config.ablated_flag}={value}")
            print(f"{'─'*70}")

            cli_flags = self._build_cli_flags(value)
            variant_fold_results = []

            for fold_idx, (train_paths, test_paths) in enumerate(self.folds):
                results, decisions = self._run_one_fold(
                    fold_idx, train_paths, test_paths, cli_flags
                )
                variant_fold_results.append(results)

                n = results['total']
                print(f"  [Fold {fold_idx+1}] "
                      f"Medley: {results['medley_solved']}/{n} "
                      f"PAR-2={results['medley_par2']:.1f}  "
                      f"CSBS: {results['csbs_solved']}/{n}  "
                      f"VBS: {results['vbs_solved']}/{n}")

                perfold_rows.append({
                    'experiment': self.config.code,
                    'variant_idx': vidx,
                    'value': value,
                    'fold': fold_idx,
                    'total': n,
                    'medley_solved': results['medley_solved'],
                    'medley_par2': f"{results['medley_par2']:.2f}",
                    'vbs_solved': results['vbs_solved'],
                    'vbs_par2': f"{results['vbs_par2']:.2f}",
                    'csbs_solved': results['csbs_solved'],
                    'csbs_par2': f"{results['csbs_par2']:.2f}",
                    'sbs_solved': results['sbs_solved'],
                    'sbs_par2': f"{results['sbs_par2']:.2f}",
                })

            # Aggregate variant metrics
            totals = {}
            for key in variant_fold_results[0].keys():
                totals[key] = sum(r[key] for r in variant_fold_results)

            n = totals['total']
            avg_par2 = totals['medley_par2'] / n if n > 0 else 0
            solved_pct = totals['medley_solved'] / n * 100 if n > 0 else 0
            vs_csbs = ((totals['medley_par2'] - totals['csbs_par2'])
                       / totals['csbs_par2'] * 100) if totals['csbs_par2'] > 0 else 0
            vs_vbs = ((totals['medley_par2'] - totals['vbs_par2'])
                      / totals['vbs_par2'] * 100) if totals['vbs_par2'] > 0 else 0

            print(f"\n  AGGREGATE --{self.config.ablated_flag}={value}:")
            print(f"    Medley: {totals['medley_solved']}/{n} ({solved_pct:.1f}%) "
                  f"PAR-2={totals['medley_par2']:.1f} "
                  f"avg={avg_par2:.3f} "
                  f"vs_csbs={vs_csbs:+.1f}% vs_vbs={vs_vbs:+.1f}%")

            summary_rows.append({
                'experiment': self.config.code,
                'variant_idx': vidx,
                'value': value,
                'total': n,
                'avg_par2': f"{avg_par2:.3f}",
                'solved': totals['medley_solved'],
                'solved_pct': f"{solved_pct:.1f}",
                'vs_csbs_pct': f"{vs_csbs:+.1f}",
                'vs_vbs_pct': f"{vs_vbs:+.1f}",
            })

        # Write per-fold CSV
        perfold_fields = ['experiment', 'variant_idx', 'value', 'fold', 'total',
                          'medley_solved', 'medley_par2', 'vbs_solved', 'vbs_par2',
                          'csbs_solved', 'csbs_par2', 'sbs_solved', 'sbs_par2']
        with open(perfold_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=perfold_fields)
            w.writeheader()
            w.writerows(perfold_rows)
        print(f"\nPer-fold CSV: {perfold_csv}")

        # Write summary CSV
        summary_fields = ['experiment', 'variant_idx', 'value', 'total',
                          'avg_par2', 'solved', 'solved_pct', 'vs_csbs_pct', 'vs_vbs_pct']
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Summary CSV:  {summary_csv}")

        print(f"\n{'='*70}")
        print(f"COMPLETE: {self.config.code}")
        print(f"{'='*70}")

    def _build_cli_flags(self, ablated_value) -> dict:
        """Build CLI flags dict from reference + ablated value override."""
        flags = dict(self.config.reference)
        flags[self.config.ablated_flag] = ablated_value
        return flags

    def _run_one_fold(self, fold_idx: int, train_paths: List[str],
                      test_paths: List[str], cli_flags: dict) -> Tuple[dict, list]:
        """Train + evaluate one fold. Returns (results, decisions)."""
        work_dir = Path(tempfile.mkdtemp(
            prefix=f"{self.config.code}_f{fold_idx}_",
            dir=os.environ.get('TMPDIR', '/tmp')
        ))
        staging_dir = work_dir / "staging"
        output_dir = work_dir / "output"

        classifier = cli_flags.get('classifier', 'neighbor')
        print(f"\n  [Fold {fold_idx+1}] {classifier}: "
              f"train={len(train_paths)}, test={len(test_paths)}")

        try:
            # Create staging
            create_fold_staging(self.data, train_paths, staging_dir)

            # Train per logic
            for logic in self.config.logics:
                pkl = self._train_fold(staging_dir, output_dir, logic, cli_flags)
                if pkl:
                    print(f"    Trained {logic} -> {pkl.name}")
                else:
                    print(f"    FAILED {logic}")

            # Evaluate
            results, decisions = self._evaluate_fold(
                test_paths, train_paths, output_dir, classifier,
                cli_flags.get('timeout', TIMEOUT_S)
            )
        finally:
            # Clean up staging to save disk
            shutil.rmtree(work_dir, ignore_errors=True)

        return results, decisions

    def _train_fold(self, staging_dir: Path, output_dir: Path,
                    logic: str, cli_flags: dict) -> Optional[Path]:
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
            '--feature_setting', 'bow',
            '--save_classifier', str(classifier_pkl),
        ]

        # Inject all CLI flags from reference + ablated value
        for key, flag in FLAG_MAP.items():
            if key in cli_flags and key != 'extra':
                cmd.extend([flag, str(cli_flags[key])])

        # Handle --extra (bool) specially
        # Medley uses type=bool which means bool("False") == True
        # So we pass --extra true/false and the wrapper handles it
        extra_val = cli_flags.get('extra', True)
        if isinstance(extra_val, str):
            extra_val = extra_val.lower() == 'true'
        if not extra_val:
            cmd.append('--no-extra')
        # If extra is True (default), don't pass anything — Medley defaults to True

        try:
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
                print(f"    ERROR training {cli_flags.get('classifier')}/{logic}: "
                      f"{stderr_tail}")
                return None
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT training {cli_flags.get('classifier')}/{logic}")
            return None

        return classifier_pkl if classifier_pkl.exists() else None

    def _evaluate_fold(self, test_paths: List[str], train_paths: List[str],
                       output_dir: Path, classifier_name: str,
                       timeout_s: int = TIMEOUT_S) -> Tuple[dict, list]:
        """Load trained classifiers and evaluate on test instances."""
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
            info = self.data[fp]
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

        for logic in self.config.logics:
            pkl_path = output_dir / f"classifier_{logic}.pkl"
            if not pkl_path.exists():
                print(f"    WARNING: {pkl_path} not found, skipping {logic}")
                continue

            with open(pkl_path, 'rb') as f:
                clf = dill.load(f)
            classifiers_by_logic[logic] = clf

            # Compute running mean from training instances of this logic
            train_logic = [fp for fp in train_paths
                           if self.data[fp]['logic'] == logic]
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
            print(f"    {logic}: classifier loaded, mean from {count} instances")

        # Evaluate each test instance
        decisions = []

        for fp in sorted(test_paths):
            info = self.data[fp]
            logic = info['logic']
            results['total'] += 1

            # VBS (oracle)
            best_cfg = min(
                info['runs'],
                key=lambda c: par2(info['runs'][c]['status'],
                                   info['runs'][c]['runtime_ms'], timeout_s)
            )
            vbs_p = par2(info['runs'][best_cfg]['status'],
                         info['runs'][best_cfg]['runtime_ms'], timeout_s)
            results['vbs_par2'] += vbs_p
            if info['runs'][best_cfg]['status'] in ('sat', 'unsat'):
                results['vbs_solved'] += 1

            # SBS
            if sbs in info['runs']:
                sp = par2(info['runs'][sbs]['status'],
                          info['runs'][sbs]['runtime_ms'], timeout_s)
            else:
                sp = 2 * timeout_s
            results['sbs_par2'] += sp
            if sbs in info['runs'] and info['runs'][sbs]['status'] in ('sat', 'unsat'):
                results['sbs_solved'] += 1

            # CSBS
            csbs_cfg = csbs.get(logic)
            if csbs_cfg and csbs_cfg in info['runs']:
                cp = par2(info['runs'][csbs_cfg]['status'],
                          info['runs'][csbs_cfg]['runtime_ms'], timeout_s)
            else:
                cp = 2 * timeout_s
            results['csbs_par2'] += cp
            if (csbs_cfg and csbs_cfg in info['runs']
                    and info['runs'][csbs_cfg]['status'] in ('sat', 'unsat')):
                results['csbs_solved'] += 1

            # Medley prediction
            clf = classifiers_by_logic.get(logic)
            if clf is None or not Path(fp).exists():
                results['medley_par2'] += 2 * timeout_s
                continue

            features = np.array(get_syntactic_count_features(str(fp)))
            mean = train_means[logic]
            normalized = features / (mean + 1e-9)

            # get_ordering with high count → epsilon * decay^count ≈ 0 → exploit
            ordering = clf.get_ordering(normalized, 999999, fp)
            predicted_solver = ordering[0]
            predicted_cfg = SOLVER_TO_CONFIG.get(predicted_solver, predicted_solver)

            if predicted_cfg in info['runs']:
                run = info['runs'][predicted_cfg]
                mp = par2(run['status'], run['runtime_ms'], timeout_s)
                results['medley_par2'] += mp
                if run['status'] in ('sat', 'unsat'):
                    results['medley_solved'] += 1
                decisions.append((fp, predicted_cfg, run['status'],
                                  run['runtime_ms']))
            else:
                results['medley_par2'] += 2 * timeout_s
                decisions.append((fp, predicted_cfg, 'missing', 0))

        return results, decisions


# ── SbatchGenerator ───────────────────────────────────────────────────────────

class SbatchGenerator:
    """Generate sbatch submission scripts for ablation experiments."""

    TEMPLATE = """#!/bin/bash
#SBATCH --job-name=abl_{code}
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sbatch/{code}_%j.out
#SBATCH --error=sbatch/{code}_%j.err

set -euo pipefail
cd /dcs/23/u5573765/cs351/smt-ablation-medley
source /dcs/large/u5573765/env.sh

echo "Ablation: {code} — $(date)"
echo "Job: $SLURM_JOB_ID  Node: $(hostname)"

python3 ablation_harness.py _execute ablations/{yaml_filename}

echo "COMPLETE — $(date)"
"""

    def __init__(self, config: AblationConfig):
        self.config = config

    def generate(self) -> str:
        """Generate sbatch script content."""
        yaml_filename = self.config.yaml_path.name if self.config.yaml_path else ""
        return self.TEMPLATE.format(
            code=self.config.code,
            yaml_filename=yaml_filename,
        )

    def write(self) -> Path:
        """Write sbatch script to file and return path."""
        SBATCH_DIR.mkdir(parents=True, exist_ok=True)
        script_path = SBATCH_DIR / f"{self.config.code}.sbatch"
        with open(script_path, 'w') as f:
            f.write(self.generate())
        script_path.chmod(0o755)
        return script_path


# ── ReportGenerator ───────────────────────────────────────────────────────────

class ReportGenerator:
    """Generate markdown reports from CSV results."""

    def __init__(self, code: str):
        self.code = code
        self.perfold_csv = RESULTS_DIR / f"{code}_perfold.csv"
        self.summary_csv = RESULTS_DIR / f"{code}_summary.csv"

    def generate(self) -> str:
        """Generate markdown report from CSVs."""
        lines = []

        # Load YAML config for context
        yaml_path = self._find_yaml()
        if yaml_path:
            loader = ConfigLoader(yaml_path)
            config = loader.load()
            lines.append(f"# Ablation Report: {config.code}")
            lines.append(f"**{config.name}**\n")
            lines.append(f"**Hypothesis:** {config.hypothesis}\n")
            lines.append(f"**Ablated flag:** `--{config.ablated_flag}`\n")
        else:
            lines.append(f"# Ablation Report: {self.code}\n")

        # Summary table
        if self.summary_csv.exists():
            lines.append("## Summary\n")
            with open(self.summary_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                lines.append("| Variant | Value | Total | Solved | Solved% | "
                             "Avg PAR-2 | vs CSBS | vs VBS |")
                lines.append("|---------|-------|-------|--------|---------|"
                             "-----------|---------|--------|")
                for r in rows:
                    ref_marker = " *" if yaml_path and str(r['value']) == str(
                        config.reference.get(config.ablated_flag)) else ""
                    lines.append(
                        f"| {r['variant_idx']} | {r['value']}{ref_marker} | "
                        f"{r['total']} | {r['solved']} | {r['solved_pct']}% | "
                        f"{r['avg_par2']} | {r['vs_csbs_pct']}% | "
                        f"{r['vs_vbs_pct']}% |"
                    )
                lines.append("\n\\* = reference value\n")
        else:
            lines.append("*No summary CSV found.*\n")

        # Per-fold details
        if self.perfold_csv.exists():
            lines.append("## Per-Fold Results\n")
            with open(self.perfold_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                lines.append("| Value | Fold | Total | Medley Solved | "
                             "Medley PAR-2 | VBS Solved | CSBS Solved |")
                lines.append("|-------|------|-------|---------------|"
                             "--------------|------------|-------------|")
                for r in rows:
                    lines.append(
                        f"| {r['value']} | {r['fold']} | {r['total']} | "
                        f"{r['medley_solved']} | {r['medley_par2']} | "
                        f"{r['vbs_solved']} | {r['csbs_solved']} |"
                    )
        else:
            lines.append("*No per-fold CSV found.*\n")

        return "\n".join(lines)

    def _find_yaml(self) -> Optional[Path]:
        """Find the YAML config file for this experiment code."""
        for yaml_path in ABLATIONS_DIR.glob("*.yaml"):
            if yaml_path.name.startswith(self.code):
                return yaml_path
        return None


# ── CLI Subcommands ───────────────────────────────────────────────────────────

def cmd_run(args):
    """Validate → generate sbatch → submit (or dry-run)."""
    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = BASE_DIR / yaml_path

    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        sys.exit(1)

    loader = ConfigLoader(yaml_path)
    config = loader.load()

    print(f"Validating {config.code}: {config.name}")
    print(f"  Flag: --{config.ablated_flag}")
    print(f"  Values: {config.ablation_values}")

    errors = loader.validate(config)
    if errors:
        print(f"\nVALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("  Validation: PASSED")

    gen = SbatchGenerator(config)
    script_content = gen.generate()

    if args.dry_run:
        print(f"\n--- Generated sbatch (dry-run) ---")
        print(script_content)
        print("--- end ---")
        print("\nDry run complete. No job submitted.")
    else:
        script_path = gen.write()
        print(f"\nSbatch script: {script_path}")
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Submitted: {result.stdout.strip()}")
        else:
            print(f"ERROR submitting: {result.stderr.strip()}")
            sys.exit(1)


def cmd_execute(args):
    """Internal: run inside sbatch job."""
    yaml_path = Path(args.yaml)
    if not yaml_path.is_absolute():
        yaml_path = BASE_DIR / yaml_path

    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        sys.exit(1)

    loader = ConfigLoader(yaml_path)
    config = loader.load()

    errors = loader.validate(config)
    if errors:
        print(f"VALIDATION ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    runner = AblationRunner(config)
    runner.run()


def cmd_report(args):
    """Read CSVs → markdown report."""
    code = args.code
    gen = ReportGenerator(code)

    if not gen.summary_csv.exists() and not gen.perfold_csv.exists():
        print(f"No results found for {code}")
        print(f"  Expected: {gen.summary_csv}")
        print(f"  Expected: {gen.perfold_csv}")
        sys.exit(1)

    report = gen.generate()
    print(report)


def cmd_list(args):
    """Scan results/ dir → list all experiments."""
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return

    summaries = sorted(RESULTS_DIR.glob("*_summary.csv"))
    if not summaries:
        print("No experiment results found in results/")
        return

    print(f"{'Code':<16} {'Flag':<20} {'Variants':>8} {'Status'}")
    print("-" * 60)

    for summary_csv in summaries:
        code = summary_csv.stem.replace("_summary", "")
        perfold_csv = RESULTS_DIR / f"{code}_perfold.csv"

        # Count variants from summary
        with open(summary_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Try to get flag name from YAML
        flag = "?"
        for yaml_path in ABLATIONS_DIR.glob("*.yaml"):
            if yaml_path.name.startswith(code):
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                flag = raw['ablation']['flag']
                break

        has_perfold = "complete" if perfold_csv.exists() else "summary only"
        print(f"{code:<16} --{flag:<18} {len(rows):>8} {has_perfold}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Medley CLI-Space Ablation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  run <yaml>              Validate → generate sbatch → submit
  run <yaml> --dry-run    Validate only, print sbatch script
  report <code>           Read CSVs → markdown report
  list                    Scan results/ → list experiments
  _execute <yaml>         (internal) Run inside sbatch job
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommand')

    # run
    run_parser = subparsers.add_parser('run', help='Validate and submit ablation job')
    run_parser.add_argument('yaml', help='Path to ablation YAML config')
    run_parser.add_argument('--dry-run', action='store_true',
                            help='Validate and print sbatch without submitting')

    # _execute (internal)
    exec_parser = subparsers.add_parser('_execute',
                                        help='(internal) Execute inside sbatch job')
    exec_parser.add_argument('yaml', help='Path to ablation YAML config')

    # report
    report_parser = subparsers.add_parser('report',
                                          help='Generate markdown report from CSVs')
    report_parser.add_argument('code', help='Experiment code (e.g. ABL-MDY-H01)')

    # list
    subparsers.add_parser('list', help='List all experiments with results')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'run':
        cmd_run(args)
    elif args.command == '_execute':
        cmd_execute(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'list':
        cmd_list(args)


if __name__ == "__main__":
    main()
