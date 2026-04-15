#!/usr/bin/env python3
"""
Stage 1 — MachSMT CLI Ablation Harness (CSV Output)

Ablates MachSMT's CLI knobs one-at-a-time from the production config.
Uses family-aware GroupKFold, per-logic training, and outputs CSV results.

Usage:
    python ablation_harness.py run ablations/templates/ml_core_sweep.yaml          # Submit sbatch job
    python ablation_harness.py run ablations/templates/ml_core_sweep.yaml --dry-run # Validate only
    python ablation_harness.py report ABL-ML01                                      # View report
    python ablation_harness.py list                                                 # List experiments
"""

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# MachSMT path — must be inserted before importing machsmt
MACHSMT_PATH = Path("/dcs/23/u5573765/cs351/smt-project/artifacts/machsmt/MachSMT")
if not MACHSMT_PATH.exists():
    MACHSMT_PATH = Path(__file__).parent / "artifacts" / "machsmt" / "MachSMT"
sys.path.insert(0, str(MACHSMT_PATH))

DB_PATH = Path("/dcs/large/u5573765/db/results.sqlite")
RESULTS_DIR = Path(__file__).parent / "ablations" / "results"
JOBS_DIR = Path(__file__).parent / "ablations" / "jobs"
LOGS_DIR = Path(__file__).parent / "ablations" / "logs"

# Ablatable knobs (may be overridden per-variant)
ABLATABLE_KNOBS = {
    "ml_core", "pwc", "greedy", "logic_filter", "k",
    "min_datapoints", "max_score", "rng", "feature_timeout",
}

# Fixed knobs (hardware/scoring convention — never ablated)
FIXED_KNOBS = {"cores", "par_n", "use_gpu"}

# sbatch hardware settings (must match production)
SBATCH_PARTITION = "tiger"
SBATCH_CPUS = 4
SBATCH_MEM = "32G"
SBATCH_TIME = "08:00:00"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Parsed and validated experiment configuration."""
    code: str
    hypothesis: str
    # Reference config (all knobs)
    reference: Dict[str, Any]
    # Which knob to ablate and its values
    knob: str
    values: list
    # Execution params
    suite_name: str
    portfolio_id: int
    timeout_s: int
    logics: List[str]


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Parse YAML config and validate."""

    def load(self, yaml_path: str) -> AblationConfig:
        """Parse YAML file and validate knob config."""
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        exp = raw["experiment"]
        ref = raw["reference"]
        abl = raw["ablation"]
        exe = raw["execution"]

        config = AblationConfig(
            code=exp["code"],
            hypothesis=exp.get("hypothesis", ""),
            reference=ref,
            knob=abl["knob"],
            values=abl["values"],
            suite_name=exe["suite_name"],
            portfolio_id=exe["portfolio_id"],
            timeout_s=exe["timeout_s"],
            logics=exe.get("logics", []),
        )

        self._validate(config)
        return config

    def _validate(self, config: AblationConfig):
        """Validate knob exists in reference, values are correct type."""
        all_knobs = set(config.reference.keys())

        if config.knob not in all_knobs:
            raise ValueError(
                f"Ablation knob '{config.knob}' not found in reference config. "
                f"Available: {sorted(all_knobs)}"
            )

        if config.knob not in ABLATABLE_KNOBS:
            raise ValueError(
                f"Knob '{config.knob}' is in the fixed set {FIXED_KNOBS} "
                f"and cannot be ablated."
            )

        ref_value = config.reference[config.knob]
        ref_type = type(ref_value)

        # Type-check each ablation value against reference
        for v in config.values:
            if ref_type == bool:
                if not isinstance(v, bool):
                    raise ValueError(
                        f"Value {v!r} for knob '{config.knob}' should be bool, "
                        f"got {type(v).__name__}"
                    )
            elif ref_type == int:
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"Value {v!r} for knob '{config.knob}' should be numeric, "
                        f"got {type(v).__name__}"
                    )
            elif ref_type == str:
                if not isinstance(v, str):
                    raise ValueError(
                        f"Value {v!r} for knob '{config.knob}' should be str, "
                        f"got {type(v).__name__}"
                    )

        # Check reference value is included in ablation values
        if ref_value not in config.values:
            print(
                f"WARNING: Reference value {ref_value!r} for '{config.knob}' "
                f"is not in ablation values {config.values}. "
                f"Delta-vs-ref will be computed against variant index 0."
            )

        if not config.logics:
            raise ValueError("execution.logics must list at least one logic")

    def build_variant_config(self, config: AblationConfig,
                             variant_value: Any) -> Dict[str, Any]:
        """Build full config dict for one variant (reference + override)."""
        variant = dict(config.reference)
        variant[config.knob] = variant_value
        return variant


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Execute ablation experiment with k-fold CV and CSV output."""

    def __init__(self, config: AblationConfig, loader: ConfigLoader,
                 db_path: Path = DB_PATH):
        self.config = config
        self.loader = loader
        self.db_path = db_path

    def run(self):
        """Full orchestration. Writes CSV results."""
        print(f"{'='*70}")
        print(f"Experiment: {self.config.code}")
        print(f"  Knob: {self.config.knob}")
        print(f"  Values: {self.config.values}")
        print(f"  Logics: {self.config.logics}")
        print(f"{'='*70}")

        data, timeout_s, families = self._load_run_data()
        print(f"Loaded {len(data)} benchmarks from {self.config.suite_name}")

        # Filter by requested logics
        if self.config.logics:
            data = {
                p: d for p, d in data.items()
                if d["logic"] in self.config.logics
            }
            families = {
                p: f for p, f in families.items() if p in data
            }
            print(f"Filtered to {len(data)} benchmarks for logics: {self.config.logics}")

        if not data:
            raise RuntimeError("No benchmark data after filtering")

        # Pre-parse benchmarks and cache features.
        # For feature_timeout ablation, we need separate caches per value.
        sorted_paths = sorted(data.keys())
        if self.config.knob == "feature_timeout":
            from machsmt.config import args as _ft_args
            self._feature_caches = {}
            for ft_val in self.config.values:
                _ft_args.feature_timeout = int(ft_val)
                print(f"Pre-parsing benchmarks with feature_timeout={ft_val}...")
                self._feature_caches[ft_val] = self._precompute_features(sorted_paths)
            # Default cache for reference value
            self._feature_cache = self._feature_caches[
                self.config.reference["feature_timeout"]
            ]
        else:
            self._feature_caches = None
            print("Pre-parsing all benchmarks (one-time cost)...")
            self._feature_cache = self._precompute_features(sorted_paths)
        print(f"Cached features for {len(self._feature_cache)} benchmarks")

        # Group paths by logic
        by_logic = defaultdict(list)
        for path in sorted(data.keys()):
            by_logic[data[path]["logic"]].append(path)

        all_runs_rows = []
        all_summary_rows = []

        for vi, val in enumerate(self.config.values):
            variant_cfg = self.loader.build_variant_config(self.config, val)
            k = variant_cfg["k"]

            print(f"\n{'='*60}")
            print(f"Variant {vi}: {self.config.knob}={val}")
            print(f"  k-fold={k}")
            print(f"{'='*60}")

            # Switch feature cache for feature_timeout ablation
            if self._feature_caches is not None:
                self._feature_cache = self._feature_caches[val]

            # Generate family-aware folds
            folds = self._make_group_kfold(data, families, by_logic, k)

            # Per-fold, per-logic results
            fold_results = defaultdict(lambda: defaultdict(dict))
            # fold_results[fold][logic] = {n_instances, total_par2, solved}

            for fold_idx, (train_paths, test_paths) in enumerate(folds):
                self._apply_config(variant_cfg)

                for logic in self.config.logics:
                    logic_train = [p for p in train_paths if data[p]["logic"] == logic]
                    logic_test = [p for p in test_paths if data[p]["logic"] == logic]

                    if not logic_test:
                        continue

                    n_instances = len(logic_test)
                    total_par2 = 0.0
                    solved = 0

                    # Train MachSMT on this logic's training set
                    ms = self._train_logic(logic_train, data, timeout_s, variant_cfg)

                    # Evaluate on test set
                    for path in logic_test:
                        score = self._evaluate_one(
                            ms, path, data[path], timeout_s
                        )
                        total_par2 += score
                        if score < timeout_s:
                            solved += 1

                    avg_par2 = total_par2 / n_instances if n_instances else 0
                    solve_rate = solved / n_instances if n_instances else 0

                    fold_results[fold_idx][logic] = {
                        "n_instances": n_instances,
                        "total_par2": total_par2,
                        "avg_par2": avg_par2,
                        "solved": solved,
                        "solve_rate": solve_rate,
                    }

                    all_runs_rows.append({
                        "variant_index": vi,
                        "variant_value": val,
                        "fold": fold_idx,
                        "logic": logic,
                        "n_instances": n_instances,
                        "total_par2": round(total_par2, 2),
                        "avg_par2": round(avg_par2, 4),
                        "solved": solved,
                        "solve_rate": round(solve_rate, 4),
                    })

                    print(
                        f"  Fold {fold_idx} {logic}: "
                        f"avg_par2={avg_par2:.3f}, "
                        f"solved={solved}/{n_instances} "
                        f"({solve_rate*100:.1f}%)"
                    )

            # Compute summary (aggregate across folds per logic)
            for logic in self.config.logics:
                fold_metrics = [
                    fold_results[f][logic]
                    for f in range(len(folds))
                    if logic in fold_results[f]
                ]
                if not fold_metrics:
                    continue

                par2_values = [m["avg_par2"] for m in fold_metrics]
                sr_values = [m["solve_rate"] for m in fold_metrics]

                mean_par2 = float(np.mean(par2_values))
                std_par2 = float(np.std(par2_values, ddof=1)) if len(par2_values) > 1 else 0.0
                mean_sr = float(np.mean(sr_values))

                # Delta vs reference (variant index 0 by default,
                # or the variant matching reference value)
                delta = 0.0  # placeholder; computed after all variants

                all_summary_rows.append({
                    "variant_index": vi,
                    "variant_value": val,
                    "logic": logic,
                    "mean_par2": round(mean_par2, 4),
                    "std_par2": round(std_par2, 4),
                    "mean_solve_rate": round(mean_sr, 4),
                    "n_folds": len(fold_metrics),
                    "delta_vs_ref": 0.0,  # filled below
                })

        # Compute delta_vs_ref against the reference variant
        ref_value = self.config.reference[self.config.knob]
        ref_par2_by_logic = {}
        for row in all_summary_rows:
            if row["variant_value"] == ref_value:
                ref_par2_by_logic[row["logic"]] = row["mean_par2"]

        # If reference value not in variants, use variant_index=0
        if not ref_par2_by_logic:
            for row in all_summary_rows:
                if row["variant_index"] == 0:
                    ref_par2_by_logic[row["logic"]] = row["mean_par2"]

        for row in all_summary_rows:
            ref_p2 = ref_par2_by_logic.get(row["logic"], 0)
            row["delta_vs_ref"] = round(row["mean_par2"] - ref_p2, 4)

        # Write CSVs
        self._write_runs_csv(all_runs_rows)
        self._write_summary_csv(all_summary_rows)

        print(f"\nExperiment {self.config.code} completed.")
        print(f"  Runs CSV:    {RESULTS_DIR / f'{self.config.code}_runs.csv'}")
        print(f"  Summary CSV: {RESULTS_DIR / f'{self.config.code}_summary.csv'}")

    def _load_run_data(self) -> Tuple[dict, int, dict]:
        """Load solver runs and family info from DB.

        Returns:
            data: {file_path: {logic, runs: {solver_config: (status, runtime_ms)}}}
            timeout_s: portfolio timeout
            families: {file_path: family_name}
        """
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # Get timeout
        row = cur.execute(
            "SELECT timeout_s FROM portfolios WHERE id = ?",
            (self.config.portfolio_id,)
        ).fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Portfolio {self.config.portfolio_id} not found")
        timeout_s = row[0]

        # Load runs
        runs = cur.execute("""
            SELECT
                i.file_path,
                c.name AS config_name,
                r.solver,
                r.status,
                r.runtime_ms,
                i.logic,
                i.family
            FROM runs r
            JOIN instances i ON r.instance_id = i.id
            JOIN configs c ON r.config_id = c.id
            WHERE r.portfolio_id = ?
              AND i.suite_name = ?
        """, (self.config.portfolio_id, self.config.suite_name)).fetchall()

        conn.close()

        data = {}
        families = {}
        for file_path, config_name, solver, status, runtime_ms, logic, family in runs:
            solver_config = f"{solver}::{config_name}"
            if file_path not in data:
                data[file_path] = {"logic": logic, "runs": {}}
                families[file_path] = family
            data[file_path]["runs"][solver_config] = (status, runtime_ms)

        return data, timeout_s, families

    def _make_group_kfold(self, data: dict, families: dict,
                          by_logic: dict, k: int) -> List[Tuple[list, list]]:
        """Create family-aware GroupKFold splits.

        Groups benchmarks by family, then uses sklearn GroupKFold
        to ensure all benchmarks from the same family are in the same fold.
        Split is done per-logic, then merged.

        Returns list of (train_paths, test_paths) tuples.
        """
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=k)
        folds = [[] for _ in range(k)]  # folds[i] = set of test paths

        for logic, paths in by_logic.items():
            paths_sorted = sorted(paths)
            groups = [families.get(p, "unknown") for p in paths_sorted]
            X_dummy = np.zeros(len(paths_sorted))

            for fold_idx, (train_idx, test_idx) in enumerate(
                gkf.split(X_dummy, groups=groups)
            ):
                folds[fold_idx].extend([paths_sorted[i] for i in test_idx])

        all_paths = sorted(data.keys())
        result = []
        for fold_idx in range(k):
            test_set = set(folds[fold_idx])
            test_paths = [p for p in all_paths if p in test_set]
            train_paths = [p for p in all_paths if p not in test_set]
            result.append((train_paths, test_paths))

        return result

    def _apply_config(self, variant_cfg: Dict[str, Any]):
        """Set CONFIG_OBJ attributes and re-seed random state.

        MachSMT only seeds numpy/random at import time, so we must
        re-seed explicitly before each training run.
        """
        import random
        from machsmt.config import args as CONFIG_OBJ

        CONFIG_OBJ.ml_core = str(variant_cfg["ml_core"])
        CONFIG_OBJ.pwc = bool(variant_cfg["pwc"])
        CONFIG_OBJ.greedy = bool(variant_cfg["greedy"])
        CONFIG_OBJ.logic_filter = bool(variant_cfg["logic_filter"])
        CONFIG_OBJ.k = int(variant_cfg["k"])
        CONFIG_OBJ.min_datapoints = int(variant_cfg["min_datapoints"])
        CONFIG_OBJ.max_score = int(variant_cfg["max_score"])
        CONFIG_OBJ.rng = int(variant_cfg["rng"])
        CONFIG_OBJ.feature_timeout = int(variant_cfg["feature_timeout"])
        # Fixed knobs
        CONFIG_OBJ.cores = int(variant_cfg["cores"])
        CONFIG_OBJ.par_n = int(variant_cfg["par_n"])
        CONFIG_OBJ.use_gpu = bool(variant_cfg["use_gpu"])

        # Re-seed random state — MachSMT only seeds at import time
        rng = int(variant_cfg["rng"])
        random.seed(rng)
        np.random.seed(rng)

    def _precompute_features(self, all_paths: list) -> dict:
        """Parse all benchmarks once and cache {path: (features, logic)}.

        Processes sequentially to minimize memory — only one benchmark's
        tokens are in memory at a time. ThreadPool doesn't help due to GIL.
        """
        from machsmt.benchmark import Benchmark

        cache = {}
        for i, path in enumerate(all_paths):
            b = Benchmark(path)
            b.parse()  # tokenize, extract features, then deletes tokens
            cache[path] = {
                "features": b.features,
                "logic": b.logic,
            }
            del b  # free the Benchmark object immediately
            if (i + 1) % 500 == 0 or (i + 1) == len(all_paths):
                print(f"  Parsed {i+1}/{len(all_paths)} benchmarks")

        return cache

    def _make_cached_benchmark(self, path: str) -> Any:
        """Create a Benchmark with pre-cached features (no file parsing)."""
        from machsmt.benchmark import Benchmark

        b = Benchmark(path)
        cached = self._feature_cache[path]
        b.features = cached["features"]
        b.logic = cached["logic"]
        b.parsed = True
        return b

    def _train_logic(self, train_paths: list, data: dict,
                     timeout_s: int, variant_cfg: dict) -> Optional[Any]:
        """Train MachSMT model on one logic's training data.

        Builds a DataBase from cached features (no re-parsing) and
        creates MachSMT from it.

        Returns trained MachSMT instance, or None on failure.
        """
        if len(train_paths) < variant_cfg["min_datapoints"]:
            print(f"    Skipping: only {len(train_paths)} training samples "
                  f"(min={variant_cfg['min_datapoints']})")
            return None

        try:
            from machsmt import MachSMT
            from machsmt.database import DataBase
            from machsmt.solver import Solver

            # Build DataBase from cached features — no file parsing needed
            db = DataBase(build_on_init=False)
            for path in train_paths:
                b = self._make_cached_benchmark(path)
                db.benchmarks[path] = b
                for sc, (status, runtime_ms) in data[path]["runs"].items():
                    score = _compute_par2(status, runtime_ms, timeout_s)
                    if sc not in db.solvers:
                        db.solvers[sc] = Solver(sc)
                    db.solvers[sc].add_benchmark(b, score)
                    b.add_solver(db.solvers[sc], score)

            ms = MachSMT(db, train_on_init=True)
            return ms
        except Exception as e:
            print(f"    WARNING: Training failed: {e}")
            return None

    def _evaluate_one(self, ms, path: str, benchmark_data: dict,
                      timeout_s: int) -> float:
        """Evaluate a single benchmark. Returns PAR-2 score."""
        if ms is None:
            return 2 * timeout_s

        try:
            b = self._make_cached_benchmark(path)
            predictions = ms.predict(b)  # returns a list (one per benchmark)
            prediction = predictions[0] if predictions else None

            # prediction is a Solver object; use .name to match DB keys
            pred_key = prediction.name if prediction else None

            if pred_key and pred_key in benchmark_data["runs"]:
                status, runtime_ms = benchmark_data["runs"][pred_key]
                return _compute_par2(status, runtime_ms, timeout_s)
            else:
                return 2 * timeout_s
        except Exception:
            return 2 * timeout_s

    def _write_runs_csv(self, rows: list):
        """Write per-(variant, fold, logic) CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.config.code}_runs.csv"
        fieldnames = [
            "variant_index", "variant_value", "fold", "logic",
            "n_instances", "total_par2", "avg_par2", "solved", "solve_rate",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_summary_csv(self, rows: list):
        """Write aggregated summary CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.config.code}_summary.csv"
        fieldnames = [
            "variant_index", "variant_value", "logic",
            "mean_par2", "std_par2", "mean_solve_rate",
            "n_folds", "delta_vs_ref",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# ReportGenerator (CSV-based)
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate markdown reports from CSV results."""

    def generate(self, experiment_code: str) -> str:
        """Generate markdown report from summary CSV."""
        summary_path = RESULTS_DIR / f"{experiment_code}_summary.csv"
        runs_path = RESULTS_DIR / f"{experiment_code}_runs.csv"

        if not summary_path.exists():
            return f"No results found for '{experiment_code}' at {summary_path}"

        # Read summary
        with open(summary_path, "r") as f:
            reader = csv.DictReader(f)
            summary_rows = list(reader)

        lines = [
            f"# Ablation Report: {experiment_code}",
            "",
        ]

        # Try to find the template for hypothesis
        template_dir = Path(__file__).parent / "ablations" / "templates"
        for tpl in template_dir.glob("*.yaml"):
            with open(tpl) as f:
                raw = yaml.safe_load(f)
            if raw.get("experiment", {}).get("code") == experiment_code:
                lines.append(f"**Hypothesis**: {raw['experiment'].get('hypothesis', 'N/A')}")
                lines.append(f"**Knob**: {raw['ablation']['knob']}")
                lines.append(f"**Values**: {raw['ablation']['values']}")
                lines.append("")
                break

        # Group by logic
        logics = sorted(set(r["logic"] for r in summary_rows))

        for logic in logics:
            logic_rows = [r for r in summary_rows if r["logic"] == logic]
            lines.append(f"## {logic}")
            lines.append("")
            lines.append(
                "| Variant | Value | Mean PAR-2 | Std PAR-2 | "
                "Solve Rate | Folds | Delta vs Ref |"
            )
            lines.append(
                "|---------|-------|------------|-----------|"
                "------------|-------|--------------|"
            )
            for r in sorted(logic_rows, key=lambda x: int(x["variant_index"])):
                delta = float(r["delta_vs_ref"])
                delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
                sr_pct = f"{float(r['mean_solve_rate'])*100:.1f}%"
                lines.append(
                    f"| {r['variant_index']} | {r['variant_value']} "
                    f"| {r['mean_par2']} | {r['std_par2']} "
                    f"| {sr_pct} | {r['n_folds']} | {delta_str} |"
                )
            lines.append("")

        # Per-fold detail if runs CSV exists
        if runs_path.exists():
            with open(runs_path, "r") as f:
                reader = csv.DictReader(f)
                runs_rows = list(reader)

            lines.append("## Per-Fold Details")
            lines.append("")
            lines.append(
                "| Variant | Value | Fold | Logic | Instances | "
                "Avg PAR-2 | Solved | Solve Rate |"
            )
            lines.append(
                "|---------|-------|------|-------|-----------|"
                "-----------|--------|------------|"
            )
            for r in runs_rows:
                sr_pct = f"{float(r['solve_rate'])*100:.1f}%"
                lines.append(
                    f"| {r['variant_index']} | {r['variant_value']} "
                    f"| {r['fold']} | {r['logic']} | {r['n_instances']} "
                    f"| {r['avg_par2']} | {r['solved']} | {sr_pct} |"
                )
            lines.append("")

        return "\n".join(lines)

    def list_experiments(self) -> str:
        """List all experiments with summary CSVs."""
        if not RESULTS_DIR.exists():
            return "No results directory found."

        summaries = sorted(RESULTS_DIR.glob("*_summary.csv"))
        if not summaries:
            return "No experiment results found."

        lines = [
            "# Ablation Experiments",
            "",
            "| Code | Summary CSV | Rows |",
            "|------|-------------|------|",
        ]

        for path in summaries:
            code = path.stem.replace("_summary", "")
            with open(path, "r") as f:
                n_rows = sum(1 for _ in f) - 1  # subtract header
            lines.append(f"| {code} | {path.name} | {n_rows} |")

        # Also check for templates without results yet
        template_dir = Path(__file__).parent / "ablations" / "templates"
        if template_dir.exists():
            result_codes = {
                p.stem.replace("_summary", "") for p in summaries
            }
            pending = []
            for tpl in sorted(template_dir.glob("*.yaml")):
                with open(tpl) as f:
                    raw = yaml.safe_load(f)
                code = raw.get("experiment", {}).get("code", "")
                if code and code not in result_codes:
                    pending.append(code)

            if pending:
                lines.append("")
                lines.append(f"**Pending (no results yet):** {', '.join(pending)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sbatch generation
# ---------------------------------------------------------------------------

def _generate_sbatch(config: AblationConfig, yaml_path: str) -> Path:
    """Generate sbatch script for this experiment."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    sbatch_path = JOBS_DIR / f"{config.code}.sbatch"
    harness_path = Path(__file__).resolve()
    yaml_abs = Path(yaml_path).resolve()

    script = f"""#!/bin/bash
#SBATCH --job-name={config.code}
#SBATCH --output={LOGS_DIR.resolve()}/{config.code}_%j.out
#SBATCH --error={LOGS_DIR.resolve()}/{config.code}_%j.err
#SBATCH --time={SBATCH_TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={SBATCH_CPUS}
#SBATCH --mem={SBATCH_MEM}
#SBATCH --partition={SBATCH_PARTITION}

source /dcs/large/u5573765/env.sh
ulimit -s unlimited

cd {harness_path.parent}
python {harness_path} run {yaml_abs} --execute
"""

    with open(sbatch_path, "w") as f:
        f.write(script)
    sbatch_path.chmod(0o755)

    return sbatch_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_par2(status: str, runtime_ms: int, timeout_s: int) -> float:
    """PAR-2 score: runtime in seconds if solved, else 2 * timeout."""
    if status in ("sat", "unsat"):
        return runtime_ms / 1000.0
    return 2 * timeout_s


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Validate config, then either submit sbatch or execute locally."""
    loader = ConfigLoader()
    config = loader.load(args.yaml_path)

    print(f"Experiment: {config.code}")
    print(f"  Knob: {config.knob}")
    print(f"  Values: {config.values}")
    print(f"  Reference {config.knob}={config.reference[config.knob]}")
    print(f"  Suite: {config.suite_name}, Portfolio: {config.portfolio_id}")
    print(f"  Logics: {config.logics}")
    print(f"  K-fold: {config.reference['k']}")

    if args.dry_run:
        print("\n[DRY RUN] Config validated successfully. No execution.")
        return 0

    if args.execute:
        # Called from within sbatch — actually run the experiment
        # Clear sys.argv so MachSMT's own argparse doesn't choke on our args
        sys.argv = [sys.argv[0]]
        runner = ExperimentRunner(config, loader)
        runner.run()

        # Auto-print report
        reporter = ReportGenerator()
        report = reporter.generate(config.code)
        print(f"\n{report}")
        return 0

    # Default: generate sbatch script and submit
    sbatch_path = _generate_sbatch(config, args.yaml_path)
    print(f"\nGenerated sbatch: {sbatch_path}")

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"sbatch submission failed: {result.stderr.strip()}")
        return 1

    print(f"Submitted: {result.stdout.strip()}")
    return 0


def cmd_report(args):
    """Generate report for an experiment."""
    reporter = ReportGenerator()
    report = reporter.generate(args.experiment_code)
    print(report)
    return 0


def cmd_list(args):
    """List all experiments."""
    reporter = ReportGenerator()
    summary = reporter.list_experiments()
    print(summary)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 — MachSMT CLI Ablation Harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run an ablation experiment")
    run_parser.add_argument("yaml_path", help="Path to YAML config file")
    run_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config without running or submitting",
    )
    run_parser.add_argument(
        "--execute", action="store_true",
        help="Execute locally (used by sbatch script internally)",
    )

    # report
    report_parser = subparsers.add_parser("report", help="Generate experiment report")
    report_parser.add_argument("experiment_code", help="Experiment code (e.g., ABL-ML01)")

    # list
    subparsers.add_parser("list", help="List all experiments")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "list":
        return cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
