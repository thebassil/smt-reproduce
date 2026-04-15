#!/usr/bin/env python3
"""
Grackle Ablation Harness — CLI for running and reporting ablation experiments.

Subcommands:
    run <yaml>           Submit ablation experiment to SLURM
    run <yaml> --dry-run Generate files without submitting
    report <experiment>  Generate comparison report from results
    list                 Show status of all experiments
"""

import argparse
import csv
import difflib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from fly_templates import (
    KNOB_TO_FLY_KEYS,
    ARRAY_MAP,
    generate_fly_content,
    generate_fly_files,
    get_fly_filenames,
    get_solver_logic_dirs,
)

# Paths
ABLATION_DIR = Path(__file__).resolve().parent
EVAL_GRACKLE_DIR = Path("/dcs/23/u5573765/cs351/eval-grackle")
GRACKLE_RUNS_DIR = EVAL_GRACKLE_DIR / "grackle_runs"

# Default wall time for sbatch (18h = 12h Grackle timeout + 6h buffer for cvc5_qflia)
DEFAULT_WALL_TIME = "18:00:00"

# Shared assets to symlink from eval-grackle
SHARED_SYMLINKS = {
    "seeds":                       GRACKLE_RUNS_DIR / "seeds",
    "benchmarks_QF_BV.txt":        GRACKLE_RUNS_DIR / "benchmarks_QF_BV.txt",
    "benchmarks_QF_LIA.txt":       GRACKLE_RUNS_DIR / "benchmarks_QF_LIA.txt",
    "benchmarks_QF_NRA.txt":       GRACKLE_RUNS_DIR / "benchmarks_QF_NRA.txt",
    "shared_folds_k5.json":        EVAL_GRACKLE_DIR / "shared_folds_k5.json",
    "grackle_cli_evaluator.py":    EVAL_GRACKLE_DIR / "grackle_cli_evaluator.py",
    "grackle_kfold_evaluator.py":  EVAL_GRACKLE_DIR / "grackle_kfold_evaluator.py",
    "cvc5_inits.txt":              GRACKLE_RUNS_DIR / "cvc5_inits.txt",
    "z3_inits.txt":                GRACKLE_RUNS_DIR / "z3_inits.txt",
}


def _safe_value_str(value: Any) -> str:
    """Convert a value to a filesystem-safe string."""
    s = str(value)
    # Replace dots, spaces, etc. for directory names
    s = s.replace(" ", "_").replace("/", "_").replace(".", "_")
    return s


def _variant_dir_name(experiment: str, value: Any) -> str:
    """Generate the run directory name for a variant."""
    return f"{experiment}-{_safe_value_str(value)}"


def _compute_wall_time(experiment_config: dict, value: Any) -> str:
    """Compute wall time for a variant.

    Uses wall_time_override from the YAML if present, otherwise default.
    """
    overrides = experiment_config.get("wall_time_override")
    if overrides and isinstance(overrides, dict):
        val_str = str(value)
        if val_str in overrides:
            return overrides[val_str]
        # Try numeric key
        try:
            if int(value) in overrides or float(value) in overrides:
                return overrides.get(int(value), overrides.get(float(value)))
        except (ValueError, TypeError):
            pass
    return DEFAULT_WALL_TIME


def _generate_train_sbatch(
    experiment: str,
    value: Any,
    run_dir: Path,
    wall_time: str,
) -> str:
    """Generate the training sbatch script content."""
    solver_logic_dirs = get_solver_logic_dirs()
    fly_filenames = get_fly_filenames()

    dirs_str = " ".join(solver_logic_dirs)
    flies_str = " ".join(fly_filenames)
    run_dir_abs = str(run_dir.resolve())

    return f"""#!/bin/bash
#SBATCH --job-name=grackle-abl-{experiment}-{_safe_value_str(value)}
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --time={wall_time}
#SBATCH --array=0-5
#SBATCH --chdir={run_dir_abs}
#SBATCH --output=grackle_%A_%a.out
#SBATCH --error=grackle_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

# ============================================================================
# Grackle Ablation: {experiment}={value}
#
# Array tasks:
#   0 = cvc5 x QF_BV    1 = cvc5 x QF_LIA   2 = cvc5 x QF_NRA
#   3 = z3   x QF_BV    4 = z3   x QF_LIA   5 = z3   x QF_NRA
# ============================================================================

set -euo pipefail

source /dcs/large/u5573765/env.sh
export PYPROVE_BENCHMARKS=/dcs/large/u5573765/data/benchmarks/suite_9k

DIRS=({dirs_str})
FLIES=({flies_str})

IDX=${{SLURM_ARRAY_TASK_ID}}
RUN_DIR="${{DIRS[$IDX]}}"
FLY_FILE="${{FLIES[$IDX]}}"

echo "=============================================="
echo "Grackle Ablation: {experiment}={value}"
echo "Run: ${{RUN_DIR}}"
echo "=============================================="
echo "Start:     $(date)"
echo "Node:      $(hostname)"
echo "CPUs:      ${{SLURM_CPUS_PER_TASK}}"
echo "Fly file:  ${{FLY_FILE}}"
echo "Task ID:   ${{IDX}}"
echo ""

cd "{run_dir_abs}/${{RUN_DIR}}"

fly-grackle.py "${{FLY_FILE}}" 2>&1 | tee run.log

echo ""
echo "=============================================="
echo "Grackle Complete: ${{RUN_DIR}}"
echo "End: $(date)"
echo "=============================================="

echo ""
echo "Output files:"
ls -la db-*.json 2>/dev/null || echo "  (no db JSON files yet)"
ls -la confs/ 2>/dev/null || echo "  (no confs directory yet)"
echo ""
echo "Final configs found:"
grep -i "FINAL CONFIGURATIONS" run.log 2>/dev/null || echo "  (check run.log for details)"
"""


def _generate_eval_sbatch(
    experiment: str,
    value: Any,
    run_dir: Path,
    results_dir: Path,
) -> str:
    """Generate the evaluation sbatch script content."""
    run_dir_abs = str(run_dir.resolve())
    results_dir_abs = str(results_dir.resolve())

    return f"""#!/bin/bash
#SBATCH --job-name=grackle-eval-{experiment}-{_safe_value_str(value)}
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=00:30:00
#SBATCH --chdir={run_dir_abs}
#SBATCH --output=grackle_eval_%j.out
#SBATCH --error=grackle_eval_%j.err

# ============================================================================
# Grackle Evaluation: {experiment}={value}
# ============================================================================

set -euo pipefail

source /dcs/large/u5573765/env.sh

cd "{run_dir_abs}"

echo "=== Grackle CLI Evaluation: {experiment}={value} ==="
echo "Start: $(date)"

# Check all 6 runs produced output
MISSING=0
for dir in {{cvc5,z3}}_{{qfbv,qflia,qfnra}}; do
    if [ -f "${{dir}}/db-trains-cache.json" ]; then
        NCONF=$(python3 -c "import json; print(len(json.load(open('${{dir}}/db-trains-cache.json'))))")
        echo "  ${{dir}}: ${{NCONF}} configs"
    else
        echo "  MISSING: ${{dir}}/db-trains-cache.json"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo "ERROR: $MISSING db-trains-cache.json files missing. Cannot evaluate."
    exit 1
fi

echo ""

python3 grackle_cli_evaluator.py \\
    --runs-dir . --folds shared_folds_k5.json \\
    --output-dir "{results_dir_abs}" --timeout 30

echo ""
echo "Done: $(date)"
"""


def _load_experiment(yaml_path: str) -> dict:
    """Load and validate an experiment YAML file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    required_keys = ["experiment", "knob", "values"]
    for key in required_keys:
        if key not in config:
            print(f"ERROR: Missing required key '{key}' in {yaml_path}")
            sys.exit(1)

    if config["knob"] not in KNOB_TO_FLY_KEYS:
        print(f"ERROR: Unknown knob '{config['knob']}'. "
              f"Valid: {list(KNOB_TO_FLY_KEYS.keys())}")
        sys.exit(1)

    return config


def _create_symlinks(run_dir: Path):
    """Create symlinks for shared assets in the run directory and solver subdirs."""
    for name, target in SHARED_SYMLINKS.items():
        link_path = run_dir / name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target)

    # Grackle CDs into solver subdirs; seeds/ must be accessible from there
    seeds_target = SHARED_SYMLINKS["seeds"]
    for solver_dir_name in get_solver_logic_dirs():
        solver_dir = run_dir / solver_dir_name
        solver_dir.mkdir(parents=True, exist_ok=True)
        seeds_link = solver_dir / "seeds"
        if seeds_link.exists() or seeds_link.is_symlink():
            seeds_link.unlink()
        seeds_link.symlink_to(seeds_target)


def _coerce_value(value: Any, knob: str) -> Any:
    """Coerce YAML value to the correct Python type for .fly generation."""
    if knob in ("atavistic", "trainer_restarts"):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if knob == "runner_penalty":
        return int(float(value))
    if knob in ("tops", "best", "rank", "grackle_timeout", "trainer_timeout",
                "runner_timeout"):
        return int(value)
    return value


def cmd_run(args):
    """Execute the 'run' subcommand."""
    config = _load_experiment(args.yaml_file)
    experiment = config["experiment"]
    knob = config["knob"]
    values = config["values"]
    dry_run = args.dry_run

    runs_base = ABLATION_DIR / "runs"
    results_base = ABLATION_DIR / "results"

    print(f"{'[DRY RUN] ' if dry_run else ''}Ablation experiment: {experiment}")
    print(f"  Knob: {knob}")
    print(f"  Values: {values}")
    print(f"  Reference: {config.get('reference_value', 'N/A')}")
    print()

    submitted = []

    for value in values:
        coerced = _coerce_value(value, knob)
        variant = _variant_dir_name(experiment, value)
        run_dir = runs_base / variant
        results_dir = results_base / variant

        wall_time = _compute_wall_time(config, value)

        print(f"--- Variant: {variant} (wall_time={wall_time}) ---")

        # Create directories
        run_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Symlink shared assets
        _create_symlinks(run_dir)

        # Generate .fly files
        fly_files = generate_fly_files(run_dir, knob, coerced)
        print(f"  Generated {len(fly_files)} .fly files")

        # Show .fly diff for the first solver/logic pair
        ref_content = generate_fly_content("cvc5", "QF_BV")
        mod_content = generate_fly_content("cvc5", "QF_BV", knob, coerced)
        if ref_content != mod_content:
            print(f"  .fly change (cvc5_qfbv):")
            diff = difflib.unified_diff(
                ref_content.splitlines(),
                mod_content.splitlines(),
                lineterm="",
            )
            for line in diff:
                if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                    continue
                if line.startswith("-") or line.startswith("+"):
                    print(f"    {line}")
        else:
            print(f"  .fly: identical to reference (this IS the reference value)")

        # Generate sbatch scripts
        train_sbatch = _generate_train_sbatch(experiment, value, run_dir, wall_time)
        eval_sbatch = _generate_eval_sbatch(experiment, value, run_dir, results_dir)

        train_path = run_dir / "run_grackle.sbatch"
        eval_path = run_dir / "eval_grackle.sbatch"
        train_path.write_text(train_sbatch)
        eval_path.write_text(eval_sbatch)
        print(f"  Generated sbatch scripts")

        if dry_run:
            print(f"  [DRY RUN] Would submit: sbatch {train_path}")
            print(f"  [DRY RUN] Then: sbatch --dependency=afterok:<train_job> {eval_path}")
        else:
            # Submit training job
            try:
                result = subprocess.run(
                    ["sbatch", "--parsable", str(train_path)],
                    capture_output=True, text=True, check=True,
                    cwd=str(run_dir),
                )
                train_job_id = result.stdout.strip()
                print(f"  Submitted training: job {train_job_id}")

                # Submit eval job with dependency (afterany: cvc5_qfbv may
                # OOM on exit despite completing; eval validates db files)
                result = subprocess.run(
                    ["sbatch", "--parsable",
                     f"--dependency=afterany:{train_job_id}",
                     str(eval_path)],
                    capture_output=True, text=True, check=True,
                    cwd=str(run_dir),
                )
                eval_job_id = result.stdout.strip()
                print(f"  Submitted eval: job {eval_job_id} (after {train_job_id})")

                job_info = {
                    "experiment": experiment,
                    "knob": knob,
                    "value": value,
                    "variant": variant,
                    "train_job_id": train_job_id,
                    "eval_job_id": eval_job_id,
                    "wall_time": wall_time,
                    "submitted_at": datetime.now().isoformat(),
                }
                submitted.append(job_info)

                # Save job info
                jobs_path = run_dir / "jobs.json"
                with open(jobs_path, "w") as f:
                    json.dump(job_info, f, indent=2)

            except subprocess.CalledProcessError as e:
                print(f"  ERROR submitting: {e.stderr}")
                continue
            except FileNotFoundError:
                print(f"  ERROR: sbatch not found. Are you on a SLURM cluster?")
                sys.exit(1)

        print()

    # Summary
    print("=" * 60)
    if dry_run:
        print(f"[DRY RUN] Would submit {len(values)} variants for '{experiment}'")
        print(f"Total SLURM jobs: {len(values)} training arrays + "
              f"{len(values)} eval = {2 * len(values)}")
    else:
        print(f"Submitted {len(submitted)}/{len(values)} variants for '{experiment}'")
        for job in submitted:
            print(f"  {job['variant']}: train={job['train_job_id']} "
                  f"eval={job['eval_job_id']}")


def cmd_report(args):
    """Execute the 'report' subcommand."""
    experiment = args.experiment
    results_base = ABLATION_DIR / "results"

    # Find all result directories for this experiment
    pattern = f"{experiment}-*"
    result_dirs = sorted(results_base.glob(pattern))

    if not result_dirs:
        print(f"No results found for experiment '{experiment}'")
        print(f"Searched: {results_base / pattern}")
        return

    print(f"# Ablation Report: {experiment}")
    print()

    # Collect data from each variant
    rows = []
    for rd in result_dirs:
        summary_path = rd / "cli_kfold_k5_summary.csv"
        if not summary_path.exists():
            continue

        variant = rd.name
        value = variant[len(experiment) + 1:]  # strip "experiment-" prefix

        with open(summary_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["strategy"] == "GRACKLE":
                    rows.append({
                        "variant": variant,
                        "value": value,
                        "avg_par2": float(row["avg_par2"]),
                        "solved_pct": float(row["solved_pct"]),
                        "total_par2": float(row["total_par2"]),
                        "solved": int(row["solved"]),
                    })

    if not rows:
        print("No completed results found (missing cli_kfold_k5_summary.csv)")
        return

    # Find reference row (load YAML to identify reference value)
    yaml_path = ABLATION_DIR / "experiments" / f"{experiment}.yaml"
    ref_value = None
    if yaml_path.exists():
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        ref_value = _safe_value_str(config.get("reference_value"))

    ref_par2 = None
    ref_solved = None
    for r in rows:
        if ref_value and r["value"] == ref_value:
            ref_par2 = r["avg_par2"]
            ref_solved = r["solved_pct"]
            break

    # Print table
    print(f"| {'Value':<20} | {'Avg PAR-2':>12} | {'Solved%':>10} | "
          f"{'ΔPAR-2':>12} | {'ΔSolved%':>10} |")
    print(f"|{'-' * 22}|{'-' * 14}|{'-' * 12}|{'-' * 14}|{'-' * 12}|")

    for r in rows:
        delta_par2 = ""
        delta_solved = ""
        marker = ""
        if ref_par2 is not None:
            dp = r["avg_par2"] - ref_par2
            ds = r["solved_pct"] - ref_solved
            delta_par2 = f"{dp:+.3f}"
            delta_solved = f"{ds:+.2f}%"
        if ref_value and r["value"] == ref_value:
            marker = " (ref)"

        print(f"| {r['value'] + marker:<20} | {r['avg_par2']:>12.3f} | "
              f"{r['solved_pct']:>9.2f}% | {delta_par2:>12} | "
              f"{delta_solved:>10} |")

    print()

    # Per-logic breakdown if available
    for rd in result_dirs:
        perlogic_path = rd / "cli_kfold_k5_perlogic.csv"
        if perlogic_path.exists():
            variant = rd.name
            value = variant[len(experiment) + 1:]
            print(f"### {value}")
            with open(perlogic_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["strategy"] == "GRACKLE":
                        print(f"  {row['logic']}: PAR-2={float(row['avg_par2']):.3f} "
                              f"Solved={row['solved']}/{row['total']} "
                              f"({float(row['solved_pct']):.1f}%)")
            print()


def cmd_list(args):
    """Execute the 'list' subcommand."""
    runs_base = ABLATION_DIR / "runs"
    results_base = ABLATION_DIR / "results"

    if not runs_base.exists():
        print("No runs directory found. Run an experiment first.")
        return

    print(f"{'Experiment':<25} {'Value':<15} {'Status':<15} "
          f"{'Train Job':<12} {'Eval Job':<12} {'PAR-2':>10} {'Solved%':>10}")
    print("-" * 100)

    for run_dir in sorted(runs_base.iterdir()):
        if not run_dir.is_dir():
            continue

        variant = run_dir.name
        jobs_path = run_dir / "jobs.json"

        # Parse experiment name and value
        parts = variant.split("-", 1)
        if len(parts) == 2:
            exp_name, value = parts
        else:
            exp_name, value = variant, "?"

        train_job = eval_job = "-"
        if jobs_path.exists():
            with open(jobs_path) as f:
                jobs = json.load(f)
            train_job = jobs.get("train_job_id", "-")
            eval_job = jobs.get("eval_job_id", "-")

        # Check for results
        results_dir = results_base / variant
        summary_path = results_dir / "cli_kfold_k5_summary.csv"

        status = "pending"
        par2_str = "-"
        solved_str = "-"

        if summary_path.exists():
            status = "completed"
            with open(summary_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["strategy"] == "GRACKLE":
                        par2_str = f"{float(row['avg_par2']):.3f}"
                        solved_str = f"{float(row['solved_pct']):.1f}%"
        elif jobs_path.exists():
            # Check if any SLURM output files exist (indicates running/ran)
            slurm_outs = list(run_dir.glob("grackle_*.out"))
            if slurm_outs:
                status = "running"

        print(f"{exp_name:<25} {value:<15} {status:<15} "
              f"{train_job:<12} {eval_job:<12} {par2_str:>10} {solved_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Grackle Ablation Harness CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # run
    run_parser = subparsers.add_parser("run", help="Run ablation experiment")
    run_parser.add_argument("yaml_file", help="Path to experiment YAML file")
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Generate files without submitting to SLURM")

    # report
    report_parser = subparsers.add_parser("report",
                                          help="Generate comparison report")
    report_parser.add_argument("experiment", help="Experiment name")

    # list
    subparsers.add_parser("list", help="List all experiments and their status")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
