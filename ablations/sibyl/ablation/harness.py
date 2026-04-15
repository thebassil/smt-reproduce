#!/usr/bin/env python3
"""
Sibyl CLI Ablation Harness

Subcommands:
    run   <yaml>            Validate, generate sbatch scripts, and submit jobs
    run   <yaml> --dry-run  Validate and show what would be submitted
    report <experiment_code> Generate markdown report from CSV results
    list                     List all experiments from CSV files
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "ablation"
REFERENCE_PATH = ABLATION_DIR / "reference_9k.json"
CONFIGS_DIR = ABLATION_DIR / "configs"
RESULTS_DIR = Path("/dcs/large/u5573765/ablation_results/sibyl_9k")
TEMPLATE_PATH = ABLATION_DIR / "sbatch_template.sh"


# ── helpers ──────────────────────────────────────────────────────

def load_reference() -> dict:
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def load_experiment_yaml(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def validate_experiment(exp: dict, ref: dict) -> List[str]:
    """Validate a YAML experiment definition against the reference config."""
    errors = []

    for field in ("experiment_code", "knob", "values"):
        if field not in exp:
            errors.append(f"Missing required field: {field}")
    if errors:
        return errors

    knob = exp["knob"]

    if knob not in ref["algorithmic"]:
        errors.append(
            f"Knob '{knob}' not found in reference algorithmic config. "
            f"Valid knobs: {list(ref['algorithmic'].keys())}"
        )
        return errors

    # Check reference value is present
    ref_val = ref["algorithmic"][knob]
    values = exp["values"]

    # Normalise for comparison (handle lists, booleans, etc.)
    def norm(v):
        if isinstance(v, list):
            return json.dumps(sorted(v))
        return str(v).lower()

    ref_norm = norm(ref_val)
    value_norms = [norm(v) for v in values]

    if ref_norm not in value_norms:
        errors.append(
            f"Reference value '{ref_val}' for knob '{knob}' is not "
            f"among ablation values {values}. It must be included."
        )

    if len(values) < 2:
        errors.append(f"Need at least 2 ablation values, got {len(values)}")

    return errors


def build_run_config(ref: dict, exp: dict, value: Any) -> dict:
    """Build a concrete run config: reference + one knob override."""
    cfg = copy.deepcopy(ref)

    # Remove metadata
    for k in list(cfg.keys()):
        if k.startswith("_"):
            del cfg[k]

    # Apply the single knob override
    knob = exp["knob"]
    cfg["algorithmic"][knob] = value

    # Metadata
    cfg["experiment_code"] = exp["experiment_code"]
    cfg["knob"] = knob
    cfg["value"] = value

    return cfg


def value_slug(value: Any) -> str:
    """Make a filesystem-safe slug from a value."""
    if isinstance(value, list):
        return "_".join(str(v) for v in value).replace(" ", "").lower()
    return str(value).replace(" ", "_").lower()


def render_sbatch(cfg: dict, run_config_path: str, results_dir: str,
                  logic: str) -> str:
    """Render the sbatch template with concrete values."""
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    replacements = {
        "{{EXPERIMENT_CODE}}": cfg["experiment_code"],
        "{{KNOB}}": cfg["knob"],
        "{{VALUE}}": str(cfg["value"]),
        "{{VALUE_SLUG}}": value_slug(cfg["value"]),
        "{{LOGIC}}": logic,
        "{{TIMESTAMP}}": datetime.now(timezone.utc).isoformat(),
        "{{RESULTS_DIR}}": results_dir,
        "{{RUN_CONFIG}}": run_config_path,
        "{{PROJECT_ROOT}}": str(PROJECT_ROOT),
    }
    result = template
    for placeholder, val in replacements.items():
        result = result.replace(placeholder, val)
    return result


# ── subcommands ──────────────────────────────────────────────────

def cmd_run(args):
    """Validate, generate, and submit sbatch jobs for an experiment."""
    ref = load_reference()
    exp = load_experiment_yaml(args.yaml)

    errors = validate_experiment(exp, ref)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        return 1

    code = exp["experiment_code"]
    knob = exp["knob"]
    values = exp["values"]
    logics = ref["data"]["logics"]

    print(f"Experiment: {code}")
    print(f"Knob:       {knob}")
    print(f"Values:     {values}")
    print(f"Reference:  {ref['algorithmic'][knob]}")
    print(f"Logics:     {logics}")
    print(f"Dry run:    {args.dry_run}")
    print()

    exp_results_dir = RESULTS_DIR / code
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    submitted = []

    for value in values:
        slug = value_slug(value)

        for logic in logics:
            variant_dir = exp_results_dir / slug / logic
            variant_dir.mkdir(parents=True, exist_ok=True)

            # Write run config JSON
            run_cfg = build_run_config(ref, exp, value)
            config_path = variant_dir / "run_config.json"
            with open(config_path, "w") as f:
                json.dump(run_cfg, f, indent=2)

            # Render sbatch script
            sbatch_content = render_sbatch(
                run_cfg, str(config_path), str(variant_dir), logic,
            )
            sbatch_path = variant_dir / "run.sbatch"
            with open(sbatch_path, "w") as f:
                f.write(sbatch_content)
            os.chmod(sbatch_path, 0o755)

            label = f"{slug}/{logic}"

            if args.dry_run:
                print(f"  [{label}] DRY RUN — would submit: {sbatch_path}")
            else:
                print(f"  [{label}] Submitting: {sbatch_path}")
                result = subprocess.run(
                    ["sbatch", str(sbatch_path)],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"    Submitted: job {job_id}")
                    submitted.append({
                        "value": value, "logic": logic, "job_id": job_id,
                    })
                    with open(variant_dir / "slurm_job_id.txt", "w") as f:
                        f.write(job_id + "\n")
                else:
                    print(f"    FAILED: {result.stderr.strip()}")
                    return 1

    # Write experiment manifest
    manifest = {
        "experiment_code": code,
        "knob": knob,
        "values": values,
        "logics": logics,
        "reference_value": ref["algorithmic"][knob],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "jobs": submitted,
        "dry_run": args.dry_run,
    }
    with open(exp_results_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_jobs = len(values) * len(logics)
    print(f"\n{'DRY RUN complete' if args.dry_run else f'Submitted {len(submitted)} jobs'} "
          f"({n_jobs} total = {len(values)} values × {len(logics)} logics).")
    print(f"Results dir: {exp_results_dir}")
    return 0


def cmd_report(args):
    """Generate a markdown report from CSV results."""
    code = args.code
    exp_dir = RESULTS_DIR / code

    if not exp_dir.exists():
        print(f"ERROR: No results directory for experiment '{code}'")
        return 1

    # Load manifest
    manifest_path = exp_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"experiment_code": code, "knob": "unknown"}

    # Collect summaries
    summaries = []
    for summary_csv in sorted(exp_dir.rglob("summary.csv")):
        with open(summary_csv) as f:
            for row in csv.DictReader(f):
                summaries.append(row)

    if not summaries:
        print(f"No summary CSVs found in {exp_dir}/**/summary.csv")
        print("Jobs may still be running. Check with: squeue -u $USER")
        return 1

    # Collect per-fold details
    perfold_rows = []
    for perfold_csv in sorted(exp_dir.rglob("perfold.csv")):
        with open(perfold_csv) as f:
            for row in csv.DictReader(f):
                perfold_rows.append(row)

    knob = manifest.get("knob", summaries[0].get("knob", "unknown"))
    ref_val = manifest.get("reference_value", "?")

    # Group summaries by logic
    logics = sorted(set(s.get("logic", "?") for s in summaries))

    lines = [
        f"# Ablation Report: {code}",
        "",
        f"**Knob**: `{knob}`  ",
        f"**Reference value**: `{ref_val}`  ",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        "",
    ]

    for logic in logics:
        logic_summaries = [s for s in summaries if s.get("logic") == logic]
        if not logic_summaries:
            continue

        lines += [
            f"## {logic} — Cross-Fold Summary",
            "",
            "| Value | Sibyl PAR-2 | Solved | Solved % | vs SBS | Closeness to VBS | Wall (s) |",
            "|-------|-------------|--------|----------|--------|------------------|----------|",
        ]

        for s in logic_summaries:
            val = s.get("value", "?")
            marker = " **(ref)**" if str(val).lower() == str(ref_val).lower() else ""
            lines.append(
                f"| {val}{marker} "
                f"| {float(s.get('sibyl_par2', 0)):.1f} "
                f"| {s.get('sibyl_solved', '?')} "
                f"| {float(s.get('sibyl_solved_pct', 0)):.1f}% "
                f"| {float(s.get('sibyl_vs_sbs_pct', 0)):+.1f}% "
                f"| {float(s.get('closeness_to_vbs_pct', 0)):.1f}% "
                f"| {float(s.get('total_wall_seconds', 0)):.0f} |"
            )

        # Baselines
        s0 = logic_summaries[0]
        lines += [
            "",
            f"### {logic} Baselines",
            "",
            "| Strategy | PAR-2 | Solved | Solved % |",
            "|----------|-------|--------|----------|",
            f"| VBS | {float(s0.get('vbs_par2', 0)):.1f} | {s0.get('vbs_solved', '?')} | {float(s0.get('vbs_solved_pct', 0)):.1f}% |",
            f"| SBS | {float(s0.get('sbs_par2', 0)):.1f} | {s0.get('sbs_solved', '?')} | {float(s0.get('sbs_solved_pct', 0)):.1f}% |",
            "",
        ]

    # Per-fold detail
    if perfold_rows:
        lines += [
            "## Per-Fold Detail",
            "",
            "| Logic | Value | Fold | Sibyl PAR-2 | Solved | VBS PAR-2 | SBS PAR-2 | Wall (s) |",
            "|-------|-------|------|-------------|--------|-----------|-----------|----------|",
        ]
        for r in perfold_rows:
            lines.append(
                f"| {r.get('logic', '?')} "
                f"| {r.get('value', '?')} "
                f"| {r.get('fold', '?')} "
                f"| {float(r.get('sibyl_par2', 0)):.1f} "
                f"| {r.get('sibyl_solved', '?')} "
                f"| {float(r.get('vbs_par2', 0)):.1f} "
                f"| {float(r.get('sbs_par2', 0)):.1f} "
                f"| {float(r.get('wall_seconds', 0)):.0f} |"
            )

    report_text = "\n".join(lines) + "\n"

    report_path = exp_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport written to: {report_path}")
    return 0


def cmd_list(args):
    """List all experiments from CSV files."""
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return 0

    experiments = []
    for exp_dir in sorted(RESULTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        code = exp_dir.name
        manifest_path = exp_dir / "manifest.json"

        # Count completed variants
        variants_total = 0
        variants_done = 0
        for summary_csv in exp_dir.rglob("summary.csv"):
            variants_done += 1
        for run_config in exp_dir.rglob("run_config.json"):
            variants_total += 1

        # Load manifest
        knob = "?"
        ref_val = "?"
        submitted_at = "?"
        if manifest_path.exists():
            with open(manifest_path) as f:
                m = json.load(f)
            knob = m.get("knob", "?")
            ref_val = m.get("reference_value", "?")
            submitted_at = m.get("submitted_at", "?")[:19]

        status = ("complete" if variants_done == variants_total and variants_total > 0
                  else f"{variants_done}/{variants_total}")

        experiments.append({
            "code": code,
            "knob": knob,
            "ref": str(ref_val)[:12],
            "status": status,
            "submitted": submitted_at,
        })

    if not experiments:
        print("No experiments found.")
        return 0

    print(f"{'Code':<25} {'Knob':<18} {'Ref':<14} {'Status':<12} {'Submitted':<20}")
    print("-" * 89)
    for e in experiments:
        print(f"{e['code']:<25} {e['knob']:<18} {e['ref']:<14} {e['status']:<12} {e['submitted']:<20}")

    print(f"\n{len(experiments)} experiment(s) found in {RESULTS_DIR}")
    return 0


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sibyl CLI Ablation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py run configs/epochs.yaml --dry-run
  python harness.py run configs/pool_type.yaml
  python harness.py report ABL_POOL_TYPE
  python harness.py list
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Validate + generate + submit sbatch jobs")
    p_run.add_argument("yaml", help="Path to experiment YAML config")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Validate only, no submission")

    # report
    p_report = sub.add_parser("report", help="Generate markdown report from CSV results")
    p_report.add_argument("code", help="Experiment code (e.g. ABL_EPOCHS)")

    # list
    sub.add_parser("list", help="List all experiments from CSV files")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "list":
        return cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
