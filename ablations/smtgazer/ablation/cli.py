#!/usr/bin/env python3
"""
Ablation CLI entry point.

Usage:
  python -m ablation.cli run <yaml> [--dry-run] [--values V1,V2,...] [--force]
  python -m ablation.cli report <code|knob|all>
  python -m ablation.cli list [--status {all|completed|running|failed}]
"""

import argparse
import json
import sys
from pathlib import Path

from ablation.config import ALGORITHMIC_KNOBS, experiment_code, load_reference, load_template
from ablation.report import report_all, report_knob, report_single
from ablation.runner import cmd_run, is_completed


def cmd_list(status_filter="all"):
    """List all experiments with their status."""
    reference = load_reference()
    results_base = Path(reference["paths"]["results_base"])
    k_folds = reference["convention"]["k_folds"]
    template_dir = Path(__file__).parent / "experiments"

    print(f"{'Experiment':<35} {'Knob':<28} {'Value':<10} {'Status':<12} {'Job ID':<12}")
    print("-" * 100)

    total = 0
    completed = 0
    running = 0
    failed = 0
    pending = 0

    for knob in ALGORITHMIC_KNOBS:
        template_path = template_dir / f"{knob}.yaml"
        if not template_path.exists():
            continue

        template = load_template(template_path)

        for value in template["values"]:
            code = experiment_code(knob, value)
            results_dir = results_base / code

            # Determine status
            meta_path = results_dir / "experiment_meta.json"
            summary_exists = is_completed(results_dir, k_folds)

            if summary_exists:
                status = "completed"
                completed += 1
            elif meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                job_id = meta.get("job_id", "-")
                meta_status = meta.get("status", "unknown")
                if meta_status == "submit_failed":
                    status = "failed"
                    failed += 1
                else:
                    status = "running"
                    running += 1
            else:
                status = "pending"
                job_id = "-"
                pending += 1

            total += 1

            # Apply filter
            if status_filter != "all" and status != status_filter:
                continue

            ref_marker = " *" if value == template["reference_value"] else ""
            job_str = "-"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                job_str = str(meta.get("job_id", "-"))

            print(f"{code:<35} {knob:<28} {str(value) + ref_marker:<10} {status:<12} {job_str:<12}")

    print("-" * 100)
    print(f"Total: {total}  |  Completed: {completed}  |  Running: {running}  |  Failed: {failed}  |  Pending: {pending}")
    print(f"(* = reference value)")


def main():
    parser = argparse.ArgumentParser(
        prog="ablation",
        description="SMTGazer ablation study harness",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    run_parser = subparsers.add_parser("run", help="Submit ablation experiments")
    run_parser.add_argument("yaml", help="Path to experiment YAML template")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate and print without submitting")
    run_parser.add_argument("--values", type=str, help="Comma-separated list of values to run")
    run_parser.add_argument("--force", action="store_true", help="Rerun even if completed")

    # report
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("target", help="Experiment code, knob name, or 'all'")

    # list
    list_parser = subparsers.add_parser("list", help="List experiments and status")
    list_parser.add_argument(
        "--status",
        choices=["all", "completed", "running", "failed", "pending"],
        default="all",
        help="Filter by status",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        values = None
        if args.values:
            values = [v.strip() for v in args.values.split(",")]
        return cmd_run(args.yaml, dry_run=args.dry_run, values=values, force=args.force)

    elif args.command == "report":
        if args.target == "all":
            report_all()
        elif args.target in ALGORITHMIC_KNOBS:
            report_knob(args.target)
        else:
            report_single(args.target)
        return 0

    elif args.command == "list":
        cmd_list(args.status)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
