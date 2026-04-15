#!/usr/bin/env python3
"""
Ablation report generator.

Reads CSV results from completed experiments and generates:
  - Per-experiment markdown summaries
  - Per-knob comparison tables
  - Full ablation comparison CSV
"""

import csv
from pathlib import Path

from ablation.config import ALGORITHMIC_KNOBS, experiment_code, load_reference, load_template


def _load_summary(results_dir, k_folds=5):
    """Load kfold summary CSV. Returns dict {strategy: {metric: value}}."""
    path = results_dir / "results" / f"kfold_k{k_folds}_summary.csv"
    if not path.exists():
        return None
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["strategy"]] = {
                "total_par2": float(row["total_par2"]),
                "avg_par2": float(row["avg_par2"]),
                "solved": int(row["solved"]),
                "solved_pct": float(row["solved_pct"]),
            }
    return rows


def report_single(code):
    """Generate markdown for one experiment."""
    reference = load_reference()
    results_base = Path(reference["paths"]["results_base"])
    results_dir = results_base / code
    k_folds = reference["convention"]["k_folds"]

    summary = _load_summary(results_dir, k_folds)
    if summary is None:
        print(f"No results found for {code}")
        return

    print(f"# Experiment: {code}")
    print()
    print(f"| Strategy | Avg PAR-2 | Solved % |")
    print(f"|----------|-----------|----------|")
    for strat in ["VBS", "SMTGAZER", "CSBS", "SBS", "RANDOM"]:
        if strat in summary:
            s = summary[strat]
            print(f"| {strat} | {s['avg_par2']:.3f} | {s['solved_pct']:.1f}% |")


def report_knob(knob):
    """Generate comparison table across values of one knob."""
    reference = load_reference()
    results_base = Path(reference["paths"]["results_base"])
    k_folds = reference["convention"]["k_folds"]

    # Find the template
    template_dir = Path(__file__).parent / "experiments"
    template_path = template_dir / f"{knob}.yaml"
    if not template_path.exists():
        print(f"No template found for knob: {knob}")
        return

    template = load_template(template_path)
    ref_value = template["reference_value"]
    values = template["values"]

    print(f"# Ablation: {knob}")
    print(f"Reference value: {ref_value}")
    print()
    print(f"| Value | SMTGazer Avg PAR-2 | SMTGazer Solved % | Delta vs Ref | Status |")
    print(f"|-------|--------------------|--------------------|--------------|--------|")

    ref_avg_par2 = None

    for value in values:
        code = experiment_code(knob, value)
        results_dir = results_base / code
        summary = _load_summary(results_dir, k_folds)

        if summary is None:
            marker = " (ref)" if value == ref_value else ""
            print(f"| {value}{marker} | - | - | - | NOT RUN |")
            continue

        sg = summary.get("SMTGAZER", {})
        avg_par2 = sg.get("avg_par2", 0)
        solved_pct = sg.get("solved_pct", 0)

        if value == ref_value:
            ref_avg_par2 = avg_par2

        if ref_avg_par2 is not None:
            delta = avg_par2 - ref_avg_par2
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "-"

        marker = " **(ref)**" if value == ref_value else ""
        print(f"| {value}{marker} | {avg_par2:.3f} | {solved_pct:.1f}% | {delta_str} | DONE |")


def report_all():
    """Generate full ablation report across all knobs."""
    reference = load_reference()
    results_base = Path(reference["paths"]["results_base"])
    k_folds = reference["convention"]["k_folds"]
    template_dir = Path(__file__).parent / "experiments"

    print("# SMTGazer Ablation Study — Full Results")
    print()

    all_rows = []

    for knob in ALGORITHMIC_KNOBS:
        template_path = template_dir / f"{knob}.yaml"
        if not template_path.exists():
            continue

        template = load_template(template_path)
        ref_value = template["reference_value"]

        # Get reference result for delta computation
        ref_code = experiment_code(knob, ref_value)
        ref_summary = _load_summary(results_base / ref_code, k_folds)
        ref_avg = ref_summary["SMTGAZER"]["avg_par2"] if ref_summary and "SMTGAZER" in ref_summary else None

        for value in template["values"]:
            code = experiment_code(knob, value)
            results_dir = results_base / code
            summary = _load_summary(results_dir, k_folds)

            if summary is None:
                continue

            sg = summary.get("SMTGAZER", {})
            avg_par2 = sg.get("avg_par2", 0)
            solved_pct = sg.get("solved_pct", 0)
            delta = avg_par2 - ref_avg if ref_avg is not None else None

            all_rows.append({
                "experiment_code": code,
                "knob": knob,
                "value": value,
                "is_reference": value == ref_value,
                "smtgazer_avg_par2": avg_par2,
                "smtgazer_solved_pct": solved_pct,
                "delta_vs_reference": delta,
            })

        print(f"\n## {knob}")
        report_knob(knob)
        print()

    # Write comparison CSV
    if all_rows:
        csv_path = results_base / "ablation_comparison.csv"
        fieldnames = [
            "experiment_code", "knob", "value", "is_reference",
            "smtgazer_avg_par2", "smtgazer_solved_pct", "delta_vs_reference",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nComparison CSV: {csv_path}")
