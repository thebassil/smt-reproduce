#!/usr/bin/env python3
"""Register ablation experiments in DB and launch SLURM sweep jobs.

Usage:
    python scripts/launch_sweep.py --dry-run              # print what would be submitted
    python scripts/launch_sweep.py --register-only        # register experiments, write map, no submit
    python scripts/launch_sweep.py --submit               # register + submit all
    python scripts/launch_sweep.py --submit --axis=model  # only model axis
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add repo root to path
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from pipeline.db import create_experiment

DB_PATH = "/dcs/large/u5573765/db/results.sqlite"  # override with --db
PORTFOLIO_ID = 7
SUITE_NAME = "final_suite"
LOGICS = ["QF_LIA", "QF_BV", "QF_NRA"]
SLURM_DIR = REPO_DIR / "scripts" / "slurm"
SWEEP_MAP_PATH = REPO_DIR / "sweep_experiments.json"

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Card definitions: core vs extended, CPU vs GPU
# ---------------------------------------------------------------------------

CORE_MODELS_CPU = ["adaboost_ehm", "knn", "rf", "xgboost", "lightgbm"]
CORE_MODELS_GPU = ["gat_mlp", "gcn", "gin", "graphsage", "graph_transformer"]
EXTENDED_MODELS_CPU = [
    "autofolio", "collaborative_filtering", "conformal", "contextual_bandits",
    "cost_sensitive", "gp_regression", "greedy_logic", "instance_clustering",
    "lambdarank", "mosap", "online_learning", "ordinal_regression",
    "pwc", "quantile_regression", "solver_logic_pwc", "stacking", "survival",
]
EXTENDED_MODELS_GPU = [
    "deep_kernel", "hypernetworks", "meta_learning", "moe",
    "neural_processes", "rl_scheduling", "transformer",
]

CORE_FEATURISERS_CPU = [
    "machsmt_162", "static_expanded", "static_light_bow",
    "entropy", "proof_complexity",
]
EXTENDED_FEATURISERS_CPU = [
    "community", "community_modularity_only", "community_spectral",
    "dynamic_probes", "dynamic_probes_05s", "dynamic_probes_1s",
    "dynamic_probes_2s", "dynamic_probes_5s",
    "entropy_no_mi", "hash_kernel", "hypergraph_features",
    "random_walk_kernel", "spectral",
    "static_expanded_k10", "static_expanded_k50", "static_expanded_k100",
    "static_expanded_pca_only", "static_expanded_poly_only",
    "static_light_bow_binary", "static_light_bow_no_arith",
    "static_light_bow_no_bv", "static_light_bow_theory",
    "structural_width", "structural_width_approx",
    "tda", "wl_kernel", "wl_kernel_h1", "wl_kernel_h2", "wl_kernel_h5",
]
CORE_FEATURISERS_GPU = []  # sibyl_ast_ud used as ref_gnn fixed featuriser
EXTENDED_FEATURISERS_GPU = [
    "aig", "ast_path", "ast_path_len5", "ast_path_len10", "ast_path_len20",
    "clause_hypergraph", "contrastive_embed", "contrastive_embed_supervised_only",
    "formula2vec", "lcg", "lcg_star", "lig", "llm_embed", "llm_embed_finetuned",
    "neuroback", "sibyl_ast_ud", "smt_dag", "subgraph_pool",
    "tess", "transformer_embed", "transformer_embed_graph_transformer",
    "tree_lstm", "tree_lstm_flat_lstm",
    "tripartite", "tripartite_no_posenc", "vcg", "vcg_star", "vig",
]

CORE_POLICIES = [
    "top1_full", "top2_split", "top2_split_50_50", "top2_split_70_30",
    "top2_split_80_20", "top3_split", "exponential_timer",
    "confidence_gate", "softmax_sampling", "inverse_runtime_proportional",
    "pairwise_voting", "presolver_then_select", "cluster_dispatch",
]
EXTENDED_POLICIES = [
    "cluster_dispatch_smac", "confidence_gate_tau05", "confidence_gate_tau09",
    "inverse_runtime_k2", "inverse_runtime_k5",
    "pairwise_voting_unweighted", "presolver_frac05", "presolver_frac20",
    "probability_proportional", "rank", "select_argmin",
    "softmax_sampling_t01", "softmax_sampling_t5",
    "survival_curve_k3", "survival_curve_schedule",
    "survival_risk_averse", "survival_risk_neutral",
]

SWEEPS = {
    # ── CORE: ref_vector ──────────────────────────────────────────────
    "core_model_cpu": {
        "axis": "model", "tier": "core",
        "cards": CORE_MODELS_CPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "machsmt_162", "policy_card": "select_argmin"},
        "sbatch": "sweep_vector.sbatch",
        "description": "Core VECTOR model sweep (CPU, tiger)",
    },
    "core_model_gpu": {
        "axis": "model", "tier": "core",
        "cards": CORE_MODELS_GPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "sibyl_ast_ud", "policy_card": "select_argmin"},
        "sbatch": "sweep_gnn.sbatch",
        "description": "Core GNN model sweep (GPU, gecko)",
    },
    "core_featuriser_cpu": {
        "axis": "featuriser", "tier": "core",
        "cards": CORE_FEATURISERS_CPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "model_card": "adaboost_ehm", "policy_card": "select_argmin"},
        "sbatch": "sweep_featuriser.sbatch",
        "description": "Core featuriser sweep (CPU, tiger)",
    },
    "core_policy": {
        "axis": "policy", "tier": "core",
        "cards": CORE_POLICIES,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "machsmt_162", "model_card": "adaboost_ehm"},
        "sbatch": "sweep_policy.sbatch",
        "description": "Core policy sweep (CPU, tiger)",
    },
    # ── EXTENDED: ref_vector ──────────────────────────────────────────
    "ext_model_cpu": {
        "axis": "model", "tier": "extended",
        "cards": EXTENDED_MODELS_CPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "machsmt_162", "policy_card": "select_argmin"},
        "sbatch": "sweep_vector.sbatch",
        "description": "Extended VECTOR model sweep (CPU, tiger)",
    },
    "ext_model_gpu": {
        "axis": "model", "tier": "extended",
        "cards": EXTENDED_MODELS_GPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "sibyl_ast_ud", "policy_card": "select_argmin"},
        "sbatch": "sweep_gnn.sbatch",
        "description": "Extended GPU model sweep (GPU, gecko)",
    },
    "ext_featuriser_cpu": {
        "axis": "featuriser", "tier": "extended",
        "cards": EXTENDED_FEATURISERS_CPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "model_card": "adaboost_ehm", "policy_card": "select_argmin"},
        "sbatch": "sweep_featuriser.sbatch",
        "description": "Extended CPU featuriser sweep (CPU, tiger)",
    },
    "ext_featuriser_gpu": {
        "axis": "featuriser", "tier": "extended",
        "cards": EXTENDED_FEATURISERS_GPU,
        "fixed": {"portfolio_card": "fixed_portfolio", "model_card": "gat_mlp", "policy_card": "select_argmin"},
        "sbatch": "sweep_gnn.sbatch",
        "description": "Extended GPU featuriser sweep (GPU, gecko)",
    },
    "ext_policy": {
        "axis": "policy", "tier": "extended",
        "cards": EXTENDED_POLICIES,
        "fixed": {"portfolio_card": "fixed_portfolio", "featuriser_card": "machsmt_162", "model_card": "adaboost_ehm"},
        "sbatch": "sweep_policy.sbatch",
        "description": "Extended policy sweep (CPU, tiger)",
    },
}


def register_experiments(db_path: str, sweep_filter: str | None = None) -> dict[str, int]:
    """Register all experiments in DB. Returns {card|logic -> experiment_id} map."""
    experiment_map: dict[str, int] = {}

    for sweep_name, sweep in SWEEPS.items():
        if sweep_filter and sweep["axis"] != sweep_filter:
            continue

        axis = sweep["axis"]
        fixed = sweep["fixed"]

        for card in sweep["cards"]:
            for logic in LOGICS:
                # Build the 4 card columns
                cards = dict(fixed)
                cards[f"{axis}_card"] = card

                key = f"{card}|{logic}"
                if key in experiment_map:
                    continue

                exp_id = create_experiment(
                    db_path=db_path,
                    portfolio_card=cards["portfolio_card"],
                    featuriser_card=cards["featuriser_card"],
                    model_card=cards["model_card"],
                    policy_card=cards["policy_card"],
                    portfolio_id=PORTFOLIO_ID,
                    logic=logic,
                    suite_name=SUITE_NAME,
                    description=f"{sweep_name}: {card} on {logic}",
                )
                experiment_map[key] = exp_id
                print(f"  Registered: {key} -> experiment_id={exp_id}")

    return experiment_map


def write_map(experiment_map: dict[str, int]) -> None:
    """Write experiment map to JSON for sbatch scripts to consume."""
    with open(SWEEP_MAP_PATH, "w") as f:
        json.dump(experiment_map, f, indent=2)
    print(f"\nWrote {len(experiment_map)} entries to {SWEEP_MAP_PATH}")


def submit_jobs(sweep_filter: str | None = None, dry_run: bool = False) -> None:
    """Submit SLURM array jobs."""
    for sweep_name, sweep in SWEEPS.items():
        if sweep_filter and sweep["axis"] != sweep_filter:
            continue

        sbatch_path = SLURM_DIR / sweep["sbatch"]
        n_cards = len(sweep["cards"])
        n_tasks = n_cards * len(LOGICS) * 3  # 3 seeds
        array_spec = f"0-{n_tasks - 1}"

        cmd = [
            "sbatch", "--parsable",
            f"--array={array_spec}",
            str(sbatch_path),
        ]

        print(f"\n--- {sweep['description']} ({n_tasks} tasks) ---")
        print(f"  cmd: {' '.join(cmd)}")

        if dry_run:
            print("  [DRY RUN] skipped")
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                job_id = result.stdout.strip()
                print(f"  Submitted: job {job_id}")
            else:
                print(f"  FAILED: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Launch cross-system ablation sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--register-only", action="store_true",
                        help="Register experiments and write map, don't submit")
    parser.add_argument("--submit", action="store_true", help="Register + submit")
    parser.add_argument("--axis", choices=["model", "featuriser", "policy"],
                        help="Only run one axis")
    parser.add_argument("--db", default=DB_PATH, help="DB path override")
    args = parser.parse_args()

    db_path = args.db

    if not (args.dry_run or args.register_only or args.submit):
        parser.print_help()
        return

    print("=== Cross-System Ablation Sweep Launcher ===")
    print(f"DB: {db_path}")
    print(f"Portfolio: {PORTFOLIO_ID}")
    print(f"Logics: {LOGICS}")
    print()

    if args.dry_run:
        print("--- DRY RUN (no DB writes, no submissions) ---\n")
        for sweep_name, sweep in SWEEPS.items():
            if args.axis and sweep["axis"] != args.axis:
                continue
            n_tasks = len(sweep["cards"]) * len(LOGICS) * 3
            print(f"  {sweep_name}: {len(sweep['cards'])} cards x 2 logics x 3 seeds = {n_tasks} tasks")
            for card in sweep["cards"]:
                for logic in LOGICS:
                    print(f"    {card}|{logic}")
        print(f"\nTotal experiments: would register above, submit via SLURM")
        return

    # Register experiments
    print("--- Registering experiments ---")
    experiment_map = register_experiments(db_path, args.axis)
    write_map(experiment_map)

    if args.register_only:
        print("\n--- Register-only mode: done ---")
        return

    # Submit SLURM jobs
    print("\n--- Submitting SLURM jobs ---")
    submit_jobs(args.axis)

    print("\n=== Done ===")
    print(f"Monitor: squeue -u $(whoami) --format='%.10i %.20j %.8T %.10M %.6D %R'")
    print(f"Logs:    ls -lt {REPO_DIR / 'logs/'}")


if __name__ == "__main__":
    main()
