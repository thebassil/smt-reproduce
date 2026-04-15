from __future__ import annotations
"""Layer 3: Reproduce cross-system ablation sweep.

Uses the same card definitions as the original launch_sweep.py.
Invokes `python -m pipeline.runner` with Hydra overrides — the exact
same command that ran on SLURM, now executed locally via subprocess.
"""

import sys
from pathlib import Path
from typing import Optional

from smt.backends.base import Job
from smt.backends import get_backend
from smt.config import ABLATION_AXES, resolve_db_path, REPO_ROOT
from smt.display import console

# ---------------------------------------------------------------------------
# Card definitions (verbatim from smt-ablation-cross-system/scripts/launch_sweep.py)
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

# ---------------------------------------------------------------------------
# Sweep definitions (verbatim structure from launch_sweep.py)
# ---------------------------------------------------------------------------

SWEEPS = {
    "core_model_cpu": {
        "axis": "model", "tier": "core", "reference": "ref_vector",
        "cards": CORE_MODELS_CPU,
        "fixed": {"featuriser": "machsmt_162", "policy": "select_argmin"},
    },
    "core_model_gpu": {
        "axis": "model", "tier": "core", "reference": "ref_gnn",
        "cards": CORE_MODELS_GPU,
        "fixed": {"featuriser": "sibyl_ast_ud", "policy": "select_argmin"},
    },
    "core_featuriser_cpu": {
        "axis": "featuriser", "tier": "core", "reference": "ref_vector",
        "cards": CORE_FEATURISERS_CPU,
        "fixed": {"model": "adaboost_ehm", "policy": "select_argmin"},
    },
    "core_policy": {
        "axis": "policy", "tier": "core", "reference": "ref_vector",
        "cards": CORE_POLICIES,
        "fixed": {"featuriser": "machsmt_162", "model": "adaboost_ehm"},
    },
    "ext_model_cpu": {
        "axis": "model", "tier": "extended", "reference": "ref_vector",
        "cards": EXTENDED_MODELS_CPU,
        "fixed": {"featuriser": "machsmt_162", "policy": "select_argmin"},
    },
    "ext_model_gpu": {
        "axis": "model", "tier": "extended", "reference": "ref_gnn",
        "cards": EXTENDED_MODELS_GPU,
        "fixed": {"featuriser": "sibyl_ast_ud", "policy": "select_argmin"},
    },
    "ext_featuriser_cpu": {
        "axis": "featuriser", "tier": "extended", "reference": "ref_vector",
        "cards": EXTENDED_FEATURISERS_CPU,
        "fixed": {"model": "adaboost_ehm", "policy": "select_argmin"},
    },
    "ext_featuriser_gpu": {
        "axis": "featuriser", "tier": "extended", "reference": "ref_gnn",
        "cards": EXTENDED_FEATURISERS_GPU,
        "fixed": {"model": "gat_mlp", "policy": "select_argmin"},
    },
    "ext_policy": {
        "axis": "policy", "tier": "extended", "reference": "ref_vector",
        "cards": EXTENDED_POLICIES,
        "fixed": {"featuriser": "machsmt_162", "model": "adaboost_ehm"},
    },
}

LOGICS = ["QF_LIA", "QF_BV", "QF_NRA"]
SEEDS = [42, 123, 456]


def run_cross_system(
    axis: Optional[str] = None,
    backend: str = "local",
    db_override: Optional[str] = None,
) -> None:
    db = str(resolve_db_path(db_override))
    pipeline_dir = REPO_ROOT / "pipeline"

    if not pipeline_dir.exists():
        console.print("[red]pipeline/ directory not found.[/red]")
        console.print("Copy cross-system pipeline files first. See README.")
        return

    jobs = []
    for sweep_name, sweep in SWEEPS.items():
        if axis and sweep["axis"] != axis:
            continue

        ref = sweep["reference"]
        fixed = sweep["fixed"]

        for card in sweep["cards"]:
            for logic in LOGICS:
                for seed in SEEDS:
                    # Build Hydra overrides — same as the sbatch scripts
                    overrides = [
                        f"reference={ref}",
                        f"data.db_path={db}",
                        f"data.logic={logic}",
                        f"seed={seed}",
                    ]
                    # Set the swept axis card
                    overrides.append(
                        f"{sweep['axis']}@reference.{sweep['axis']}={card}"
                    )
                    # Set fixed cards
                    for fixed_axis, fixed_card in fixed.items():
                        overrides.append(
                            f"{fixed_axis}@reference.{fixed_axis}={fixed_card}"
                        )

                    cmd = [sys.executable, "-m", "pipeline.runner"] + overrides

                    jobs.append(
                        Job(
                            name=f"{sweep_name}|{card}|{logic}|s{seed}",
                            cmd=cmd,
                            cwd=str(REPO_ROOT),
                            env={"SMT_DB_PATH": db},
                        )
                    )

    if not jobs:
        console.print("[red]No cross-system jobs to run.[/red]")
        if axis:
            console.print(f"[dim]No sweeps found for axis '{axis}'.[/dim]")
        return

    n_gpu = sum(1 for j in jobs if "gpu" in j.name)
    n_cpu = len(jobs) - n_gpu

    console.print(f"[bold]Cross-system ablation sweep[/bold]")
    console.print(f"  Total experiments: {len(jobs)}")
    console.print(f"  CPU experiments:   {n_cpu}")
    console.print(f"  GPU experiments:   {n_gpu}")
    console.print(f"  Logics: {LOGICS}")
    console.print(f"  Seeds:  {SEEDS}")
    console.print()

    if n_gpu > 0 and backend == "local":
        console.print(
            "[yellow]Warning:[/yellow] GPU experiments require PyTorch + CUDA. "
            "They will fail if not available."
        )

    executor = get_backend(backend)
    results = executor.submit(jobs)

    ok = sum(1 for r in results if r.ok)
    fail = len(results) - ok
    console.print(f"\n[bold]Completed: {ok} succeeded, {fail} failed.[/bold]")
    if fail > 0:
        console.print("[dim]Failed experiments:[/dim]")
        for r in results:
            if not r.ok:
                console.print(f"  [red]FAIL[/red] {r.job.name}")
                if r.stderr:
                    # Show last line of stderr
                    last = r.stderr.strip().split("\n")[-1]
                    console.print(f"    {last[:120]}")
