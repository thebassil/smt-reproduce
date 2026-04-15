#!/bin/bash
#
# Full ablation sweep orchestrator.
# Submits SLURM jobs with dependency chaining across all 3 axes.
#
# Usage:
#   ./run_full_ablation.sh                     # dry run (no DB write)
#   ./run_full_ablation.sh --experiment-id 5   # persist to DB
#   ./run_full_ablation.sh --dry-run           # print commands without submitting
#
# Sweep structure (per axis, vary one card while holding others at REF-VECTOR default):
#   1. Model axis:      5 VECTOR models x 3 seeds x 2 logics = 30 tasks (taskfarm)
#   2. Model axis (GNN): 5 GNN models x 3 seeds x 2 logics = 30 tasks (Avon GPU)
#   3. Policy axis:     13 policies x 3 seeds x 2 logics = 78 tasks (taskfarm)
#   4. Featuriser axis: 5 featurisers x 3 seeds x 2 logics = 30 tasks (taskfarm)
#
# Total: 168 tasks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false
EXP_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-id)
            EXP_ARGS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

submit() {
    local name="$1"
    local script="$2"
    local dep="${3:-}"

    local cmd="sbatch --parsable"
    if [ -n "${dep}" ]; then
        cmd+=" --dependency=afterany:${dep}"
    fi
    cmd+=" ${script}"
    if [ -n "${EXP_ARGS}" ]; then
        cmd+=" ${EXP_ARGS}"
    fi

    if [ "${DRY_RUN}" = true ]; then
        echo "[DRY RUN] ${name}: ${cmd}"
        echo "0"  # fake job ID
    else
        echo "Submitting ${name}..."
        JOB_ID=$(eval "${cmd}")
        echo "  ${name}: job ${JOB_ID}"
        echo "${JOB_ID}"
    fi
}

echo "=== Cross-System Ablation Sweep ==="
echo "Experiment ID: ${EXP_ARGS:-none (dry run, no DB write)}"
echo "Dry run: ${DRY_RUN}"
echo ""

# --- Phase 1: Model axis (VECTOR, taskfarm) ---
echo "--- Phase 1: VECTOR model sweep (30 tasks, taskfarm) ---"
VECTOR_JOB=$(submit "sweep_vector" "${SCRIPT_DIR}/sweep_vector.sbatch")

# --- Phase 2: Model axis (GNN, Avon GPU) ---
echo "--- Phase 2: GNN model sweep (30 tasks, Avon GPU) ---"
GNN_JOB=$(submit "sweep_gnn" "${SCRIPT_DIR}/sweep_gnn.sbatch")

# --- Phase 3: Policy axis (taskfarm, depends on nothing) ---
echo "--- Phase 3: Policy sweep (78 tasks, taskfarm) ---"
POLICY_JOB=$(submit "sweep_policy" "${SCRIPT_DIR}/sweep_policy.sbatch")

# --- Phase 4: Featuriser axis (taskfarm, depends on nothing) ---
echo "--- Phase 4: Featuriser sweep (30 tasks, taskfarm) ---"
FEAT_JOB=$(submit "sweep_featuriser" "${SCRIPT_DIR}/sweep_featuriser.sbatch")

echo ""
echo "=== All jobs submitted ==="
echo "  VECTOR models: ${VECTOR_JOB}"
echo "  GNN models:    ${GNN_JOB}"
echo "  Policies:      ${POLICY_JOB}"
echo "  Featurisers:   ${FEAT_JOB}"
echo ""
echo "Monitor: squeue -u \$(whoami) --format='%.10i %.20j %.8T %.10M %.6D %R'"
echo "Logs:    ls -lt logs/"
