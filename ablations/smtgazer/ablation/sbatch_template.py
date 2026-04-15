#!/usr/bin/env python3
"""
Generates sbatch scripts for ablation experiments.

The generated scripts use #SBATCH flags from the reference config
(partition=tiger, cpus from reference.yaml, mem-per-cpu=3000).
"""

from pathlib import Path


ABLATION_DIR = Path(__file__).parent
GENERATED_DIR = ABLATION_DIR / "generated"


def generate_sbatch(experiment_code, config, wall_time, results_dir):
    """Generate an sbatch script for one ablation experiment.

    Args:
        experiment_code: e.g. "cluster_num_15"
        config: merged config dict
        wall_time: e.g. "12:00:00"
        results_dir: Path to experiment results directory

    Returns:
        Path to the generated sbatch script.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    hw = config["hardware"]
    log_dir = results_dir / "logs"

    script = f"""#!/bin/bash
#SBATCH --job-name=abl-{experiment_code}
#SBATCH --partition={hw["partition"]}
#SBATCH --cpus-per-task={hw["cpus_per_task"]}
#SBATCH --mem-per-cpu={hw["mem_per_cpu_mb"]}
#SBATCH --time={wall_time}
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

set -e

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
source {config["paths"]["env_file"]}

echo "=== Ablation Experiment: {experiment_code} ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo ""

export SMAC_WORKERS={hw["smac_workers"]}
export FEATURE_WORKERS={hw["feature_workers"]}

CONFIG_YAML="{results_dir / "config.yaml"}"
ABLATION_DIR="{ABLATION_DIR}"

cd "$ABLATION_DIR/.."

python3 -m ablation.pipeline_ablation "$CONFIG_YAML"
RC=$?

echo ""
echo "=== Ablation Experiment Complete ==="
echo "Exit code: $RC"
echo "End: $(date)"
exit $RC
"""

    script_path = GENERATED_DIR / f"{experiment_code}.sbatch"
    with open(script_path, "w") as f:
        f.write(script)
    script_path.chmod(0o755)

    return script_path
