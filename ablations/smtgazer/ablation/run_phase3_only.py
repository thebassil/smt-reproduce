#!/usr/bin/env python3
"""
Run ONLY Phase 3 (K-Fold CV) for a completed SMTGazer experiment.
Skips Phase 1 (setup) and Phase 2 (training) since workdirs already exist.

Usage:
  python3 -m ablation.run_phase3_only /path/to/experiment/config.yaml
"""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.pipeline_ablation import (
    load_experiment_config,
    phase_kfold,
    setup_workdir,
    patch_artifacts,
)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        return 1

    config_path = sys.argv[1]
    cfg = load_experiment_config(config_path)

    print("=" * 70)
    print(f"PHASE 3 ONLY: {cfg.experiment_code}")
    print(f"  k_folds: {cfg.k_folds}")
    print(f"  Workdir: {cfg.workdir}")
    print(f"  Results: {cfg.results_dir}")
    print("=" * 70)

    if cfg.k_folds <= 0:
        print(f"ERROR: k_folds={cfg.k_folds} — must be > 0")
        return 1

    # Re-run setup to ensure workdir + patched scripts are intact
    print("\n=== Re-ensuring workdir setup ===")
    setup_workdir(cfg)
    patch_artifacts(cfg)

    # Run Phase 3
    print(f"\n=== Phase 3: K-Fold CV (k={cfg.k_folds}) ===")
    if not phase_kfold(cfg):
        print("\nK-FOLD FAILED")
        return 1

    # Update completion marker
    marker = cfg.results_dir / "pipeline_complete"
    marker.write_text("completed_phase3\n")
    print(f"  Updated completion marker: {marker}")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
