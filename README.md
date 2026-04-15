# SMT Algorithm Selection — Reproducibility Package

Reproducibility CLI for the dissertation: *Algorithm Selection for SMT Solvers*.

This tool reproduces all experimental results from the dissertation:
- **5 selector system evaluations** (MachSMT, Sibyl, Grackle, Medley, SMTGazer)
- **48 subsystem ablation experiments** (per-system knob sweeps)
- **1,134 cross-system ablation experiments** (featuriser × model × policy combinations)
- **218,804 ground-truth solver runs** (Z3 + CVC5 on 9,500 SMT instances)

## Quick Start

```bash
# Clone and install
git clone https://github.com/thebassil/smt-reproduce.git
cd smt-reproduce
pip install -e .

# Download the pre-computed database (~28 MB)
smt download

# Inspect all results (no computation needed)
smt reproduce systems --inspect
smt reproduce subsystem --inspect
smt reproduce cross-system --inspect
smt reproduce ground-truth --inspect
```

## Commands

### Inspect pre-computed results (no computation)

```bash
smt status                                    # What data is available locally

smt reproduce systems --inspect               # Compare all 5 selector systems
smt reproduce systems --inspect --system machsmt  # Single system detail

smt reproduce subsystem --inspect             # All subsystem ablation results
smt reproduce subsystem --inspect --system grackle

smt reproduce cross-system --inspect          # Cross-system overview (826 core experiments)
smt reproduce cross-system --inspect --axis model      # Rank models by PAR-2
smt reproduce cross-system --inspect --axis featuriser  # Rank featurisers
smt reproduce cross-system --inspect --axis policy      # Rank policies

smt reproduce ground-truth --inspect          # Database stats: 218k runs, per-logic breakdown
```

### Re-run experiments

```bash
# Re-train all 5 selector systems (~hours)
smt reproduce systems
smt reproduce systems --system machsmt        # Single system

# Re-run all 48 subsystem ablation experiments (~hours)
smt reproduce subsystem
smt reproduce subsystem --system machsmt      # Single system

# Re-run cross-system ablation sweep (~days for all 1,134 experiments)
smt reproduce cross-system
smt reproduce cross-system --axis model       # Only model axis
```

### Execution backends

```bash
smt --backend local reproduce cross-system    # Default: local multiprocessing
smt --backend slurm reproduce cross-system    # Submit to SLURM cluster
smt --backend docker reproduce cross-system   # Run inside Docker container
```

### Ground-truth reproduction

```bash
# Re-execute Z3 and CVC5 on all benchmarks (~150 CPU-hours)
smt download --benchmarks
smt reproduce ground-truth
```

This requires solver binaries (included in Docker image, or set `SMT_Z3` and `SMT_CVC5` env vars).

## Project Structure

```
smt-reproduce/
├── src/smt/           # CLI tool (Click-based)
│   ├── cli.py         # Command definitions
│   ├── config.py      # Path resolution (auto-detects DCS environment)
│   ├── backends/      # local / slurm / docker execution
│   ├── layers/        # Orchestration per experimental layer
│   └── inspect/       # Read-only result display
│
├── systems/           # Verbatim eval-* worktree code (5 selector systems)
├── ablations/         # Verbatim smt-ablation-* worktree code (5 systems)
├── pipeline/          # Cross-system ablation pipeline (Hydra-based)
├── conf/              # Hydra YAML configs (60+ featurisers, 34 models, 29 policies)
├── final_data/        # Pre-computed result CSVs
├── data/              # Database + benchmarks (downloaded, .gitignored)
└── Dockerfile         # Container with solvers pre-installed
```

## Data

| Dataset | Size | Contents |
|---------|------|----------|
| `results.sqlite` | ~28 MB | 218,804 solver runs, 9,500 instances, 118 configs, 8 portfolios |
| `final_data/` | ~5 MB | Pre-computed CSV results for all experiments |
| Benchmarks | ~2 GB | 9,500 SMT-LIB instances (optional, for ground-truth only) |

## Key Metrics

| System | PAR-2 | Solved % |
|--------|-------|----------|
| Grackle | 141,290 | 89.9% |
| Medley | 167,384 | 86.2% |
| MachSMT | 252,727 | 78.6% |
| Sibyl | 254,937 | 79.2% |
| VBS (oracle) | 119,659 | 90.6% |
| SBS (best single) | 175,853 | 86.4% |

## Requirements

- Python 3.9+
- `pip install -e .` for inspect-only mode (click + rich)
- `pip install -e ".[train]"` for re-running experiments (adds numpy, sklearn, xgboost, etc.)
- `pip install -e ".[gpu]"` for GNN-based experiments (adds PyTorch, torch_geometric)
- Docker for containerised reproduction (optional)

## Environment

On Warwick DCS, the tool auto-detects the shared database at `/dcs/large/u5573765/db/results.sqlite`. On other machines, use `smt download` or set `SMT_DB_PATH`.
