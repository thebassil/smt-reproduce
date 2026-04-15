# SMT Algorithm Selection — Reproducibility Package

**Repository:** https://github.com/thebassil/smt-reproduce

Reproducibility CLI for the dissertation: *Algorithm Selection for SMT Solvers*.

This tool reproduces all experimental results from the dissertation:
- **5 selector system evaluations** (MachSMT, Sibyl, Grackle, Medley, SMTGazer)
- **48 subsystem ablation experiments** (per-system knob sweeps)
- **1,134 cross-system ablation experiments** (featuriser x model x policy combinations)
- **218,804 ground-truth solver runs** (Z3 + CVC5 on 9,500 SMT instances)

## Quick Start

```bash
# 1. Clone (or unzip the submitted archive)
git clone https://github.com/thebassil/smt-reproduce.git
cd smt-reproduce

# 2. Install (requires Python 3.9+, only needs click and rich)
pip install -e .

# 3. Download the pre-computed database (3.5 MB download, 28 MB uncompressed)
#    This fetches results.sqlite from the GitHub release automatically.
smt download

# 4. Inspect all results — no computation needed
smt reproduce systems --inspect
smt reproduce subsystem --inspect
smt reproduce cross-system --inspect
smt reproduce ground-truth --inspect
```

### If you received this as a ZIP file

```bash
unzip smt-reproduce.zip
cd smt-reproduce
pip install -e .
smt download          # Downloads the 28 MB database from GitHub
smt status            # Verify everything is available
```

**Note:** The database (`results.sqlite`, 28 MB) is hosted as a GitHub Release asset
at https://github.com/thebassil/smt-reproduce/releases/tag/v0.1.0 because it is
too large for the ZIP submission. The `smt download` command fetches it automatically.
If `gh` (GitHub CLI) is not available, it falls back to a direct HTTPS download — no
authentication required.

## Commands Reference

### Inspect pre-computed results (no computation, no special dependencies)

```bash
smt status                                    # Show what data is available locally

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

### Re-run experiments (requires `pip install -e ".[train]"`)

```bash
# Re-train all 5 selector systems (~hours on a standard machine)
smt reproduce systems
smt reproduce systems --system machsmt        # Single system only

# Re-run all 48 subsystem ablation experiments (~hours)
smt reproduce subsystem
smt reproduce subsystem --system machsmt

# Re-run cross-system ablation sweep (~days for all 1,134 experiments)
smt reproduce cross-system
smt reproduce cross-system --axis model       # Only model axis (faster)
```

### Execution backends

```bash
smt --backend local reproduce cross-system    # Default: sequential on local machine
smt --backend slurm reproduce cross-system    # Generate and submit SLURM array jobs
smt --backend docker reproduce cross-system   # Run inside Docker container
```

## Project Structure

```
smt-reproduce/
├── README.md          # This file
├── pyproject.toml     # Package definition, entry point: smt = "smt.cli:smt"
├── Dockerfile         # Container with Z3 + CVC5 solvers pre-installed
├── Makefile           # Convenience targets (make install, make inspect-all)
│
├── src/smt/           # CLI tool source code (Click-based)
│   ├── cli.py         # Command definitions
│   ├── config.py      # Path resolution (auto-detects Warwick DCS or standalone)
│   ├── display.py     # Rich table formatting
│   ├── download.py    # GitHub Release asset download
│   ├── backends/      # Execution backends: local, SLURM, Docker
│   ├── layers/        # Run-mode orchestration per experimental layer
│   └── inspect/       # Read-only result display per layer
│
├── systems/           # Selector system code (verbatim from evaluation worktrees)
│   ├── machsmt/       # MachSMT: neural network runtime predictor
│   ├── sibyl/         # Sibyl: GNN-based solver selector
│   ├── grackle/       # Grackle: greedy set-cover portfolio builder
│   ├── medley/        # Medley: k-NN with exponential timer
│   └── smtgazer/      # SMTGazer: SMAC-optimised portfolio
│
├── ablations/         # Per-system ablation code (verbatim from ablation worktrees)
│   ├── machsmt/       # 9 ablation templates (ml_core, features, seeds, etc.)
│   ├── sibyl/         # 8 ablation templates (dropout, epochs, edge_sets, etc.)
│   ├── grackle/       # 11 ablation templates (best-k, timeout, rank, etc.)
│   ├── medley/        # 13 ablation templates (classifier, hyperparameters)
│   └── smtgazer/      # 7 ablation templates (cluster_num, smac_trials, etc.)
│
├── pipeline/          # Cross-system ablation pipeline (Hydra-based)
│   ├── runner.py      # Main entry point: python -m pipeline.runner
│   ├── compose.py     # SystemPipeline(featuriser, model, policy)
│   ├── db.py          # SQLite database I/O
│   ├── evaluate.py    # K-fold cross-validation, PAR-2 metrics
│   ├── types.py       # Pydantic data models and protocols
│   ├── featurisers/   # 37 feature extraction implementations
│   ├── models/        # 36 ML model implementations
│   └── policies/      # 16 solver selection policy implementations
│
├── conf/              # Hydra YAML configs for all pipeline cards
│   ├── config.yaml    # Top-level config (portfolio, logic, timeout, etc.)
│   ├── featuriser/    # 60+ featuriser configs
│   ├── model/         # 34 model configs
│   ├── policy/        # 29 policy configs
│   └── reference/     # ref_vector (CPU) and ref_gnn (GPU) reference systems
│
├── final_data/        # Pre-computed result CSVs (committed to repo)
│   ├── 01_system_reproductions/   # K-fold results for each of the 5 systems
│   ├── 02_subsystem_ablations/    # Ablation sweep results per system
│   ├── 03_cross_system_ablations/ # Core cross-system ablation CSV (826 rows)
│   └── 04_cross_system_ablations_extended/  # Extended featuriser experiments
│
├── data/              # Downloaded at runtime (.gitignored)
│   └── results.sqlite # Central database (fetched by `smt download`)
│
├── scripts/           # Original SLURM submission scripts
└── shared_folds_k5.json  # Shared k-fold split definitions for reproducibility
```

## Data

| Dataset | Size | Location | How to get |
|---------|------|----------|------------|
| Pre-computed CSVs | ~5 MB | `final_data/` | Included in repo |
| Central database | 28 MB | `data/results.sqlite` | `smt download` (from GitHub Release) |
| Benchmark instances | ~2 GB | Not included | Optional, for ground-truth reproduction only |

The database contains 218,804 solver execution results (Z3 + CVC5 on 9,500 SMT-LIB instances
across 118 solver configurations), plus all ablation experiment metadata and ML selector decisions.

## Key Results

| System | Total PAR-2 | Solved % | Notes |
|--------|-------------|----------|-------|
| Grackle | 141,290 | 89.9% | Greedy set-cover (best overall) |
| Medley | 167,384 | 86.2% | k-NN + exponential timer |
| SBS (best single solver) | 175,853 | 86.4% | Baseline: z3_case_split |
| MachSMT | 252,727 | 78.6% | Neural runtime predictor |
| Sibyl | 254,937 | 79.2% | GNN-based selector |
| VBS (oracle) | 119,659 | 90.6% | Upper bound: always picks best |

## Requirements

| Mode | Install command | What you need |
|------|----------------|---------------|
| Inspect only | `pip install -e .` | Python 3.9+, click, rich |
| Re-run experiments | `pip install -e ".[train]"` | + numpy, scikit-learn, xgboost, hydra |
| GPU experiments | `pip install -e ".[gpu]"` | + PyTorch, torch_geometric |
| Docker | `docker build -t smt-reproduce .` | Docker |

## External Libraries

The following external libraries are used (all installable via pip):
- **click** (CLI framework), **rich** (terminal tables)
- **hydra-core** + **omegaconf** (configuration management)
- **numpy**, **scipy**, **scikit-learn** (ML fundamentals)
- **xgboost**, **lightgbm** (gradient boosting models)
- **pydantic** (data validation)
- **torch**, **torch_geometric** (optional, for GNN models only)

All dependencies are declared in `pyproject.toml` and installed automatically by pip.
