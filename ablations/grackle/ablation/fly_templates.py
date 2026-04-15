#!/usr/bin/env python3
"""
.fly file generator for Grackle ablation experiments.

Reads reference values from reproducibility.json, accepts a single knob override,
and generates 6 .fly files (one per solver x logic combination).
"""

import json
from pathlib import Path
from typing import Any, Dict

SOLVERS = ["cvc5", "z3"]
LOGICS = ["QF_BV", "QF_LIA", "QF_NRA"]

# Maps array index to (solver, logic) — same order as production sbatch
ARRAY_MAP = [
    ("cvc5", "QF_BV"),   # 0
    ("cvc5", "QF_LIA"),  # 1
    ("cvc5", "QF_NRA"),  # 2
    ("z3",   "QF_BV"),   # 3
    ("z3",   "QF_LIA"),  # 4
    ("z3",   "QF_NRA"),  # 5
]

# Trainer class templates keyed by alias
TRAINER_CLASSES = {
    "ParamILS":  "grackle.trainer.{solver}.paramils.{Solver}ParamilsTrainer",
    "Smac3":     "grackle.trainer.{solver}.smac3.{Solver}Smac3Trainer",
    "Smac3HPO":  "grackle.trainer.{solver}.smac3.{Solver}Smac3TrainerHPO",
    "ROAR":      "grackle.trainer.{solver}.smac3.{Solver}Smac3TrainerROAR",
}

# Reference values for all ablatable knobs
REFERENCE_FLY = {
    "cores": 4,
    "tops": 8,
    "best": 1,
    "rank": 1,
    "inits": "../{solver}_inits.txt",
    "timeout": 43200,
    "atavistic": False,
    # selection is omitted in reference .fly (uses Grackle default)
    "runner.prefix": "{solver}-",
    "trains.data": "../benchmarks_{logic}.txt",
    "trains.runner": "grackle.runner.{solver}.{Solver}Runner",
    "trains.runner.timeout": 30,
    "trains.runner.penalty": 100000000,
    "trainer": "ParamILS",  # alias — expanded during generation
    "trainer.runner": "grackle.runner.{solver}.{Solver}Runner",
    "trainer.restarts": True,
    "trainer.runner.timeout": 30,
    "trainer.runner.cores": 1,
    "trainer.runner.penalty": 100000000,
    "trainer.timeout": 3600,
}

# Map from knob name (in YAML) to .fly key(s) that get modified
KNOB_TO_FLY_KEYS = {
    "tops":              ["tops"],
    "best":              ["best"],
    "rank":              ["rank"],
    "selection":         ["selection"],
    "atavistic":         ["atavistic"],
    "grackle_timeout":   ["timeout"],
    "trainer":           ["trainer"],
    "trainer_timeout":   ["trainer.timeout"],
    "trainer_restarts":  ["trainer.restarts"],
    "runner_timeout":    ["trains.runner.timeout", "trainer.runner.timeout"],
    "runner_penalty":    ["trains.runner.penalty", "trainer.runner.penalty"],
}


def _format_value(key: str, value: Any) -> str:
    """Format a Python value for .fly file output."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        # Use integer notation if it's a whole number
        if value == int(value):
            return str(int(value))
        return str(value)
    return str(value)


def _expand_solver_logic(template: str, solver: str, logic: str) -> str:
    """Expand {solver}, {Solver}, {logic} placeholders."""
    return template.format(
        solver=solver,
        Solver=solver.capitalize(),
        logic=logic,
    )


def generate_fly_content(
    solver: str,
    logic: str,
    knob: str = None,
    value: Any = None,
) -> str:
    """Generate the content of a single .fly file.

    Args:
        solver: "cvc5" or "z3"
        logic: "QF_BV", "QF_LIA", or "QF_NRA"
        knob: ablation knob name (None = reference config)
        value: override value for the knob
    """
    # Start from reference values
    fly = dict(REFERENCE_FLY)

    # Apply the knob override
    if knob is not None and value is not None:
        fly_keys = KNOB_TO_FLY_KEYS.get(knob, [knob])
        for fk in fly_keys:
            fly[fk] = value

    # Resolve trainer alias to full class name
    trainer_alias = fly["trainer"]
    if trainer_alias in TRAINER_CLASSES:
        trainer_class = _expand_solver_logic(
            TRAINER_CLASSES[trainer_alias], solver, logic
        )
    else:
        trainer_class = trainer_alias
    fly["trainer"] = trainer_class

    # Expand solver/logic placeholders in template values
    for key in ["inits", "runner.prefix", "trains.data", "trains.runner",
                "trainer.runner"]:
        if key in fly:
            fly[key] = _expand_solver_logic(str(fly[key]), solver, logic)

    # Build .fly file content in the same order as the reference
    lines = []

    # Core parameters
    lines.append(f"cores = {_format_value('cores', fly['cores'])}")
    lines.append(f"tops = {_format_value('tops', fly['tops'])}")
    lines.append(f"best = {_format_value('best', fly['best'])}")
    lines.append(f"rank = {_format_value('rank', fly['rank'])}")
    lines.append(f"inits = {fly['inits']}")
    lines.append(f"timeout = {_format_value('timeout', fly['timeout'])}")
    lines.append(f"atavistic = {_format_value('atavistic', fly['atavistic'])}")

    # Selection — only emit if explicitly set (reference omits it)
    if "selection" in fly and fly["selection"] != "default":
        lines.append(f"selection = {fly['selection']}")
    elif knob == "selection":
        # For selection ablation, always emit the key even for "default"
        lines.append(f"selection = {fly['selection']}")

    lines.append("")

    # Runner
    lines.append(f"runner.prefix = {fly['runner.prefix']}")
    lines.append("")

    # Trains
    lines.append(f"trains.data = {fly['trains.data']}")
    lines.append(f"trains.runner = {fly['trains.runner']}")
    lines.append(f"trains.runner.timeout = {_format_value('trains.runner.timeout', fly['trains.runner.timeout'])}")
    lines.append(f"trains.runner.penalty = {_format_value('trains.runner.penalty', fly['trains.runner.penalty'])}")
    lines.append("")

    # Trainer
    lines.append(f"trainer = {fly['trainer']}")
    lines.append(f"trainer.runner = {fly['trainer.runner']}")
    lines.append(f"trainer.restarts = {_format_value('trainer.restarts', fly['trainer.restarts'])}")
    lines.append(f"trainer.runner.timeout = {_format_value('trainer.runner.timeout', fly['trainer.runner.timeout'])}")
    lines.append(f"trainer.runner.cores = {_format_value('trainer.runner.cores', fly['trainer.runner.cores'])}")
    lines.append(f"trainer.runner.penalty = {_format_value('trainer.runner.penalty', fly['trainer.runner.penalty'])}")
    lines.append(f"trainer.timeout = {_format_value('trainer.timeout', fly['trainer.timeout'])}")

    return "\n".join(lines) + "\n"


def generate_fly_files(
    run_dir: Path,
    knob: str = None,
    value: Any = None,
) -> Dict[str, Path]:
    """Generate all 6 .fly files for a run directory.

    Creates solver_logic/ subdirectories with .fly files inside.

    Returns:
        dict mapping "solver_logic" to the path of the generated .fly file
    """
    fly_files = {}
    for solver, logic in ARRAY_MAP:
        logic_lower = logic.lower().replace("_", "")
        dir_name = f"{solver}_{logic_lower}"
        subdir = run_dir / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        fly_name = f"{dir_name}.fly"
        fly_path = subdir / fly_name
        content = generate_fly_content(solver, logic, knob, value)
        fly_path.write_text(content)
        fly_files[dir_name] = fly_path

    return fly_files


def get_solver_logic_dirs() -> list:
    """Return the 6 solver_logic directory names in array order."""
    dirs = []
    for solver, logic in ARRAY_MAP:
        logic_lower = logic.lower().replace("_", "")
        dirs.append(f"{solver}_{logic_lower}")
    return dirs


def get_fly_filenames() -> list:
    """Return the 6 .fly filenames in array order."""
    return [f"{d}.fly" for d in get_solver_logic_dirs()]
