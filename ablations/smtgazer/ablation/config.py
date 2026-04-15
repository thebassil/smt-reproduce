#!/usr/bin/env python3
"""
YAML config loader and validation for ablation experiments.

Provides:
  load_reference(path)             → dict
  load_template(path)              → dict
  merge_config(reference, knob, value) → dict with one knob overridden
  validate_one_at_a_time(config, reference) → raises if >1 knob differs
  experiment_code(knob, value)     → canonical experiment directory name
"""

from pathlib import Path

import yaml


ALGORITHMIC_KNOBS = [
    "cluster_num",
    "portfolio_size",
    "seed",
    "smac_n_trials",
    "smac_w1",
    "kmeans_n_init",
    "smac_internal_cv_splits",
]


def load_reference(path=None):
    """Load the reference YAML config. Returns a flat dict of all parameters."""
    if path is None:
        path = Path(__file__).parent / "reference.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Flatten into a single dict with section prefixes for non-algorithmic
    config = {}
    config.update(raw["algorithmic"])
    config["hardware"] = raw["hardware"]
    config["convention"] = raw["convention"]
    config["paths"] = raw["paths"]
    return config


def load_template(path):
    """Load an experiment template YAML. Returns dict with keys:
    experiment_base, ablation_knob, reference_value, values, wall_time_override.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    required = ["experiment_base", "ablation_knob", "reference_value", "values"]
    for key in required:
        if key not in raw:
            raise ValueError(f"Template {path} missing required key: {key}")

    if raw["ablation_knob"] not in ALGORITHMIC_KNOBS:
        raise ValueError(
            f"Unknown ablation knob: {raw['ablation_knob']}. "
            f"Must be one of: {ALGORITHMIC_KNOBS}"
        )

    if raw["reference_value"] not in raw["values"]:
        raise ValueError(
            f"Reference value {raw['reference_value']} not in values list: {raw['values']}"
        )

    raw.setdefault("wall_time_override", {})
    return raw


def merge_config(reference, knob, value):
    """Create a new config dict with exactly one algorithmic knob overridden."""
    if knob not in ALGORITHMIC_KNOBS:
        raise ValueError(f"Cannot ablate non-algorithmic knob: {knob}")

    config = dict(reference)
    config[knob] = value
    return config


def validate_one_at_a_time(config, reference):
    """Verify that exactly one algorithmic knob differs from reference.
    Raises ValueError if the invariant is violated.
    Returns the (knob, value) that differs, or None if config == reference.
    """
    diffs = []
    for knob in ALGORITHMIC_KNOBS:
        ref_val = reference[knob]
        cfg_val = config[knob]
        # Compare with type coercion for float/int mismatches
        if type(ref_val) != type(cfg_val):
            if isinstance(ref_val, (int, float)) and isinstance(cfg_val, (int, float)):
                if float(ref_val) != float(cfg_val):
                    diffs.append((knob, cfg_val, ref_val))
            else:
                diffs.append((knob, cfg_val, ref_val))
        elif ref_val != cfg_val:
            diffs.append((knob, cfg_val, ref_val))

    if len(diffs) == 0:
        return None  # This is the reference control point
    if len(diffs) == 1:
        return (diffs[0][0], diffs[0][1])
    raise ValueError(
        f"One-at-a-time invariant violated: {len(diffs)} knobs differ from reference: "
        + ", ".join(f"{k}={v} (ref={r})" for k, v, r in diffs)
    )


def experiment_code(knob, value):
    """Generate a canonical experiment directory name."""
    # Normalize floats: 0.5 → "0.5", integers stay as-is
    if isinstance(value, float):
        val_str = f"{value:g}"
    else:
        val_str = str(value)
    return f"{knob}_{val_str}"


def get_wall_time(template, value, reference):
    """Get wall time for a specific experiment value.
    Checks template overrides first, falls back to reference hardware wall_time.
    """
    overrides = template.get("wall_time_override", {})
    # YAML may parse numeric keys as int/float
    for override_val, override_time in overrides.items():
        if type(override_val) != type(value):
            if isinstance(override_val, (int, float)) and isinstance(value, (int, float)):
                if float(override_val) == float(value):
                    return override_time
        elif override_val == value:
            return override_time
    return reference["hardware"]["wall_time"]
