"""Configuration loading: configs/*.yaml -> dot-accessible objects;
supports CLI overrides.

Single responsibility: configuration management only.
(Implemented; ready to use.)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_ns(obj):
    """Recursively convert dictionaries to SimpleNamespace objects."""
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{k: _to_ns(v) for k, v in obj.items()}
        )

    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]

    return obj


def load_config(path="configs/default.yaml", overrides=None):
    """Load YAML -> SimpleNamespace.

    Supports dot notation for accessing nested configuration values,
    e.g. cfg.model.name.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of CLI-style overrides.
            Example:
                {"train.epochs": 5, "model.name": "RN50"}
    """
    with open(Path(path), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for dotted, val in (overrides or {}).items():
        d, keys = cfg, dotted.split(".")

        for k in keys[:-1]:
            d = d[k]

        d[keys[-1]] = val

    return _to_ns(cfg)


def parse_cli():
    """Parse command-line arguments.

    Supported arguments:

        --config PATH
            Specify the configuration file.

        --set a.b=value
            Override a configuration value.
            Can be specified multiple times.

    Example:
        python train.py \
            --config configs/default.yaml \
            --set train.epochs=5 \
            --set train.batch_size=256
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a configuration value, e.g. --set train.epochs=5",
    )

    args = p.parse_args()

    ov = {}

    for item in args.set:
        k, v = item.split("=", 1)
        ov[k] = yaml.safe_load(v)  # Automatically infer the value type

    return load_config(args.config, ov)