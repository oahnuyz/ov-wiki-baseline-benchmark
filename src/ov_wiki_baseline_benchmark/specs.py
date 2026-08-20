"""Load and validate the thirteen fixed experiment specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    dataset: str
    raw_dataset: str
    expected_qas: int
    expected_documents: int
    options: dict[str, Any]


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_configs_dir() -> Path:
    return repository_root() / "configs"


def load_specs(configs_dir: Path | None = None) -> dict[str, ExperimentSpec]:
    directory = configs_dir or default_configs_dir()
    specs: dict[str, ExperimentSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Experiment config must be a mapping: {path}")
        experiment_id = str(value.get("id") or "").strip()
        dataset = str(value.get("dataset") or "").strip()
        raw_dataset = str(value.get("raw_dataset") or "").strip()
        expected = value.get("expected")
        options = value.get("options") or {}
        if (
            not experiment_id
            or path.stem != experiment_id
            or experiment_id in specs
            or not dataset
            or not raw_dataset
            or not isinstance(expected, dict)
            or not isinstance(expected.get("qas"), int)
            or not isinstance(expected.get("documents"), int)
            or not isinstance(options, dict)
        ):
            raise ValueError(f"Invalid experiment config: {path}")
        specs[experiment_id] = ExperimentSpec(
            id=experiment_id,
            dataset=dataset,
            raw_dataset=raw_dataset,
            expected_qas=expected["qas"],
            expected_documents=expected["documents"],
            options=options,
        )
    if len(specs) != 13:
        raise ValueError(f"Expected exactly 13 experiment configs, got {len(specs)}")
    return specs
