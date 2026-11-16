"""ELT-Bench tasks, what the agent sees plus the hidden answers.

the agent only gets `config` and `models`, row counts and answers stay with the grader
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from elt_rl import paths


@dataclass(frozen=True)
class ModelSpec:
    """one target table from data_model.yaml"""

    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class EltTask:
    name: str
    task_dir: Path
    config: dict[str, Any]  # config.yaml, where the sources are
    models: tuple[ModelSpec, ...]  # data_model.yaml, tables to build
    # hidden from the agent
    expected_row_counts: dict[str, int]  # raw table -> rows, stage 1
    sort_keys: dict[str, list[str]]  # model -> key columns, stage 2
    ground_truth_dir: Path

    # ELT-Bench mixes case (retail_complains has model clients but file Clients) so ignore it

    def ground_truth(self, model: str) -> pd.DataFrame:
        """answer rows for one model, all strings like the official grader reads them"""
        files = {p.stem.lower(): p for p in self.ground_truth_dir.glob("*.csv")}
        return _read_ground_truth(files[model.lower()])

    def sort_key(self, model: str) -> list[str]:
        keys = {name.lower(): cols for name, cols in self.sort_keys.items()}
        return keys.get(model.lower(), [])


def _parse_models(text: str) -> tuple[ModelSpec, ...]:
    try:
        data_model = yaml.safe_load(text)
        return tuple(
            ModelSpec(name=m["name"], columns=tuple(c["name"] for c in m.get("columns", [])))
            for m in data_model["models"]
        )
    except yaml.YAMLError:
        pass
    # world_development_indicators has an unquoted ':' so it's not valid yaml
    # only need names so go by indent, shallowest "- name:" lines are models, deeper are columns
    entries = [
        (len(line) - len(line.lstrip()), line.split("name:", 1)[1].strip())
        for line in text.splitlines()
        if line.lstrip().startswith("- name:")
    ]
    model_indent = min(indent for indent, _ in entries)
    models: list[tuple[str, list[str]]] = []
    for indent, name in entries:
        if indent == model_indent:
            models.append((name, []))
        else:
            models[-1][1].append(name)
    return tuple(ModelSpec(name=name, columns=tuple(cols)) for name, cols in models)


@functools.cache
def _read_ground_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True)


@functools.cache
def _eval_json(name: str) -> dict[str, Any]:
    return json.loads((paths.EVAL_DIR / name).read_text())


def list_tasks(destination: str = "snowflake") -> list[str]:
    return sorted(p.name for p in (paths.ELT_BENCH_DIR / "elt-bench" / destination).iterdir())


def load_task(
    name: str, destination: str = "snowflake", ground_truth: str = "gt_snowflake"
) -> EltTask:
    """load one task. `ground_truth` picks which answers, ELT-Bench has a set per warehouse"""
    task_dir = paths.task_dir(name, destination)
    return EltTask(
        name=name,
        task_dir=task_dir,
        config=yaml.safe_load((task_dir / "config.yaml").read_text()),
        models=_parse_models((task_dir / "data_model.yaml").read_text()),
        expected_row_counts=dict(_eval_json("table.json")[name]),
        sort_keys=dict(_eval_json("sort_key.json").get(name, {})),
        ground_truth_dir=paths.GROUND_TRUTH_DIR / ground_truth / name,
    )
