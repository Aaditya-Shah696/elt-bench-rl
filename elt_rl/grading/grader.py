"""grade a namespace the way ELT-Bench does.

stage 1, every raw table is there with the right row count
stage 2, every target table matches the answer column by column after sorting both by the key

table names ignore case since ELT-Bench mixes them (eg "Air_Carriers")
"""

import io
from dataclasses import dataclass

import pandas as pd

from elt_rl.destinations.base import Connector
from elt_rl.grading.comparator import check_correctness, sort_by_keys
from elt_rl.tasks import EltTask


@dataclass(frozen=True)
class RawTableCheck:
    table: str
    expected_rows: int
    actual_rows: int | None  # None means the table wasn't there

    @property
    def ok(self) -> bool:
        return self.actual_rows == self.expected_rows


@dataclass(frozen=True)
class ModelCheck:
    model: str
    key_columns: tuple[str, ...]
    found: bool
    matched: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()
    missed: tuple[str, ...] = ()
    error: str | None = None

    @property
    def exact(self) -> bool:
        return self.found and self.error is None and not self.unmatched and not self.missed

    @property
    def column_fraction(self) -> float:
        """share of non-key columns that match, 0 if the key itself is wrong.

        copying ids from a source is free so they don't earn credit, but they have to be right
        or the rows don't line up
        """
        if not self.found or self.error is not None:
            return 0.0
        keys = {k.lower() for k in self.key_columns}
        if any(c.lower() in keys for c in self.unmatched + self.missed):
            return 0.0
        scored = [c for c in self.matched + self.unmatched + self.missed if c.lower() not in keys]
        if not scored:
            return 1.0 if self.exact else 0.0
        return sum(c.lower() not in keys for c in self.matched) / len(scored)


@dataclass(frozen=True)
class Grade:
    task: str
    raw_tables: tuple[RawTableCheck, ...]
    models: tuple[ModelCheck, ...]

    @property
    def load_fraction(self) -> float:
        return sum(t.ok for t in self.raw_tables) / len(self.raw_tables) if self.raw_tables else 0.0

    @property
    def success(self) -> bool:
        """official ELT-Bench pass, everything loaded and every table exact"""
        return all(t.ok for t in self.raw_tables) and all(m.exact for m in self.models)


def _csv_round_trip(df: pd.DataFrame) -> pd.DataFrame:
    # the official grader round trips results through csv as strings, so we do too
    # otherwise stuff like 5 vs 5.0 wouldn't line up the same way
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, keep_default_na=True)


def check_raw_tables(task: EltTask, connector: Connector) -> tuple[RawTableCheck, ...]:
    existing = {t.lower(): t for t in connector.list_tables()}
    checks = []
    for table, expected in task.expected_row_counts.items():
        actual = existing.get(table.lower())
        rows = connector.count_rows(actual) if actual is not None else None
        checks.append(RawTableCheck(table=table, expected_rows=expected, actual_rows=rows))
    return tuple(checks)


def check_model(
    task: EltTask, connector: Connector, model: str, existing: dict[str, str]
) -> ModelCheck:
    keys = tuple(task.sort_key(model))
    actual = existing.get(model.lower())
    if actual is None:
        return ModelCheck(model=model, key_columns=keys, found=False)
    try:
        mine = _csv_round_trip(connector.fetch_table(actual))
        result = check_correctness(
            sort_by_keys(task.ground_truth(model), list(keys)), sort_by_keys(mine, list(keys))
        )
    except Exception as e:  # noqa: BLE001 - broken table just scores 0
        return ModelCheck(model=model, key_columns=keys, found=True, error=repr(e))
    return ModelCheck(
        model=model,
        key_columns=keys,
        found=True,
        matched=tuple(result["matched"]),
        unmatched=tuple(result["unmatched"]),
        missed=tuple(result["missed"]),
    )


def grade(task: EltTask, connector: Connector) -> Grade:
    existing = {t.lower(): t for t in connector.list_tables()}
    return Grade(
        task=task.name,
        raw_tables=check_raw_tables(task, connector),
        models=tuple(check_model(task, connector, m.name, existing) for m in task.models),
    )
