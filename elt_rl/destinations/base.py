"""what any warehouse has to support.

adding another one (snowflake etc) is just a new file implementing these, nothing else changes
"""

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class Namespace:
    """one rollout's own corner of the warehouse.

    raw tables and final tables both go in `schema`, same as ELT-Bench. user/password is the
    login the agent gets, it can only touch this namespace
    """

    database: str
    schema: str
    user: str | None = None
    password: str | None = None


class Connector(Protocol):
    """how the grader reads a namespace"""

    def list_tables(self) -> list[str]:
        """tables and views in the schema, names as the warehouse stores them"""
        ...

    def count_rows(self, table: str) -> int: ...

    def fetch_table(self, table: str) -> pd.DataFrame: ...

    def close(self) -> None: ...


class Destination(Protocol):
    name: str

    def provision(self, task: str, run_id: str) -> Namespace:
        """make a fresh empty namespace for one rollout"""
        ...

    def clone(self, source: Namespace, run_id: str) -> Namespace:
        """copy a namespace, eg one with the raw tables already loaded"""
        ...

    def teardown(self, namespace: Namespace) -> None:
        """drop the namespace and everything in it"""
        ...

    def connect(self, namespace: Namespace) -> Connector: ...

    # what the sandbox gets, seen from inside the sandbox network

    def agent_env(self, namespace: Namespace) -> dict[str, str]:
        """env vars for the sandbox. must set ELT_WAREHOUSE_URL + ELT_WAREHOUSE_SCHEMA for elt-load"""
        ...

    def agent_config(self, namespace: Namespace) -> dict[str, Any]:
        """the warehouse block shown to the agent in config.yaml"""
        ...

    def dbt_profile(self, namespace: Namespace) -> dict[str, Any]:
        """profiles.yml contents, profile name `elt`"""
        ...

    def agent_files(self, namespace: Namespace) -> dict[str, str]:
        """extra files the sandbox needs to log in (eg a key), path -> content"""
        ...

    def prompt_note(self) -> str:
        """a line for the prompt, which sql dialect + how to query"""
        ...
