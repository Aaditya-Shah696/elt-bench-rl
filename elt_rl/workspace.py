"""what goes into the sandbox at the start of a rollout.

only the agent-visible half of the task goes in (config, data model, schema docs). answers,
row counts and sort keys never leave the host
"""

import copy
from typing import Any

import yaml

from elt_rl import paths
from elt_rl.destinations.base import Destination, Namespace
from elt_rl.sources import READER_PASSWORD, READER_USER
from elt_rl.tasks import EltTask

WORKSPACE = "/workspace"
TASK_DIR = f"{WORKSPACE}/task"
SUBMISSION_DIR = f"{WORKSPACE}/elt"
DBT_DIR = f"{SUBMISSION_DIR}/dbt"
LOAD_SPEC = f"{SUBMISSION_DIR}/load.yaml"
PROFILES = "/root/.dbt/profiles.yml"

API_URL = "http://elt-api:5005"
FILES_URL = "http://elt-files:8000"
# destination blocks in ELT-Bench's config.yaml, swapped for our own warehouse block
_DESTINATION_KEYS = ("Airbyte", "snowflake", "databricks", "redshift")

DBT_PROJECT = """\
name: elt
version: "1.0"
profile: elt
model-paths: ["models"]
models:
  elt:
    +materialized: table
"""


def agent_config(task: EltTask, destination: Destination, ns: Namespace) -> dict[str, Any]:
    """config.yaml as the agent sees it"""
    config = copy.deepcopy(task.config)
    for key in _DESTINATION_KEYS:
        config.pop(key, None)

    # flat files normally come from google drive, point them at the local copy instead
    for f in config.get("flat_files") or []:
        local = paths.SOURCE_DATA_DIR / task.name / f"{f['table']}.{f['format']}"
        if local.exists():
            f["path"] = f"{FILES_URL}/{task.name}/{local.name}"

    # read-only login instead of ELT-Bench's superuser, so sources can't be changed (see sources.py)
    pg = (config.get("postgres") or {}).get("config")
    if pg is not None:
        pg["user"], pg["password"] = READER_USER, READER_PASSWORD

    # ELT-Bench leaves the api address to its docs, spell it out
    api = (config.get("custom_api") or {}).get("config")
    if api is not None:
        api["base_url"] = API_URL
        api["endpoints"] = {t: f"/{task.name}/{t}" for t in api.get("tables", [])}

    config["warehouse"] = destination.agent_config(ns)
    return config


def initial_files(task: EltTask, destination: Destination, ns: Namespace) -> dict[str, str]:
    files = {
        f"{TASK_DIR}/config.yaml": yaml.safe_dump(
            agent_config(task, destination, ns), sort_keys=False
        ),
        f"{TASK_DIR}/data_model.yaml": (task.task_dir / "data_model.yaml").read_text(),
        f"{DBT_DIR}/dbt_project.yml": DBT_PROJECT,
        f"{DBT_DIR}/models/.keep": "",
        PROFILES: yaml.safe_dump(destination.dbt_profile(ns), sort_keys=False),
        **destination.agent_files(ns),
    }
    schemas = paths.SCHEMAS_DIR / task.name
    if schemas.is_dir():
        for f in sorted(schemas.glob("*.csv")):
            files[f"{TASK_DIR}/schemas/{f.name}"] = f.read_text()
    return files
