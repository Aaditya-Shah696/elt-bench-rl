"""fill the local source services with ELT-Bench data.

runs ELT-Bench's own data.sh per task (postgres + s3), then loads the mongo collections from
setup/mongo.jsonl. the api serves its files directly and flat files get downloaded at sync
time, so those need nothing

    uv run python scripts/load_sources.py --only books
    uv run python scripts/load_sources.py            # all 100 tasks (17 GB, slow)

needs psql, aws and the services running. also sets up the read-only source login the agent
gets (--readers-only does just that, for tasks loaded earlier)
"""

import argparse
import json
import math
import os
import shutil
import subprocess

import pandas as pd
from pymongo import MongoClient

from elt_rl.paths import SETUP_DIR, SOURCE_DATA_DIR
from elt_rl.sources import grant_reader
from elt_rl.tasks import load_task

# homebrew doesn't put psql on PATH by default
EXTRA_BIN_DIRS = ["/opt/homebrew/opt/libpq/bin", "/usr/local/opt/libpq/bin"]


def tool_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *EXTRA_BIN_DIRS])
    for tool in ("psql", "aws"):
        if shutil.which(tool, path=env["PATH"]) is None:
            raise SystemExit(f"`{tool}` not found; install it first (see README)")
    return env


def load_postgres_and_s3(task: str, env: dict[str, str]) -> None:
    # data.sh wants the folder that has data/<task>/ and has to run from its own folder
    result = subprocess.run(
        ["bash", "data.sh", str(SETUP_DIR)],
        cwd=SETUP_DIR / "sources" / task,
        env=env,
        capture_output=True,
        text=True,
        check=False,  # checked below so the error shows stderr
    )
    if result.returncode != 0:
        raise RuntimeError(f"{task}: data.sh failed\n{result.stderr[-2000:]}")


def load_mongo(task: str, collections: list[str], client: MongoClient) -> None:
    # same as ELT-Bench's setup/mongo.py but for one task
    client.drop_database(task)
    db = client[task]
    for name in collections:
        df = pd.read_csv(SOURCE_DATA_DIR / task / f"{name}.csv")
        rows = [
            {k: v for k, v in row.items() if not (isinstance(v, float) and math.isnan(v))}
            for row in df.to_dict(orient="records")
        ]
        db[name].insert_many(rows)
        # mixed numbers and strings, store as strings so airbyte doesn't make it DECIMAL
        if task == "card_games" and name == "cards":
            db[name].update_many(
                {"loyalty": {"$type": "number"}},
                [{"$set": {"loyalty": {"$toString": "$loyalty"}}}],
            )


def grant_reader_for(task: str) -> None:
    # the agent gets this read-only login instead of the superuser one
    pg = (load_task(task).config.get("postgres") or {}).get("config")
    if pg:
        grant_reader(pg["database"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated tasks (default: all)")
    parser.add_argument(
        "--readers-only",
        action="store_true",
        help="just (re)create the read-only source logins, for tasks loaded before they existed",
    )
    args = parser.parse_args()

    all_tasks = sorted(p.name for p in (SETUP_DIR / "sources").iterdir() if p.is_dir())
    tasks = [t.strip() for t in args.only.split(",") if t.strip()] or all_tasks
    unknown = set(tasks) - set(all_tasks)
    if unknown:
        raise SystemExit(f"unknown tasks: {sorted(unknown)}")

    if args.readers_only:
        for task in tasks:
            grant_reader_for(task)
        return

    env = tool_env()
    mongo_tasks = json.loads((SETUP_DIR / "mongo.jsonl").read_text())
    client = MongoClient("mongodb://localhost:27017/?directConnection=true")
    try:
        for task in tasks:
            load_postgres_and_s3(task, env)
            if task in mongo_tasks:
                load_mongo(task, mongo_tasks[task], client)
            grant_reader_for(task)
            print(f"loaded {task}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
