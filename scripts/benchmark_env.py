"""how long the environment itself takes, no model involved.

runs the scripted books oracle a few times in each mode and prints mean seconds per phase. the
model's own sampling time is on top of this (and is most of an episode, see docs/design.md)

    uv run python scripts/benchmark_env.py --runs 3
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_env_integration import ORACLE_LOAD, ORACLE_MODELS, call
from test_snowflake import to_snowflake

from elt_rl.destinations import get_destination
from elt_rl.env import EltRollout, EnvConfig
from elt_rl.golden import drop_goldens
from elt_rl.tasks import load_task


def script(loaded: bool, dialect=lambda sql: sql) -> list[dict]:
    steps = [] if loaded else [call("write_file", path="elt/load.yaml", content=ORACLE_LOAD)]
    if not loaded:
        steps.append(call("bash", command="elt-load elt/load.yaml"))
    steps += [
        call("write_file", path=f"elt/dbt/models/{m}.sql", content=dialect(s))
        for m, s in ORACLE_MODELS.items()
    ]
    return [*steps, call("bash", command="dbt run --project-dir elt/dbt"), call("submit")]


async def one(task, destination, config) -> dict[str, float]:
    t0 = time.monotonic()
    rollout = await EltRollout.create(task, destination, config)
    times = {"setup": rollout.setup_seconds}
    try:
        env = rollout.message_env(config)
        await env.initial_observation()
        t1 = time.monotonic()
        # the reference sql is postgres flavoured, translate it for snowflake
        dialect = to_snowflake if config.destination == "snowflake" else (lambda s: s)
        for turn in script(config.start_loaded, dialect):
            step = await env.step(turn)
            if step.episode_done:
                break
        times["tools + replay"] = round(time.monotonic() - t1, 2)
        times |= {
            k.removeprefix("time/"): v for k, v in step.metrics.items() if k.startswith("time/")
        }
        times["reward"] = step.reward
    finally:
        await rollout.cleanup()
    times["total"] = round(time.monotonic() - t0, 2)
    return times


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--destination", default="postgres")
    args = parser.parse_args()
    task, destination = load_task("books"), get_destination(args.destination)

    report = {}
    try:
        for mode, loaded in (("full episode", False), ("transform only", True)):
            config = EnvConfig(destination=args.destination, start_loaded=loaded)
            runs = [await one(task, destination, config) for _ in range(args.runs)]
            keys = runs[-1].keys()
            report[mode] = {
                k: round(statistics.mean(r.get(k, 0.0) for r in runs[1:] or runs), 2) for k in keys
            }
            report[mode]["first setup"] = runs[0]["setup"]
    finally:
        await drop_goldens()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
