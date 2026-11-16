"""pre-loaded copies of each task's raw tables, for transform-only episodes.

the first time a task is needed its raw tables get loaded once (with the reference load
config) into a golden namespace. after that every rollout and every replay just clones it,
which on postgres is a file copy and on snowflake a zero-copy clone. skips the slowest and
least interesting part of an episode, so training can focus on the transform stage
"""

import asyncio
import logging
import uuid
from collections import defaultdict

import yaml

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.reference import reference_load_spec
from elt_rl.sandbox import DockerSandbox
from elt_rl.tasks import EltTask
from elt_rl.workspace import agent_config

logger = logging.getLogger(__name__)

_golden: dict[tuple[str, str], tuple[Destination, Namespace]] = {}
_locks: defaultdict[tuple[int, str], asyncio.Lock] = defaultdict(asyncio.Lock)


def _lock(name: str) -> asyncio.Lock:
    # asyncio locks belong to one event loop, so key by loop too
    return _locks[(id(asyncio.get_running_loop()), name)]


async def golden_namespace(task: EltTask, destination: Destination) -> Namespace:
    """the task's pre-loaded namespace, loading it the first time"""
    key = (destination.name, task.name)
    async with _lock(f"make:{key}"):
        if key in _golden:
            return _golden[key][1]
        ns = await asyncio.to_thread(
            destination.provision, task.name, f"gold_{uuid.uuid4().hex[:10]}"
        )
        try:
            await _load(task, destination, ns)
        except BaseException:
            await asyncio.to_thread(destination.teardown, ns)
            raise
        _golden[key] = (destination, ns)
        return ns


async def _load(task: EltTask, destination: Destination, ns: Namespace) -> None:
    spec = reference_load_spec(agent_config(task, destination, ns))
    sandbox = await DockerSandbox.create(env=destination.agent_env(ns))
    try:
        await sandbox.put_files(
            {"/workspace/load.yaml": yaml.safe_dump(spec), **destination.agent_files(ns)}
        )
        r = await sandbox.run_command("elt-load /workspace/load.yaml", timeout=1800)
    finally:
        await sandbox.cleanup()
    connector = destination.connect(ns)
    try:
        found = {t.lower() for t in await asyncio.to_thread(connector.list_tables)}
    finally:
        connector.close()
    missing = {t.lower() for t in task.expected_row_counts} - found
    if r.exit_code != 0 or missing:
        raise RuntimeError(f"couldn't pre-load {task.name}, missing {missing}: {r.stderr[-2000:]}")


async def clone_namespace(destination: Destination, base: Namespace, run_id: str) -> Namespace:
    # postgres won't copy a template that's busy, so one copy of a given base at a time
    async with _lock(f"clone:{base.database}"):
        return await asyncio.to_thread(destination.clone, base, run_id)


async def drop_goldens() -> None:
    for destination, ns in list(_golden.values()):
        try:
            await asyncio.to_thread(destination.teardown, ns)
        except Exception as e:  # noqa: BLE001
            logger.warning("couldn't drop golden namespace %s: %s", ns.database, e)
    _golden.clear()
