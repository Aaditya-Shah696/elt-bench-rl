"""replay grading, rerun the submitted pipeline from scratch and grade what comes out.

the agent's own warehouse can have hand made tables or leftovers from earlier tries, so it's
never graded directly. instead its /workspace/elt is copied into a clean sandbox (fresh image,
agent can't have tampered with it) pointed at an empty namespace, then elt-load + dbt run
"""

import asyncio
import dataclasses
import logging
import time
import uuid
from dataclasses import dataclass, field

import yaml

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.golden import clone_namespace
from elt_rl.grading.grader import Grade, grade
from elt_rl.grading.lint import hardcoded_models
from elt_rl.grading.reward import DEFAULT_WEIGHTS, RewardWeights, compute_reward
from elt_rl.sandbox import DockerSandbox
from elt_rl.tasks import EltTask
from elt_rl.workspace import DBT_DIR, LOAD_SPEC, PROFILES, SUBMISSION_DIR, WORKSPACE

logger = logging.getLogger(__name__)

MAX_SUBMISSION_BYTES = 20 * 1024 * 1024


@dataclass
class ReplayResult:
    grade: Grade | None
    has_load_spec: bool = False
    has_dbt_project: bool = False
    load_ok: bool = False
    dbt_ok: bool = False
    error: str | None = None
    logs: dict[str, str] = field(default_factory=dict)
    hardcoded: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        return {
            "replay/hardcoded_models": float(len(self.hardcoded)),
            "replay/has_load_spec": float(self.has_load_spec),
            "replay/has_dbt_project": float(self.has_dbt_project),
            "replay/load_ok": float(self.load_ok),
            "replay/dbt_ok": float(self.dbt_ok),
            "replay/error": float(self.error is not None),
        } | {f"time/replay_{k}": v for k, v in self.seconds.items()}


async def replay(
    task: EltTask,
    destination: Destination,
    submission_tar: bytes | None,
    run_id: str,
    base: Namespace | None = None,
    load_timeout: int = 900,
    dbt_timeout: int = 600,
) -> ReplayResult:
    """run a submission (tar of the elt/ folder) into a fresh namespace and grade it.

    with `base` (transform only episodes) the fresh namespace is a copy of that pre-loaded one
    and the submission's load step is skipped
    """
    if submission_tar is None:
        return ReplayResult(grade=None, error="no submission folder")

    result = ReplayResult(grade=None)
    clock = time.monotonic()

    def lap(name: str) -> None:
        nonlocal clock
        now = time.monotonic()
        result.seconds[name] = round(now - clock, 2)
        clock = now

    if base is None:
        ns = await asyncio.to_thread(destination.provision, task.name, f"{run_id}_r")
    else:
        ns = await clone_namespace(destination, base, f"{run_id}_r")
    sandbox: DockerSandbox | None = None
    try:
        sandbox = await DockerSandbox.create(env=destination.agent_env(ns))
        await sandbox.put_tar(submission_tar, WORKSPACE)
        await sandbox.put_files(
            {PROFILES: yaml.safe_dump(destination.dbt_profile(ns)), **destination.agent_files(ns)}
        )
        check = await sandbox.run_command(
            f"test -f {LOAD_SPEC} && echo load; test -f {DBT_DIR}/dbt_project.yml && echo dbt"
        )
        result.has_load_spec = "load" in check.stdout.split()
        result.has_dbt_project = "dbt" in check.stdout.split()
        lap("setup")

        if base is not None:
            result.load_ok = True  # the copy already has the raw tables
        elif result.has_load_spec:
            r = await sandbox.run_command(f"elt-load {LOAD_SPEC}", timeout=load_timeout)
            result.load_ok = r.exit_code == 0
            result.logs["load"] = (r.stdout + r.stderr)[-4000:]
            lap("load")
        if result.has_dbt_project:
            r = await sandbox.run_command(
                f"dbt run --project-dir {DBT_DIR} --profiles-dir /root/.dbt", timeout=dbt_timeout
            )
            result.dbt_ok = r.exit_code == 0
            result.logs["dbt"] = (r.stdout + r.stderr)[-4000:]
            lap("dbt")

        result.grade = await asyncio.to_thread(_grade_namespace, task, destination, ns)
        lap("grade")
    except Exception as e:
        logger.exception("replay failed for %s", run_id)
        result.error = f"{type(e).__name__}: {e}"
    finally:
        if sandbox is not None:
            await sandbox.cleanup()
        await asyncio.to_thread(destination.teardown, ns)
        lap("teardown")
    return result


def _zero_out(grade: Grade, flagged: dict[str, str]) -> Grade:
    """flagged models count as broken, no credit for them"""
    if not flagged:
        return grade
    lowered = {k.lower(): v for k, v in flagged.items()}
    models = tuple(
        dataclasses.replace(m, error=f"hard-coded: {lowered[m.model.lower()]}")
        if m.model.lower() in lowered
        else m
        for m in grade.models
    )
    return dataclasses.replace(grade, models=models)


def _grade_namespace(task, destination, ns) -> Grade:
    connector = destination.connect(ns)
    try:
        return grade(task, connector)
    finally:
        connector.close()


class EltReward:
    """the cookbook reward_fn, called once when the episode ends.

    replay=False grades the agent's own namespace in place, quicker but easy to fool, only
    meant for comparisons
    """

    def __init__(
        self,
        task: EltTask,
        destination: Destination,
        sandbox: DockerSandbox,
        namespace,
        run_id: str,
        replay: bool = True,
        weights: RewardWeights = DEFAULT_WEIGHTS,
        submitted=lambda: True,
        base: Namespace | None = None,
    ):
        self.task, self.destination, self.sandbox = task, destination, sandbox
        self.namespace, self.run_id = namespace, run_id
        self.replay, self.weights, self._submitted = replay, weights, submitted
        self.base = base
        self.last: ReplayResult | None = None

    async def __call__(self, history) -> tuple[float, dict[str, float]]:
        if self.replay:
            try:
                tar = await self.sandbox.get_tar(SUBMISSION_DIR, MAX_SUBMISSION_BYTES)
            except ValueError as e:
                tar, err = None, str(e)
            else:
                err = None
            result = await replay(self.task, self.destination, tar, self.run_id, self.base)
            if err:
                result.error = err
            if result.grade is not None:
                result.hardcoded = hardcoded_models(self.task, tar)
                result.grade = _zero_out(result.grade, result.hardcoded)
        else:
            g = await asyncio.to_thread(
                _grade_namespace, self.task, self.destination, self.namespace
            )
            result = ReplayResult(grade=g)
        self.last = result

        metrics = result.metrics()
        metrics["submitted"] = float(self._submitted())
        if result.grade is None:
            return 0.0, metrics | {"reward": 0.0, "task_success": 0.0}
        reward, grade_metrics = compute_reward(result.grade, self.weights)
        return reward, metrics | grade_metrics


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
