"""the RL environment, wired into tinker-cookbook the same way recipes/harbor_rl is.

one rollout = one sandbox + one warehouse namespace + tools + a reward that replays the
submission. EltEnvGroupBuilder makes group_size of them for one task, the dataset hands
out groups per training step
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

import chz
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.message_env import MessageStepResult
from tinker_cookbook.rl.rollout_limits import (
    ParseErrorPolicy,
    RolloutLimits,
    TerminationRewardPolicy,
)
from tinker_cookbook.rl.rollout_presets import RolloutConfig
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.agent_tool_message_env import AgentToolMessageEnv

from elt_rl.destinations import get_destination
from elt_rl.destinations.base import Destination, Namespace
from elt_rl.golden import clone_namespace, golden_namespace
from elt_rl.grading.replay import EltReward, new_run_id
from elt_rl.grading.reward import DEFAULT_WEIGHTS, RewardWeights
from elt_rl.prompts import system_prompt, user_prompt
from elt_rl.sandbox import DockerSandbox
from elt_rl.sources import GUARD
from elt_rl.tasks import EltTask, load_task
from elt_rl.tools import EltTools
from elt_rl.workspace import initial_files

logger = logging.getLogger(__name__)

TRANSFORM_ONLY = RewardWeights(load=0.0, transform=1.0)


@chz.chz
class EnvConfig:
    destination: str = "postgres"
    max_turns: int = 40
    max_tool_calls: int = 100
    max_trajectory_tokens: int = 64 * 1024
    command_timeout: int = 300
    # grade a rerun of the submission (default) or just what's in the agent's warehouse
    replay: bool = True
    # transform-only episodes: raw tables are already there (cloned from a pre-loaded copy) and
    # only the dbt models are graded. faster, and focuses training on the hard part
    start_loaded: bool = False
    # zero the group + reload sources if anyone changed source data (see sources.py)
    guard_sources: bool = True
    # the cookbook's agentic preset zeroes out rollouts that hit a limit. off by default since
    # early on almost nothing finishes and load credit is most of the signal
    zero_reward_on_limit: bool = False

    def rollout_config(self) -> RolloutConfig:
        return RolloutConfig(
            limits=RolloutLimits(
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_tool_calls=self.max_tool_calls,
            ),
            parse_errors=ParseErrorPolicy(max_consecutive=2),
            termination=TerminationRewardPolicy(
                zero_reward_on_limit=self.zero_reward_on_limit,
                skip_grading_on_timeout=True,
                grader_timeout_seconds=1800,
            ),
            # tools share one sandbox, so run them in the order the model asked
            tool_execution="sequential",
        )


NUDGE = (
    "No tool call found in your last message. Call a tool to keep working, "
    "or call submit if your pipeline is done."
)


@dataclass
class NudgingToolMessageEnv(AgentToolMessageEnv):
    """the cookbook's tool env, except a turn with no tool call gets a nudge instead of ending
    the episode (up to max_nudges times). a garbled or cut off tool call otherwise ends the
    whole task, which happened in the first snowflake rollout"""

    max_nudges: int = 2
    _nudges: int = dataclasses.field(default=0, init=False)

    async def step(self, message: Message) -> MessageStepResult:
        no_call = not message.get("tool_calls") and not message.get("unparsed_tool_calls")
        last_turn = self._turn_count + 1 >= self.max_turns
        if no_call and self._nudges < self.max_nudges and not last_turn:
            self._nudges += 1
            self._turn_count += 1
            self.history.append(message)
            self.history.append({"role": "user", "content": NUDGE})
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=self.history,
                metrics={"nudged": 1.0},
                logs={},
            )
        return await super().step(message)


@dataclass
class EltRollout:
    """everything one rollout owns. create() sets it up, cleanup() tears it down"""

    task: EltTask
    destination: Destination
    run_id: str
    namespace: Namespace
    sandbox: DockerSandbox
    tools: EltTools
    reward: EltReward
    loaded: bool = False
    setup_seconds: float = 0.0

    @classmethod
    async def create(
        cls, task: EltTask, destination: Destination, config: EnvConfig
    ) -> "EltRollout":
        run_id = new_run_id()
        start = time.monotonic()
        base = await golden_namespace(task, destination) if config.start_loaded else None
        if base is None:
            ns = await asyncio.to_thread(destination.provision, task.name, run_id)
        else:
            ns = await clone_namespace(destination, base, run_id)
        sandbox = None
        try:
            sandbox = await DockerSandbox.create(env=destination.agent_env(ns))
            await sandbox.put_files(initial_files(task, destination, ns))
        except BaseException:
            if sandbox is not None:
                await sandbox.cleanup()
            await asyncio.to_thread(destination.teardown, ns)
            raise
        tools = EltTools(sandbox, command_timeout=config.command_timeout)
        reward = EltReward(
            task,
            destination,
            sandbox,
            ns,
            run_id,
            replay=config.replay,
            submitted=lambda: tools.submitted,
            base=base,
            # loading is done for it, so all the credit is for the transform
            weights=TRANSFORM_ONLY if base is not None else DEFAULT_WEIGHTS,
        )
        rollout = cls(task, destination, run_id, ns, sandbox, tools, reward)
        rollout.loaded = base is not None
        rollout.setup_seconds = round(time.monotonic() - start, 2)
        return rollout

    def plain_messages(self) -> list[Message]:
        """system + user prompt, for driving the env without a renderer (tests)"""
        return [
            {"role": "system", "content": system_prompt(self.destination)},
            {"role": "user", "content": user_prompt(self.task, self.loaded)},
        ]

    def initial_messages(self, renderer: Renderer) -> list[Message]:
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in self.tools.all],
            system_prompt=system_prompt(self.destination),
        )
        return prefix + [{"role": "user", "content": user_prompt(self.task, self.loaded)}]

    def message_env(self, config: EnvConfig) -> AgentToolMessageEnv:
        """the message level env, no tokens involved. what the scripted test drives"""
        return NudgingToolMessageEnv(
            tools=self.tools.all,
            initial_messages=self.plain_messages(),
            max_turns=config.max_turns,
            reward_fn=self.reward,
            tool_execution="sequential",
        )

    def token_env(self, renderer: Renderer, config: EnvConfig) -> Env:
        """the real thing tinker trains on"""
        env = build_agent_tool_env(
            renderer=renderer,
            tools=self.tools.all,
            initial_messages=self.initial_messages(renderer),
            reward_fn=self.reward,
            rollout_config=config.rollout_config(),
        )
        # same settings the cookbook worked out, just the nudging version of its message env
        inner = env.message_env
        assert isinstance(inner, AgentToolMessageEnv)
        env.message_env = NudgingToolMessageEnv(
            **{f.name: getattr(inner, f.name) for f in dataclasses.fields(inner) if f.init}
        )
        return env

    async def grade_if_needed(self) -> tuple[float, dict[str, float]]:
        """grade now if the env never got to. the cookbook's runner can stop an episode itself
        (turn or token budget) and then the env's reward_fn never runs. (0, {}) if it did"""
        if self.reward.last is not None:
            return 0.0, {"time/env_setup": self.setup_seconds}
        reward, metrics = await self.reward([])
        return reward, metrics | {
            "graded_after_runner_stop": 1.0,
            "time/env_setup": self.setup_seconds,
        }

    async def cleanup(self) -> None:
        try:
            await self.sandbox.cleanup()
        finally:
            await asyncio.to_thread(self.destination.teardown, self.namespace)


class EltEnvGroupBuilder(EnvGroupBuilder):
    """group_size rollouts of the same task, rewards get compared within the group"""

    def __init__(
        self,
        task: EltTask,
        renderer: Renderer,
        group_size: int,
        config: EnvConfig,
        destination: Destination | None = None,
    ):
        self.task = task
        self.renderer = renderer
        self.group_size = group_size
        self.config = config
        self.destination = destination or get_destination(config.destination)
        self.rollouts: list[EltRollout] = []
        self._rollout_for: dict[int, EltRollout] = {}

    async def make_envs(self) -> Sequence[Env]:
        if self.config.guard_sources:
            await asyncio.to_thread(GUARD.baseline, self.task)
        results = await asyncio.gather(
            *(
                EltRollout.create(self.task, self.destination, self.config)
                for _ in range(self.group_size)
            ),
            return_exceptions=True,
        )
        # append, not replace. the cookbook's retry strategy calls make_envs again for a fresh
        # env and everything made here still needs cleaning up later
        created = [r for r in results if isinstance(r, EltRollout)]
        self.rollouts.extend(created)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            await self.cleanup()
            raise errors[0]
        envs = [r.token_env(self.renderer, self.config) for r in created]
        self._rollout_for.update({id(e): r for e, r in zip(envs, created, strict=True)})
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        # added on top of what the env gave, so this only grades episodes the env didn't
        results = list(
            await asyncio.gather(*(self._rollout_for[id(e)].grade_if_needed() for e in env_group))
        )
        if not self.config.guard_sources:
            return results
        changed = await asyncio.to_thread(GUARD.changed, self.task)
        if not changed:
            return [(r, m | {"source_tampered": 0.0}) for r, m in results]
        # someone changed source data. cancel every reward in the group (nothing to gain from
        # sabotaging siblings) and put the sources back for the next group
        logger.error("sources of %s changed during a group: %s", self.task.name, changed)
        await asyncio.to_thread(GUARD.restore, self.task)
        return [
            (-sum(t.reward for t in traj.transitions), {"source_tampered": 1.0})
            for traj in trajectory_group
        ]

    async def cleanup(self) -> None:
        for r in self.rollouts:
            try:
                await r.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.warning("cleanup failed for %s: %s", r.run_id, e)
        self.rollouts = []
        self._rollout_for = {}

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.name]


class EltDataset(RLDataset):
    def __init__(
        self,
        tasks: list[EltTask],
        renderer: Renderer,
        batch_size: int,
        group_size: int,
        config: EnvConfig,
    ):
        self.tasks, self.renderer, self.config = tasks, renderer, config
        self.batch_size, self.group_size = batch_size, group_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        # fresh builders every call since a builder holds its rollouts until cleanup.
        # wraps around so a few tasks can fill many steps
        n = len(self.tasks)
        return [
            EltEnvGroupBuilder(
                self.tasks[(index * self.batch_size + i) % n],
                self.renderer,
                self.group_size,
                self.config,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return max(1, -(-len(self.tasks) // self.batch_size))


@chz.chz
class EltDatasetBuilder(RLDatasetBuilder):
    tasks: list[str]
    model_name: str
    batch_size: int = 1
    group_size: int = 4
    renderer_name: str | None = None
    env: EnvConfig = chz.field(default_factory=EnvConfig)

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = get_renderer(
            self.renderer_name or model_info.get_recommended_renderer_name(self.model_name),
            tokenizer,
        )
        tasks = [load_task(t) for t in self.tasks]
        return EltDataset(tasks, renderer, self.batch_size, self.group_size, self.env), None
