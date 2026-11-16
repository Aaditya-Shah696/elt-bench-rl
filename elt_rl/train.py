"""RL training on ELT-Bench with tinker, same shape as tinker-cookbook's harbor_rl/train.py.

    uv run python -m elt_rl.train max_steps=1                          # one step on books
    uv run python -m elt_rl.train tasks=[books,address] group_size=8   # any chz field works

logs, rollout transcripts and metrics go to log_path (default ~/tinker-runs/elt_rl/<run>)
"""

import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, hyperparam_utils, model_info
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from elt_rl.env import EltDatasetBuilder, EnvConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # model. the 4b is the cheap debugging one, Qwen/Qwen3.6-35B-A3B is the real target
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    # per turn, qwen thinks before it acts so this needs room
    max_tokens: int = 8192
    temperature: float = 1.0

    # data + env
    tasks: list[str] = chz.field(default_factory=lambda: ["books"])
    env: EnvConfig = chz.field(
        default_factory=lambda: EnvConfig(max_turns=50, max_trajectory_tokens=56 * 1024)
    )

    # training
    group_size: int = 4
    groups_per_batch: int = 1
    learning_rate: float | None = None  # None = the cookbook's calibrated lora lr for the model
    max_steps: int | None = 1
    # async with stable_reinforce, the cookbook default. 1 = sampling can run one step ahead
    max_steps_off_policy: int = 1
    # a sandbox flake shouldn't kill the run, retry it. a stuck rollout gets cut after 30 min
    max_retries: int = 2
    per_rollout_timeout: float = 1800

    # logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 5
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cfg: CLIConfig) -> None:
    lr = cfg.learning_rate or hyperparam_utils.get_lr(cfg.model_name)
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    run_name = (
        f"elt-{cfg.model_name.replace('/', '-')}-{'+'.join(cfg.tasks)}-"
        f"g{cfg.group_size}x{cfg.groups_per_batch}-lr{lr:.1e}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or cli_utils.runs_path("elt_rl", run_name)

    dataset_builder = EltDatasetBuilder(
        tasks=cfg.tasks,
        model_name=cfg.model_name,
        renderer_name=renderer_name,
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        env=cfg.env,
    )
    config = Config(
        learning_rate=lr,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="elt_rl",
        renderer_name=renderer_name,
        lora_rank=cfg.lora_rank,
        load_checkpoint_path=cfg.load_checkpoint_path,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        log_path=log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        session_name=cfg.wandb_name or run_name,
        eval_every=0,  # no held-out set yet
        save_every=cfg.save_every,
        max_steps=cfg.max_steps,
        # same reward rules on the trainer side as the env side
        termination=cfg.env.rollout_config().termination,
        rollout_error_tolerance=RetryOnFailure(
            max_retries=cfg.max_retries, per_rollout_timeout=cfg.per_rollout_timeout
        ),
        async_config=AsyncConfig(
            max_steps_off_policy=cfg.max_steps_off_policy,
            groups_per_batch=cfg.groups_per_batch,
        ),
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    logger.info("logging to %s", log_path)
    await main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
