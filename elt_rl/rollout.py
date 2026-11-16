"""one episode with a real model, no training. for checking prompts and for the official rollout.

    uv run python -m elt_rl.rollout task=books
    uv run python -m elt_rl.rollout model_name=Qwen/Qwen3.6-35B-A3B checkpoint=tinker://...

writes the full transcript, reward and replay logs to a json file under ~/tinker-runs/elt_rl
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker_cookbook import cli_utils, model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import format_content_as_string
from tinker_cookbook.rl.rollouts import do_single_rollout

from elt_rl.destinations import get_destination
from elt_rl.env import EltRollout, EnvConfig
from elt_rl.tasks import load_task

logger = logging.getLogger(__name__)


@chz.chz
class RolloutConfig:
    task: str = "books"
    model_name: str = "Qwen/Qwen3.5-4B"
    checkpoint: str | None = None  # tinker:// sampler weights, None = base model
    renderer_name: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0
    env: EnvConfig = chz.field(
        default_factory=lambda: EnvConfig(max_turns=50, max_trajectory_tokens=56 * 1024)
    )
    out: str | None = None


def _message_json(m) -> dict:
    out = {"role": m["role"], "content": format_content_as_string(m.get("content") or "")}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {"name": tc.function.name, "arguments": tc.function.arguments} for tc in m["tool_calls"]
        ]
    return out


async def run(cfg: RolloutConfig) -> dict:
    task = load_task(cfg.task)
    destination = get_destination(cfg.env.destination)
    renderer = get_renderer(
        cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name),
        tokenizer_utils.get_tokenizer(cfg.model_name),
    )
    service = tinker.ServiceClient()
    if cfg.checkpoint:
        sampler = await service.create_sampling_client_async(model_path=cfg.checkpoint)
    else:
        sampler = await service.create_sampling_client_async(base_model=cfg.model_name)
    policy = TinkerTokenCompleter(
        sampler, max_tokens=cfg.max_tokens, temperature=cfg.temperature, context_window=65536
    )

    rollout = await EltRollout.create(task, destination, cfg.env)
    start = time.time()
    try:
        env = rollout.token_env(renderer, cfg.env)
        trajectory = await do_single_rollout(policy, env)
        history = env.message_env.history  # type: ignore[attr-defined]
        extra_reward, extra_metrics = await rollout.grade_if_needed()
    finally:
        await rollout.cleanup()

    last = trajectory.transitions[-1]
    result = {
        "task": cfg.task,
        "model": cfg.model_name,
        "checkpoint": cfg.checkpoint,
        "reward": sum(t.reward for t in trajectory.transitions) + extra_reward,
        "stop_reason": trajectory.stop_reason,
        "turns": len(trajectory.transitions),
        "sampled_tokens": sum(len(t.ac.tokens) for t in trajectory.transitions),
        "seconds": round(time.time() - start, 1),
        "metrics": last.metrics | extra_metrics,
        "replay_logs": rollout.reward.last.logs if rollout.reward.last else {},
        "messages": [_message_json(m) for m in history],
    }
    out = Path(
        cfg.out
        or cli_utils.runs_path(
            "elt_rl", "rollouts", f"{cfg.task}-{datetime.now():%Y-%m-%d-%H-%M-%S}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "messages"}, indent=2, default=str))
    print(f"transcript: {out}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(chz.entrypoint(RolloutConfig)))
