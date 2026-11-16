"""the same oracle episode, but through the token level loop tinker actually trains on.

a scripted TokenCompleter turns each scripted message into tokens with the real renderer, the
cookbook's rollout runner feeds them to the env, which has to parse the tool calls back out.
needs the public Qwen tokenizer from huggingface (no login), still no tinker or snowflake
"""

import pytest
from test_env_integration import CONFIG, oracle_script
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollouts import do_single_rollout

from elt_rl.destinations import PostgresDestination
from elt_rl.env import EltRollout

pytestmark = pytest.mark.services

MODEL = "Qwen/Qwen3.5-4B"


@pytest.fixture(scope="module")
def renderer():
    try:
        tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"couldn't get the {MODEL} tokenizer: {e}")
    return get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer)


class ScriptedPolicy(TokenCompleter):
    """plays back fixed assistant messages as tokens, like a model that knows the answer"""

    def __init__(self, renderer, messages):
        self.renderer, self.messages = renderer, list(messages)

    async def __call__(self, model_input, stop, *, max_tokens=None) -> TokensWithLogprobs:
        message = self.messages.pop(0)
        convo = [{"role": "user", "content": "go"}, message]
        tokens, weights = self.renderer.build_supervised_example(convo)
        # weights mark the assistant's own tokens, that's what a model would have sampled
        action = [t for t, w in zip(tokens.to_ints(), weights.tolist(), strict=True) if w > 0]
        return TokensWithLogprobs(tokens=action, maybe_logprobs=[0.0] * len(action))


async def test_oracle_through_token_loop(require_sandbox, books, renderer):
    rollout = await EltRollout.create(books, PostgresDestination.from_env(), CONFIG)
    try:
        env = rollout.token_env(renderer, CONFIG)
        trajectory = await do_single_rollout(ScriptedPolicy(renderer, oracle_script()), env)
    finally:
        await rollout.cleanup()

    last = trajectory.transitions[-1]
    assert last.episode_done
    assert last.metrics.get("parse_error", 0.0) == 0.0
    assert rollout.tools.submitted  # the submit call made it through parsing
    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    assert last.metrics["task_success"] == 1.0


async def test_runner_stopped_episode_still_gets_graded(require_sandbox, books, renderer):
    # the bug from the first snowflake rollout: the cookbook's runner hit its own turn limit
    # before the env did, so the env's reward never ran and the attempt silently scored 0
    from test_env_integration import ORACLE_LOAD, call
    from tinker_cookbook.rl.rollout_limits import RolloutLimits
    from tinker_cookbook.rl.rollout_runner import run_rollout

    from elt_rl.env import EltEnvGroupBuilder

    builder = EltEnvGroupBuilder(books, renderer, group_size=1, config=CONFIG)
    try:
        (env,) = await builder.make_envs()
        script = [
            call("write_file", path="elt/load.yaml", content=ORACLE_LOAD),
            call("bash", command="echo still going"),
            call("bash", command="echo and going"),
        ]
        trajectory = await run_rollout(
            ScriptedPolicy(renderer, script), env, limits=RolloutLimits(max_turns=2)
        )
        assert trajectory.stop_reason == "max_turns"
        assert sum(t.reward for t in trajectory.transitions) == 0.0  # env never graded

        ((reward, metrics),) = await builder.compute_group_rewards([trajectory], [env])
        assert reward == pytest.approx(0.2)  # the load.yaml it wrote still counts
        assert metrics["graded_after_runner_stop"] == 1.0
    finally:
        await builder.cleanup()
