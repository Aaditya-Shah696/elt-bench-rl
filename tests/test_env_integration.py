"""credential free integration test, grading item 1.

scripted policies stand in for the model and drive the real env loop: real sandbox, real
tools, real warehouse, replay grading. no tinker, no snowflake. the oracle does books right
and should get 1.0, the cheaters should get nothing
"""

import json

import pytest
from conftest import FIXTURES, reference_sql
from tinker_cookbook.renderers.base import ToolCall

from elt_rl.destinations import PostgresDestination
from elt_rl.env import EltRollout, EnvConfig
from elt_rl.grading.replay import EltReward

pytestmark = pytest.mark.services

CONFIG = EnvConfig(max_turns=30)


def call(name: str, **arguments) -> dict:
    """one assistant turn that calls one tool"""
    tc = ToolCall(
        id=f"call_{name}",
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)),
    )
    return {"role": "assistant", "content": "", "tool_calls": [tc]}


async def run_episode(rollout: EltRollout, script: list[dict], config: EnvConfig = CONFIG):
    """feed scripted turns into the env until it says done, return the last step + tool outputs"""
    env = rollout.message_env(config)
    await env.initial_observation()
    outputs = []
    for turn in script:
        step = await env.step(turn)
        outputs.append(step.next_messages[-1]["content"])
        if step.episode_done:
            return step, outputs
    raise AssertionError("script ran out before the episode ended")


ORACLE_MODELS = reference_sql("books")
ORACLE_LOAD = (FIXTURES / "books_reference" / "load.yaml").read_text()


def oracle_script() -> list[dict]:
    script = [
        call("bash", command="cat task/config.yaml"),
        call("write_file", path="elt/load.yaml", content=ORACLE_LOAD),
        call("bash", command="elt-load elt/load.yaml"),
    ]
    for model, sql in ORACLE_MODELS.items():
        script.append(call("write_file", path=f"elt/dbt/models/{model}.sql", content=sql))
    script += [
        call("bash", command="dbt run --project-dir elt/dbt"),
        call("run_sql", query="select count(*) from books.authors"),
        call("submit"),
    ]
    return script


@pytest.fixture
async def rollout(require_sandbox, books):
    r = await EltRollout.create(books, PostgresDestination.from_env(), CONFIG)
    yield r
    await r.cleanup()


async def test_oracle_policy_scores_one(rollout):
    step, outputs = await run_episode(rollout, oracle_script())
    load_out = json.loads(outputs[2])
    assert load_out["exit_code"] == 0, load_out
    assert "15/15 streams loaded" in load_out["stdout"]
    dbt_out = json.loads(outputs[-3])
    assert dbt_out["exit_code"] == 0, dbt_out["stdout"][-2000:]
    assert "9235" in outputs[-2]  # authors row count, straight from the warehouse

    assert step.episode_done
    assert step.metrics["submitted"] == 1.0
    assert step.metrics["replay/load_ok"] == 1.0 and step.metrics["replay/dbt_ok"] == 1.0
    assert step.reward == pytest.approx(1.0), rollout.reward.last
    assert step.metrics["task_success"] == 1.0


async def test_hand_made_tables_dont_count(rollout):
    # loads + builds everything by hand in the warehouse but leaves no pipeline behind
    sql = ";\n".join(f'CREATE TABLE books."{m}" AS {s}' for m, s in ORACLE_MODELS.items())
    script = [
        call(
            "bash", command=f"cat > /tmp/l.yaml <<'EOF'\n{ORACLE_LOAD}\nEOF\nelt-load /tmp/l.yaml"
        ),
        call("bash", command=f"psql -v ON_ERROR_STOP=1 <<'EOF'\n{sql};\nEOF"),
        call("submit"),
    ]
    step, outputs = await run_episode(rollout, script)
    assert json.loads(outputs[1])["exit_code"] == 0, outputs[1]
    assert step.reward == 0.0
    assert step.metrics["replay/has_load_spec"] == 0.0

    # grading the warehouse in place would have been fooled into a perfect score
    in_place = EltReward(
        rollout.task,
        rollout.destination,
        rollout.sandbox,
        rollout.namespace,
        rollout.run_id,
        replay=False,
    )
    reward, _ = await in_place([])
    assert reward == pytest.approx(1.0)


async def test_load_only_gets_load_credit(rollout):
    script = [
        call("write_file", path="elt/load.yaml", content=ORACLE_LOAD),
        call("submit"),
    ]
    step, _ = await run_episode(rollout, script)
    assert step.reward == pytest.approx(0.2)
    assert step.metrics["replay/has_dbt_project"] == 1.0  # the scaffold is there, just no models


async def test_running_out_of_turns_still_grades(require_sandbox, books):
    config = EnvConfig(max_turns=2)
    r = await EltRollout.create(books, PostgresDestination.from_env(), config)
    try:
        env = r.message_env(config)
        await env.initial_observation()
        await env.step(call("write_file", path="elt/load.yaml", content=ORACLE_LOAD))
        step = await env.step(call("bash", command="echo still working"))
        assert step.episode_done and step.metrics["max_turns"] == 1.0
        assert step.metrics["submitted"] == 0.0
        assert step.reward == pytest.approx(0.2)
    finally:
        await r.cleanup()


async def test_sandbox_boundaries(rollout):
    script = [
        call("write_file", path="/etc/evil", content="x"),
        call("write_file", path="elt/../task/config.yaml", content="x"),
        # no internet, so the public answer files can't be downloaded
        call("bash", command="curl -sS -m 5 https://huggingface.co >/dev/null && echo ONLINE"),
        # answers and grading files were never copied in
        call(
            "bash",
            command="ls /workspace/task; find / -name 'table.json' -o -name 'gt_*' 2>/dev/null | head",
        ),
        # the agent's login only works on its own database
        call(
            "bash",
            command="psql -d postgres -tAc 'select datname from pg_database' | grep elt_ | "
            "while read d; do psql -d $d -tAc 'select 1' >/dev/null 2>&1 && echo CAN_OPEN $d; done",
        ),
        call("run_sql", query="create table books.x (a int)"),
        call("submit"),
    ]
    _, outputs = await run_episode(rollout, script)
    assert outputs[0].startswith("error") and outputs[1].startswith("error")
    assert "ONLINE" not in json.loads(outputs[2])["stdout"]
    listing = json.loads(outputs[3])["stdout"].split()
    assert sorted(listing) == ["config.yaml", "data_model.yaml", "schemas"]
    opened = json.loads(outputs[4])["stdout"].split()
    assert opened == ["CAN_OPEN", rollout.namespace.database]
    assert "read-only transaction" in outputs[5]


async def test_turn_without_tool_call_gets_nudged(rollout):
    env = rollout.message_env(CONFIG)
    await env.initial_observation()
    text_only = {"role": "assistant", "content": "let me think about the plan first"}
    first = await env.step(text_only)
    assert not first.episode_done and first.metrics["nudged"] == 1.0
    assert first.next_messages[-1]["role"] == "user"  # the nudge
    loaded = await env.step(call("write_file", path="elt/load.yaml", content=ORACLE_LOAD))
    assert not loaded.episode_done
    await env.step(text_only)  # second nudge
    last = await env.step(text_only)  # out of nudges, this ends it and grades
    assert last.episode_done and last.reward == pytest.approx(0.2)
