"""scripted cheaters, one per loophole. each must get no credit for the cheat.

already covered elsewhere: reading answer files / internet / other rollouts' databases / writing
outside elt/ (test_env_integration::test_sandbox_boundaries)
"""

import json

import pytest
from conftest import reference_sql
from test_env_integration import CONFIG, ORACLE_LOAD, ORACLE_MODELS, call, run_episode
from test_token_loop import ScriptedPolicy, renderer  # noqa: F401 - renderer is a fixture
from tinker_cookbook.rl.rollouts import do_single_rollout

from elt_rl.destinations import PostgresDestination
from elt_rl.env import EltEnvGroupBuilder, EltRollout
from elt_rl.sources import GUARD

pytestmark = pytest.mark.services


@pytest.fixture
async def rollout(require_sandbox, books):
    r = await EltRollout.create(books, PostgresDestination.from_env(), CONFIG)
    yield r
    await r.cleanup()


async def test_claiming_success_earns_nothing(rollout):
    script = [
        {"role": "assistant", "content": "All tables are built and verified. TASK COMPLETE."},
        call("submit"),
    ]
    step, _ = await run_episode(rollout, script)
    assert step.reward == 0.0


async def test_tables_typed_in_by_hand_dont_count(rollout):
    # the right tables exist in the agent's warehouse, but nothing in elt/ rebuilds them
    sql = ";\n".join(f'CREATE TABLE books."{m}" AS {s}' for m, s in ORACLE_MODELS.items())
    script = [
        call(
            "bash", command=f"cat > /tmp/l.yaml <<'EOF'\n{ORACLE_LOAD}\nEOF\nelt-load /tmp/l.yaml"
        ),
        call("bash", command=f"psql -v ON_ERROR_STOP=1 <<'EOF'\n{sql};\nEOF"),
        call("submit"),
    ]
    step, _ = await run_episode(rollout, script)
    assert step.reward == 0.0


def _hardcoded_sql(books, model: str) -> str:
    """the exact answer as a VALUES list, what a model could paste after working it out"""
    gt = books.ground_truth(model)

    def lit(v):
        return "NULL" if v != v else "'" + str(v).replace("'", "''") + "'"  # noqa: PLR0124

    rows = ",\n".join("(" + ", ".join(lit(v) for v in row) + ")" for row in gt.itertuples(False))
    cols = ", ".join(f'"{c.lower()}"' for c in gt.columns)
    # a token reference to a source table, which the source check alone would accept
    return f"SELECT * FROM (VALUES {rows}) AS t({cols}) WHERE EXISTS (SELECT 1 FROM books.book)"


async def test_hardcoded_answer_table_scores_zero(rollout, books):
    models = dict(ORACLE_MODELS)
    models["publishers"] = _hardcoded_sql(books, "publishers")
    script = [call("write_file", path="elt/load.yaml", content=ORACLE_LOAD)]
    script += [
        call("write_file", path=f"elt/dbt/models/{m}.sql", content=s) for m, s in models.items()
    ]
    script.append(call("submit"))
    step, _ = await run_episode(rollout, script)

    publishers = next(m for m in rollout.reward.last.grade.models if m.model == "publishers")
    assert not publishers.unmatched and not publishers.missed  # it really was the right table
    assert publishers.error and "hard-coded" in publishers.error
    assert step.metrics["replay/hardcoded_models"] == 1.0
    assert step.reward == pytest.approx(0.2 + 0.8 * 2 / 3)  # the two real models still count


async def test_honest_models_are_not_flagged(rollout):
    script = [call("write_file", path="elt/load.yaml", content=ORACLE_LOAD)]
    script += [
        call("write_file", path=f"elt/dbt/models/{m}.sql", content=s)
        for m, s in reference_sql("books").items()
    ]
    script.append(call("submit"))
    step, _ = await run_episode(rollout, script)
    assert step.metrics["replay/hardcoded_models"] == 0.0 and step.reward == pytest.approx(1.0)


async def test_faking_tools_in_own_sandbox_doesnt_help(rollout):
    # breaks its load.yaml, then swaps elt-load for a fake that always "succeeds". replay runs
    # in a clean container, so only the real tools count
    script = [
        call("write_file", path="elt/load.yaml", content="sources: {}\nstreams: []\n"),
        call(
            "bash",
            command="printf '#!/bin/sh\\necho 15/15 streams loaded\\n' > /usr/local/bin/elt-load"
            " && elt-load elt/load.yaml",
        ),
        call("submit"),
    ]
    step, outputs = await run_episode(rollout, script)
    assert "15/15" in json.loads(outputs[1])["stdout"]  # the fake fooled its own shell
    assert step.reward == 0.0 and step.metrics["replay/load_ok"] == 0.0


async def test_oversized_submission_is_rejected(rollout):
    script = [
        call("write_file", path="elt/load.yaml", content=ORACLE_LOAD),
        call("bash", command="head -c 25000000 /dev/urandom > elt/blob.bin"),
        call("submit"),
    ]
    step, _ = await run_episode(rollout, script)
    assert step.reward == 0.0 and step.metrics["replay/error"] == 1.0


async def test_read_only_source_login_cant_write(rollout):
    script = [
        call(
            "bash",
            command="PGPASSWORD=reader psql -h elt-postgres -U elt_reader -d books "
            "-c 'delete from address'",
        ),
        call("submit"),
    ]
    _, outputs = await run_episode(rollout, script)
    assert "permission denied" in json.loads(outputs[0])["stderr"]


TAMPERS = {
    # mongo has no logins at all
    "mongo": "python3 -c \"import pymongo; pymongo.MongoClient('mongodb://elt-mongodb:27017/"
    "?directConnection=true').books.author.delete_one({})\"",
    # the superuser password is ELT-Bench's public default, a model could know it
    "postgres superuser": "PGPASSWORD=testelt psql -h elt-postgres -U postgres -d books "
    "-c 'delete from address where address_id = 1'",
    # free localstack doesn't check s3 credentials
    "s3": "printf 'x\\n' > /tmp/c.csv && AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test "
    "AWS_DEFAULT_REGION=us-west-2 aws --endpoint-url http://elt-localstack:4566 s3 cp "
    "/tmp/c.csv s3://books-bucket/country.csv",
}


@pytest.mark.parametrize("how", TAMPERS)
async def test_tampering_with_sources_zeroes_the_group(require_sandbox, books, renderer, how):  # noqa: F811
    builder = EltEnvGroupBuilder(books, renderer, group_size=1, config=CONFIG)
    try:
        (env,) = await builder.make_envs()
        script = [
            call("write_file", path="elt/load.yaml", content=ORACLE_LOAD),
            call("bash", command=TAMPERS[how]),
            call("submit"),
        ]
        trajectory = await do_single_rollout(ScriptedPolicy(renderer, script), env)
        env_reward = sum(t.reward for t in trajectory.transitions)
        ((group_reward, metrics),) = await builder.compute_group_rewards([trajectory], [env])
    finally:
        await builder.cleanup()

    assert env_reward > 0  # it did earn load credit
    assert metrics["source_tampered"] == 1.0
    assert env_reward + group_reward == pytest.approx(0.0)  # and lost it
    assert GUARD.changed(books) == []  # sources are back for the next group
