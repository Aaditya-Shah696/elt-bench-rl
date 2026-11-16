"""snowflake end to end. skips until scripts/snowflake_setup.py has been run.

the oracle episode is the proof that grading on snowflake is right, not just running: the same
books pipeline (translated to snowflake sql) must score exactly 1.0 there too
"""

import json

import pytest
from conftest import FIXTURES, reference_sql
from sqlalchemy import text
from test_env_integration import call, run_episode

from elt_rl.destinations import SnowflakeDestination
from elt_rl.env import EltRollout, EnvConfig
from elt_rl.grading.replay import new_run_id
from elt_rl.sandbox import DockerSandbox

pytestmark = pytest.mark.services


@pytest.fixture(scope="module")
def snowflake() -> SnowflakeDestination:
    try:
        return SnowflakeDestination.from_env()
    except RuntimeError as e:
        pytest.skip(str(e))


def to_snowflake(sql: str) -> str:
    """the two postgres-only bits in our books sql. COUNT(*) FILTER (WHERE x) -> COUNT_IF(x),
    and no COLLATE "C" (snowflake already compares text byte by byte)"""
    sql = sql.replace(' COLLATE "C"', "")
    marker = "COUNT(*) FILTER (WHERE "
    while (start := sql.find(marker)) != -1:
        i, depth = start + len(marker), 1
        while depth:  # find the matching close paren, the condition has its own parens
            depth += {"(": 1, ")": -1}.get(sql[i], 0)
            i += 1
        cond = sql[start + len(marker) : i - 1]
        sql = sql[:start] + f"COUNT_IF({cond})" + sql[i:]
    return sql


def test_translation():
    assert to_snowflake("COUNT(*) FILTER (WHERE EXTRACT(YEAR FROM d) = 1) / 2") == (
        "COUNT_IF(EXTRACT(YEAR FROM d) = 1) / 2"
    )
    assert "FILTER" not in "".join(to_snowflake(s) for s in reference_sql("books").values())


def test_admin_login(snowflake):
    engine = snowflake._engine()
    try:
        with engine.connect() as c:
            role, wh = c.execute(text("select current_role(), current_warehouse()")).one()
    finally:
        engine.dispose()
    assert (role, wh) == (snowflake.role, snowflake.warehouse)


def test_namespace_lifecycle(snowflake):
    ns = snowflake.provision("books", f"t_{new_run_id()}")
    try:
        connector = snowflake.connect(ns)
        try:
            assert connector.list_tables() == []
        finally:
            connector.close()
    finally:
        snowflake.teardown(ns)
    engine = snowflake._engine()
    try:
        with engine.connect() as c:
            users = c.execute(text(f"show users like '{ns.user}'")).fetchall()
    finally:
        engine.dispose()
    assert users == []


async def test_sandbox_reaches_snowflake_through_proxy(require_sandbox, snowflake):
    ns = snowflake.provision("books", f"t_{new_run_id()}")
    sandbox = await DockerSandbox.create(env=snowflake.agent_env(ns))
    try:
        await sandbox.put_files(snowflake.agent_files(ns))
        r = await sandbox.run_command("run_sql 'select current_user(), current_database()'")
        direct = await sandbox.run_command(
            "env -u HTTPS_PROXY -u https_proxy run_sql 'select 1'", timeout=60
        )
    finally:
        await sandbox.cleanup()
        snowflake.teardown(ns)
    assert ns.user in r.stdout and ns.database in r.stdout, r.stdout + r.stderr
    assert direct.exit_code != 0  # no proxy, no way out


async def test_oracle_scores_one_on_snowflake(require_sandbox, snowflake, books):
    config = EnvConfig(destination="snowflake", max_turns=30)
    rollout = await EltRollout.create(books, snowflake, config)
    load = (FIXTURES / "books_reference" / "load.yaml").read_text()
    script = [
        call("write_file", path="elt/load.yaml", content=load),
        call("bash", command="elt-load elt/load.yaml"),
    ]
    for model, sql in reference_sql("books").items():
        script.append(
            call("write_file", path=f"elt/dbt/models/{model}.sql", content=to_snowflake(sql))
        )
    script += [call("bash", command="dbt run --project-dir elt/dbt"), call("submit")]
    try:
        step, outputs = await run_episode(rollout, script, config)
    finally:
        await rollout.cleanup()

    assert json.loads(outputs[1])["exit_code"] == 0, outputs[1]
    dbt = json.loads(outputs[-2])
    assert dbt["exit_code"] == 0, dbt["stdout"][-3000:]
    assert step.reward == pytest.approx(1.0), rollout.reward.last
    assert step.metrics["task_success"] == 1.0
