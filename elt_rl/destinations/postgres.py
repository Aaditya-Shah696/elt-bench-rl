"""postgres warehouse, one database + one login per rollout.

defaults match elt-warehouse in docker/compose.yml, override with the ELT_WAREHOUSE_* env vars.
the grader connects as the admin from the host, agents connect as their own role from the
sandbox network
"""

import logging
import os
import re
import secrets
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from elt_rl.destinations.base import Namespace

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^[a-z0-9_]{1,50}$")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


class PostgresConnector:
    def __init__(self, engine: Engine, namespace: Namespace):
        self._engine = engine
        self._schema = namespace.schema

    def _qualified(self, table: str) -> str:
        return f"{_quote(self._schema)}.{_quote(table)}"

    def list_tables(self) -> list[str]:
        # views count too, dbt makes views by default
        sql = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :schema AND table_type IN ('BASE TABLE', 'VIEW')"
        )
        with self._engine.connect() as c:
            return [row[0] for row in c.execute(sql, {"schema": self._schema})]

    def count_rows(self, table: str) -> int:
        with self._engine.connect() as c:
            return c.execute(text(f"SELECT count(*) FROM {self._qualified(table)}")).scalar_one()

    def fetch_table(self, table: str) -> pd.DataFrame:
        with self._engine.connect() as c:
            return pd.read_sql(text(f"SELECT * FROM {self._qualified(table)}"), c)

    def close(self) -> None:
        self._engine.dispose()


@dataclass(frozen=True)
class PostgresDestination:
    host: str = "localhost"
    port: int = 5434
    user: str = "postgres"
    password: str = "warehouse"
    # how sandboxes reach it on the docker network
    network_host: str = "elt-warehouse"
    network_port: int = 5432
    name: str = "postgres"

    @classmethod
    def from_env(cls) -> "PostgresDestination":
        return cls(
            host=os.environ.get("ELT_WAREHOUSE_HOST", cls.host),
            port=int(os.environ.get("ELT_WAREHOUSE_PORT", cls.port)),
            user=os.environ.get("ELT_WAREHOUSE_USER", cls.user),
            password=os.environ.get("ELT_WAREHOUSE_PASSWORD", cls.password),
        )

    def url(self, database: str) -> str:
        return (
            f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{database}"
        )

    def _admin(self, *statements: str, database: str = "postgres") -> None:
        # create/drop database can't run inside a transaction
        engine = create_engine(self.url(database), isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as c:
                for sql in statements:
                    c.execute(text(sql))
        finally:
            engine.dispose()

    @staticmethod
    def _database_name(run_id: str) -> str:
        if not _RUN_ID.match(run_id):
            raise ValueError(f"run_id must match {_RUN_ID.pattern}: {run_id!r}")
        return f"elt_{run_id}"

    def _new_namespace(self, run_id: str, schema: str) -> Namespace:
        name = self._database_name(run_id)
        # role has the same name as the database, it owns it and nothing else
        return Namespace(name, schema, user=name, password=secrets.token_hex(16))

    def _create_owned_database(self, ns: Namespace, template: str) -> None:
        assert ns.user and ns.password
        self._admin(f"CREATE ROLE {_quote(ns.user)} LOGIN PASSWORD '{ns.password}'")
        try:
            self._admin(
                f"CREATE DATABASE {_quote(ns.database)} OWNER {_quote(ns.user)} {template}",
                # nobody else gets to connect, the admin still can
                f"REVOKE ALL ON DATABASE {_quote(ns.database)} FROM PUBLIC",
            )
        except Exception:
            self._admin(f"DROP ROLE IF EXISTS {_quote(ns.user)}")
            raise

    def provision(self, task: str, run_id: str) -> Namespace:
        ns = self._new_namespace(run_id, task)
        # C collation so text sorts like snowflake, needs template0 to set it
        self._create_owned_database(
            ns, "TEMPLATE template0 ENCODING 'UTF8' LC_COLLATE 'C' LC_CTYPE 'C'"
        )
        self._admin(
            f"CREATE SCHEMA {_quote(ns.schema)} AUTHORIZATION {_quote(ns.user or '')}",
            database=ns.database,
        )
        return ns

    def clone(self, source: Namespace, run_id: str) -> Namespace:
        # copying from a template is way faster than reloading, source can't have open connections
        ns = self._new_namespace(run_id, source.schema)
        self._create_owned_database(ns, f"TEMPLATE {_quote(source.database)}")
        # hand the copied schema + tables to the new role. not REASSIGN OWNED, that also moves
        # shared stuff like the source database itself
        owner = _quote(ns.user or "")
        self._admin(
            f"ALTER SCHEMA {_quote(ns.schema)} OWNER TO {owner}",
            f"""DO $$ DECLARE r record; BEGIN
                FOR r IN SELECT c.relname FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = '{ns.schema}' AND c.relkind IN ('r', 'v', 'm', 'p')
                LOOP EXECUTE format('ALTER TABLE %I.%I OWNER TO %s',
                                    '{ns.schema}', r.relname, '{owner}');
                END LOOP; END $$""",
            database=ns.database,
        )
        return ns

    def teardown(self, namespace: Namespace) -> None:
        self._admin(f"DROP DATABASE IF EXISTS {_quote(namespace.database)} WITH (FORCE)")
        if namespace.user:
            try:
                self._admin(f"DROP ROLE IF EXISTS {_quote(namespace.user)}")
            except Exception as e:  # noqa: BLE001 - a leftover role is harmless, don't fail cleanup
                logger.warning("couldn't drop role %s: %s", namespace.user, e)

    def connect(self, namespace: Namespace) -> PostgresConnector:
        return PostgresConnector(create_engine(self.url(namespace.database)), namespace)

    def agent_env(self, namespace: Namespace) -> dict[str, str]:
        user, password = namespace.user or self.user, namespace.password or self.password
        return {
            "ELT_WAREHOUSE_URL": (
                f"postgresql+psycopg2://{user}:{password}"
                f"@{self.network_host}:{self.network_port}/{namespace.database}"
            ),
            "ELT_WAREHOUSE_SCHEMA": namespace.schema,
            # so plain `psql` just works
            "PGHOST": self.network_host,
            "PGPORT": str(self.network_port),
            "PGDATABASE": namespace.database,
            "PGUSER": user,
            "PGPASSWORD": password,
        }

    def agent_config(self, namespace: Namespace) -> dict[str, Any]:
        return {
            "type": "postgres",
            "host": self.network_host,
            "port": self.network_port,
            "database": namespace.database,
            "schema": namespace.schema,
            "user": namespace.user or self.user,
            "password": namespace.password or self.password,
        }

    def dbt_profile(self, namespace: Namespace) -> dict[str, Any]:
        c = self.agent_config(namespace)
        output = {
            "type": "postgres",
            "host": c["host"],
            "port": c["port"],
            "user": c["user"],
            "password": c["password"],
            "dbname": c["database"],
            "schema": c["schema"],
            "threads": 4,
        }
        return {"elt": {"target": "dev", "outputs": {"dev": output}}}

    def agent_files(self, namespace: Namespace) -> dict[str, str]:
        return {}

    def prompt_note(self) -> str:
        return (
            "The warehouse is PostgreSQL. `psql` is already connected to it, and `run_sql` "
            "(the tool, or the run_sql shell command) runs read-only queries."
        )
