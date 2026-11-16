"""snowflake warehouse, one database + one key-pair login per rollout.

needs the one-time setup from scripts/snowflake_setup.py, which writes the SNOWFLAKE_* values
to .env. the grader connects as ELT_RL_ADMIN from the host. sandboxes connect as their own
service user through the egress proxy (the only way out of the sandbox network)
"""

import atexit
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from sqlalchemy import Engine, create_engine, text

from elt_rl.destinations.base import Namespace
from elt_rl.paths import REPO_DIR

logger = logging.getLogger(__name__)

_RUN_ID = re.compile(r"^[a-z0-9_]{1,50}$")
SANDBOX_KEY = "/root/.snowflake/rsa_key.p8"
PROXY = "http://elt-egress:3128"
LOCAL_HOSTS = "elt-postgres,elt-mongodb,elt-localstack,elt-api,elt-files,elt-warehouse,localhost"


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _dotenv() -> dict[str, str]:
    env = REPO_DIR / ".env"
    if not env.exists():
        return {}
    pairs = (line.split("=", 1) for line in env.read_text().splitlines() if "=" in line)
    return {k.strip(): v.strip() for k, v in pairs if not k.strip().startswith("#")}


def _new_key() -> str:
    """private key (pem) for one rollout's login"""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()


def _public_body(private_pem: str) -> str:
    """the base64 public key snowflake wants, no header lines"""
    key = serialization.load_pem_private_key(private_pem.encode(), password=None)
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return "".join(pem.decode().strip().splitlines()[1:-1])


class SnowflakeConnector:
    def __init__(self, engine: Engine, namespace: Namespace, owns_engine: bool = True):
        self._engine = engine
        self._ns = namespace
        self._owns_engine = owns_engine

    def _qualified(self, table: str) -> str:
        return f"{_q(self._ns.database)}.{_q(self._ns.schema)}.{_q(table)}"

    def list_tables(self) -> list[str]:
        sql = text(
            f"SELECT table_name FROM {_q(self._ns.database)}.information_schema.tables "
            "WHERE table_schema = :schema AND table_type IN ('BASE TABLE', 'VIEW')"
        )
        with self._engine.connect() as c:
            return [row[0] for row in c.execute(sql, {"schema": self._ns.schema})]

    def count_rows(self, table: str) -> int:
        with self._engine.connect() as c:
            return c.execute(text(f"SELECT count(*) FROM {self._qualified(table)}")).scalar_one()

    def fetch_table(self, table: str) -> pd.DataFrame:
        with self._engine.connect() as c:
            return pd.read_sql(text(f"SELECT * FROM {self._qualified(table)}"), c)

    def close(self) -> None:
        if self._owns_engine:
            self._engine.dispose()


# one pooled admin engine per destination. every new snowflake connection is a full key-pair
# login (a second or two), so reusing them matters when a rollout does a dozen admin steps
_SHARED: dict["SnowflakeDestination", Engine] = {}
_SHARED_LOCK = threading.Lock()


@atexit.register
def _close_shared() -> None:
    for engine in _SHARED.values():
        engine.dispose()
    _SHARED.clear()


@dataclass(frozen=True)
class SnowflakeDestination:
    account: str
    user: str
    role: str
    warehouse: str
    private_key_path: str
    name: str = "snowflake"

    @classmethod
    def from_env(cls) -> "SnowflakeDestination":
        env = {**_dotenv(), **os.environ}
        keys = ["ACCOUNT", "USER", "ROLE", "WAREHOUSE", "PRIVATE_KEY_PATH"]
        missing = [f"SNOWFLAKE_{k}" for k in keys if not env.get(f"SNOWFLAKE_{k}")]
        if missing:
            raise RuntimeError(
                f"snowflake isn't set up ({', '.join(missing)} missing), "
                "run scripts/snowflake_setup.py first"
            )
        return cls(*(env[f"SNOWFLAKE_{k}"] for k in keys))

    # host side, admin login

    def _engine(self, database: str | None = None) -> Engine:
        url = f"snowflake://{self.user}@{self.account}/"
        if database:
            url += _q(database)
        return create_engine(
            url,
            connect_args={
                "private_key_file": str(Path(self.private_key_path).expanduser()),
                "authenticator": "SNOWFLAKE_JWT",
                "role": self.role,
                "warehouse": self.warehouse,
            },
        )

    def _shared(self) -> Engine:
        with _SHARED_LOCK:
            if self not in _SHARED:
                _SHARED[self] = self._engine()
            return _SHARED[self]

    def _admin(self, *statements: str) -> None:
        # every statement names its database explicitly, so one shared engine does for all
        with self._shared().connect() as c:
            for sql in statements:
                c.execute(text(sql))

    def _new_namespace(self, run_id: str, schema: str) -> Namespace:
        if not _RUN_ID.match(run_id):
            raise ValueError(f"run_id must match {_RUN_ID.pattern}: {run_id!r}")
        name = f"ELT_{run_id}".upper()
        # for snowflake the namespace secret is the login's private key, not a password
        return Namespace(name, schema.upper(), user=name, password=_new_key())

    def _create_login(self, ns: Namespace) -> None:
        public = _public_body(ns.password or "")
        role, user = _q(ns.user or ""), _q(ns.user or "")
        self._admin(
            f"CREATE ROLE {role}",
            # admin keeps access through the role hierarchy, for grading and cleanup
            f"GRANT ROLE {role} TO ROLE {_q(self.role)}",
            f"GRANT USAGE ON WAREHOUSE {_q(self.warehouse)} TO ROLE {role}",
            f"CREATE USER {user} TYPE = SERVICE DEFAULT_ROLE = {role} "
            f"DEFAULT_WAREHOUSE = {_q(self.warehouse)} RSA_PUBLIC_KEY = '{public}'",
            f"GRANT ROLE {role} TO USER {user}",
        )

    def provision(self, task: str, run_id: str) -> Namespace:
        ns = self._new_namespace(run_id, task)
        self._create_login(ns)
        try:
            self._admin(
                f"CREATE DATABASE {_q(ns.database)}",
                f"CREATE SCHEMA {_q(ns.database)}.{_q(ns.schema)}",
                f"GRANT OWNERSHIP ON SCHEMA {_q(ns.database)}.{_q(ns.schema)} "
                f"TO ROLE {_q(ns.user or '')}",
                f"GRANT OWNERSHIP ON DATABASE {_q(ns.database)} TO ROLE {_q(ns.user or '')}",
            )
        except Exception:
            self.teardown(ns)
            raise
        return ns

    def clone(self, source: Namespace, run_id: str) -> Namespace:
        # zero copy clone, then hand the copy and everything in it to the new login
        ns = self._new_namespace(run_id, source.schema)
        self._create_login(ns)
        db, schema, role = _q(ns.database), _q(ns.schema), _q(ns.user or "")
        try:
            self._admin(
                f"CREATE DATABASE {db} CLONE {_q(source.database)}",
                f"GRANT OWNERSHIP ON ALL TABLES IN SCHEMA {db}.{schema} TO ROLE {role} "
                "COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON ALL VIEWS IN SCHEMA {db}.{schema} TO ROLE {role} "
                "COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON SCHEMA {db}.{schema} TO ROLE {role} COPY CURRENT GRANTS",
                f"GRANT OWNERSHIP ON DATABASE {db} TO ROLE {role} COPY CURRENT GRANTS",
            )
        except Exception:
            self.teardown(ns)
            raise
        return ns

    def teardown(self, namespace: Namespace) -> None:
        for sql in (
            f"DROP DATABASE IF EXISTS {_q(namespace.database)}",
            f"DROP USER IF EXISTS {_q(namespace.user or '')}",
            f"DROP ROLE IF EXISTS {_q(namespace.user or '')}",
        ):
            try:
                self._admin(sql)
            except Exception as e:  # noqa: BLE001 - keep cleaning up the rest
                logger.warning("snowflake cleanup step failed (%s): %s", sql, e)

    def connect(self, namespace: Namespace) -> SnowflakeConnector:
        # the connector qualifies every table with its database, so the shared engine works
        return SnowflakeConnector(self._shared(), namespace, owns_engine=False)

    # what the sandbox gets

    def agent_env(self, namespace: Namespace) -> dict[str, str]:
        url = (
            f"snowflake://{namespace.user}@{self.account}/{namespace.database}/{namespace.schema}"
            f"?warehouse={self.warehouse}&role={namespace.user}"
        )
        return {
            "ELT_WAREHOUSE_URL": url,
            "ELT_WAREHOUSE_SCHEMA": namespace.schema,
            "ELT_WAREHOUSE_KEY_FILE": SANDBOX_KEY,
            # snowflake is on the internet, the proxy is the only way there
            "HTTPS_PROXY": PROXY,
            "https_proxy": PROXY,
            "NO_PROXY": LOCAL_HOSTS,
            "no_proxy": LOCAL_HOSTS,
        }

    def agent_files(self, namespace: Namespace) -> dict[str, str]:
        return {SANDBOX_KEY: namespace.password or ""}

    def agent_config(self, namespace: Namespace) -> dict[str, Any]:
        return {
            "type": "snowflake",
            "account": self.account,
            "user": namespace.user,
            "role": namespace.user,
            "warehouse": self.warehouse,
            "database": namespace.database,
            "schema": namespace.schema,
            "private_key_path": SANDBOX_KEY,
        }

    def dbt_profile(self, namespace: Namespace) -> dict[str, Any]:
        output = {
            "type": "snowflake",
            "account": self.account,
            "user": namespace.user,
            "role": namespace.user,
            "private_key_path": SANDBOX_KEY,
            "warehouse": self.warehouse,
            "database": namespace.database,
            "schema": namespace.schema,
            "threads": 4,
        }
        return {"elt": {"target": "dev", "outputs": {"dev": output}}}

    def prompt_note(self) -> str:
        return (
            "The warehouse is Snowflake (unquoted names are case-insensitive). Use `run_sql` "
            "(the tool, or the run_sql shell command) to query it."
        )
