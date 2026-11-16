"""keeping the source systems honest.

the agent gets the source connection details, so it could change source data. that can't give
it the answers, but it can sabotage the other rollouts in its group, and since rewards are
compared within a group that would raise its own advantage. so:

- prevent: the agent's postgres login is a read-only role (api + flat files are read-only mounts)
- detect: mongo has no logins and free localstack can't enforce s3 permissions, so a group's
  sources get fingerprinted before and after. if anything changed the whole group scores 0
  (nothing to gain from sabotage) and the task's sources get reloaded
"""

import logging
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from pymongo import MongoClient
from sqlalchemy import create_engine, text

from elt_rl.paths import REPO_DIR
from elt_rl.tasks import EltTask

logger = logging.getLogger(__name__)

READER_USER = "elt_reader"
READER_PASSWORD = "reader"
SOURCE_PG = "postgresql+psycopg://postgres:testelt@localhost:5433"
MONGO = "mongodb://localhost:27017/?directConnection=true&serverSelectionTimeoutMS=5000"
S3 = "http://localhost:4566"


def grant_reader(database: str) -> None:
    """read-only login on one source database, created if missing"""
    engine = create_engine(f"{SOURCE_PG}/{database}", isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as c:
            exists = c.execute(
                text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": READER_USER}
            ).first()
            if not exists:
                c.execute(text(f"CREATE ROLE {READER_USER} LOGIN PASSWORD '{READER_PASSWORD}'"))
            c.execute(text(f'GRANT CONNECT ON DATABASE "{database}" TO {READER_USER}'))
            c.execute(text(f"GRANT USAGE ON SCHEMA public TO {READER_USER}"))
            c.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {READER_USER}"))
    finally:
        engine.dispose()


def _postgres(task: EltTask) -> dict[str, str]:
    cfg = (task.config.get("postgres") or {}).get("config")
    if not cfg:
        return {}
    engine = create_engine(f"{SOURCE_PG}/{cfg['database']}")
    try:
        with engine.connect() as c:
            # write counters move on any insert/update/delete, plus a plain count
            rows = c.execute(
                text(
                    "SELECT relname, n_tup_ins, n_tup_upd, n_tup_del FROM pg_stat_user_tables "
                    "WHERE relname = ANY(:t)"
                ),
                {"t": cfg["tables"]},
            ).all()
            out = {f"pg:{r[0]}": f"{r[1]}/{r[2]}/{r[3]}" for r in rows}
            for t in cfg["tables"]:
                out[f"pg:{t}:count"] = str(c.execute(text(f'SELECT count(*) FROM "{t}"')).scalar())
            return out
    finally:
        engine.dispose()


def _mongo(task: EltTask) -> dict[str, str]:
    cfg = (task.config.get("mongodb") or {}).get("config")
    if not cfg:
        return {}
    client = MongoClient(MONGO)
    try:
        hashes = client[cfg["database"]].command("dbHash", collections=cfg["tables"])
    finally:
        client.close()
    return {f"mongo:{k}": v for k, v in hashes["collections"].items()} | {
        "mongo:collections": ",".join(sorted(hashes["collections"]))
    }


def _s3(task: EltTask) -> dict[str, str]:
    data = (task.config.get("aws_s3") or {}).get("data") or []
    out: dict[str, str] = {}
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    for bucket in sorted({urlparse(d["path"]).netloc for d in data}):
        resp = requests.get(f"{S3}/{bucket}", params={"list-type": "2"}, timeout=30)
        resp.raise_for_status()
        for obj in ET.fromstring(resp.content).findall("s3:Contents", ns):
            key = obj.findtext("s3:Key", namespaces=ns)
            out[f"s3:{bucket}/{key}"] = f"{obj.findtext('s3:ETag', namespaces=ns)}"
    return out


def fingerprint(task: EltTask) -> dict[str, str]:
    return _postgres(task) | _mongo(task) | _s3(task)


class SourceGuard:
    """remembers each task's source fingerprint and checks it after every group"""

    def __init__(self) -> None:
        self._baseline: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()

    def baseline(self, task: EltTask) -> None:
        with self._lock:
            if task.name not in self._baseline:
                self._baseline[task.name] = fingerprint(task)

    def changed(self, task: EltTask) -> list[str]:
        before = self._baseline.get(task.name)
        if before is None:
            return []
        now = fingerprint(task)
        return sorted(k for k in before.keys() | now.keys() if before.get(k) != now.get(k))

    def restore(self, task: EltTask) -> None:
        """reload the task's sources from the original files, then take a new baseline"""
        with self._lock:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_DIR / "scripts" / "load_sources.py"),
                    "--only",
                    task.name,
                ],
                check=True,
                capture_output=True,
            )
            self._baseline[task.name] = fingerprint(task)


GUARD = SourceGuard()
