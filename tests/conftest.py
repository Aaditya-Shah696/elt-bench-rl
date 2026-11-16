import socket
import subprocess
from pathlib import Path

import pytest

from elt_rl.tasks import EltTask, load_task

FIXTURES = Path(__file__).parent / "fixtures"
SERVICE_PORTS = {
    "source postgres": 5433,
    "warehouse": 5434,
    "mongodb": 27017,
    "s3": 4566,
    "api": 5005,
    "files": 8008,
}


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


@pytest.fixture(scope="session")
def require_services() -> None:
    down = [name for name, port in SERVICE_PORTS.items() if not _port_open(port)]
    if down:
        pytest.skip(f"services not running ({', '.join(down)}); run scripts/services.sh up")


@pytest.fixture(scope="session")
def require_sandbox(require_services) -> None:
    image = subprocess.run(
        ["docker", "image", "inspect", "elt-rl-agent"], check=False, capture_output=True
    )
    if image.returncode != 0:
        pytest.skip("sandbox image missing, run: docker build -t elt-rl-agent docker/agent")


@pytest.fixture(scope="session")
def books() -> EltTask:
    return load_task("books")


def reference_sql(task: str) -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted((FIXTURES / f"{task}_reference").glob("*.sql"))}
