"""local docker sandbox, one container per rollout.

implements tinker-cookbook's SandboxInterface (it only ships modal + sandboxfusion) by shelling
out to the docker cli, so no extra python deps. containers join the internal sandbox network,
which reaches the sources + warehouse but not the internet
"""

import asyncio
import io
import shlex
import tarfile
import uuid

from tinker_cookbook.sandbox import SandboxResult

IMAGE = "elt-rl-agent"
NETWORK = "elt-docker_sandbox"
LABEL = "elt-rl.sandbox"
DEFAULT_MAX_OUTPUT = 128 * 1024


async def _docker(*args: str, stdin: bytes | None = None, timeout: float | None = None):
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out, err


def _text(raw: bytes, limit: int) -> str:
    return raw[:limit].decode(errors="replace")


class DockerSandbox:
    def __init__(self, container: str):
        self._container = container

    @classmethod
    async def create(
        cls,
        env: dict[str, str] | None = None,
        image: str = IMAGE,
        network: str = NETWORK,
        memory: str = "4g",
    ) -> "DockerSandbox":
        name = f"elt-sbx-{uuid.uuid4().hex[:12]}"
        args = ["run", "-d", "--rm", "--name", name, "--network", network, "--label", LABEL]
        args += ["--memory", memory]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        code, _, err = await _docker(*args, image, timeout=120)
        if code != 0:
            raise RuntimeError(f"docker run failed: {err.decode(errors='replace')}")
        return cls(name)

    @property
    def sandbox_id(self) -> str:
        return self._container

    async def send_heartbeat(self, timeout: int = 30) -> None:
        pass  # local containers don't expire

    async def run_command(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 60,
        max_output_bytes: int | None = None,
    ) -> SandboxResult:
        limit = max_output_bytes or DEFAULT_MAX_OUTPUT
        args = ["exec"]
        if workdir:
            args += ["-w", workdir]
        # timeout runs inside the container so a hung command actually dies, exit code 124
        args += [self._container, "timeout", "-k", "5", str(timeout), "bash", "-c", command]
        try:
            code, out, err = await _docker(*args, timeout=timeout + 30)
        except TimeoutError:
            return SandboxResult(stdout="", stderr=f"timed out after {timeout}s", exit_code=124)
        return SandboxResult(stdout=_text(out, limit), stderr=_text(err, limit), exit_code=code)

    async def read_file(
        self, path: str, max_bytes: int | None = None, timeout: int = 60
    ) -> SandboxResult:
        cmd = (
            f"head -c {int(max_bytes)} {shlex.quote(path)}"
            if max_bytes
            else "cat " + shlex.quote(path)
        )
        return await self.run_command(cmd, timeout=timeout, max_output_bytes=max_bytes)

    async def write_file(
        self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60
    ) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        q = shlex.quote(path)
        script = f'mkdir -p "$(dirname {q})" && cat > {q}' + (
            f" && chmod +x {q}" if executable else ""
        )
        code, out, err = await _docker(
            "exec", "-i", self._container, "sh", "-c", script, stdin=data, timeout=timeout
        )
        return SandboxResult(stdout=_text(out, 4096), stderr=_text(err, 4096), exit_code=code)

    async def put_files(self, files: dict[str, str | bytes], timeout: int = 60) -> None:
        """write a bunch of files in one go (absolute paths), faster than write_file each"""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for path, content in files.items():
                data = content.encode() if isinstance(content, str) else content
                info = tarfile.TarInfo(path.lstrip("/"))
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        await self.put_tar(buf.getvalue(), "/", timeout=timeout)

    async def put_tar(self, data: bytes, dest: str, timeout: int = 60) -> None:
        code, _, err = await _docker(
            "exec",
            "-i",
            self._container,
            "sh",
            "-c",
            f"mkdir -p {shlex.quote(dest)} && tar -xf - -C {shlex.quote(dest)}",
            stdin=data,
            timeout=timeout,
        )
        if code != 0:
            raise RuntimeError(f"untar into sandbox failed: {err.decode(errors='replace')}")

    async def get_tar(self, path: str, max_bytes: int, timeout: int = 60) -> bytes | None:
        """tar of a directory, None if it doesn't exist. raises if bigger than max_bytes"""
        parent, _, name = path.rstrip("/").rpartition("/")
        q, qp = shlex.quote(name), shlex.quote(parent or "/")
        code, out, err = await _docker(
            "exec",
            self._container,
            "sh",
            "-c",
            f"cd {qp} && [ -d {q} ] || exit 3; tar -cf - {q} | head -c {max_bytes + 1}",
            timeout=timeout,
        )
        if code == 3:
            return None
        if code != 0:
            raise RuntimeError(f"tar from sandbox failed: {err.decode(errors='replace')}")
        if len(out) > max_bytes:
            raise ValueError(f"{path} is bigger than {max_bytes} bytes")
        return out

    async def cleanup(self) -> None:
        await _docker("rm", "-f", self._container, timeout=60)
