"""the tools the model gets. all of them act inside its own sandbox"""

import json
import posixpath
import uuid
from typing import Annotated

from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

from elt_rl.workspace import SUBMISSION_DIR, WORKSPACE

MAX_OUTPUT_CHARS = 8000


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    # keep the start and the end, errors usually show up at the end
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars cut] ...\n{text[-half:]}"


def _resolve(path: str) -> str:
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(WORKSPACE, path))


class EltTools:
    def __init__(self, sandbox: SandboxInterface, command_timeout: int = 300):
        self._sandbox = sandbox
        self._command_timeout = command_timeout
        self.submitted = False

    @property
    def all(self) -> list:
        return [self.bash, self.write_file, self.read_file, self.run_sql, self.submit]

    @tool
    async def bash(
        self, command: Annotated[str, "The bash command to run, from /workspace."]
    ) -> ToolResult:
        """Run a bash command in the sandbox and get its exit code and output."""
        r = await self._sandbox.run_command(
            command, workdir=WORKSPACE, timeout=self._command_timeout
        )
        out = json.dumps(
            {"exit_code": r.exit_code, "stdout": _clip(r.stdout), "stderr": _clip(r.stderr)}
        )
        return simple_tool_result(out)

    @tool
    async def write_file(
        self,
        path: Annotated[
            str, "File path under /workspace/elt (absolute or relative to /workspace)."
        ],
        content: Annotated[str, "The full file content."],
    ) -> ToolResult:
        """Create or overwrite a file in your submission folder /workspace/elt."""
        full = _resolve(path)
        if not full.startswith(SUBMISSION_DIR + "/"):
            return simple_tool_result(f"error: you can only write under {SUBMISSION_DIR}/")
        r = await self._sandbox.write_file(full, content)
        if r.exit_code != 0:
            return simple_tool_result(f"error: {r.stderr.strip()}")
        return simple_tool_result(f"wrote {len(content)} chars to {full}")

    @tool
    async def read_file(
        self, path: Annotated[str, "File path (absolute or relative to /workspace)."]
    ) -> ToolResult:
        """Read a text file from the sandbox."""
        r = await self._sandbox.read_file(_resolve(path), max_bytes=4 * MAX_OUTPUT_CHARS)
        if r.exit_code != 0:
            return simple_tool_result(f"error: {r.stderr.strip()}")
        return simple_tool_result(_clip(r.stdout))

    @tool
    async def run_sql(
        self, query: Annotated[str, "A read-only SQL query to run against your warehouse."]
    ) -> ToolResult:
        """Run a read-only SQL query against the warehouse and see the result."""
        tmp = f"/tmp/q_{uuid.uuid4().hex[:8]}.sql"
        await self._sandbox.write_file(tmp, query)
        # run_sql in the image speaks whatever warehouse ELT_WAREHOUSE_URL points at
        r = await self._sandbox.run_command(f"run_sql -f {tmp}; rm -f {tmp}", timeout=120)
        return simple_tool_result(_clip(r.stdout + r.stderr))

    @tool
    async def submit(self) -> ToolResult:
        """Submit your pipeline (the files in /workspace/elt) for grading. This ends the task."""
        self.submitted = True
        return simple_tool_result(
            "submitted, your pipeline will now be re-run and graded", should_stop=True
        )
