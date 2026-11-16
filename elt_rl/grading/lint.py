"""catch target tables typed in by hand instead of computed from the sources.

the model can't see the answers, but it could work them out with run_sql and paste the result
into a model as literals. that table would match, without a pipeline that works on new data.
a target model gets flagged when its sql reads no source table at all, or when a sql file in
the dbt project holds far more literals than real transformations need (honest models for
books have 7-9, hard-coding even a 10 row table takes ~50). flagged models score 0.

it's a heuristic, a determined model could still smuggle values in, see the design doc
"""

import io
import re
import tarfile
from pathlib import PurePosixPath

from elt_rl.tasks import EltTask

MAX_LITERALS = 40
_LITERAL = re.compile(r"'(?:[^']|'')*'|(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _sql_files(submission_tar: bytes) -> dict[str, str]:
    """path -> text for every .sql file in the submitted dbt project"""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(submission_tar)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".sql") and "/dbt/" in member.name:
                f = tar.extractfile(member)
                if f is not None:
                    out[member.name] = _COMMENT.sub("", f.read().decode(errors="replace"))
    return out


def hardcoded_models(task: EltTask, submission_tar: bytes | None) -> dict[str, str]:
    """target model -> why it looks hand typed"""
    if submission_tar is None:
        return {}
    files = _sql_files(submission_tar)
    models = {PurePosixPath(p).stem.lower(): sql for p, sql in files.items() if "/models/" in p}
    sources = {t.lower() for t in task.expected_row_counts}
    word = re.compile(r"[a-z_][a-z0-9_]*")

    # a macro (not a model) full of values counts against every model, any of them could use it
    stuffed = [
        p
        for p, sql in files.items()
        if "/models/" not in p and len(_LITERAL.findall(sql)) > MAX_LITERALS
    ]
    flagged = {}
    for spec in task.models:
        sql = models.get(spec.name.lower())
        if sql is None:
            continue  # no file, so dbt didn't build it and grading already gives 0
        names = set(word.findall(sql.lower()))
        if not (names & sources or "ref(" in sql or "source(" in sql):
            flagged[spec.name] = "reads no source table"
        elif len(_LITERAL.findall(sql)) > MAX_LITERALS or stuffed:
            flagged[spec.name] = f"too many literal values ({', '.join(stuffed) or 'model'})"
    return flagged
