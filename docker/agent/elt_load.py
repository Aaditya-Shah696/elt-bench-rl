#!/usr/bin/env python3
"""elt-load, the extract + load tool inside the sandbox.

describe what to copy in a yaml file and this copies it into the warehouse, like a tiny airbyte.
the warehouse login comes from the environment, nothing to set for it

    elt-load /workspace/elt/load.yaml

format:

    sources:                       # one entry per source system, any names you like
      postgres:   {type: postgres, host: ..., port: 5432, database: ..., user: ..., password: ...}
      mongodb:    {type: mongodb, connection_string: ..., database: ...}
      aws_s3:     {type: aws_s3, endpoint_url: ..., access_key_id: ..., secret_access_key: ..., region: ...}
      custom_api: {type: custom_api, base_url: http://elt-api:5005}
      flat_files: {type: flat_files}
    streams:                       # top level, one entry per table to copy
      - {source: postgres, table: address}                 # schema defaults to public
      - {source: mongodb, collection: author}
      - {source: aws_s3, path: "s3://bucket/country.csv"}  # csv, jsonl, json or parquet
      - {source: custom_api, path: /books/publisher}
      - {source: flat_files, path: "http://elt-files:8000/books/order_status.csv"}

types are named like the sections of config.yaml (postgres, mongodb, aws_s3, custom_api,
flat_files). each stream becomes a warehouse table named after its table, collection or file,
or `name:` if set. names are lowercased on postgres (uppercased on snowflake) so they work
unquoted. tables are replaced on every run
"""

import csv
import io
import json
import os
import sys
import time
from pathlib import PurePosixPath
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml
from elt_warehouse import warehouse_engine
from sqlalchemy import create_engine

FORMATS = ("csv", "jsonl", "json", "parquet")

# the config.yaml section names plus the obvious short forms
TYPES = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "aws_s3": "s3",
    "s3": "s3",
    "custom_api": "api",
    "api": "api",
    "flat_files": "file",
    "flat_file": "file",
    "file": "file",
}
# what each kind of stream must say, so a missing key gets a real message
NEEDS = {
    "postgres": ["table"],
    "mongodb": ["collection"],
    "s3": ["path"],
    "api": ["path"],
    "file": ["path"],
}


class ConfigError(Exception):
    pass


def read_bytes(raw: bytes, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(io.BytesIO(raw))
    if fmt == "jsonl":
        return pd.read_json(io.BytesIO(raw), lines=True)
    if fmt == "json":
        return pd.DataFrame(json.loads(raw))
    if fmt == "parquet":
        return pd.read_parquet(io.BytesIO(raw))
    raise ConfigError(f"format must be one of {FORMATS}, got {fmt!r}")


def guess_format(path: str, stream: dict) -> str:
    fmt = stream.get("format") or PurePosixPath(urlparse(path).path).suffix.lstrip(".")
    if fmt not in FORMATS:
        raise ConfigError(f"can't tell the format of {path!r}, add format: one of {FORMATS}")
    return fmt


def default_name(stream: dict) -> str:
    for key in ("table", "collection", "path"):
        if stream.get(key):
            return PurePosixPath(urlparse(str(stream[key])).path).stem.split(".")[-1]
    raise ConfigError("stream needs a name")


def extract_postgres(src: dict, stream: dict) -> pd.DataFrame:
    url = (
        f"postgresql+psycopg2://{src['user']}:{src['password']}"
        f"@{src['host']}:{src.get('port', 5432)}/{src['database']}"
    )
    schema = stream.get("schema") or src.get("schema") or "public"
    engine = create_engine(url)
    try:
        with engine.connect() as c:
            return pd.read_sql_table(stream["table"], c, schema=schema)
    finally:
        engine.dispose()


def extract_mongodb(src: dict, stream: dict) -> pd.DataFrame:
    from pymongo import MongoClient

    client = MongoClient(src["connection_string"], serverSelectionTimeoutMS=10_000)
    try:
        docs = list(client[src["database"]][stream["collection"]].find())
    finally:
        client.close()
    df = pd.DataFrame(docs)
    if "_id" in df:
        df["_id"] = df["_id"].astype(str)
    # nested stuff goes in as json text, cast it in sql if needed
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, dict | list)).any():
            df[col] = df[col].map(
                lambda v: json.dumps(v, default=str) if isinstance(v, dict | list) else v
            )
    return df


def extract_s3(src: dict, stream: dict) -> pd.DataFrame:
    import boto3

    parsed = urlparse(stream["path"])
    if parsed.scheme != "s3":
        raise ConfigError(f"s3 path must look like s3://bucket/key, got {stream['path']!r}")
    client = boto3.client(
        "s3",
        endpoint_url=src.get("endpoint_url"),
        aws_access_key_id=src.get("access_key_id"),
        aws_secret_access_key=src.get("secret_access_key"),
        region_name=src.get("region", "us-east-1"),
    )
    obj = client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
    return read_bytes(obj["Body"].read(), guess_format(stream["path"], stream))


def extract_api(src: dict, stream: dict) -> pd.DataFrame:
    path = stream["path"]
    url = path if urlparse(path).scheme else src["base_url"].rstrip("/") + "/" + path.lstrip("/")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def extract_file(src: dict, stream: dict) -> pd.DataFrame:
    url = stream["path"]
    if urlparse(url).scheme not in ("http", "https"):
        raise ConfigError(f"flat file path must be an http(s) url, got {url!r}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return read_bytes(resp.content, guess_format(url, stream))


EXTRACTORS = {
    "postgres": extract_postgres,
    "mongodb": extract_mongodb,
    "s3": extract_s3,
    "api": extract_api,
    "file": extract_file,
}


def copy_into_postgres(table, conn, keys, data_iter):
    # pandas recipe, COPY is way faster than row by row inserts
    buf = io.StringIO()
    csv.writer(buf).writerows(data_iter)
    buf.seek(0)
    cols = ", ".join('"' + k.replace('"', '""') + '"' for k in keys)
    name = '"' + table.name.replace('"', '""') + '"'
    if table.schema:
        name = '"' + table.schema.replace('"', '""') + '".' + name
    with conn.connection.cursor() as cur:
        cur.copy_expert(f"COPY {name} ({cols}) FROM STDIN WITH CSV", buf)


def _missing(v) -> bool:
    # pd.isna chokes on lists, this doesn't
    return v is None or (isinstance(v, float) and v != v)  # noqa: PLR0124 - nan check


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    # a column mixing numbers and strings can't be typed, store it as text
    for col in df.columns:
        if df[col].dtype == object:
            kinds = {type(v) for v in df[col] if not _missing(v)}
            if len(kinds) > 1:
                df[col] = df[col].map(lambda v: None if _missing(v) else str(v))
    return df


def load(df: pd.DataFrame, name: str, engine, schema: str) -> None:
    # names get the warehouse's own case (lower on postgres, upper on snowflake), so they
    # work unquoted in sql
    upper = engine.dialect.name == "snowflake"
    names = [str(c).upper() if upper else str(c).lower() for c in df.columns]
    if len(set(names)) != len(names):
        raise ConfigError(f"columns {list(df.columns)} clash once case is folded")
    df = tidy(df.set_axis(names, axis=1))
    if upper:
        from snowflake.connector.pandas_tools import write_pandas

        raw = engine.raw_connection()
        try:
            write_pandas(
                raw.driver_connection,
                df,
                table_name=name.upper(),
                schema=schema.upper(),
                auto_create_table=True,
                overwrite=True,
                quote_identifiers=True,
                use_logical_type=True,
            )
        finally:
            raw.close()
        return
    method = copy_into_postgres if engine.dialect.name == "postgresql" else "multi"
    df.to_sql(name.lower(), engine, schema=schema, if_exists="replace", index=False, method=method)


def check_spec(spec) -> tuple[dict, list]:
    """the shape of the whole file, with messages that say how to fix it"""
    if not isinstance(spec, dict):
        raise ConfigError("the file must be a mapping with `sources:` and `streams:` at the top")
    sources, streams = spec.get("sources"), spec.get("streams")
    if not isinstance(sources, dict) or not sources:
        raise ConfigError("`sources:` must be a mapping of name -> {type: ..., ...}")
    if streams is None:
        nested = [n for n, s in sources.items() if isinstance(s, dict) and "streams" in s]
        hint = f" (found it inside source {nested[0]!r}, move it out)" if nested else ""
        raise ConfigError(f"`streams:` must be a top level list next to `sources:`{hint}")
    if not isinstance(streams, list) or not all(isinstance(s, dict) for s in streams):
        raise ConfigError("`streams:` must be a list of {source: ..., ...} entries")
    return sources, streams


def resolve(sources: dict, stream: dict) -> tuple[dict, str]:
    name = stream.get("source")
    if name not in sources:
        raise ConfigError(f"source {name!r} isn't defined, sources has {list(sources)}")
    src = sources[name]
    kind = TYPES.get(str(src.get("type")).lower()) if isinstance(src, dict) else None
    if kind is None:
        raise ConfigError(
            f"source {name!r} needs type: one of postgres, mongodb, aws_s3, custom_api, flat_files"
        )
    if kind == "file" and "url" in stream and "path" not in stream:
        stream = {**stream, "path": stream["url"]}
    missing = [k for k in NEEDS[kind] if not stream.get(k)]
    if missing:
        raise ConfigError(f"a {src['type']} stream needs {missing}")
    return src, kind


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if len(argv) != 2:
        print("usage: elt-load <load.yaml>   (elt-load --help for the format)", file=sys.stderr)
        return 2
    try:
        with open(argv[1]) as f:
            sources, streams = check_spec(yaml.safe_load(f))
    except (OSError, yaml.YAMLError, ConfigError) as e:
        print(f"bad config: {e}\n(elt-load --help shows the format)", file=sys.stderr)
        return 2

    engine = warehouse_engine()
    schema = os.environ["ELT_WAREHOUSE_SCHEMA"]
    failed = 0
    for stream in streams:
        start = time.time()
        try:
            src, kind = resolve(sources, stream)
            if kind == "file" and "path" not in stream:
                stream = {**stream, "path": stream["url"]}
            name = stream.get("name") or default_name(stream)
            df = EXTRACTORS[kind](src, stream)
            load(df, name, engine, schema)
            print(f"ok    {name.lower()}: {len(df)} rows ({time.time() - start:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {stream}: {type(e).__name__}: {e}"[:2000], file=sys.stderr)
    engine.dispose()
    print(f"{len(streams) - failed}/{len(streams)} streams loaded")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
