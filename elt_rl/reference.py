"""the correct extract + load config for any task, built straight from its config.yaml.

used to check every task is loadable in our setup (so a failed load is the model's fault, not
ours) and later for pre-loaded snapshots when training only the transform stage
"""

from typing import Any


def reference_load_spec(agent_config: dict[str, Any]) -> dict[str, Any]:
    """load.yaml that copies every source table in the (agent-visible) config"""
    sources: dict[str, Any] = {}
    streams: list[dict[str, Any]] = []

    pg = (agent_config.get("postgres") or {}).get("config")
    if pg:
        sources["postgres"] = {
            "type": "postgres",
            **{k: pg[k] for k in ("host", "port", "database", "user", "password")},
            "schema": pg.get("schema", "public"),
        }
        streams += [{"source": "postgres", "table": t} for t in pg["tables"]]

    mongo = (agent_config.get("mongodb") or {}).get("config")
    if mongo:
        sources["mongodb"] = {
            "type": "mongodb",
            "connection_string": mongo["connection_string"],
            "database": mongo["database"],
        }
        streams += [{"source": "mongodb", "collection": t} for t in mongo["tables"]]

    s3 = agent_config.get("aws_s3")
    if s3:
        sources["aws_s3"] = {
            "type": "aws_s3",
            "endpoint_url": s3["AWS_ENDPOINT_URL"],
            "access_key_id": s3["AWS_ACCESS_KEY_ID"],
            "secret_access_key": s3["AWS_SECRET_ACCESS_KEY"],
            "region": s3["AWS_DEFAULT_REGION"],
        }
        streams += [{"source": "aws_s3", "path": d["path"], "name": d["table"]} for d in s3["data"]]

    api = (agent_config.get("custom_api") or {}).get("config")
    if api:
        sources["custom_api"] = {"type": "custom_api", "base_url": api["base_url"]}
        streams += [
            {"source": "custom_api", "path": path, "name": t}
            for t, path in api["endpoints"].items()
        ]

    files = agent_config.get("flat_files")
    if files:
        sources["flat_files"] = {"type": "flat_files"}
        streams += [
            {"source": "flat_files", "path": f["path"], "format": f["format"], "name": f["table"]}
            for f in files
        ]

    return {"sources": sources, "streams": streams}
