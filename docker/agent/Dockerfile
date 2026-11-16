# the sandbox the agent works in, one container per rollout
# trimmed down from ELT-Bench's elt-swe image, everything's baked in so rollouts never download
#
#   docker build -t elt-rl-agent docker/agent

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl jq postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        awscli \
        boto3 \
        dbt-core \
        dbt-postgres \
        dbt-snowflake \
        pandas \
        psycopg2-binary \
        pymongo \
        pyyaml \
        requests \
        "snowflake-connector-python[pandas]" \
        snowflake-sqlalchemy \
        sqlalchemy

# our loader, the agent configures it with a yaml file
COPY elt_load.py /usr/local/bin/elt-load
COPY run_sql.py /usr/local/bin/run_sql
COPY elt_warehouse.py /usr/local/bin/elt_warehouse.py
RUN chmod +x /usr/local/bin/elt-load /usr/local/bin/run_sql

# dbt phones home by default, there's no internet anyway so don't bother trying
ENV DBT_SEND_ANONYMOUS_USAGE_STATS=False

WORKDIR /workspace
CMD ["sleep", "infinity"]
