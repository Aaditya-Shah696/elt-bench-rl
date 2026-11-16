"""how the sandbox tools connect to the warehouse, shared by elt-load and run_sql"""

import os

from sqlalchemy import create_engine


def warehouse_engine(read_only: bool = False):
    url = os.environ["ELT_WAREHOUSE_URL"]
    connect_args = {}
    if url.startswith("postgresql") and read_only:
        connect_args["options"] = "-c default_transaction_read_only=on"
    if url.startswith("snowflake"):
        # the snowflake driver won't take a key path in the url, it goes here instead
        connect_args["private_key_file"] = os.environ["ELT_WAREHOUSE_KEY_FILE"]
        connect_args["authenticator"] = "SNOWFLAKE_JWT"
    return create_engine(url, connect_args=connect_args)
