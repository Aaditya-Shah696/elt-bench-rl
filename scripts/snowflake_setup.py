"""one-time snowflake setup: makes a key pair, writes .env, prints the sql to run in snowsight.

    uv run python scripts/snowflake_setup.py --account ABCDEFG-XY12345

snowflake blocks password-only logins for scripts now, so the admin login is a service user
with a key pair. the private key stays in ~/.elt-rl/snowflake/, never in the repo
"""

import argparse
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

REPO = Path(__file__).resolve().parent.parent
KEY_DIR = Path.home() / ".elt-rl" / "snowflake"
KEY_FILE = KEY_DIR / "admin_key.p8"

ADMIN = "ELT_RL_ADMIN"
WAREHOUSE = "ELT_RL_WH"


def public_key_body(private_pem: bytes) -> str:
    key = serialization.load_pem_private_key(private_pem, password=None)
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    # snowflake wants just the base64 part, no header lines
    return "".join(pem.decode().strip().splitlines()[1:-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True, help="account identifier, eg ABCDEFG-XY12345")
    args = parser.parse_args()

    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        KEY_FILE.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        os.chmod(KEY_FILE, 0o600)
    public = public_key_body(KEY_FILE.read_bytes())

    env = REPO / ".env"
    lines = [
        line
        for line in (env.read_text().splitlines() if env.exists() else [])
        if not line.startswith("SNOWFLAKE_")
    ]
    lines += [
        f"SNOWFLAKE_ACCOUNT={args.account}",
        f"SNOWFLAKE_USER={ADMIN}",
        f"SNOWFLAKE_ROLE={ADMIN}",
        f"SNOWFLAKE_WAREHOUSE={WAREHOUSE}",
        f"SNOWFLAKE_PRIVATE_KEY_PATH={KEY_FILE}",
    ]
    env.write_text("\n".join(lines) + "\n")

    print(f"""wrote {env} and the private key {KEY_FILE}

now paste this into a snowsight worksheet and click "run all":
------------------------------------------------------------------------------------------
use role accountadmin;

-- a role that can make one database + login per rollout, nothing else
create role if not exists {ADMIN};
grant create database on account to role {ADMIN};
grant create role on account to role {ADMIN};
grant create user on account to role {ADMIN};
grant role {ADMIN} to role sysadmin;

-- smallest warehouse, sleeps after 60s idle so it barely costs anything
create warehouse if not exists {WAREHOUSE}
  warehouse_size = xsmall auto_suspend = 60 auto_resume = true initially_suspended = true;
grant ownership on warehouse {WAREHOUSE} to role {ADMIN} copy current grants;

-- the login our code uses, key pair only
create user if not exists {ADMIN}
  type = service default_role = {ADMIN} default_warehouse = {WAREHOUSE}
  rsa_public_key = '{public}';
grant role {ADMIN} to user {ADMIN};
------------------------------------------------------------------------------------------
then check it with: uv run pytest tests/test_snowflake.py
""")


if __name__ == "__main__":
    main()
