#!/usr/bin/env python3
"""run_sql, query your warehouse from bash. same thing the run_sql tool uses

    run_sql 'select count(*) from books.book'
    run_sql -f query.sql

works for any warehouse, it connects with ELT_WAREHOUSE_URL. on postgres it's read-only
"""

import sys

import pandas as pd
from elt_warehouse import warehouse_engine
from sqlalchemy import text

MAX_ROWS = 50


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "-f":
        with open(argv[2]) as f:
            sql = f.read()
    elif len(argv) >= 2 and argv[1] not in ("-h", "--help"):
        sql = " ".join(argv[1:])
    else:
        print(__doc__)
        return 2
    sql = sql.strip().rstrip(";").strip()

    engine = warehouse_engine(read_only=True)
    try:
        with engine.connect() as c:
            result = c.execute(text(sql))
            if not result.returns_rows:
                print("ok")
                return 0
            df = pd.DataFrame(result.fetchmany(MAX_ROWS + 1), columns=list(result.keys()))
    except Exception as e:  # noqa: BLE001 - the message is the useful part
        msg = str(getattr(e, "orig", None) or e).strip()
        print(f"ERROR: {msg}"[:3000], file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.head(MAX_ROWS).to_string(index=False, max_colwidth=60))
    if len(df) > MAX_ROWS:
        print(f"... more than {MAX_ROWS} rows, only the first {MAX_ROWS} shown")
    else:
        print(f"({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
