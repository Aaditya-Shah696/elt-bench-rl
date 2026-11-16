from elt_rl.destinations.base import Connector, Destination, Namespace
from elt_rl.destinations.postgres import PostgresDestination
from elt_rl.destinations.snowflake import SnowflakeDestination

# adding a warehouse = one file implementing Destination + one line here
DESTINATIONS: dict[str, type] = {
    "postgres": PostgresDestination,
    "snowflake": SnowflakeDestination,
}


def get_destination(name: str) -> Destination:
    if name not in DESTINATIONS:
        raise ValueError(f"unknown destination {name!r}, available: {sorted(DESTINATIONS)}")
    return DESTINATIONS[name].from_env()


__all__ = [
    "Connector",
    "Destination",
    "Namespace",
    "PostgresDestination",
    "SnowflakeDestination",
    "get_destination",
]
