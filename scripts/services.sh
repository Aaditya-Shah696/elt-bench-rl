#!/usr/bin/env bash
# start/stop the sources + warehouse
#
#   scripts/services.sh up      # start and wait till healthy
#   scripts/services.sh down    # stop, keeps the data
#   scripts/services.sh reset   # stop and wipe the data
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# compose resolves relative paths from docker/ so give it an absolute one
export ELT_BENCH_DIR="$(cd "${ELT_BENCH_DIR:-$REPO_DIR/../ELT-Bench}" && pwd)"
COMPOSE=(docker compose -f "$REPO_DIR/docker/compose.yml")

case "${1:-up}" in
  up)    "${COMPOSE[@]}" up -d --wait ;;
  down)  "${COMPOSE[@]}" down ;;
  reset) "${COMPOSE[@]}" down -v ;;
  *)     echo "usage: $0 [up|down|reset]" >&2; exit 1 ;;
esac
