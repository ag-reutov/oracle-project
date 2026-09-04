#!/usr/bin/env bash
# Apply (or refresh) the Metabase read-only Postgres role against a running
# local database. Safe to re-run; updates the password from env/.env.
#
# Usage (from repo root):
#   ./scripts/create_metabase_reader.sh
#
# Requires the dota_predictor_postgres container to be running
# (docker compose up -d postgres).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      METABASE_READER_USER=*|METABASE_READER_PASSWORD=*)
        key="${line%%=*}"
        value="${line#*=}"
        export "$key=$value"
        ;;
    esac
  done < .env
fi

METABASE_READER_USER="${METABASE_READER_USER:-metabase_reader}"
METABASE_READER_PASSWORD="${METABASE_READER_PASSWORD:-change-me-locally}"

CONTAINER="${DOTA_POSTGRES_CONTAINER:-dota_predictor_postgres}"

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
  echo "error: container '$CONTAINER' is not running" >&2
  echo "start it with: docker compose up -d postgres" >&2
  exit 1
fi

docker exec \
  -e METABASE_READER_USER="$METABASE_READER_USER" \
  -e METABASE_READER_PASSWORD="$METABASE_READER_PASSWORD" \
  "$CONTAINER" \
  sh /docker-entrypoint-initdb.d/02-create-metabase-reader.sh

echo "Applied read-only role '${METABASE_READER_USER}' on '${CONTAINER}'."
echo "Use this account only for Metabase / analytics — never for app writes."
