#!/bin/sh
# Creates a dedicated read-only login for Metabase (and other analytics/QA).
# Runs on first Postgres volume init via /docker-entrypoint-initdb.d.
# Re-apply against an existing volume with scripts/create_metabase_reader.sh.
#
# Required env (defaults match .env.example / docker-compose.yml):
#   METABASE_READER_USER
#   METABASE_READER_PASSWORD
#   POSTGRES_USER / POSTGRES_DB (set by the official Postgres image)

set -eu

METABASE_READER_USER="${METABASE_READER_USER:-metabase_reader}"
METABASE_READER_PASSWORD="${METABASE_READER_PASSWORD:-change-me-locally}"
POSTGRES_DB="${POSTGRES_DB:-dota_predictor}"

# Escape single quotes for safe inclusion in SQL string literals.
sql_quote() {
  printf "%s" "$1" | sed "s/'/''/g"
}

USER_SQL=$(sql_quote "$METABASE_READER_USER")
PASS_SQL=$(sql_quote "$METABASE_READER_PASSWORD")

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
DO \$\$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '${USER_SQL}'
  ) THEN
    CREATE ROLE ${METABASE_READER_USER}
      LOGIN
      PASSWORD '${PASS_SQL}'
      NOSUPERUSER
      NOCREATEDB
      NOCREATEROLE
      NOINHERIT
      NOREPLICATION
      NOBYPASSRLS;
  ELSE
    ALTER ROLE ${METABASE_READER_USER}
      WITH LOGIN
      PASSWORD '${PASS_SQL}'
      NOSUPERUSER
      NOCREATEDB
      NOCREATEROLE
      NOINHERIT
      NOREPLICATION
      NOBYPASSRLS;
  END IF;
END
\$\$;

GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${METABASE_READER_USER};

-- Analytical tables live in public (see storage.schema / alembic migrations).
GRANT USAGE ON SCHEMA public TO ${METABASE_READER_USER};
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${METABASE_READER_USER};
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ${METABASE_READER_USER};

-- Future tables/views created by the app role remain readable.
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
  GRANT SELECT ON TABLES TO ${METABASE_READER_USER};
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
  GRANT SELECT ON SEQUENCES TO ${METABASE_READER_USER};

-- Analytical `research` schema (views over the canonical warehouse). It is
-- created later by the Alembic migration, so grant only when it exists; the
-- migration's own GRANTS block keeps it readable after that.
DO \$\$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = 'research')
  THEN
    GRANT USAGE ON SCHEMA research TO ${METABASE_READER_USER};
    GRANT SELECT ON ALL TABLES IN SCHEMA research TO ${METABASE_READER_USER};
  END IF;
END
\$\$;
EOSQL

echo "metabase_reader role ready: user=${METABASE_READER_USER} db=${POSTGRES_DB}"
