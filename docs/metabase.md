# Metabase (local analytics / QA)

Metabase is an **external** analytics tool. It is not part of the prediction
runtime, does not change application behavior, and must connect to the Dota
database only through the dedicated **read-only** role `metabase_reader`.

Do **not** enter the writable `dota_predictor` app credentials in Metabase.

## Prerequisites

- Docker Compose (same stack as `docker-compose.yml`)
- Local Postgres from this repo (`docker compose up -d postgres`)

## 1. Create / refresh the read-only role

On a **fresh** Postgres volume, `docker/postgres/init/02-create-metabase-reader.sh`
runs automatically at first init.

On an **existing** volume (typical for this project), apply or refresh the role:

```bash
./scripts/create_metabase_reader.sh
```

Password and username come from `.env` (see `.env.example`):

```
METABASE_READER_USER=metabase_reader
METABASE_READER_PASSWORD=change-me-locally
```

Replace the placeholder password locally; never commit real secrets.

## 2. Start Metabase

```bash
docker compose up -d metabase
```

Metabase stores its own application state in the Docker volume `metabase-data`
(H2 via `MB_DB_FILE`). That is separate from the Dota analytical database.
`MB_DB_*` is **not** pointed at Postgres.

## 3. Open Metabase

http://localhost:3030

(Host port **3030** — the Next.js frontend already uses **3000**.)

## 4. First-run "Add database" values

Complete Metabase's setup UI manually (not automated in this slice). When
adding the Dota database, use:

| Field    | Value              | Notes                                      |
|----------|--------------------|--------------------------------------------|
| Database | PostgreSQL         |                                            |
| Host     | `postgres`         | Compose service name (same Docker network) |
| Port     | `5432`             | Internal container port                    |
| Database | `dota_predictor`   | Analytical DB, not `dota_predictor_test`   |
| Username | `metabase_reader`  | Read-only only                             |
| Password | *(from `.env`)*    | `METABASE_READER_PASSWORD`                 |

This connection **must remain read-only**. If Metabase can write, the role or
grants are wrong — fix privileges; do not fall back to the app user.

## 6. Research schema

The read-only analytical layer lives in the `research` schema (views over the
canonical warehouse — see `docs/research/research_layer.md`):

- `research.matches` — one row per canonical match, with effective
  event/tier/classification semantics and population booleans.
- `research.player_matches` — one row per player appearance.
- `research.draft_events` — one row per draft pick/ban.
- `research.t12_matches` / `research.pro_matches` /
  `research.t12_draft_matches` — reusable main-event populations.

The `metabase_reader` role is granted `USAGE` on the `research` schema and
`SELECT` on every research view (by the Alembic migration, and refreshed by
`./scripts/create_metabase_reader.sh`). Metabase can query these relations
directly, read-only.

## 7. Stop

```bash
docker compose stop metabase
```
