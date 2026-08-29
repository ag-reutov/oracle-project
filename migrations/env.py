import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from dota_predictor.storage.schema import METADATA
from dota_predictor.utils.env import load_project_env

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = METADATA

# Connection string is resolved from configuration (DATABASE_URL), never
# hard-coded in alembic.ini. Falls back to TEST_DATABASE_URL only when
# ALEMBIC_USE_TEST_DB is explicitly set, so `alembic upgrade head` can be
# pointed at the test database in CI/tests without a second config file.
load_project_env(Path(__file__).resolve().parents[1])
_env_var = "TEST_DATABASE_URL" if os.environ.get("ALEMBIC_USE_TEST_DB") else "DATABASE_URL"
_database_url = os.environ.get(_env_var)
if _database_url:
    config.set_main_option("sqlalchemy.url", _database_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
