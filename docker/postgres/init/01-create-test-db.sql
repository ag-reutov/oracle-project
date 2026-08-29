-- Runs once when the postgres data volume is first initialized.
-- POSTGRES_DB (from docker-compose.yml) creates the dev database
-- automatically; this creates the separate, dedicated test database so
-- DATABASE_URL and TEST_DATABASE_URL never point at the same database.
CREATE DATABASE dota_predictor_test;
