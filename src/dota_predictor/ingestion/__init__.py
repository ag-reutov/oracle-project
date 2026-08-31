"""STRATZ historical league-match ingestion."""

from dota_predictor.ingestion.pipeline import (
    ingest_league,
    ingest_leagues,
    ingest_matches_by_id,
)

__all__ = ["ingest_league", "ingest_leagues", "ingest_matches_by_id"]
