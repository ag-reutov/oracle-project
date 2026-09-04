"""Analytical research layer over the canonical Dota warehouse.

A thin set of PostgreSQL views (see `views.py`) exposing effective
event/tier semantics and reusable match/player/draft relations so ordinary
research questions can be answered from SQL, Metabase, or Python without
reconstructing league/event/tier/qualifier semantics by hand.
"""
