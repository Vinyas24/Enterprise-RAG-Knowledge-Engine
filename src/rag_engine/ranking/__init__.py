"""Ranking slice: re-ranking of merged retrieval candidates.

Populated in later phases. The re-ranker is an interface so a hosted provider
(e.g. Cohere) and a local cross-encoder are interchangeable.
"""
