"""Guardrails slice: input checks, output grounding checks, access control.

Populated in later phases. Authorization is resolved before retrieval so that
filters are applied by the stores, never after the fact.
"""
