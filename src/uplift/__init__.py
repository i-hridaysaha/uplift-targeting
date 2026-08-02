"""Causal uplift targeting.

Estimate the per-customer incremental effect of a treatment from randomized
experiment data, rank customers by that effect, and turn the ranking into a
budgeted spend policy with a projected incremental-value range.

Submodules are added as the phased roadmap progresses:

- ``data``   dataset loading, tidy schema, arm-balanced splits, randomization checks
- ``models`` baselines, meta-learners, direct uplift models
- ``eval``   Qini / AUUC / decile / incremental-value harness
- ``api``    FastAPI serving layer
"""

__version__ = "0.1.0"
