# Uplift Targeting

Estimate the incremental effect of an action per customer, then turn that into a
targeting policy with a dollar value attached.

Most models answer *what will happen* or *what is this*. A churn model tells you
who will leave; it does not tell you who stays **because** you intervened. Those
are different people, and the gap between them is where the budget goes. This
project lives in that gap: it is decision science, not prediction.

Given a randomized treatment, a control, and an outcome, it estimates the uplift
(the incremental effect of the treatment) for each individual, ranks people by
that uplift, and produces a spend policy — who to treat at a given budget, and
the projected incremental value of doing so.

> **Status: under construction.** The project is built in phases. Results,
> figures, and the model leaderboard are added only once they are measured — no
> numbers are claimed ahead of the run. See the roadmap below.

## What it will ship

- A multi-model bakeoff (naive baselines, S/T/X meta-learners, direct uplift
  models) evaluated on the same footing.
- An offline evaluation harness: Qini and AUUC curves, uplift by decile, and
  incremental value at a fixed budget, cross-validated with confidence bands.
- A chosen model with recorded reasoning — or an honest finding that the
  baselines win.
- A FastAPI service (`/score`, `/policy`, `/health`) and a live Streamlit demo.

Everything runs on public datasets, open-source libraries, and free hosting.

## Data

- **Hillstrom MineThatData Email Challenge** — ~64k customers, randomized email
  arms, visit / conversion / spend outcomes. The small, readable case.
- **Criteo Uplift Prediction dataset** — ~13M rows, 12 anonymized features,
  randomized treatment flag. The scale case, subsampled where needed.

Both carry a real randomized treatment/control split (no confounding to untangle)
and load through the `scikit-uplift` datasets loader for reproducibility.

## Tech stack

Python 3.12, [`uv`](https://docs.astral.sh/uv/) for env and locking, Ruff for
lint and format, `pytest` for tests. Modeling on LightGBM plus
`scikit-uplift` / `causalml` / `econml`. FastAPI for serving, Streamlit for the
demo. Dependencies are added phase by phase as they are used.

## Getting started

```bash
uv sync
uv run pytest
uv run ruff check
```

`uv sync` provisions Python 3.12 and installs the current dependency set into a
local `.venv`.

## Repository layout

```
src/uplift/        core package
  data/            dataset loading, schema, splits, randomization checks
  models/          baselines, meta-learners, direct uplift models
  eval/            Qini / AUUC / decile / incremental-value harness
  api/             FastAPI serving layer
app/               Streamlit demo
tests/             unit tests
notebooks/         EDA
reports/figures/   generated figures
data/              raw and processed datasets (gitignored, regenerated)
models/            persisted model artifacts (gitignored)
```

## Roadmap

1. Project setup — env, tooling, folder structure. ✓
2. Data research and design (no code). ✓
3. Data ingestion and randomization checks. ✓
4. EDA. ✓
5. Evaluation harness and naive baselines. ✓
6. Meta-learners (S, T, X). **(next)**
7. Direct uplift models.
8. Evaluation, model selection, explainability.
9. Serving (FastAPI).
10. Streamlit demo.
11. Docs, deploy, release.

## Limitations

Uplift is never labeled per row — you never see both outcomes for the same
person — so all evaluation is aggregate and noisy. Estimates speak only to the
population and the exact treatment that ran. The dollar figure rests on stated
cost and value assumptions and ships as a range, never a single confident number.
The public datasets are anonymized ad and marketing data, so this is a
methodology demonstration, not a live retention system.

## Author

Hriday Saha
