# PROJECT.md

**Working title:** Uplift Lab (rename freely, e.g. `uplift-targeting`, `who-to-treat`, `LiftLens`)

**One line:** Estimate the incremental effect of an action per customer, then turn that into a targeting policy with a dollar value attached.

---

## The why

Every model in the portfolio so far answers "what will happen" or "what is this." Fraud gets caught. CVEs get mapped. Policies get answered. Revenue gets forecast. None of them answer "what should we actually do about it."

A churn model tells you who will leave. It does not tell you who stays *because* you intervened. Those are different people. The space between them is where the budget goes, and almost no data science portfolio touches it.

This project lives in that space. It is decision science, not prediction. That is the point.

## What this is

A causal targeting system built on randomized experiment data. Given a treatment, a control, and an outcome, it estimates the uplift (the incremental effect of the treatment) for each individual, ranks people by that uplift, and produces a spend policy: who to treat at a given budget, and the projected incremental value of doing so. It ships as a clean repo, a served API, and a live demo, in the same shape as the other projects.

## Scope

### In scope for v1

- One binary treatment against a control, one outcome, on data where the treatment was randomized.
- A multi-model bakeoff. No single model is assumed best. Several are fitted and compared, and the winner is chosen on measured results.
- Meta-learners (S, T, X learners) over a shared base learner, direct uplift models (uplift tree / forest, class transformation), and two naive baselines (treat everyone, target by predicted response).
- A proper offline evaluation harness: Qini and AUUC curves, uplift by decile, and incremental value at a fixed budget, all cross-validated with confidence bands.
- Explainability on the chosen model.
- A FastAPI service and a Streamlit demo, deployed on a free tier.
- Free resources only. Public datasets, open-source libraries, free hosting. No paid APIs or services anywhere in the stack.

### Out of scope for v1

- Observational causal inference. We stay on randomized data, so no propensity weighting, no backdoor adjustment, no DAGs.
- Multi-arm or continuous treatments.
- Spillovers, network effects, and long-term downstream outcomes.
- Off-policy reinforcement learning, contextual bandits, and any online experiment loop.
- Cloud infra, feature stores, orchestration, streaming.

Naming these keeps the write-up honest and stops a reviewer from asking why they are missing. They are choices, not gaps.

## Limitations (lead with these, they are where the project earns its seniority)

- **No per-row ground truth.** You never see both outcomes for the same person. A customer either got the treatment or did not. Uplift is never labeled at the row level, so there is no per-customer error to report. This is the nature of the problem, not a flaw in the build. It is also why the evaluation looks nothing like a forecast. Everything is aggregate.
- **Aggregate metrics are noisy.** Qini and AUUC have high variance, swing with the treated-to-control balance, and a model can look good by luck. The harness reports cross-validated Qini with confidence bands, not a single number.
- **An honest negative result is likely.** On some datasets, uplift modeling barely beats "just target the people most likely to respond." If that happens, that finding is the result, and it reads as senior. This mirrors the FCF outcome in FinSight, where seasonal-naive won.
- **Thin signal.** Uplift is a difference of two noisy quantities, often tiny next to the base rate. The "sleeping dogs," people a treatment pushes away, matter most and are the hardest to detect reliably on public data.
- **Bounded validity.** Estimates speak only to the population and the exact treatment that ran. Acting on the policy changes who gets targeted, so incremental value is an offline projection, never a promise.
- **The dollar figure rests on assumptions.** Value per conversion and cost per contact are inputs, not measurements. Incremental value ships as a range with the assumptions stated, never a single confident number. The public datasets are ad and marketing data with anonymized features, so the whole thing is framed as a methodology demonstration, not a live retention system.

## Data

Both datasets are public, free, and carry a real randomized treatment and control split, so there is no confounding to untangle.

- **Hillstrom MineThatData Email Challenge.** ~64k customers, treatment arms (no email, men's email, women's email), outcomes visit / conversion / spend. Small, interpretable, fast to iterate. Used as the primary readable case. The two email arms can be collapsed to a single binary treatment for v1.
- **Criteo Uplift Prediction dataset.** ~13M rows, 12 anonymized features, treatment flag, visit and conversion labels. Large and realistic. Used as the scale case, subsampled where compute demands it (documented when done).

Both are available through the `scikit-uplift` datasets loader, which keeps ingestion reproducible. A randomization sanity check (covariate balance across arms) runs before any modeling and is recorded.

## Modeling (the bakeoff)

The base learner is LightGBM, kept in the existing stack, with a swap to a simpler learner as a robustness check. Models under test:

- **Baselines:** treat everyone, and target by predicted response (a plain classifier, not an uplift model). These set the bar the uplift models must clear.
- **Meta-learners:** S-learner, T-learner, X-learner, over the shared base learner. Optional R-learner if the earlier phases leave budget.
- **Direct uplift models:** uplift tree / random forest, and the class-transformation (transformed outcome) approach.

Libraries: `scikit-uplift`, `causalml`, `econml`, `lightgbm`, all open-source.

**Model selection.** The winner is chosen on cross-validated Qini / AUUC across both datasets, with ties broken on stability of the uplift ranking and on interpretability. Selection is logged as a leaderboard, the same instinct used for MASE in FinSight, pointed at a decision instead of a forecast. If no model beats the baselines, that is stated plainly and becomes a headline finding.

## Evaluation

- Uplift by decile, to see whether high-scored people actually respond more to treatment.
- Qini curve and Qini coefficient.
- AUUC (area under the uplift curve).
- Cross-validation with confidence bands on every headline number.
- Incremental value at a fixed budget, reported as a range under stated cost and value assumptions.
- Every model measured against both naive baselines on the same footing.

## Explainability

SHAP on the chosen model, with the uplift-specific caveat stated openly. For meta-learners, SHAP is applied to the base learners and the difference is interpreted carefully, since a SHAP value on a two-model difference is not a SHAP value on a single fitted target. The point is to name the drivers of uplift honestly, not to overclaim precision.

## Demo

A Streamlit app over the FastAPI service, deployed free (Streamlit Community Cloud or Hugging Face Spaces). It lets a viewer pick a dataset or segment, see the uplift score distribution, read the Qini curve and decile chart, move a budget slider, and watch the resulting targeting policy and its projected incremental-value range update. Real screenshots of real results are captured for the write-up.

## Tech stack and environment

Python 3.12, `uv` with `pyproject.toml` and `uv.lock`, Ruff for lint and format, `pytest` for tests. Modeling on LightGBM plus `scikit-uplift` / `causalml` / `econml`. SHAP for explainability. FastAPI for serving, Streamlit for the demo. Everything open-source, hosting on a free tier.

## Deliverables

- A clean public GitHub repo under `i-hridaysaha`, single author.
- A reproducible pipeline from raw data to results, driven by `uv run` commands.
- A model leaderboard across both datasets, with an explicit chosen model and the reasoning.
- A FastAPI service (`/score`, `/policy`, `/health`) and a live Streamlit demo.
- A `hridaysaha.com` project page write-up sourced only from measured results.
- Real result screenshots for the write-up.

## Definition of done

- The bakeoff runs end to end and produces a leaderboard on both datasets.
- A model is chosen with recorded reasoning, or the baselines win and that is documented as the finding.
- The API serves the chosen policy and the demo runs live and is verified in a browser.
- The repo installs clean from a fresh checkout, ruff is clean, tests pass.
- Limitations and assumptions are written down, not implied.

## Phased roadmap

Work is split across separate chats, one phase per chat, with a review stop between phases. Phases are sized for roughly even token budget. The heavy modeling work is deliberately split across two phases (6 and 7) so no single chat blows the budget. TODO.md phases map 1:1 to the per-chat kickoff prompts.

1. **Project setup.** Repo, `uv` env, `pyproject.toml`, Ruff, folder structure, and the project-memory files scaffolded (PROJECT.md, DATA.md, MODELING.md, EVAL.md, TODO.md, STATUS.md, DECISIONS.md, ENGINEERING.md). No modeling.
2. **Data research and design, no code.** Confirm datasets and schema, define the binary-treatment mapping, the randomization-check plan, the evaluation metrics on paper, the cost and value assumption ranges, and the train/eval splitting strategy. Decisions recorded.
3. **Data ingestion.** Load both datasets via `scikit-uplift`, build a tidy schema, create splits that preserve arm balance, run and record the covariate-balance randomization checks, persist processed data.
4. **EDA (in-depth narrative).** A full data-storytelling EDA report, not just plots: covariate balance, base rates, naive ATE with CIs, uplift-by-segment / by-feature-decile (who responds), value-per-conversion, data quality, and modelling takeaways. Analysis only, no pipeline changes. Internal foundation for a later standalone EDA blog (blog deferred). Expanded from the original "figures only" line — see DECISIONS D19.
5. **Evaluation harness and baselines.** Build and unit-test the Qini / AUUC / decile / incremental-value functions. Implement the two naive baselines. This is the measuring stick, built before any uplift model.
6. **Meta-learners.** S, T, and X learners over LightGBM. Fit, score, log to the leaderboard.
7. **Direct uplift models.** Uplift tree / forest and class transformation, optional R-learner. Fit, score, log to the leaderboard.
8. **Evaluation, selection, explainability.** Cross-validated Qini with confidence bands, full leaderboard across both datasets, pick the winner with tie-breaks, run SHAP on it, and record the honest baseline comparison. Report.
9. **Serving.** Persist the chosen model, build the FastAPI service (`/score`, `/policy` with the incremental-value range, `/health`), with tests.
10. **Streamlit demo.** Dataset and segment selection, uplift distribution, Qini curve, decile chart, budget slider driving the policy and value range. Live-verified in a browser.
11. **Docs, deploy, release.** README (pitch, results table, architecture diagram, install guide, case study, limitations, future work), capture screenshots, deploy on a free tier, run a final repo audit (no secrets, clean history, license present).

## Blogs (deferred, on purpose)

Blog topics are not chosen now. The project comes first. Once it is done, blog recommendations are drawn from the actual results, the scope, and the process, so the posts stand on real findings rather than a plan. Two to three posts are expected.


