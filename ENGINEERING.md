# Engineering conventions

How code is written in this repository. This is the source of truth; when a
choice is not covered here, prefer the simplest option that a reader can follow.

## Language and environment

- Python 3.12, pinned in `.python-version` and enforced by `requires-python`.
- `uv` owns the environment. `uv sync` installs from `pyproject.toml`, and
  `uv.lock` is committed so a fresh checkout reproduces the exact versions.
- Dependencies are added phase by phase, when a phase first needs them, rather
  than all at once. This keeps installs small and easy to debug. Runtime deps go
  in `[project.dependencies]`; tooling goes in the `dev` dependency group.

## Style and formatting

- Ruff is the single tool for both lint and format. Configuration lives in
  `pyproject.toml`; there is no separate config file.
- Line length is 100. Run `uv run ruff format` and `uv run ruff check` before
  every commit; the tree stays clean.
- Imports are sorted by Ruff (isort rules). Standard library, third party, then
  first party.

## Layout

- `src` layout. The importable package is `uplift`, installed editable by
  `uv sync`, so tests and the API import it the same way an installed user would.
- One responsibility per submodule: `data`, `models`, `eval`, `api`. Keep pure
  logic (transforms, metrics, model wrappers) separate from I/O and side effects,
  and push file and network access to the edges.
- No business logic in `__init__.py` files beyond exports and the version.

## Typing and docstrings

- Type-hint public functions and any function whose signature is not obvious.
  Do not chase 100% coverage on trivial internals.
- Every module gets a one-line docstring saying what it is for. Public functions
  get a short docstring that says what they return and why, not a restatement of
  the code.

## Reproducibility

- Anything with randomness (splits, model fits, subsampling) takes an explicit
  seed and defaults to a fixed value. The same command produces the same result.
- The pipeline is driven by `uv run` commands, not manual notebook state.
  Notebooks are for exploration only; anything that feeds a result is code.
- Data splits and processed datasets are written to disk so downstream steps read
  the same inputs. Their provenance is recorded in the data notes.

## Data and configuration

- `data/` (raw and processed) and `models/` are gitignored. Raw data is treated
  as immutable; processed data is derived and regenerated from raw by the code.
- No secrets or credentials in the repo. Cost-per-contact and value-per-
  conversion are modeling **assumptions**, kept as explicit, documented config
  values — never hardcoded deep inside a function.

## Testing

- `pytest`, tests under `tests/`. Unit-test the evaluation math and the data
  transforms first — they are the parts a silent bug would corrupt invisibly.
- Unit tests do not hit the network. Dataset downloads are cached; tests that
  need data use small fixtures or cached files.
- A change that fixes a bug adds the test that would have caught it.

## Git and commits

- Single author, `Hriday Saha`. Commit messages are imperative and prefixed by
  type: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
- Commit and push only after a phase is reviewed and approved.
- Never commit generated artifacts, data, `.venv`, or local-only notes. The
  repository is public; treat every commit as public.
