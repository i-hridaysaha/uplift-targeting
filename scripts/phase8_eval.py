"""Phase 8 evaluation, selection, and explainability -- the bakeoff verdict.

Read-only over the tidy processed datasets in ``data/processed/``. For each
dataset it:

1. Scores the **whole field** (two baselines + S/T/X meta-learners + the three
   direct models) out-of-fold with cross-validated Qini/AUUC bands, writing
   ``reports/phase8_leaderboard.csv``.
2. Confirms every model once on the untouched 20% hold-out, writing
   ``reports/phase8_holdout.csv`` -- the tie-break basis.
3. Applies the MODELING.md selection rule (beat both baselines on CV Qini; ties
   broken on stability then interpretability) and prints the suggested winner with
   its reasoning.
4. Explains the chosen uplift model's drivers: TreeSHAP (LightGBM ``pred_contrib``,
   no extra dependency -- D26) for a LightGBM-backed winner, permutation importance
   for the hand-rolled forest.
5. Regenerates the report figures in ``assets/phase8/`` (SVG + PNG).

Run: ``uv run python scripts/phase8_eval.py``
     ``uv run python scripts/phase8_eval.py --datasets hillstrom``  (fast case only)

Nothing here modifies processed data or the shipped pipeline. Palette is Okabe-Ito
(colorblind-safe), matching the Phase 4 EDA figures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift.data.schema import CRITEO_FEATURES, HILLSTROM_FEATURES, TREATMENT_COL
from uplift.data.splits import HOLDOUT_FOLD, SEED
from uplift.eval.explain import permutation_uplift_importance, treeshap_uplift
from uplift.eval.metrics import qini_curve, uplift_by_decile
from uplift.models import (
    ClassTransformation,
    SLearner,
    TLearner,
    UpliftForest,
    UpliftTree,
    XLearner,
    holdout_board,
    holdout_uplift,
    leaderboard,
    model_scorer,
)
from uplift.models.leaderboard import FOLD_COL, MODELS, Scorer

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
ASSETS = Path("assets/phase8")
PRIMARY_OUTCOME = "visit"

FEATURES: dict[str, list[str]] = {
    "hillstrom": HILLSTROM_FEATURES,
    "criteo": CRITEO_FEATURES,
}
# Per-tree row cap for the forest (D25): full bootstrap on small Hillstrom, capped
# on Criteo so 50 pure-numpy trees per fold stay fast.
FOREST_MAX_SAMPLES: dict[str, int | None] = {"hillstrom": None, "criteo": 150_000}

# The uplift models (not the baselines) and how to rebuild one to explain it.
UPLIFT_CLASSES: dict[str, type] = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "class_transformation": ClassTransformation,
    "uplift_tree": UpliftTree,
    "uplift_forest": UpliftForest,
}
# Interpretability rank for the tie-break (lower = more interpretable / SHAP-able).
INTERPRETABILITY: dict[str, int] = {
    "class_transformation": 0,
    "s_learner": 1,
    "t_learner": 2,
    "x_learner": 2,
    "uplift_tree": 3,
    "uplift_forest": 4,
}
TREESHAP_MODELS = (SLearner, ClassTransformation)


# --- Figure setup (Okabe-Ito, matches Phase 4) -----------------------------

OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
}
mpl.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#B0B0B0",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to ``assets/phase8/`` as both SVG (vector) and PNG (raster)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# --- The field --------------------------------------------------------------


def full_field(dataset: str) -> dict[str, Scorer]:
    """The whole bakeoff for one dataset: two baselines + S/T/X + three direct models."""
    return {
        "treat_everyone": MODELS["treat_everyone"],
        "response_model": MODELS["response_model"],
        "s_learner": MODELS["s_learner"],
        "t_learner": MODELS["t_learner"],
        "x_learner": MODELS["x_learner"],
        "class_transformation": model_scorer(ClassTransformation),
        "uplift_tree": model_scorer(UpliftTree),
        "uplift_forest": model_scorer(UpliftForest, max_samples=FOREST_MAX_SAMPLES[dataset]),
    }


def new_uplift_model(name: str, dataset: str, seed: int = SEED) -> object:
    """Build an unfitted instance of one uplift model with its per-dataset kwargs."""
    cls = UPLIFT_CLASSES[name]
    if name == "uplift_forest":
        return cls(seed=seed, max_samples=FOREST_MAX_SAMPLES[dataset])
    return cls(seed=seed)


# --- Selection rule ---------------------------------------------------------


def pick_winner(cv: pd.DataFrame, holdout: pd.DataFrame) -> tuple[str, str]:
    """Suggest a winner and explain why, following the MODELING.md rule.

    A model must beat the response-model bar on cross-validated Qini to count. Among
    those, the highest CV Qini wins; when the top two bands overlap (mean gap below
    the larger std) the tie breaks on stability (lower std) then interpretability.
    If nothing beats the bar, the response baseline "wins" -- the honest-negative
    result, meaning uplift modeling did not pay on this dataset.
    """
    cvi = cv.set_index("model")
    bar = float(cvi.loc["response_model", "qini_mean"])
    candidates = [m for m in UPLIFT_CLASSES if m in cvi.index]
    beats = [m for m in candidates if cvi.loc[m, "qini_mean"] > bar]
    if not beats:
        return "response_model", (
            f"no uplift model clears the response bar ({bar:+.4f}); "
            "honest-negative -- uplift modeling does not pay here."
        )

    beats.sort(key=lambda m: cvi.loc[m, "qini_mean"], reverse=True)
    top = beats[0]
    top_mean, top_std = cvi.loc[top, "qini_mean"], cvi.loc[top, "qini_std"]
    overlap = [
        m
        for m in beats
        if top_mean - cvi.loc[m, "qini_mean"] < max(top_std, cvi.loc[m, "qini_std"])
    ]
    winner = min(overlap, key=lambda m: (cvi.loc[m, "qini_std"], INTERPRETABILITY[m]))
    hold = float(holdout.set_index("model").loc[winner, "qini"])
    tie = "" if len(overlap) == 1 else f" (tie-break over {overlap} on stability+interpretability)"
    return winner, (
        f"beats the response bar ({bar:+.4f}) with CV Qini "
        f"{cvi.loc[winner, 'qini_mean']:+.4f} +/- {cvi.loc[winner, 'qini_std']:.4f}, "
        f"hold-out {hold:+.4f}{tie}."
    )


# --- Boards -----------------------------------------------------------------


def run_boards(name: str, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the (cross-validated, hold-out) leaderboards for one dataset."""
    frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
    field = full_field(name)
    cv = leaderboard(frame, FEATURES[name], PRIMARY_OUTCOME, models=field, seed=seed)
    hold = holdout_board(frame, FEATURES[name], PRIMARY_OUTCOME, models=field, seed=seed)
    cv.insert(0, "dataset", name)
    hold.insert(0, "dataset", name)
    return cv, hold


# --- Figures ----------------------------------------------------------------


def fig_leaderboard(cv: pd.DataFrame, name: str) -> None:
    """Bar of cross-validated Qini per model with std error bars and the response bar."""
    order = list(full_field(name).keys())
    d = cv.set_index("model").loc[order]
    bar = float(d.loc["response_model", "qini_mean"])
    colors = [
        OKABE["gray"]
        if m in ("treat_everyone", "response_model")
        else (OKABE["green"] if d.loc[m, "qini_mean"] > bar else OKABE["sky"])
        for m in order
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(order, d["qini_mean"], yerr=d["qini_std"], color=colors, capsize=3, alpha=0.9)
    ax.axhline(bar, color=OKABE["vermillion"], ls="--", lw=1.2, label=f"response bar {bar:+.4f}")
    ax.set_ylabel("Qini (normalized, 5-fold CV)")
    ax.set_title(f"{name.capitalize()} -- cross-validated Qini across the field")
    ax.tick_params(axis="x", rotation=35)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(loc="upper right")
    save(fig, f"cv_leaderboard_{name}")


def fig_holdout_qini(frame: pd.DataFrame, name: str, winner: str, seed: int = SEED) -> None:
    """Hold-out Qini curves: chosen uplift model vs response vs the random diagonal."""
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    y = hold[PRIMARY_OUTCOME].to_numpy()
    t = hold[TREATMENT_COL].to_numpy()
    field = full_field(name)
    n = len(hold)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    show = {"response_model": OKABE["orange"]}
    if winner not in show:
        show[winner] = OKABE["vermillion"]
    for model_name, color in show.items():
        u = holdout_uplift(frame, FEATURES[name], PRIMARY_OUTCOME, field[model_name], seed=seed)
        x, curve = qini_curve(y, u, t)
        ax.plot(x / n, curve, color=color, lw=2.0, label=model_name)
    ax.plot([0, 1], [0, curve[-1]], color=OKABE["gray"], ls=":", lw=1.3, label="random")
    ax.set_xlabel("fraction of population targeted")
    ax.set_ylabel("cumulative incremental visits (size-corrected)")
    ax.set_title(f"{name.capitalize()} -- hold-out Qini curve")
    ax.legend(loc="lower right")
    save(fig, f"holdout_qini_{name}")


def fig_decile(frame: pd.DataFrame, name: str, winner: str, uplift: np.ndarray) -> None:
    """Observed uplift by predicted-uplift decile for the chosen model on the hold-out."""
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    table = uplift_by_decile(
        hold[PRIMARY_OUTCOME].to_numpy(), uplift, hold[TREATMENT_COL].to_numpy()
    )
    colors = [OKABE["green"] if v >= 0 else OKABE["vermillion"] for v in table["observed_uplift"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(table["decile"], table["observed_uplift"], color=colors, alpha=0.9)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xlabel("decile of predicted uplift (1 = highest)")
    ax.set_ylabel("observed uplift  (visit rate: treated - control)")
    ax.set_title(f"{name.capitalize()} -- {winner}: observed uplift by decile (hold-out)")
    ax.set_xticks(table["decile"])
    save(fig, f"decile_{name}_{winner}")


def fig_drivers(table: pd.DataFrame, value_col: str, name: str, winner: str, kind: str) -> None:
    """Horizontal bar of the top uplift drivers (SHAP contribution or permutation drop)."""
    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(top["feature"].astype(str), top[value_col], color=OKABE["blue"], alpha=0.9)
    ax.set_xlabel(kind)
    ax.set_title(f"{name.capitalize()} -- {winner}: uplift drivers")
    ax.grid(axis="y", visible=False)
    save(fig, f"drivers_{name}_{winner}")


# --- Explainability ---------------------------------------------------------


def explain_winner(frame: pd.DataFrame, name: str, winner: str, seed: int = SEED) -> None:
    """Fit the chosen uplift model on dev, rank its uplift drivers, and plot them.

    TreeSHAP (LightGBM ``pred_contrib``) for the S-learner / class-transformation;
    permutation importance for anything else (notably the hand-rolled forest).
    """
    features = FEATURES[name]
    dev = frame[frame[FOLD_COL] != HOLDOUT_FOLD]
    hold = frame[frame[FOLD_COL] == HOLDOUT_FOLD]
    model = new_uplift_model(winner, name, seed=seed)
    model.fit(dev[features], dev[TREATMENT_COL].to_numpy(), dev[PRIMARY_OUTCOME].to_numpy())

    if isinstance(model, TREESHAP_MODELS):
        table = treeshap_uplift(model, hold[features])
        table.to_csv(REPORTS / f"phase8_drivers_{name}.csv", index=False)
        units = "uplift units" if isinstance(model, ClassTransformation) else "log-odds margin"
        fig_drivers(table, "mean_abs_shap", name, winner, f"mean |TreeSHAP| ({units})")
        print(f"  drivers (TreeSHAP, {units}): {list(table['feature'].head(5))}")
    else:
        table = permutation_uplift_importance(
            model,
            hold[features],
            hold[TREATMENT_COL].to_numpy(),
            hold[PRIMARY_OUTCOME].to_numpy(),
            seed=seed,
        )
        table.to_csv(REPORTS / f"phase8_drivers_{name}.csv", index=False)
        fig_drivers(table, "importance", name, winner, "Qini drop when feature shuffled")
        print(f"  drivers (permutation): {list(table['feature'].head(5))}")


# --- Orchestration ----------------------------------------------------------


def _format(cv: pd.DataFrame, hold: pd.DataFrame) -> str:
    """Render the CV band and the hold-out number side by side, one row per model."""
    h = hold.set_index("model")["qini"]
    lines = [f"{'model':<21} {'CV Qini':>18} {'hold-out':>10}"]
    for row in cv.itertuples():
        cvq = f"{row.qini_mean:+.4f} +/- {row.qini_std:.4f}"
        lines.append(f"{row.model:<21} {cvq:>18} {h[row.model]:>+10.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the full-field bakeoff verdict and persist boards + figures."""
    parser = argparse.ArgumentParser(description="Phase 8 evaluation, selection, explainability.")
    parser.add_argument("--datasets", default="hillstrom,criteo")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    cv_all, hold_all = [], []

    for name in names:
        start = time.perf_counter()
        cv, hold = run_boards(name, seed=args.seed)
        cv_all.append(cv)
        hold_all.append(hold)

        winner, why = pick_winner(cv, hold)
        print(f"\n=== {name} ({time.perf_counter() - start:.1f}s) ===")
        print(_format(cv, hold))
        print(f"winner: {winner} -- {why}")

        fig_leaderboard(cv, name)
        frame = pd.read_parquet(PROCESSED / f"{name}.parquet")
        fig_holdout_qini(frame, name, winner, seed=args.seed)
        if winner in UPLIFT_CLASSES:
            uplift = holdout_uplift(
                frame, FEATURES[name], PRIMARY_OUTCOME, full_field(name)[winner], seed=args.seed
            )
            fig_decile(frame, name, winner, uplift)
            explain_winner(frame, name, winner, seed=args.seed)
        else:
            print("  (winner is a baseline -- no uplift model to explain here)")

    pd.concat(cv_all, ignore_index=True).to_csv(REPORTS / "phase8_leaderboard.csv", index=False)
    pd.concat(hold_all, ignore_index=True).to_csv(REPORTS / "phase8_holdout.csv", index=False)
    print(f"\nwrote {REPORTS}/phase8_leaderboard.csv, {REPORTS}/phase8_holdout.csv, {ASSETS}/*")


if __name__ == "__main__":
    main()
