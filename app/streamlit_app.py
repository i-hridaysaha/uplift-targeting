"""Streamlit demo for the uplift targeting bakeoff.

Runs in-process over the ``uplift`` library (no HTTP call): it loads the persisted
policy bundles for their labels and the slim hold-out reference tables
(``scripts/phase9_persist.py`` writes both to ``models/``), then reuses the
Phase-5 ``eval`` harness through ``uplift.api.demo`` to draw four live views --
the score distribution, the Qini curve, uplift-by-decile, and a priced targeting
policy driven by a budget slider.

Pick a dataset (and, on Hillstrom, a readable segment); Criteo's ``f0..f11`` are
anonymized so it is shown whole-population. Hillstrom's chosen model is the
honest-negative response baseline -- its weak, near-diagonal Qini is shown as-is
and labeled, not hidden.

Run: ``uv run streamlit run app/streamlit_app.py``
Needs ``models/`` populated first: ``uv run python scripts/phase9_persist.py``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless, thread-safe: Streamlit runs the script off the main thread,
# and a GUI backend (the macOS default) segfaults there and on a headless server.

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection above)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from uplift.api.artifacts import SCORE_COL, load_bundles  # noqa: E402
from uplift.api.demo import (  # noqa: E402
    ALL,
    apply_segment,
    decile_table,
    format_segment_value,
    load_reference,
    policy_for_frame,
    qini_score,
    qini_xy,
    segment_features,
    segment_label,
    segment_options,
    value_sweep,
)
from uplift.eval.value import (  # noqa: E402
    COST_PER_CONTACT,
    COST_PER_CONTACT_RANGE,
    VALUE_PER_CONVERSION,
    VALUE_PER_CONVERSION_RANGE,
)

MODELS_DIR = os.environ.get("UPLIFT_MODELS_DIR", "models")

# Okabe-Ito, matching the Phase-4/8 report figures (colourblind-safe; D21).
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#9aa0a6"

DATASET_ORDER = ["hillstrom", "criteo"]
DATASET_LABEL = {"hillstrom": "Hillstrom (Readable Case)", "criteo": "Criteo (Scale Case)"}

# One shared matplotlib look, so every chart reads as one system (clean spines,
# muted grid, consistent type). Set once at import.
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#d0d0d0",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#3c4043",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#eceff1",
        "grid.linewidth": 1.0,
        "xtick.color": "#5f6368",
        "ytick.color": "#5f6368",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "#3c4043",
        "font.size": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }
)

_CSS = """
<style>
  .block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1280px; }
  .hero-title { font-size: 2.1rem; font-weight: 800; letter-spacing: .03em;
                text-transform: uppercase; margin: 0 0 .15rem; }
  .hero-sub { color: #9aa0a6; font-size: 1.03rem; line-height: 1.45; margin: 0 0 .1rem; }
  .verdict { display: flex; align-items: center; gap: .7rem; flex-wrap: wrap;
             margin: .1rem 0 .35rem; }
  .verdict-msg { font-size: 1.03rem; line-height: 1.4; }
  .pill { display: inline-block; padding: .2rem .72rem; border-radius: 999px; font-size: .74rem;
          font-weight: 700; letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; }
  .pill-green { background: rgba(0,158,115,.18); color: #1fbf8f;
                border: 1px solid rgba(0,158,115,.45); }
  .pill-amber { background: rgba(230,159,0,.16); color: #e6a100;
                border: 1px solid rgba(230,159,0,.45); }
  .muted { color: #9aa0a6; font-size: .9rem; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
  [data-testid="stMetricLabel"] p { opacity: .7; font-size: .82rem; }
  h3 { font-weight: 700; letter-spacing: -0.01em; }
  /* Make the two chart cards in a row the same height (tops and bottoms aligned). */
  [data-testid="stHorizontalBlock"] { align-items: stretch; }
  [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }
</style>
"""


# --- Cached loaders ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _labels(models_dir: str) -> dict[str, dict[str, object]]:
    """Load bundle labels (model name, score type, honest-negative flag) per dataset."""
    return {
        name: {
            "model_name": b.model_name,
            "score_type": b.score_type,
            "honest_negative": b.honest_negative,
        }
        for name, b in load_bundles(models_dir).items()
    }


@st.cache_data(show_spinner=False)
def _reference(models_dir: str, dataset: str) -> pd.DataFrame:
    """Load a dataset's demo hold-out reference table (cached)."""
    return load_reference(models_dir, dataset)


# --- Small presentation helpers ---------------------------------------------


def _pretty_model(model_name: str) -> str:
    """De-jargon the internal model name for a caption (``uplift_forest`` -> ``Uplift forest``)."""
    return model_name.replace("_", " ").capitalize()


def _score_phrase(honest: bool) -> str:
    """What customers are ranked by, in plain words."""
    return "predicted response probability" if honest else "predicted uplift"


def _verdict_html(honest: bool) -> str:
    """A one-line verdict banner: a coloured pill plus what it means, no model jargon."""
    if honest:
        pill = '<span class="pill pill-amber">Honest-negative</span>'
        msg = (
            "Uplift modeling did not beat plain response targeting on this dataset, so customers "
            f"are ranked by <b>{_score_phrase(True)}</b>, not uplift."
        )
    else:
        pill = '<span class="pill pill-green">Uplift wins</span>'
        msg = (
            "Uplift modeling beats response targeting here, so customers are ranked by "
            f"<b>{_score_phrase(False)}</b>."
        )
    return f'<div class="verdict">{pill}<span class="verdict-msg">{msg}</span></div>'


# --- Figures ----------------------------------------------------------------


def _score_distribution_fig(scores: np.ndarray, score_type: str):
    """Histogram of the per-customer targeting scores."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(scores, bins=40, color=BLUE, alpha=0.9)
    ax.axvline(0.0, color=GREY, lw=1.0, ls="--")
    ax.set_xlabel(
        "predicted uplift" if score_type == "uplift" else "predicted response probability"
    )
    ax.set_ylabel("customers")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def _qini_fig(reference: pd.DataFrame):
    """Qini curve (fraction targeted vs cumulative incremental visits) with the random diagonal."""
    x, y = qini_xy(reference)
    n = len(reference)
    frac = x / n if n else x
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(frac, y, color=BLUE, lw=2.2, label="model")
    ax.plot([0.0, 1.0], [0.0, y[-1]], color=GREY, lw=1.3, ls="--", label="random targeting")
    ax.set_xlabel("fraction of customers targeted")
    ax.set_ylabel("cumulative incremental visits")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _decile_fig(reference: pd.DataFrame):
    """Observed uplift within each decile of predicted score (good model: decreasing)."""
    table = decile_table(reference)
    observed = table["observed_uplift"].to_numpy()
    colors = [VERMILLION if v < 0 else GREEN for v in np.nan_to_num(observed)]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.bar(table["decile"], observed, color=colors, alpha=0.92, width=0.72)
    ax.axhline(0.0, color="#5f6368", lw=1.0)
    ax.set_xlabel("decile of predicted score  (1 = highest-scored customers)")
    ax.set_ylabel("observed uplift (visit)")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def _value_fig(reference: pd.DataFrame, current_fraction: float, value_pt: float, cost_pt: float):
    """Net incremental value across targeted fractions, with the current policy marked."""
    fractions = np.linspace(0.0, 1.0, 101)
    xs, values = value_sweep(reference, fractions, value_pt, cost_pt)
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.plot(xs, values, color=BLUE, lw=2.2)
    ax.fill_between(xs, 0, values, where=values >= 0, color=GREEN, alpha=0.12)
    ax.fill_between(xs, 0, values, where=values < 0, color=VERMILLION, alpha=0.12)
    ax.axhline(0.0, color="#5f6368", lw=1.0)
    ax.axvline(current_fraction, color=ORANGE, lw=1.6, ls="--", label="current policy")
    ax.set_xlabel("fraction of customers targeted")
    ax.set_ylabel("net incremental value ($)")
    ax.legend(loc="lower center")
    fig.tight_layout()
    return fig


def _chart_card(title: str, caption: str, fig) -> None:
    """Render a titled, captioned chart inside a bordered card (uniform height across columns)."""
    with st.container(border=True):
        st.markdown(f"##### {title}")
        st.caption(caption)
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# --- App --------------------------------------------------------------------


def main() -> None:
    """Render the demo."""
    st.set_page_config(page_title="Uplift targeting", page_icon="🎯", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    labels = _labels(MODELS_DIR)
    if not labels:
        st.error(
            f"No policy bundles found in `{MODELS_DIR}/`. Build them first:\n\n"
            "`uv run python scripts/phase9_persist.py`"
        )
        st.stop()

    available = [d for d in DATASET_ORDER if d in labels] + [
        d for d in labels if d not in DATASET_ORDER
    ]

    # --- Sidebar: dataset, segment, assumptions ---
    st.sidebar.header("Controls")
    # A ?dataset= query param sets the initial view, so a dataset is deep-linkable
    # (shareable in the write-up / deploy). The selectbox stays the source of truth.
    qp_dataset = st.query_params.get("dataset")
    default_index = available.index(qp_dataset) if qp_dataset in available else 0
    dataset = st.sidebar.selectbox(
        "Dataset", available, index=default_index, format_func=lambda d: DATASET_LABEL.get(d, d)
    )
    meta = labels[dataset]
    honest = bool(meta["honest_negative"])
    reference = _reference(MODELS_DIR, dataset)

    selections: dict[str, object] = {}
    seg_feats = segment_features(dataset)
    if seg_feats:
        st.sidebar.subheader("Segment")
        options = segment_options(reference, dataset)
        for feature in seg_feats:
            values = options.get(feature, [])
            selections[feature] = st.sidebar.selectbox(
                segment_label(feature),
                [ALL, *values],
                format_func=lambda v, f=feature: format_segment_value(f, v),
                key=f"seg_{dataset}_{feature}",
            )
    else:
        st.sidebar.caption(
            "Criteo's features are anonymized (`f0..f11`), so there is no readable segment; "
            "the whole hold-out is shown."
        )

    st.sidebar.subheader("Dollar Assumptions")
    st.sidebar.caption("Assumptions, not measurements. The value band always sweeps the ranges.")
    value_pt = st.sidebar.number_input(
        "Value per Conversion ($)",
        min_value=1.0,
        value=float(VALUE_PER_CONVERSION),
        step=5.0,
        help=rf"Sensitivity range \${VALUE_PER_CONVERSION_RANGE[0]:.0f}"
        rf"-\${VALUE_PER_CONVERSION_RANGE[1]:.0f}.",
    )
    cost_pt = st.sidebar.number_input(
        "Cost per Contact ($)",
        min_value=0.001,
        value=float(COST_PER_CONTACT),
        step=0.05,
        format="%.3f",
        help=rf"Notional range \${COST_PER_CONTACT_RANGE[0]:.2f}"
        rf"-\${COST_PER_CONTACT_RANGE[1]:.2f}.",
    )

    # --- Hero + verdict ---
    st.markdown('<div class="hero-title">Uplift targeting</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Estimate each customer\'s incremental response to a treatment, '
        "rank by it, and price a targeting policy. Everything below is computed live on an "
        "untouched 20% hold-out.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(f"### {DATASET_LABEL.get(dataset, dataset)}")
    st.markdown(_verdict_html(honest), unsafe_allow_html=True)

    # --- Segment filter ---
    filtered = apply_segment(reference, selections)
    n_all, n_seg = len(reference), len(filtered)
    scope = "whole hold-out" if n_seg == n_all else "segment"
    st.markdown(
        f'<span class="muted">Ranking model: <b>{_pretty_model(str(meta["model_name"]))}</b> '
        f"&nbsp;·&nbsp; {n_seg:,} of {n_all:,} hold-out customers ({scope})</span>",
        unsafe_allow_html=True,
    )
    if n_seg == 0:
        st.warning("No customers match this segment. Widen the filter.")
        st.stop()
    if n_seg < 200:
        st.caption("Small segment; the Qini and decile numbers below are noisy.")
    st.write("")

    # --- Views: two aligned cards, then a full-width decile card ---
    left, right = st.columns(2, gap="large")
    with left:
        _chart_card(
            "Score Distribution",
            "How the per-customer targeting score is spread.",
            _score_distribution_fig(filtered[SCORE_COL].to_numpy(), str(meta["score_type"])),
        )
    with right:
        q = qini_score(filtered)
        _chart_card(
            "Qini Curve",
            f"Normalized Qini = **{q.normalized:+.4f}** (higher is better).",
            _qini_fig(filtered),
        )

    _chart_card(
        "Uplift by Decile",
        "Observed treatment effect within each score decile. "
        "A good ranker steps down left to right.",
        _decile_fig(filtered),
    )

    # --- Policy (budget-driven) ---
    st.divider()
    st.markdown("### Targeting Policy")
    st.caption("Move the budget to target the top-ranked customers; the value band updates live.")
    driver = st.radio(
        "Budget Driver", ["Share of Customers (%)", "Dollar Budget ($)"], horizontal=True
    )
    if driver.startswith("Share"):
        pct = st.slider("Share of Customers Targeted (%)", 0.0, 100.0, 10.0, 0.5)
        result = policy_for_frame(
            filtered, fraction=pct / 100.0, value_per_conversion=value_pt, cost_per_contact=cost_pt
        )
    else:
        max_budget = max(1.0, round(n_seg * cost_pt))
        budget = st.slider(
            "Budget ($)",
            0.0,
            float(max_budget),
            float(min(100.0, max_budget)),
            step=max_budget / 200,
        )
        result = policy_for_frame(
            filtered, budget=budget, value_per_conversion=value_pt, cost_per_contact=cost_pt
        )

    band = result.band
    implied_budget = result.k * cost_pt
    with st.container(border=True):
        tiles = st.columns(4)
        tiles[0].metric("Customers Targeted", f"{result.k:,}", f"{result.fraction_targeted:.1%}")
        tiles[1].metric("Budget Spent", f"${implied_budget:,.0f}")
        tiles[2].metric("Incremental Conversions", f"{band.incremental_conversions:+.1f}")
        tiles[3].metric(
            "Net Value", f"${band.point:,.0f}", f"${band.low:,.0f} to ${band.high:,.0f}"
        )
        st.caption(
            rf"Value band sweeps value \${VALUE_PER_CONVERSION_RANGE[0]:.0f}"
            rf"-\${VALUE_PER_CONVERSION_RANGE[1]:.0f} x cost \${COST_PER_CONTACT_RANGE[0]:.2f}"
            rf"-\${COST_PER_CONTACT_RANGE[1]:.2f}; point uses value \${value_pt:,.2f}, "
            rf"cost \${cost_pt:.3f}. "
            "Incremental conversions are assumption-free (off the conversion Qini curve)."
        )

    _chart_card(
        "Net Value vs. How Many You Target",
        "Where the budget you picked lands on the value curve (orange line = current policy).",
        _value_fig(filtered, result.fraction_targeted, value_pt, cost_pt),
    )


if __name__ == "__main__":
    main()
