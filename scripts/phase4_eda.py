"""Phase 4 exploratory data analysis: figures and headline numbers.

Read-only analysis over the tidy processed datasets in ``data/processed/``. This
script never modifies processed data and is not part of the shipped pipeline; it
regenerates every figure in ``assets/eda/`` (SVG + PNG) and prints the numbers
that back the narrative report to stdout as JSON.

Run: ``uv run python scripts/phase4_eda.py``

Palette: Okabe-Ito, a colorblind-safe qualitative set, used consistently across
every figure. The uplift heatmap uses ColorBrewer RdBu, a colorblind-safe
diverging map, centered on zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paths -----------------------------------------------------------------

PROCESSED = Path("data/processed")
ASSETS = Path("assets/eda")

# --- Palette (Okabe-Ito, colorblind-safe) ----------------------------------

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
TREAT = OKABE["vermillion"]  # treated arm
CONTROL = OKABE["blue"]  # control arm
VISIT = OKABE["sky"]  # visit outcome
CONV = OKABE["orange"]  # conversion outcome
ACCENT = OKABE["green"]  # highlights / pass
WARN = OKABE["vermillion"]  # threshold lines
GRID = "#B0B0B0"
GRAY = "#666666"

# --- Matplotlib defaults ---------------------------------------------------

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
        "grid.color": GRID,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    }
)

SOURCE = "Source: Hillstrom (64k) and Criteo 1M subsample (seed 42) — data/processed/"

# Compact labels for the Hillstrom prior-year spend bands (avoid tick overlap).
SEG_SHORT = {
    "1) $0 - $100": "$0-100",
    "2) $100 - $200": "$100-200",
    "3) $200 - $350": "$200-350",
    "4) $350 - $500": "$350-500",
    "5) $500 - $750": "$500-750",
    "6) $750 - $1,000": "$750-1k",
    "7) $1,000 +": "$1k+",
}


# --- Helpers ---------------------------------------------------------------


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure to ``assets/eda/`` as both SVG (vector) and PNG (raster)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def prop_ci(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the normal-approx confidence interval for a proportion."""
    if n <= 0:
        return float("nan")
    return z * float(np.sqrt(p * (1.0 - p) / n))


def ate_ci(pt: float, nt: int, pc: float, nc: int, z: float = 1.96) -> float:
    """Half-width of the CI for a difference in proportions (treated - control)."""
    if nt <= 0 or nc <= 0:
        return float("nan")
    se = np.sqrt(pt * (1.0 - pt) / nt + pc * (1.0 - pc) / nc)
    return z * float(se)


def uplift_by_group(df: pd.DataFrame, group: pd.Series, outcome: str) -> pd.DataFrame:
    """Per-group treated-minus-control uplift on ``outcome`` with a 95% CI.

    Returns a frame indexed by the group values with columns ``ate``, ``ci``,
    ``nt``, ``nc`` (treated/control counts). Groups are kept in the order the
    index sorts, so callers control presentation order upstream.
    """
    rows = {}
    t = df["treatment"].to_numpy()
    y = df[outcome].to_numpy()
    g = group.to_numpy()
    for level in pd.unique(group.dropna()):
        m = g == level
        yt, yc = y[m & (t == 1)], y[m & (t == 0)]
        if len(yt) == 0 or len(yc) == 0:
            continue
        pt, pc = yt.mean(), yc.mean()
        rows[level] = {
            "ate": pt - pc,
            "ci": ate_ci(pt, len(yt), pc, len(yc)),
            "nt": len(yt),
            "nc": len(yc),
        }
    return pd.DataFrame(rows).T


def rank_deciles(s: pd.Series, q: int = 10) -> pd.Series:
    """Split a series into ``q`` equal-sized bins by rank (ties broken by order).

    Ranking first guarantees exactly ``q`` bins even when the feature is heavily
    tied/quantized, which the Criteo features are.
    """
    return pd.qcut(s.rank(method="first"), q, labels=False)


# --- Figures ---------------------------------------------------------------


def fig_arms(h: pd.DataFrame, c: pd.DataFrame) -> None:
    """fig01 — treated vs control counts and shares for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, df, title in [(axes[0], h, "Hillstrom"), (axes[1], c, "Criteo (1M)")]:
        n_c = int((df.treatment == 0).sum())
        n_t = int((df.treatment == 1).sum())
        total = n_c + n_t
        bars = ax.bar(
            ["Control", "Treated"],
            [n_c, n_t],
            color=[CONTROL, TREAT],
            width=0.62,
        )
        for b, n in zip(bars, [n_c, n_t], strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{n:,}\n{n / total:.0%}",
                ha="center",
                va="bottom",
                fontsize=9.5,
            )
        ax.set_title(title)
        ax.set_ylabel("customers" if title == "Hillstrom" else "rows")
        ax.set_ylim(0, max(n_c, n_t) * 1.18)
        ax.margins(x=0.2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Randomized arms: treated vs control", y=1.02)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig01_arms")


def fig_balance(balance_csv: str, name: str, title: str) -> None:
    """fig02/fig03 — SMD lollipop per covariate with the +/-0.1 imbalance lines."""
    b = pd.read_csv(PROCESSED / balance_csv).sort_values("smd")
    fig, ax = plt.subplots(figsize=(8.2, max(3.5, 0.34 * len(b) + 1.2)))
    y = np.arange(len(b))
    colors = [WARN if v else ACCENT for v in b["imbalanced"]]
    ax.hlines(y, 0, b["smd"], color=colors, linewidth=2, alpha=0.85)
    ax.scatter(b["smd"], y, color=colors, s=34, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=WARN, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        0.1, len(b) - 0.4, "  |SMD| = 0.1\n  imbalance flag", color=WARN, fontsize=8.5, va="top"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(b["covariate"], fontsize=8.5)
    ax.set_xlabel("standardized mean difference (treated - control)")
    ax.set_title(title)
    lim = max(0.12, float(b["abs_smd"].max()) * 1.25)
    ax.set_xlim(-lim, lim)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.03, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, name)


def fig_base_rates(stats: dict) -> None:
    """fig04 — visit vs conversion base rates per dataset, with the ~16x gap."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = ["Hillstrom", "Criteo (1M)"]
    visit = [stats["hillstrom"]["visit"]["base"], stats["criteo"]["visit"]["base"]]
    conv = [stats["hillstrom"]["conversion"]["base"], stats["criteo"]["conversion"]["base"]]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, visit, w, label="visit", color=VISIT)
    b2 = ax.bar(x + w / 2, conv, w, label="conversion", color=CONV)
    ax.set_yscale("log")
    ax.set_ylabel("base rate (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Visit is common, conversion is rare (~16x apart)")
    for bars, vals in [(b1, visit), (b2, conv)]:
        for b, v in zip(bars, vals, strict=True):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.2%}", ha="center", va="bottom", fontsize=9
            )
    for xi, dset in zip(x, ["hillstrom", "criteo"], strict=True):
        r = stats[dset]["visit_conv_ratio"]
        ax.annotate(
            f"{r:.0f}x",
            xy=(xi, np.sqrt(stats[dset]["visit"]["base"] * stats[dset]["conversion"]["base"])),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=GRAY,
        )
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig04_base_rates")


def fig_ate_forest(stats: dict) -> None:
    """fig05 — naive ATE forest plot for visit and conversion, both datasets."""
    rows = [
        ("Hillstrom - visit", "hillstrom", "visit", VISIT),
        ("Criteo - visit", "criteo", "visit", VISIT),
        ("Hillstrom - conversion", "hillstrom", "conversion", CONV),
        ("Criteo - conversion", "criteo", "conversion", CONV),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for i, (_label, dset, out, color) in enumerate(rows):
        s = stats[dset][out]
        y = len(rows) - 1 - i
        lo, hi = s["ate_lo"], s["ate_hi"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.4, solid_capstyle="round")
        ax.scatter([s["ate"]], [y], color=color, s=60, zorder=3)
        ax.text(hi, y + 0.16, f"+{s['ate'] * 100:.2f} pp", fontsize=9, color=GRAY, va="bottom")
    ax.axvline(0, color=GRAY, linewidth=0.9, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel("naive average treatment effect (treated - control), 95% CI")
    ax.set_title("Treatment lifts outcomes on average - but by how much, for whom?")
    ax.set_xlim(left=-0.004)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig05_ate_forest")


def fig_hillstrom_segments(h: pd.DataFrame, overall_ate: float) -> None:
    """fig06 — Hillstrom visit uplift across five customer slices, with 95% CIs."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    axes = axes.ravel()

    # (a) recency 1..12 as a line
    ax = axes[0]
    rec = uplift_by_group(h, h["recency"], "visit").sort_index()
    ax.errorbar(
        rec.index,
        rec["ate"],
        yerr=rec["ci"],
        fmt="o-",
        color=TREAT,
        ecolor=GRAY,
        elinewidth=1,
        capsize=2,
        markersize=4,
    )
    ax.set_title("by recency (months since purchase)")
    ax.set_xlabel("recency")
    ax.set_ylabel("visit uplift (treat - control)")

    # (b-e) categorical bars
    cats = [
        (
            "history_segment",
            "by prior-year spend band",
            sorted(h["history_segment"].cat.categories),
        ),
        ("channel", "by channel", sorted(h["channel"].cat.categories)),
        ("zip_code", "by zip segment", sorted(h["zip_code"].cat.categories)),
    ]
    for ax, (col, title, order) in zip(axes[1:4], cats, strict=True):
        u = uplift_by_group(h, h[col].astype(object), "visit").reindex(order)
        ax.bar(
            range(len(u)), u["ate"], yerr=u["ci"], color=VISIT, capsize=3, error_kw={"ecolor": GRAY}
        )
        ax.set_xticks(range(len(u)))
        if col == "history_segment":
            labs = [SEG_SHORT.get(str(x), str(x)) for x in u.index]
            ax.set_xticklabels(labs, fontsize=8, rotation=30, ha="right")
        else:
            ax.set_xticklabels([str(x) for x in u.index], fontsize=9, rotation=0)
        ax.set_title(title)
        ax.set_ylabel("visit uplift")

    # (e) newbie
    ax = axes[4]
    u = uplift_by_group(h, h["newbie"].map({0: "existing", 1: "newbie"}), "visit").reindex(
        ["existing", "newbie"]
    )
    ax.bar(range(len(u)), u["ate"], yerr=u["ci"], color=VISIT, capsize=3, error_kw={"ecolor": GRAY})
    ax.set_xticks(range(len(u)))
    ax.set_xticklabels(u.index)
    ax.set_title("by customer tenure")
    ax.set_ylabel("visit uplift")

    axes[5].axis("off")
    for ax in axes[:5]:
        ax.axhline(overall_ate, color=GRAY, linestyle="--", linewidth=1.0)
        ax.axhline(0, color="#000000", linewidth=0.7)
        ax.grid(axis="x", visible=False)
    axes[5].text(
        0.05,
        0.5,
        "Dashed line = overall\nvisit uplift "
        f"(+{overall_ate * 100:.2f} pp).\n\nBars above it = slices where\n"
        "treatment works harder.\nThis spread is the case\nfor targeting.",
        fontsize=11,
        va="center",
        color=GRAY,
    )
    fig.suptitle("Hillstrom: who responds to the email? Visit uplift by slice", y=1.01)
    fig.text(0.5, -0.02, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig06_hillstrom_lift_segments")


def fig_hillstrom_history_decile(h: pd.DataFrame, overall_ate: float) -> None:
    """fig07 — Hillstrom visit uplift across deciles of prior-year spend."""
    d = rank_deciles(h["history"])
    u = uplift_by_group(h, d, "visit").sort_index()
    # median history per decile for readable x labels
    med = h.groupby(d, observed=True)["history"].median()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.errorbar(
        u.index + 1,
        u["ate"],
        yerr=u["ci"],
        fmt="o-",
        color=TREAT,
        ecolor=GRAY,
        elinewidth=1,
        capsize=3,
        markersize=5,
    )
    ax.axhline(
        overall_ate,
        color=GRAY,
        linestyle="--",
        linewidth=1.0,
        label=f"overall (+{overall_ate * 100:.2f} pp)",
    )
    ax.set_xticks(u.index + 1)
    ax.set_xticklabels([f"D{i + 1}\n~${med[i]:.0f}" for i in u.index], fontsize=8)
    ax.set_xlabel("prior-year spend decile (low to high, with median $)")
    ax.set_ylabel("visit uplift (treat - control)")
    ax.set_title("Hillstrom: visit uplift rises with prior-year spend")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig07_hillstrom_lift_history_decile")


def criteo_feature_uplift(c: pd.DataFrame, outcome: str = "visit") -> pd.DataFrame:
    """Per-feature, per-decile uplift matrix (features x 10 deciles) for Criteo."""
    feats = [f"f{i}" for i in range(12)]
    mat = {}
    for f in feats:
        d = rank_deciles(c[f])
        u = uplift_by_group(c, d, outcome).sort_index()
        mat[f] = u["ate"].reindex(range(10)).to_numpy()
    return pd.DataFrame(mat, index=[f"D{i + 1}" for i in range(10)]).T


def fig_criteo_heatmap(mat: pd.DataFrame) -> None:
    """fig08 — Criteo visit uplift heatmap: 12 features by 10 deciles, diverging."""
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    vmax = float(np.nanmax(np.abs(mat.to_numpy())))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=9)
    ax.set_xlabel("feature decile (low to high)")
    ax.set_title("Criteo: visit uplift by feature decile (red = treatment helps more)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("visit uplift (treat - control)")
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    fig.text(0.5, -0.02, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig08_criteo_lift_heatmap")


def fig_criteo_top_features(c: pd.DataFrame, mat: pd.DataFrame) -> list[str]:
    """fig09 — the three Criteo features whose uplift varies most across deciles."""
    spread = (mat.max(axis=1) - mat.min(axis=1)).sort_values(ascending=False)
    top = list(spread.head(3).index)
    colors = [TREAT, ACCENT, OKABE["purple"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for f, col in zip(top, colors, strict=True):
        d = rank_deciles(c[f])
        u = uplift_by_group(c, d, "visit").sort_index()
        ax.errorbar(
            u.index + 1,
            u["ate"],
            yerr=u["ci"],
            fmt="o-",
            color=col,
            ecolor=col,
            elinewidth=0.8,
            capsize=2,
            markersize=4,
            alpha=0.9,
            label=f,
        )
    ax.axhline(0, color="#000000", linewidth=0.7)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("feature decile (low to high)")
    ax.set_ylabel("visit uplift (treat - control)")
    ax.set_title("Criteo: uplift swings across deciles for the top-varying features")
    ax.legend(title="feature")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig09_criteo_lift_topfeatures")
    return top


def fig_spend_converters(h: pd.DataFrame, s: dict) -> None:
    """fig10 — distribution of spend among Hillstrom converters."""
    conv = h.loc[h.conversion == 1, "spend"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.hist(conv, bins=40, color=VISIT, edgecolor="white", linewidth=0.4)
    ax.axvline(s["mean"], color=TREAT, linewidth=2, label=f"mean ${s['mean']:.2f}")
    ax.axvline(
        s["median"], color=ACCENT, linewidth=2, linestyle="--", label=f"median ${s['median']:.2f}"
    )
    ax.axvspan(
        s["q25"], s["q75"], color=WARN, alpha=0.10, label=f"IQR ${s['q25']:.0f}-${s['q75']:.0f}"
    )
    ax.set_xlabel("order value among converters ($)")
    ax.set_ylabel("converters")
    ax.set_title(f"Hillstrom: value per conversion = mean spend = ${s['mean']:.2f} (n={s['n']})")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig10_spend_converters")


def fig_criteo_feature_dist(c: pd.DataFrame) -> None:
    """fig11 — Criteo feature distributions (z-scored) to show scale and outliers."""
    feats = [f"f{i}" for i in range(12)]
    z = (c[feats] - c[feats].mean()) / c[feats].std()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bp = ax.boxplot(
        [z[f].to_numpy() for f in feats],
        tick_labels=feats,
        showfliers=True,
        flierprops={
            "marker": ".",
            "markersize": 2,
            "markerfacecolor": GRAY,
            "markeredgecolor": "none",
            "alpha": 0.3,
        },
        medianprops={"color": TREAT, "linewidth": 1.5},
        boxprops={"color": CONTROL},
        whiskerprops={"color": CONTROL},
        capprops={"color": CONTROL},
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(VISIT)
        patch.set_alpha(0.5)
    # Rasterize the outlier clouds: ~1M points would otherwise bloat the SVG to
    # ~90 MB as individual vector paths. Rasterized, the SVG embeds a small image.
    for flier in bp["fliers"]:
        flier.set_rasterized(True)
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("standardized value (z-score)")
    ax.set_title("Criteo: anonymized features are quantized and heavy-tailed")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig11_criteo_feature_dist")


def fig_hillstrom_history_dist(h: pd.DataFrame) -> None:
    """fig12 — Hillstrom prior-year spend distribution (right-skewed, long tail)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.hist(h["history"], bins=60, color=VISIT, edgecolor="white", linewidth=0.3)
    ax.axvline(
        h["history"].mean(), color=TREAT, linewidth=2, label=f"mean ${h['history'].mean():.0f}"
    )
    ax.axvline(
        h["history"].median(),
        color=ACCENT,
        linewidth=2,
        linestyle="--",
        label=f"median ${h['history'].median():.0f}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("prior-year spend ($)")
    ax.set_ylabel("customers (log scale)")
    ax.set_title(f"Hillstrom: prior-year spend is right-skewed (max ${h['history'].max():.0f})")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.04, SOURCE, ha="center", fontsize=8, color=GRAY)
    fig.tight_layout()
    save(fig, "fig12_hillstrom_history_dist")


# --- Number crunching ------------------------------------------------------


def compute_stats(h: pd.DataFrame, c: pd.DataFrame) -> dict:
    """Compute every headline number the report cites, straight from the data."""
    out: dict = {}
    for name, df in [("hillstrom", h), ("criteo", c)]:
        d: dict = {
            "rows": int(len(df)),
            "treated": int((df.treatment == 1).sum()),
            "control": int((df.treatment == 0).sum()),
            "missing": int(df.isna().sum().sum()),
        }
        for o in ("visit", "conversion"):
            nt = int((df.treatment == 1).sum())
            nc = int((df.treatment == 0).sum())
            pt = float(df.loc[df.treatment == 1, o].mean())
            pc = float(df.loc[df.treatment == 0, o].mean())
            ate = pt - pc
            ci = ate_ci(pt, nt, pc, nc)
            d[o] = {
                "base": round(float(df[o].mean()), 5),
                "pt": round(pt, 5),
                "pc": round(pc, 5),
                "ate": round(ate, 5),
                "ate_lo": round(ate - ci, 5),
                "ate_hi": round(ate + ci, 5),
            }
        d["visit_conv_ratio"] = round(float(df["visit"].mean() / df["conversion"].mean()), 2)
        out[name] = d

    conv = h.loc[h.conversion == 1, "spend"]
    n = int(conv.shape[0])
    out["spend_converters"] = {
        "n": n,
        "mean": round(float(conv.mean()), 2),
        "median": round(float(conv.median()), 2),
        "std": round(float(conv.std()), 2),
        "min": round(float(conv.min()), 2),
        "max": round(float(conv.max()), 2),
        "q25": round(float(conv.quantile(0.25)), 2),
        "q75": round(float(conv.quantile(0.75)), 2),
        "mean_ci_lo": round(float(conv.mean() - 1.96 * conv.std() / np.sqrt(n)), 2),
        "mean_ci_hi": round(float(conv.mean() + 1.96 * conv.std() / np.sqrt(n)), 2),
    }
    out["quality"] = {
        "h_conv_no_visit": int(((h.conversion == 1) & (h.visit == 0)).sum()),
        "c_conv_no_visit": int(((c.conversion == 1) & (c.visit == 0)).sum()),
        "h_spend_pos_no_conv": int(((h.spend > 0) & (h.conversion == 0)).sum()),
        "h_conv_zero_spend": int(((h.conversion == 1) & (h.spend <= 0)).sum()),
        "h_missing": int(h.isna().sum().sum()),
        "c_missing": int(c.isna().sum().sum()),
        "h_recency_range": [int(h.recency.min()), int(h.recency.max())],
        "h_history_max": round(float(h.history.max()), 2),
        "zip_has_surburban_typo": bool("Surburban" in set(map(str, h.zip_code.cat.categories))),
    }
    return out


# --- Main ------------------------------------------------------------------


def main() -> None:
    """Regenerate every Phase 4 figure and print the backing numbers as JSON."""
    h = pd.read_parquet(PROCESSED / "hillstrom.parquet")
    c = pd.read_parquet(PROCESSED / "criteo.parquet")

    stats = compute_stats(h, c)
    h_visit_ate = stats["hillstrom"]["visit"]["ate"]

    fig_arms(h, c)
    fig_balance(
        "hillstrom_balance.csv", "fig02_balance_hillstrom", "Hillstrom: covariate balance (SMD)"
    )
    fig_balance("criteo_balance.csv", "fig03_balance_criteo", "Criteo: covariate balance (SMD)")
    fig_base_rates(stats)
    fig_ate_forest(stats)
    fig_hillstrom_segments(h, h_visit_ate)
    fig_hillstrom_history_decile(h, h_visit_ate)
    mat = criteo_feature_uplift(c, "visit")
    fig_criteo_heatmap(mat)
    top = fig_criteo_top_features(c, mat)
    fig_spend_converters(h, stats["spend_converters"])
    fig_criteo_feature_dist(c)
    fig_hillstrom_history_dist(h)

    stats["criteo_top_varying_features"] = top
    stats["criteo_heatmap_absmax_uplift"] = round(float(np.nanmax(np.abs(mat.to_numpy()))), 5)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
