"""Direct uplift models: a transformed-outcome learner and a divergence tree/forest.

Unlike the meta-learners, which subtract two ordinary outcome models, these model
the treatment effect *directly*.

- **Class transformation** -- the propensity-corrected transformed outcome
  (Athey & Imbens): regress ``Y* = Y * (T - p) / (p * (1 - p))`` on the features,
  where ``p`` is the (constant, randomized) treated fraction. ``E[Y* | X]`` equals
  the true uplift ``tau(X)``, so a single regressor recovers it. The propensity
  term is what keeps it unbiased on unequal arms (Hillstrom 2:1, Criteo ~85/15);
  the plain "class variable transformation" assumes 50/50 and is biased here.

- **Uplift tree / forest** -- a decision tree whose splits maximize the divergence
  between the treated and control response rates in the children (Euclidean
  criterion), so each leaf isolates a region of similar uplift; the leaf value is
  the observed ``p_treated - p_control`` there. The forest bags such trees over
  bootstrapped rows and a random feature subset per split. Both are hand-rolled
  over numpy (no causalml/econml), consistent with the meta-learners (D24).

Every model shares the ``fit(X, treatment, y)`` / ``predict_uplift(X)`` interface
used across the bakeoff. ``X`` is feature-only; a stray ``treatment`` column is
dropped defensively. pandas ``category`` columns are consumed natively by LightGBM
in the transformed-outcome model and ordinal-encoded for the hand-rolled tree.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from uplift.data.schema import TREATMENT_COL
from uplift.data.splits import SEED
from uplift.models.meta import BASE_LGBM_PARAMS


def _features_only(x: pd.DataFrame) -> pd.DataFrame:
    """Drop a stray treatment column so it is never used as an ordinary feature."""
    return x.drop(columns=[TREATMENT_COL], errors="ignore")


class ClassTransformation:
    """Transformed-outcome learner: regress the propensity-corrected label ``Y*``.

    On randomized data the propensity ``p`` is the constant treated fraction, so
    ``Y* = Y * (T - p) / (p * (1 - p))`` has conditional mean equal to the uplift.
    A single LightGBM regressor (the shared base learner) fits it, and its
    prediction *is* the uplift -- no differencing of two models.
    """

    def __init__(self, seed: int = SEED, propensity: float | None = None, **params: object) -> None:
        self.params = {**BASE_LGBM_PARAMS, "random_state": seed, **params}
        self.model = LGBMRegressor(**self.params)
        self.propensity = propensity
        self.propensity_: float = 0.5

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Fit the base regressor on the propensity-corrected transformed outcome."""
        features = _features_only(x)
        t = np.asarray(treatment, dtype="float64")
        y = np.asarray(y, dtype="float64")
        p = float(t.mean()) if self.propensity is None else float(self.propensity)
        self.propensity_ = p
        y_star = y * (t - p) / (p * (1.0 - p))
        self.model.fit(features, y_star)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Return the regressor's prediction, which estimates ``tau(X)`` directly."""
        return self.model.predict(_features_only(x)).astype("float64")


# --- Hand-rolled uplift tree / forest --------------------------------------


def _encode(x: pd.DataFrame, categories: dict[str, pd.Index] | None) -> tuple[np.ndarray, dict]:
    """Ordinal-encode features to a float matrix the numeric tree can split on.

    Numeric columns pass through; ``category`` columns become their integer codes
    (unseen levels at predict time map to ``-1``). Returns the matrix and the
    per-column category index used, so ``predict`` encodes identically.
    """
    features = _features_only(x)
    cats: dict[str, pd.Index] = {} if categories is None else categories
    cols: list[np.ndarray] = []
    learn = categories is None
    for name in features.columns:
        col = features[name]
        if isinstance(col.dtype, pd.CategoricalDtype):
            if learn:
                cats[name] = col.cat.categories
            codes = pd.Categorical(col, categories=cats[name]).codes
            cols.append(np.asarray(codes, dtype="float64"))
        else:
            cols.append(np.asarray(col, dtype="float64"))
    return np.column_stack(cols), cats


class _Node:
    """A split node (``feature``/``bin`` set) or a leaf (``value`` set)."""

    __slots__ = ("feature", "bin", "left", "right", "value")

    def __init__(self) -> None:
        self.feature: int = -1
        self.bin: int = 0
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.value: float = 0.0


class UpliftTree:
    """Divergence uplift tree: split to separate treated/control response rates.

    Features are histogram-binned (quantile edges) so split search is a handful of
    ``bincount`` tallies per node. A split is scored by the sample-weighted squared
    child uplift ``sum_c (n_c / n) * (p_t^c - p_c^c)^2`` and kept only if it beats
    the node's own squared uplift and leaves at least ``min_arm`` treated and
    control rows on each side. Leaf value is ``p_treated - p_control`` there.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = 400,
        min_arm: int = 100,
        n_bins: int = 32,
        max_features: int | None = None,
        min_gain: float = 1e-6,
        seed: int = SEED,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_arm = min_arm
        self.n_bins = n_bins
        self.max_features = max_features
        self.min_gain = min_gain
        self.seed = seed
        self.edges_: list[np.ndarray] = []
        self.categories_: dict[str, pd.Index] = {}
        self.root_: _Node | None = None

    def _bin(self, xn: np.ndarray) -> np.ndarray:
        """Digitize each column with the fitted quantile edges into bin indices."""
        binned = np.empty(xn.shape, dtype="int32")
        for j, edges in enumerate(self.edges_):
            binned[:, j] = np.digitize(xn[:, j], edges)
        return binned

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Encode, bin, and grow the tree from all rows down to the depth/size limits."""
        xn, self.categories_ = _encode(x, None)
        t = np.asarray(treatment).astype(bool)
        ypos = np.asarray(y) == 1

        self.edges_ = []
        for j in range(xn.shape[1]):
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            self.edges_.append(np.unique(np.quantile(xn[:, j], qs)))
        xb = self._bin(xn)

        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(xb))
        self.root_ = self._grow(xb, t, ypos, idx, depth=0, rng=rng)
        return self

    def _grow(
        self,
        xb: np.ndarray,
        t: np.ndarray,
        ypos: np.ndarray,
        idx: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> _Node:
        """Recursively split ``idx``; return a leaf when no split clears the guards."""
        node = _Node()
        ti, yi = t[idx], ypos[idx]
        node.value = _uplift(ti.sum(), yi[ti].sum(), (~ti).sum(), yi[~ti].sum())

        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            return node

        parent_score = node.value * node.value
        best = self._best_split(xb, ti, yi, idx, rng)
        if best is None or best[2] - parent_score <= self.min_gain:
            return node

        feature, bin_k, _ = best
        go_left = xb[idx, feature] < bin_k
        node.feature, node.bin = feature, bin_k
        node.left = self._grow(xb, t, ypos, idx[go_left], depth + 1, rng)
        node.right = self._grow(xb, t, ypos, idx[~go_left], depth + 1, rng)
        return node

    def _best_split(
        self,
        xb: np.ndarray,
        ti: np.ndarray,
        yi: np.ndarray,
        idx: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[int, int, float] | None:
        """Return ``(feature, bin, score)`` of the highest-scoring valid split, or None."""
        n_features = xb.shape[1]
        feats = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feats = rng.choice(feats, size=self.max_features, replace=False)

        n = len(idx)
        best: tuple[int, int, float] | None = None
        for j in feats:
            b = xb[idx, j]
            nb = int(b.max()) + 1
            if nb < 2:
                continue
            tt = np.bincount(b[ti], minlength=nb).astype("float64")
            tp = np.bincount(b[ti & yi], minlength=nb).astype("float64")
            ct = np.bincount(b[~ti], minlength=nb).astype("float64")
            cp = np.bincount(b[~ti & yi], minlength=nb).astype("float64")

            l_tt, l_tp = np.cumsum(tt)[:-1], np.cumsum(tp)[:-1]
            l_ct, l_cp = np.cumsum(ct)[:-1], np.cumsum(cp)[:-1]
            r_tt, r_tp = tt.sum() - l_tt, tp.sum() - l_tp
            r_ct, r_cp = ct.sum() - l_ct, cp.sum() - l_cp

            valid = (
                (l_tt >= self.min_arm)
                & (l_ct >= self.min_arm)
                & (r_tt >= self.min_arm)
                & (r_ct >= self.min_arm)
                & (l_tt + l_ct >= self.min_samples_leaf)
                & (r_tt + r_ct >= self.min_samples_leaf)
            )
            if not valid.any():
                continue

            with np.errstate(invalid="ignore", divide="ignore"):
                u_l = l_tp / l_tt - l_cp / l_ct
                u_r = r_tp / r_tt - r_cp / r_ct
                n_l, n_r = l_tt + l_ct, r_tt + r_ct
                score = (n_l * u_l * u_l + n_r * u_r * u_r) / n

            score = np.where(valid, score, -np.inf)
            k = int(np.argmax(score))
            if best is None or score[k] > best[2]:
                best = (int(j), k + 1, float(score[k]))
        return best

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Route each row to its leaf and return the leaf uplift."""
        xn, _ = _encode(x, self.categories_)
        xb = self._bin(xn)
        out = np.empty(len(xb), dtype="float64")
        self._route(self.root_, xb, np.arange(len(xb)), out)
        return out

    def _route(self, node: _Node | None, xb: np.ndarray, idx: np.ndarray, out: np.ndarray) -> None:
        """Fill ``out[idx]`` by descending the tree with boolean index masks."""
        if node is None or node.feature < 0:
            out[idx] = 0.0 if node is None else node.value
            return
        go_left = xb[idx, node.feature] < node.bin
        self._route(node.left, xb, idx[go_left], out)
        self._route(node.right, xb, idx[~go_left], out)


class UpliftForest:
    """Bagged uplift trees: bootstrap rows, random feature subset per split, average.

    Each tree fits on a bootstrap sample (optionally capped at ``max_samples`` rows
    for large data) with ``max_features`` candidates considered per split; the
    forest uplift is the mean of the trees' leaf uplifts.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 6,
        min_samples_leaf: int = 400,
        min_arm: int = 100,
        n_bins: int = 32,
        max_features: int | None = None,
        max_samples: int | None = None,
        seed: int = SEED,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_arm = min_arm
        self.n_bins = n_bins
        self.max_features = max_features
        self.max_samples = max_samples
        self.seed = seed
        self.trees_: list[UpliftTree] = []

    def _default_max_features(self, n_features: int) -> int:
        """Per-split candidate count: the given ``max_features`` or ``sqrt(n)``."""
        if self.max_features is not None:
            return self.max_features
        return max(1, int(np.sqrt(n_features)))

    def fit(self, x: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> Self:
        """Fit ``n_estimators`` trees on bootstrap samples with per-tree seeds."""
        features = _features_only(x)
        t = np.asarray(treatment)
        y = np.asarray(y)
        n = len(features)
        draw = n if self.max_samples is None else min(self.max_samples, n)
        max_features = self._default_max_features(features.shape[1])
        rng = np.random.default_rng(self.seed)

        self.trees_ = []
        for b in range(self.n_estimators):
            rows = rng.integers(0, n, size=draw)
            tree = UpliftTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_arm=self.min_arm,
                n_bins=self.n_bins,
                max_features=max_features,
                seed=self.seed + b + 1,
            )
            tree.fit(features.iloc[rows], t[rows], y[rows])
            self.trees_.append(tree)
        return self

    def predict_uplift(self, x: pd.DataFrame) -> np.ndarray:
        """Return the mean per-row uplift across all trees."""
        preds = np.mean([tree.predict_uplift(x) for tree in self.trees_], axis=0)
        return preds.astype("float64")


def _uplift(n_t: float, pos_t: float, n_c: float, pos_c: float) -> float:
    """Response-rate difference ``p_treated - p_control`` (0.0 if an arm is empty)."""
    if n_t == 0 or n_c == 0:
        return 0.0
    return float(pos_t / n_t - pos_c / n_c)
