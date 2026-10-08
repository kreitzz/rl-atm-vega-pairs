"""Low-complexity additive payoff scores with the inherited discrete sizing head.

All imputation, quantile knots, centering, term scaling, interaction projection,
and score terciles use the supplied training rows only. The caller is responsible
for completed-outcome chronological folds. Scores are smooth; sizes remain the
50/75/100% ladder. Linear spline extrapolation avoids an additional clipping rule.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer


INTERACTIONS = (("rank_strength", "correlation"), ("rank_strength", "vix"),
                ("profit_365d", "count_365d"))
ALPHA_MULTIPLIERS = (.1, 1., 10.)
KINDS = ("linear", "gam", "gam_interactions")


def _values(frame, feature, median):
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(values), values, median)


@dataclass
class MainTerm:
    feature: str
    median: float
    constant: bool
    spline: SplineTransformer | None
    center: np.ndarray
    rms: float
    knots: np.ndarray

    def transform(self, frame):
        if self.constant or len(frame) == 0:
            return np.zeros((len(frame), len(self.center)))
        values = _values(frame, self.feature, self.median).reshape(-1, 1)
        basis = self.spline.transform(values) if self.spline is not None else values
        return (basis-self.center)/self.rms


@dataclass
class InteractionTerm:
    first: str
    second: str
    projection: np.ndarray
    center: np.ndarray
    rms: float
    constant: bool

    @property
    def name(self):
        return f"{self.first}:{self.second}"

    def transform(self, mains):
        a, b = mains[self.first], mains[self.second]
        if self.constant or len(a) == 0:
            return np.zeros((len(a), len(self.center)))
        tensor = np.einsum("ni,nj->nij", a, b).reshape(len(a), -1)
        local_mains = np.column_stack([np.ones(len(a)), a, b])
        return (tensor-local_mains@self.projection-self.center)/self.rms


def _fit_main(rows, feature, kind):
    raw = pd.to_numeric(rows[feature], errors="coerce").to_numpy(float)
    finite = raw[np.isfinite(raw)]
    median = float(np.median(finite)) if len(finite) else 0.
    values = np.where(np.isfinite(raw), raw, median)
    knots = np.unique(np.quantile(values, [0., .5, 1.]))
    constant = len(knots) == 1
    spline = None
    if kind != "linear" and not constant:
        spline = SplineTransformer(knots=knots.reshape(-1, 1), degree=2,
                                   include_bias=True, extrapolation="linear")
        basis = spline.fit_transform(values.reshape(-1, 1))
    else:
        basis = values.reshape(-1, 1)
    center = basis.mean(axis=0)
    centered = basis-center
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    constant = constant or rms <= 1e-12
    term = MainTerm(feature, median, constant, spline, center,
                    1. if constant else rms, knots)
    return term, term.transform(rows), values


def _fit_interaction(first, second, mains):
    a, b = mains[first], mains[second]
    tensor = np.einsum("ni,nj->nij", a, b).reshape(len(a), -1)
    local_mains = np.column_stack([np.ones(len(a)), a, b])
    # Project only on the two participating main terms: the result stays bivariate.
    projection = np.linalg.lstsq(local_mains, tensor, rcond=None)[0]
    residual = tensor-local_mains@projection
    center = residual.mean(axis=0)
    centered = residual-center
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    constant = rms <= 1e-10
    term = InteractionTerm(first, second, projection, center,
                           1. if constant else rms, constant)
    return term, term.transform(mains)


def tercile_size_codes(scores, edges):
    """Codes 2/3/4 mean 50/75/100%; equality goes into the higher size bucket."""
    scores, edges = np.asarray(scores, float), np.asarray(edges, float)
    if edges.shape != (2,) or edges[0] > edges[1] or not np.isfinite(edges).all():
        raise ValueError("Two finite, ordered training-score terciles are required")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite payoff scores cannot choose a size")
    return np.searchsorted(edges, scores, side="right")+2


@dataclass
class AdditiveSizingModel:
    features: list
    kind: str
    mains: dict
    interactions: dict
    ridge: Ridge
    group_slices: dict
    training_score_edges: np.ndarray
    training_mean: dict
    training_sd: dict
    feature_ranges: dict
    metadata: dict

    def term_bases(self, frame):
        main = {feature: self.mains[feature].transform(frame) for feature in self.features}
        groups = dict(main)
        groups.update({name: term.transform(main) for name, term in self.interactions.items()})
        return groups

    def design(self, frame):
        return np.column_stack(list(self.term_bases(frame).values()))

    def predict(self, frame):
        if len(frame) == 0:
            return np.empty(0, dtype=float)
        return self.ridge.predict(self.design(frame))

    def contributions(self, frame):
        groups = self.term_bases(frame)
        result = pd.DataFrame({"intercept": np.full(len(frame), float(self.ridge.intercept_))}, index=frame.index)
        for name, basis in groups.items():
            result[name] = basis@self.ridge.coef_[self.group_slices[name]]
        return result

    def size_codes(self, frame):
        return tercile_size_codes(self.predict(frame), self.training_score_edges)

    def sizes(self, frame):
        return self.size_codes(frame)/4.


def fit_model(rows, features, kind="gam", alpha_multiplier=1., target="target"):
    """Fit only supplied completed training rows; chronological CV is external."""
    features = list(features)
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind}; choose one of {KINDS}")
    if not features or len(features) != len(set(features)):
        raise ValueError("Features must be a nonempty unique list")
    forbidden = {target, "date", "end", "pair", "label_row", "finite_label", "train_eligible"}
    if forbidden.intersection(features):
        raise ValueError("Predictors cannot contain outcome or identity metadata")
    if len(rows) < 2 or not np.isfinite(alpha_multiplier) or alpha_multiplier <= 0:
        raise ValueError("At least two training rows and a positive penalty are required")
    y = pd.to_numeric(rows[target], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Training targets must all be available and finite")
    if kind == "gam_interactions" and set(sum(INTERACTIONS, ())) - set(features):
        raise ValueError("The interaction model requires all named participating features")
    mains, main_bases, means, sds, ranges = {}, {}, {}, {}, {}
    descriptions = []
    for feature in features:
        term, basis, values = _fit_main(rows, feature, kind)
        mains[feature], main_bases[feature] = term, basis
        means[feature], sds[feature] = float(values.mean()), float(values.std(ddof=0))
        ranges[feature] = [float(values.min()), float(values.max())]
        descriptions.append(dict(name=feature, kind="main", features=[feature],
            columns=basis.shape[1], constant=term.constant, median=term.median,
            knots=term.knots.tolist(), center=term.center.tolist(), group_rms=term.rms))
    interactions, groups = {}, dict(main_bases)
    if kind == "gam_interactions":
        for first, second in INTERACTIONS:
            term, basis = _fit_interaction(first, second, main_bases)
            interactions[term.name], groups[term.name] = term, basis
            descriptions.append(dict(name=term.name, kind="interaction", features=[first, second],
                columns=basis.shape[1], constant=term.constant, center=term.center.tolist(),
                group_rms=term.rms, projection=term.projection.tolist(),
                projection_scope="intercept and participating main bases only"))
    slices, position = {}, 0
    for name, basis in groups.items():
        slices[name] = slice(position, position+basis.shape[1])
        position += basis.shape[1]
    design = np.column_stack(list(groups.values()))
    ridge = Ridge(alpha=len(rows)*alpha_multiplier, solver="svd", fit_intercept=True).fit(design, y)
    predictions = ridge.predict(design)
    edges = np.quantile(predictions, [1/3, 2/3])
    metadata = dict(kind=kind, features=features, target=target, training_rows=len(rows),
        alpha_multiplier=float(alpha_multiplier), ridge_alpha=float(len(rows)*alpha_multiplier),
        basis_columns=design.shape[1], groups=descriptions, intercept=float(ridge.intercept_),
        coefficients=ridge.coef_.tolist(), training_score_edges=edges.tolist(),
        training_mean=means, training_sd=sds, feature_ranges=ranges,
        scaling="Training-centered term bases; each entire term has unit RMS Euclidean norm",
        extrapolation="linear", sizing="Training-score terciles map to discrete50/75/100% sizes; ties higher",
        training_only="Caller supplies causal completed-outcome rows and selects penalty chronologically")
    return AdditiveSizingModel(features, kind, mains, interactions, ridge, slices, edges,
                               means, sds, ranges, metadata)
