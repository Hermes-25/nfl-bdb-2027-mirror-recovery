"""Exploratory, count-aware OL prediction; no test-player outcome enters a prediction.

Run with standardized CSV columns player_id, cohort, y, n, numerical feature columns,
and a JSON fold map {"outer": {"player_id": fold}, "inner_seed": 72027}.
Outer folds must be shared/frozen before model comparison. Every model uses the same
rows and out-of-player folds. Fits are empirical Bayes/MAP, not full posterior Bayes.
"""

import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

from scipy.optimize import minimize

from scipy.special import betaln, digamma, expit, gammaln, logit

def scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)

class FoldPreprocessor:
    def fit(self, frame, columns, add_missing=True):
        self.columns = list(columns)
        raw = frame[self.columns].to_numpy(float) if self.columns else np.zeros((len(frame), 0))
        self.med = np.array([np.nanmedian(x) if np.isfinite(x).any() else 0 for x in raw.T])
        self.missing_columns = np.where(~np.isfinite(raw).all(axis=0))[0] if add_missing else np.array([], dtype=int)
        x = self._impute(raw)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1
        return self

    def _impute(self, raw):
        miss = ~np.isfinite(raw)
        imp = np.where(miss, self.med[None, :], raw)
        return np.column_stack([imp, miss[:, self.missing_columns].astype(float)])

    def transform(self, frame):
        raw = frame[self.columns].to_numpy(float) if self.columns else np.zeros((len(frame), 0))
        return (self._impute(raw) - self.mean) / self.std

class BetaBinomialRegressor:
    """y_i|p_i~Binomial(n_i,p_i); p_i~Beta(mu_i*phi,(1-mu_i)*phi).

    mu_i=logistic(intercept + standardized covariates beta). Normal ridge priors
    shrink the slopes. log(phi)~N(log(40),2^2) prevents unstable boundary estimates.
    New-player mean is mu_i; training-player posterior summaries are descriptive
    only and are never scored as held-out forecasts.
    """
    def __init__(self, penalty=10.0, family="beta_binomial"):
        self.penalty, self.family = float(penalty), family

    def fit(self, x, y, n):
        x, y, n = np.asarray(x, float), np.asarray(y, float), np.asarray(n, float)
        design = np.column_stack([np.ones(len(y)), x])
        init = np.zeros(design.shape[1] + (self.family == "beta_binomial"))
        init[0] = logit(np.clip(y.sum() / n.sum(), 1e-4, 1 - 1e-4))
        if self.family == "beta_binomial":
            init[-1] = np.log(40.0)
        logchoose = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)

        def objective(theta):
            coef = theta[:design.shape[1]]
            mu = np.clip(expit(design @ coef), 1e-10, 1 - 1e-10)
            if self.family == "beta_binomial":
                phi = np.exp(theta[-1])
                a, b = mu * phi, (1 - mu) * phi
                loglike = logchoose + betaln(y + a, n - y + b) - betaln(a, b)
                da = digamma(y + a) - digamma(a)
                db = digamma(n - y + b) - digamma(b)
                d_eta = phi * (da - db) * mu * (1 - mu)
                d_logphi = phi * (mu * da + (1 - mu) * db + digamma(phi) - digamma(n + phi))
                prior = 0.5 * ((theta[-1] - np.log(40)) / 2) ** 2
                gradphi = -d_logphi.sum() + (theta[-1] - np.log(40)) / 4
            elif self.family == "binomial":
                loglike = logchoose + y * np.log(mu) + (n - y) * np.log1p(-mu)
                d_eta, prior = y - n * mu, 0.0
            else:
                raise ValueError(self.family)
            grad = -(design.T @ d_eta)
            grad[1:] += self.penalty * coef[1:]
            val = -loglike.sum() + 0.5 * self.penalty * np.square(coef[1:]).sum() + prior
            if self.family == "beta_binomial":
                grad = np.r_[grad, gradphi]
            return float(val), grad

        bounds = [(-12, 12)] * design.shape[1]
        if self.family == "beta_binomial":
            bounds.append((np.log(0.05), np.log(100000)))
        result = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 1000, "maxls": 50, "ftol": 1e-9, "gtol": 1e-6})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Fit failed: {result.message}")
        self.coef = result.x[:design.shape[1]]
        self.phi = float(np.exp(result.x[-1])) if self.family == "beta_binomial" else None
        self.convergence = {"success": bool(result.success), "iterations": int(result.nit),
                            "objective": float(result.fun), "message": str(result.message)}
        return self

    def predict(self, x):
        return np.clip(expit(self.coef[0] + np.asarray(x) @ self.coef[1:]), 1e-8, 1 - 1e-8)

def metrics(y, n, p, phi=None):
    y, n, p = map(lambda z: np.asarray(z, float), (y, n, p))
    rate = y / n
    brier = rate * (1 - p) ** 2 + (1 - rate) * p ** 2
    ll = -rate * np.log(p) - (1 - rate) * np.log1p(-p)
    result = {"equal_player_brier": float(brier.mean()),
              "opportunity_weighted_brier": float(np.average(brier, weights=n)),
              "equal_player_logloss": float(ll.mean()),
              "opportunity_weighted_logloss": float(np.average(ll, weights=n)),
              "rate_mse": float(np.square(rate - p).mean()),
              "rate_mae": float(np.abs(rate - p).mean()),
              "predicted_mean_rate": float(p.mean()), "observed_mean_rate": float(rate.mean())}
    if phi is not None:
        a, b = p * phi, (1 - p) * phi
        logchoose = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
        result["equal_player_count_nll"] = float(-(logchoose + betaln(y + a, n - y + b) - betaln(a, b)).mean())
    return result

def inner_partition(frame, nfolds=3, seed=87007):
    """Outcome-independent, cohort-balanced ID partition of training players."""
    folds = {}
    for cohort, rows in frame.groupby("cohort", sort=True):
        ids = rows.player_id.astype(str).tolist()
        ids.sort(key=lambda z: hashlib.sha256(f"{seed}|{cohort}|{z}".encode()).hexdigest())
        folds.update({pid: i % min(nfolds, len(ids)) for i, pid in enumerate(ids)})
    return np.array([folds[str(pid)] for pid in frame.player_id])

def fit_predict(train, test, columns, penalty, family):
    pre = FoldPreprocessor().fit(train, columns, add_missing=family not in ("ridge", "mean"))
    if family in ("ridge", "mean"):
        x, rate = pre.transform(train), (train.y / train.n).to_numpy()
        model = RateRidgeRegressor(penalty).fit(x, rate)
    else:
        model = BetaBinomialRegressor(penalty, family).fit(pre.transform(train), train.y, train.n)
    return model.predict(pre.transform(test)), model, pre

class RateRidgeRegressor:
    """Historical equal-player ridge comparator with train-only median imputation."""
    def __init__(self, penalty):
        self.penalty, self.phi = penalty, None

    def fit(self, x, rate):
        intercept = float(rate.mean())
        slope = np.linalg.solve(x.T @ x + self.penalty * np.eye(x.shape[1]), x.T @ (rate - intercept))
        self.coef = np.r_[intercept, slope]
        self.convergence = {"success": True, "iterations": 1, "message": "closed-form ridge"}
        return self

    def predict(self, x):
        return np.clip(self.coef[0] + x @ self.coef[1:], 0.001, 0.999)

def nested_cv(frame, outer, feature_sets, penalties=(0.1, 1, 10, 100), seed=87007, outer_records=None):
    outputs, fits = [], []
    for fold in sorted(set(outer)):
        train, test = frame.loc[outer != fold].copy(), frame.loc[outer == fold].copy()
        if outer_records is None:
            infold = inner_partition(train, seed=seed)
        else:
            record = next(z for z in outer_records if int(z['fold']) == int(fold))
            train_ids, test_ids = set(train.player_id.astype(str)), set(test.player_id.astype(str))
            if train_ids != {str(z) for z in record['train_ids']} or test_ids != {str(z) for z in record['test_ids']}:
                raise ValueError('Outer records do not match current player population.')
            inner = {}
            for k, rec in enumerate(record['inner_folds']):
                itr, ite = {str(z) for z in rec['train_ids']}, {str(z) for z in rec['test_ids']}
                if itr & ite or itr | ite != train_ids:
                    raise ValueError('Inner train/test split fails player isolation or full coverage.')
                for pid in ite:
                    if pid in inner:
                        raise ValueError('Player appears in multiple inner evaluation folds.')
                    inner[pid] = k
            if set(inner) != train_ids:
                raise ValueError('Inner evaluation folds do not cover each outer training player exactly once.')
            infold = np.array([inner[str(pid)] for pid in train.player_id])
        for label, spec in feature_sets.items():
            columns, family = spec["columns"], spec.get("family", "beta_binomial")
            if "fixed_penalty" in spec:
                selected, curve = float(spec["fixed_penalty"]), []
            elif not columns:
                selected, curve = 0.0, []
            else:
                curve = []
                for penalty in penalties:
                    pred = np.full(len(train), np.nan)
                    for inner in sorted(set(infold)):
                        fitmask = infold != inner
                        p, _, _ = fit_predict(train.loc[fitmask], train.loc[~fitmask], columns, penalty, family)
                        pred[~fitmask] = p
                    loss = metrics(train.y, train.n, pred)["equal_player_brier"]
                    curve.append({"penalty": penalty, "inner_brier": loss})
                best = min(v["inner_brier"] for v in curve)
                # Exact ties prefer stronger shrinkage; no outer outcome guides this.
                selected = max(v["penalty"] for v in curve if v["inner_brier"] <= best + 1e-12)
            p, model, pre = fit_predict(train, test, columns, selected, family)
            for ix, probability in zip(test.index, p):
                outputs.append({"player_id": str(frame.loc[ix, "player_id"]), "cohort": int(frame.loc[ix, "cohort"]),
                                "model": label, "fold": int(fold), "y": int(frame.loc[ix, "y"]),
                                "n": int(frame.loc[ix, "n"]), "prediction": float(probability),
                                "phi": model.phi, "penalty": selected})
            fits.append({"model": label, "fold": int(fold), "train_players": len(train), "test_players": len(test),
                         "train_ids": train.player_id.tolist(), "test_ids": test.player_id.tolist(),
                         "inner_folds": [{"train_ids": train.loc[infold != k, 'player_id'].tolist(),
                                          "test_ids": train.loc[infold == k, 'player_id'].tolist()}
                                         for k in sorted(set(infold))],
                         "selected_penalty": selected, "inner_curve": curve, "phi": model.phi,
                         "coef": model.coef.tolist(), "columns": columns,
                         "missing_indicator_columns": [columns[i] for i in pre.missing_columns],
                         "convergence": model.convergence})
    pred = pd.DataFrame(outputs)
    summary = {name: metrics(rows.y, rows.n, rows.prediction,
                            rows.phi if rows.phi.notna().all() else None)
               for name, rows in pred.groupby("model")}
    return pred, fits, summary