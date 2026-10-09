"""Dense-grid functional conditioning for repeated kinematic turn curves.

This is a simple measurement-aware finite-grid analogue of functional conditional
estimation, NOT a reproduction of the sparse-grid PACE algorithm. Every fit must
occur inside the training split. Turns from one attempt are dependent: aggregate
them into one attempt curve before using this estimator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FunctionalEB:
    """Estimate repeat-attempt noise, then condition two latent curve components.

    DataFrame columns: player_id, attempt_id, c00..c14 (or another fixed grid).
    All subjects contribute one equally weighted player-average curve. The
    within-player covariance is estimated only from between-attempt deviations.
    Single-attempt athletes are conditioned with pooled attempt noise, rather
    than assigned certainty. Repeats within one session are not test-retest data.
    """
    def __init__(self, curve_columns: list[str], n_components: int = 2):
        self.curve_columns = curve_columns
        self.n_components = n_components

    @staticmethod
    def _psd(matrix):
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        return (vectors * np.maximum(values, 0)) @ vectors.T

    def fit(self, attempts: pd.DataFrame):
        cols = self.curve_columns
        clean = attempts.dropna(subset=cols)
        groups = list(clean.groupby("player_id", sort=True))
        if len(groups) < 4:
            raise ValueError("At least four training players are required")
        means = np.array([g[cols].to_numpy(float).mean(0) for _, g in groups])
        counts = np.array([len(g) for _, g in groups])
        self.mean_ = means.mean(0)
        # Covariance uses distinct attempts, not turns. Same-session attempts can
        # still share coach, instruction and sequence effects.
        within = np.zeros((len(cols), len(cols)))
        dof = 0
        repeated_players = 0
        for _, group in groups:
            values = group[cols].to_numpy(float)
            if len(values) > 1:
                residual = values - values.mean(0)
                within += residual.T @ residual
                dof += len(values) - 1
                repeated_players += 1
        self.repeated_players_ = repeated_players
        self.noise_estimable_ = repeated_players >= 5 and dof >= 5
        self.fallback_ = None if self.noise_estimable_ else "ordinary_pca_noise_unestimable"
        if self.noise_estimable_:
            raw_noise = within / dof
            # Fixed shrinkage tempers unstable high-dimensional cross-time noise.
            self.noise_ = 0.5 * raw_noise + 0.5 * np.diag(np.diag(raw_noise))
        else:
            self.noise_ = np.zeros_like(within)
        observed = np.cov(means, rowvar=False, ddof=1)
        signal = self._psd(observed - self.noise_ * np.mean(1 / counts))
        values, vectors = np.linalg.eigh(signal)
        order = np.argsort(values)[::-1][:self.n_components]
        self.eigenvalues_ = np.maximum(values[order], 1e-10)
        self.components_ = vectors[:, order]
        # Eigenvector signs are deterministic for reproducible figures.
        for k in range(self.components_.shape[1]):
            j = np.argmax(np.abs(self.components_[:, k]))
            self.components_[:, k] *= np.sign(self.components_[j, k]) or 1
        self.signal_ = (self.components_ * self.eigenvalues_) @ self.components_.T
        self.training_player_ids_ = [str(pid) for pid, _ in groups]
        return self

    def transform(self, attempts: pd.DataFrame):
        rows = []
        for pid, group in attempts.groupby("player_id", sort=True):
            valid = group.dropna(subset=self.curve_columns)
            count = len(valid)
            if count == 0:
                continue
            mean = valid[self.curve_columns].to_numpy(float).mean(0)
            noise = self.noise_ / count
            if self.noise_estimable_:
                inverse = np.linalg.pinv(self.signal_ + noise + np.eye(len(mean)) * 1e-8)
                h = self.eigenvalues_[:, None] * self.components_.T
                scores = h @ inverse @ (mean - self.mean_)
                uncertainty = np.diag(self.eigenvalues_) - h @ inverse @ h.T
            else:
                scores = self.components_.T @ (mean - self.mean_)
                uncertainty = np.full((len(scores), len(scores)), np.nan)
            row = {"player_id": pid, "n_attempts": count,
                   "noise_estimable": self.noise_estimable_, "fallback": self.fallback_}
            for k in range(len(scores)):
                row[f"functional_eb_pc{k+1}"] = float(scores[k])
                variance = uncertainty[k, k]
                row[f"functional_eb_se{k+1}"] = float(np.sqrt(max(0, variance))) if np.isfinite(variance) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


def balanced_repeatability(attempts: pd.DataFrame, value_column: str):
    """ANOVA repeatability using the first two attempts of repeated athletes.

    These repeats share session/instruction/coach effects; reported as same-session
    repeatability, never reliability of a stable trait or NFL outcome validity.
    """
    pairs = []
    for _, group in attempts.groupby("player_id", sort=True):
        sort_columns = [c for c in ["event_id", "attempt"] if c in group]
        sort_columns = sort_columns or ["attempt_id"]
        values = group.sort_values(sort_columns)[value_column].dropna().to_numpy(float)
        if len(values) >= 2:
            pairs.append(values[:2])
    values = np.array(pairs)
    if len(values) < 5:
        return {"n_repeated_players": len(values), "estimable": False}
    player_means = values.mean(axis=1)
    grand_mean = values.mean()
    between = 2 * np.sum((player_means - grand_mean) ** 2) / (len(values) - 1)
    within = np.sum((values - player_means[:, None]) ** 2) / len(values)
    if between + within <= 1e-12:
        return {"n_repeated_players":len(values), "estimable":False, "reason":"Degenerate between/within variance"}
    return {"n_repeated_players": len(values), "estimable": True,
            "icc1_1": float((between - within) / (between + within)),
            "typical_error": float(np.std(values[:, 1] - values[:, 0], ddof=1) / np.sqrt(2)),
            "mean_trial_2_minus_1": float(np.mean(values[:, 1] - values[:, 0]))}
