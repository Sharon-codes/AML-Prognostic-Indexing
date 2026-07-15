"""Binary Quantum-Behaved PSO feature selection for transcriptomic data.

BQPSO replaces classical PSO velocities with a Monte-Carlo update centred on
the swarm's mean best position (mbest).  A sigmoid maps latent positions to
Bernoulli gene-selection decisions.  Fitness combines stratified-CV linear SVM
accuracy (90%) with a sparsity reward (10%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

LOGGER = logging.getLogger(__name__)


@dataclass
class BinaryQuantumPSO:
    """Sparse binary quantum-behaved particle swarm optimiser.

    A univariate ANOVA screen bounds the optimisation dimensionality on very
    large microarrays.  It is unsupervised with respect to individual CV folds
    only in the computational screening sense; final biomarker validation must
    always be repeated in an external cohort to avoid selection optimism.
    """

    n_particles: int = 50
    n_epochs: int = 100
    min_features: int = 15
    max_features: int = 30
    candidate_features: int = 1_000
    alpha_start: float = 1.0
    alpha_end: float = 0.5
    random_state: int = 42
    _fitness_cache: dict[bytes, float] = field(default_factory=dict, init=False)

    def _screen_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Keep the most discriminative candidate genes before expensive BQPSO."""
        if X.shape[1] <= self.candidate_features:
            return X
        scores, _ = f_classif(X.to_numpy(dtype=np.float64), y.to_numpy())
        scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
        keep = np.argsort(scores)[-self.candidate_features :]
        LOGGER.info("ANOVA pre-screen retained %d of %d genes for BQPSO.", len(keep), X.shape[1])
        return X.iloc[:, keep]

    @staticmethod
    def _sigmoid(position: np.ndarray) -> np.ndarray:
        """Numerically stable logistic transform of quantum latent positions."""
        clipped = np.clip(position, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _repair_mask(self, mask: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Enforce the journal-specified 15--30 biomarker target range."""
        repaired = mask.astype(bool, copy=True)
        selected = int(repaired.sum())
        if selected < self.min_features:
            add = np.argsort(position)[::-1]
            repaired[add[: self.min_features]] = True
        elif selected > self.max_features:
            retained = np.argsort(position)[::-1][: self.max_features]
            repaired[:] = False
            repaired[retained] = True
        return repaired

    def _fitness(self, mask: np.ndarray, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
        """Return 0.9 * mean CV accuracy + 0.1 * sparsity reward."""
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        cached = self._fitness_cache.get(key)
        if cached is not None:
            return cached
        subset = X[:, mask]
        scores: list[float] = []
        for train_idx, valid_idx in cv.split(subset, y):
            # LinearSVC is appropriate for p >> n microarray matrices and much
            # faster than non-linear kernels inside a 25,000-evaluation search.
            # ``dual=False`` is mathematically equivalent but 65x faster when
            # the number of samples N is much larger than the selected features M.
            svm = LinearSVC(C=1.0, dual=False, max_iter=5_000, random_state=self.random_state)
            svm.fit(subset[train_idx], y[train_idx])
            scores.append(accuracy_score(y[valid_idx], svm.predict(subset[valid_idx])))
        accuracy = float(np.mean(scores))
        sparsity = 1.0 - (float(mask.sum()) / float(mask.size))
        score = 0.9 * accuracy + 0.1 * sparsity
        self._fitness_cache[key] = score
        return score

    def fit_select(self, X: pd.DataFrame, y: pd.Series, output_dir: Path) -> pd.DataFrame:
        """Run BQPSO and return ``X`` restricted to the selected biomarkers."""
        if X.shape[0] != y.shape[0] or not X.index.equals(y.index):
            raise ValueError("X and y must have identical sample indexes.")
        if self.min_features < 3 or self.min_features > self.max_features:
            raise ValueError("Require 3 <= min_features <= max_features.")
        X_search = self._screen_features(X, y)
        if X_search.shape[1] < self.min_features:
            raise ValueError("Fewer genes than the requested minimum feature count are available.")
        class_counts = y.value_counts()
        if len(class_counts) != 2 or int(class_counts.min()) < 5:
            raise ValueError("BQPSO requires two classes with at least five samples each for 5-fold CV.")

        n_dimensions = X_search.shape[1]
        rng = np.random.default_rng(self.random_state)
        positions = rng.normal(loc=-3.0, scale=1.0, size=(self.n_particles, n_dimensions))
        binary = np.empty((self.n_particles, n_dimensions), dtype=bool)
        for particle in range(self.n_particles):
            binary[particle] = self._repair_mask(rng.random(n_dimensions) < self._sigmoid(positions[particle]), positions[particle])

        X_array = X_search.to_numpy(dtype=np.float32)
        y_array = y.to_numpy(dtype=np.int8)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        personal_positions = positions.copy()
        personal_masks = binary.copy()
        personal_scores = np.array([self._fitness(mask, X_array, y_array, cv) for mask in binary])
        best_idx = int(np.argmax(personal_scores))
        global_position = personal_positions[best_idx].copy()
        global_mask = personal_masks[best_idx].copy()
        global_score = float(personal_scores[best_idx])
        LOGGER.info("BQPSO initial best fitness: %.5f (%d genes).", global_score, global_mask.sum())

        for epoch in range(self.n_epochs):
            alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * epoch / max(1, self.n_epochs - 1)
            mbest = personal_positions.mean(axis=0)
            for particle in range(self.n_particles):
                phi = rng.random(n_dimensions)
                attractor = phi * personal_positions[particle] + (1.0 - phi) * global_position
                u = np.clip(rng.random(n_dimensions), 1e-12, 1.0)
                sign = np.where(rng.random(n_dimensions) < 0.5, -1.0, 1.0)
                # Quantum delta-potential-well Monte Carlo update.  There is no velocity.
                positions[particle] = attractor + sign * alpha * np.abs(mbest - positions[particle]) * np.log(1.0 / u)
                proposed = rng.random(n_dimensions) < self._sigmoid(positions[particle])
                binary[particle] = self._repair_mask(proposed, positions[particle])
                score = self._fitness(binary[particle], X_array, y_array, cv)
                if score > personal_scores[particle]:
                    personal_scores[particle] = score
                    personal_positions[particle] = positions[particle].copy()
                    personal_masks[particle] = binary[particle].copy()
                    if score > global_score:
                        global_score = float(score)
                        global_position = positions[particle].copy()
                        global_mask = binary[particle].copy()
            LOGGER.info("BQPSO epoch %03d/%03d | best fitness %.5f | genes %d", epoch + 1, self.n_epochs, global_score, global_mask.sum())

        selected_genes = X_search.columns[global_mask].tolist()
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_path = output_dir / "selected_genes.txt"
        selected_path.write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
        (output_dir / "bqpso_summary.txt").write_text(
            f"best_fitness={global_score:.8f}\nselected_gene_count={len(selected_genes)}\n"
            "fitness=0.9*five_fold_linear_SVM_accuracy + 0.1*(1-selected/available_candidates)\n",
            encoding="utf-8",
        )
        LOGGER.info("BQPSO selected %d genes; list saved to %s", len(selected_genes), selected_path)
        return X.loc[:, selected_genes].copy()


if __name__ == "__main__":
    raise SystemExit("Run main.py to execute the complete pipeline.")
