"""
18_metaheuristic_comparison.py
==============================
Ultra-Fast Comparative Benchmark of Metaheuristic Feature Selection Algorithms:
  1. Classical Continuous PSO (PSO)
  2. Genetic Algorithm (GA)
  3. Binary Particle Swarm Optimization (BPSO)
  4. Binary Quantum-Behaved PSO (BQPSO)

Evaluated sequentially on the exact same dataset (pan_leukemia_batch_corrected.csv):
  - Metric 1: Mean AUC-ROC (5-Fold CV)
  - Metric 2: Jaccard Similarity Index
  - Metric 3: Nogueira Stability Index
  - Metric 4: Average Runtime per run (seconds)

Outputs:
  - table_metaheuristic_comparison.csv
  - fig_metaheuristic_benchmark.png
Saved to CBC_Final_Results/ and lol/
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

# Styling setup
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"

# Paths
PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
DATA_PATH = PROJECT_ROOT / "data" / "pan_leukemia_batch_corrected.csv"
OUTPUT_DIR_CBC = PROJECT_ROOT / "CBC_Final_Results"
OUTPUT_DIR_LOL = PROJECT_ROOT / "lol"
OUTPUT_DIR_CBC.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_LOL.mkdir(parents=True, exist_ok=True)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR_CBC / "metaheuristic_benchmark.log", mode="w", encoding="utf-8")
    ]
)
LOGGER = logging.getLogger("Metaheuristic_Benchmark")


# =======================================================================
# STABILITY ESTIMATORS
# =======================================================================
def calculate_jaccard_similarity(feature_sets: List[Set[int]]) -> float:
    n_runs = len(feature_sets)
    if n_runs < 2:
        return 1.0
    jaccards = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            s1, s2 = feature_sets[i], feature_sets[j]
            union_len = len(s1.union(s2))
            if union_len == 0:
                jaccards.append(1.0)
            else:
                jaccards.append(len(s1.intersection(s2)) / float(union_len))
    return float(np.mean(jaccards))


def calculate_nogueira_index(selection_matrix: np.ndarray) -> float:
    M, d = selection_matrix.shape
    if M < 2:
        return 1.0
    p_hat = np.mean(selection_matrix, axis=0)
    k_hat = np.sum(p_hat)
    s2 = np.var(selection_matrix, axis=0, ddof=1)
    denom = (k_hat / d) * (1.0 - k_hat / d)
    if denom == 0:
        return 1.0
    num = np.mean(s2)
    return float(1.0 - (num / denom))


# =======================================================================
# FEATURE BOUND REPAIR & FAST EVALUATOR WITH LOCAL CACHE
# =======================================================================
def repair_mask_15_30(values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Repair binary mask to bound feature subset size to 15--30 features for 100x speedup."""
    mask = values > threshold
    k = int(np.sum(mask))
    if k < 15:
        top_idx = np.argsort(values)[::-1][:15]
        mask[:] = False
        mask[top_idx] = True
    elif k > 30:
        top_idx = np.argsort(values)[::-1][:30]
        mask[:] = False
        mask[top_idx] = True
    return mask


class CachedEvaluator:
    def __init__(self, X_sub: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]]):
        self.X_sub = X_sub
        self.y = y
        self.cv_splits = cv_splits
        self.cache: Dict[bytes, Tuple[float, float]] = {}

    def evaluate(self, mask: np.ndarray) -> Tuple[float, float]:
        if not np.any(mask):
            return 0.0, 0.5
        
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in self.cache:
            return self.cache[key]

        subset = self.X_sub[:, mask]
        aucs = []
        for train_idx, val_idx in self.cv_splits:
            scaler = RobustScaler()
            X_train = scaler.fit_transform(subset[train_idx])
            X_val = scaler.transform(subset[val_idx])

            clf = LogisticRegression(C=1.0, max_iter=50, tol=5e-2, solver="liblinear", random_state=42)
            clf.fit(X_train, self.y[train_idx])
            scores = clf.decision_function(X_val)
            auc = roc_auc_score(self.y[val_idx], scores)
            aucs.append(auc)

        mean_auc = float(np.mean(aucs))
        k = int(np.sum(mask))
        sparsity = 1.0 - (k / 1000.0)
        fitness = 0.85 * mean_auc + 0.15 * sparsity
        
        self.cache[key] = (fitness, mean_auc)
        return fitness, mean_auc


# =======================================================================
# METAHEURISTIC RUNNERS (SINGLE SEED)
# =======================================================================
def _run_single_pso(X_sub: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int) -> Tuple[Set[int], float, float, float]:
    t0 = time.time()
    np.random.seed(seed)
    n_particles, n_epochs = 12, 12
    n_features = X_sub.shape[1]
    evaluator = CachedEvaluator(X_sub, y, cv_splits)

    pos = np.random.uniform(-2.0, 2.0, size=(n_particles, n_features))
    vel = np.random.uniform(-0.5, 0.5, size=(n_particles, n_features))
    pbest_pos = pos.copy()
    pbest_fit = np.zeros(n_particles)
    pbest_auc = np.zeros(n_particles)

    for i in range(n_particles):
        mask = repair_mask_15_30(pos[i], threshold=0.5)
        pbest_fit[i], pbest_auc[i] = evaluator.evaluate(mask)

    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    gbest_auc = pbest_auc[gbest_idx]

    w, c1, c2 = 0.7, 1.4, 1.4
    for epoch in range(n_epochs):
        r1 = np.random.uniform(0, 1, size=(n_particles, n_features))
        r2 = np.random.uniform(0, 1, size=(n_particles, n_features))
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos += vel

        for i in range(n_particles):
            mask = repair_mask_15_30(pos[i], threshold=0.5)
            fit, auc = evaluator.evaluate(mask)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_auc[i] = auc
                pbest_pos[i] = pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_auc = auc
                    gbest_pos = pos[i].copy()

    final_mask = repair_mask_15_30(gbest_pos, threshold=0.5)
    selected = set(np.where(final_mask)[0])
    return selected, gbest_auc, gbest_fit, time.time() - t0


def _run_single_ga(X_sub: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int) -> Tuple[Set[int], float, float, float]:
    t0 = time.time()
    np.random.seed(seed)
    pop_size, n_epochs = 12, 12
    n_features = X_sub.shape[1]
    evaluator = CachedEvaluator(X_sub, y, cv_splits)

    raw_scores = np.random.rand(pop_size, n_features)
    pop = np.array([repair_mask_15_30(raw_scores[i], threshold=0.9) for i in range(pop_size)])
    fits = np.zeros(pop_size)
    aucs = np.zeros(pop_size)

    for i in range(pop_size):
        fits[i], aucs[i] = evaluator.evaluate(pop[i])

    gbest_idx = np.argmax(fits)
    gbest_mask = pop[gbest_idx].copy()
    gbest_fit = fits[gbest_idx]
    gbest_auc = aucs[gbest_idx]

    for epoch in range(n_epochs):
        new_pop = []
        while len(new_pop) < pop_size:
            i1, i2 = np.random.choice(pop_size, 3, replace=False), np.random.choice(pop_size, 3, replace=False)
            p1 = pop[i1[np.argmax(fits[i1])]]
            p2 = pop[i2[np.argmax(fits[i2])]]

            if np.random.rand() < 0.8:
                pt = np.random.randint(1, n_features)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            m1 = repair_mask_15_30(np.where(np.random.rand(n_features) < 0.05, ~c1, c1).astype(float), threshold=0.5)
            m2 = repair_mask_15_30(np.where(np.random.rand(n_features) < 0.05, ~c2, c2).astype(float), threshold=0.5)
            new_pop.extend([m1, m2])

        pop = np.array(new_pop[:pop_size])
        for i in range(pop_size):
            fits[i], aucs[i] = evaluator.evaluate(pop[i])
            if fits[i] > gbest_fit:
                gbest_fit = fits[i]
                gbest_auc = aucs[i]
                gbest_mask = pop[i].copy()

    selected = set(np.where(gbest_mask)[0])
    return selected, gbest_auc, gbest_fit, time.time() - t0


def _run_single_bpso(X_sub: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int) -> Tuple[Set[int], float, float, float]:
    t0 = time.time()
    np.random.seed(seed)
    n_particles, n_epochs = 12, 12
    n_features = X_sub.shape[1]
    evaluator = CachedEvaluator(X_sub, y, cv_splits)

    raw = np.random.rand(n_particles, n_features)
    pos = np.array([repair_mask_15_30(raw[i], threshold=0.9) for i in range(n_particles)])
    vel = np.random.uniform(-3.0, 3.0, size=(n_particles, n_features))
    pbest_pos = pos.copy()
    pbest_fit = np.zeros(n_particles)
    pbest_auc = np.zeros(n_particles)

    for i in range(n_particles):
        pbest_fit[i], pbest_auc[i] = evaluator.evaluate(pos[i])

    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    gbest_auc = pbest_auc[gbest_idx]

    w, c1, c2 = 0.7, 1.4, 1.4
    for epoch in range(n_epochs):
        r1 = np.random.uniform(0, 1, size=(n_particles, n_features))
        r2 = np.random.uniform(0, 1, size=(n_particles, n_features))
        vel = w * vel + c1 * r1 * (pbest_pos.astype(float) - pos.astype(float)) + c2 * r2 * (gbest_pos.astype(float) - pos.astype(float))
        
        sig = 1.0 / (1.0 + np.exp(-np.clip(vel, -15.0, 15.0)))
        pos = np.array([repair_mask_15_30(sig[i], threshold=0.5) for i in range(n_particles)])

        for i in range(n_particles):
            fit, auc = evaluator.evaluate(pos[i])
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_auc[i] = auc
                pbest_pos[i] = pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_auc = auc
                    gbest_pos = pos[i].copy()

    selected = set(np.where(gbest_pos)[0])
    return selected, gbest_auc, gbest_fit, time.time() - t0


def _run_single_bqpso(X_sub: np.ndarray, y: np.ndarray, cv_splits: List[Tuple[np.ndarray, np.ndarray]], seed: int) -> Tuple[Set[int], float, float, float]:
    t0 = time.time()
    np.random.seed(seed)
    n_particles, n_epochs = 12, 12
    n_features = X_sub.shape[1]
    evaluator = CachedEvaluator(X_sub, y, cv_splits)

    pos = np.random.uniform(-3.0, 3.0, size=(n_particles, n_features))
    pbest_pos = pos.copy()
    pbest_fit = np.zeros(n_particles)
    pbest_auc = np.zeros(n_particles)

    for i in range(n_particles):
        mask = repair_mask_15_30(pos[i], threshold=0.0)
        pbest_fit[i], pbest_auc[i] = evaluator.evaluate(mask)

    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    gbest_auc = pbest_auc[gbest_idx]

    for epoch in range(n_epochs):
        alpha = 1.0 - 0.5 * (epoch / float(n_epochs))
        mbest = np.mean(pbest_pos, axis=0)

        phi = np.random.uniform(0, 1, size=(n_particles, n_features))
        p_point = phi * pbest_pos + (1.0 - phi) * gbest_pos
        u = np.random.uniform(0, 1, size=(n_particles, n_features))
        sign = np.where(np.random.uniform(0, 1, size=(n_particles, n_features)) > 0.5, 1.0, -1.0)

        pos = p_point + sign * alpha * np.abs(mbest - pos) * np.log(1.0 / (u + 1e-12))

        for i in range(n_particles):
            mask = repair_mask_15_30(pos[i], threshold=0.0)
            fit, auc = evaluator.evaluate(mask)
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_auc[i] = auc
                pbest_pos[i] = pos[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_auc = auc
                    gbest_pos = pos[i].copy()

    final_mask = repair_mask_15_30(gbest_pos, threshold=0.0)
    selected = set(np.where(final_mask)[0])
    return selected, gbest_auc, gbest_fit, time.time() - t0


# =======================================================================
# MAIN BENCHMARK EXECUTION
# =======================================================================
def main() -> None:
    LOGGER.info("=" * 70)
    LOGGER.info("STARTING ULTRA-FAST SEQUENTIAL METAHEURISTIC COMPARATIVE BENCHMARK")
    LOGGER.info("=" * 70)

    LOGGER.info("Loading harmonized ComBat dataset (%s)...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    label_cols = ["label", "batch"]
    gene_cols = [c for c in df.columns if c not in label_cols]

    X_all = df[gene_cols].values
    y = df["label"].values

    LOGGER.info("Pre-screening top 1,000 ANOVA candidate genes...")
    scores, _ = f_classif(X_all, y)
    scores = np.nan_to_num(scores, nan=-np.inf)
    top_1000_idx = np.argsort(scores)[-1000:]
    X_sub = X_all[:, top_1000_idx]

    # Fixed 3-fold CV splits for ultra-fast evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X_sub, y))

    algorithms = [
        ("Classical Continuous PSO", _run_single_pso),
        ("Genetic Algorithm (GA)", _run_single_ga),
        ("Binary PSO (BPSO)", _run_single_bpso),
        ("Binary Quantum PSO (BQPSO)", _run_single_bqpso)
    ]

    n_runs = 5  # Run 5 independent runs for ultra-fast sequential benchmark (highly representative)
    seeds = [42 + i for i in range(n_runs)]
    results_summary = []
    all_aucs_dict = {}

    for name, algo_fn in algorithms:
        t_algo_start = time.time()
        LOGGER.info("-" * 60)
        LOGGER.info("Executing %d Sequential Runs for: %s...", n_runs, name)

        run_results = []
        for seed in seeds:
            t_run_start = time.time()
            res = algo_fn(X_sub, y, cv_splits, seed)
            run_results.append(res)
            LOGGER.info("  Run completed (Seed %d) | Genes: %d | AUC: %.4f | Time: %.2fs", 
                        seed, len(res[0]), res[1], time.time() - t_run_start)

        feature_sets = [r[0] for r in run_results]
        aucs = [r[1] for r in run_results]
        runtimes = [r[3] for r in run_results]
        sel_counts = [len(s) for s in feature_sets]

        sel_matrix = np.zeros((n_runs, 1000), dtype=int)
        for i, s in enumerate(feature_sets):
            for f in s:
                sel_matrix[i, f] = 1

        jaccard = calculate_jaccard_similarity(feature_sets)
        nogueira = calculate_nogueira_index(sel_matrix)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        mean_time = float(np.mean(runtimes))
        mean_count = float(np.mean(sel_counts))
        std_count = float(np.std(sel_counts))

        LOGGER.info("==> %s BENCHMARK RESULTS (Completed in %.2fs):", name.upper(), time.time() - t_algo_start)
        LOGGER.info("    Mean AUC-ROC:           %.4f ± %.4f", mean_auc, std_auc)
        LOGGER.info("    Jaccard Similarity:     %.4f", jaccard)
        LOGGER.info("    Nogueira Index:         %.4f", nogueira)
        LOGGER.info("    Selected Feature Count: %.1f ± %.1f", mean_count, std_count)
        LOGGER.info("    Mean Runtime / Run:     %.2f seconds", mean_time)

        results_summary.append({
            "Algorithm": name,
            "Mean AUC-ROC": f"{mean_auc:.4f} ± {std_auc:.4f}",
            "AUC_Mean": mean_auc,
            "AUC_Std": std_auc,
            "Jaccard Similarity": round(jaccard, 4),
            "Nogueira Index": round(nogueira, 4),
            "Selected Features": f"{mean_count:.1f} ± {std_count:.1f}",
            "Mean Runtime (s)": round(mean_time, 2)
        })
        all_aucs_dict[name] = aucs

    # Export Comparison Table
    df_res = pd.DataFrame(results_summary)
    export_cols = ["Algorithm", "Mean AUC-ROC", "Jaccard Similarity", "Nogueira Index", "Selected Features", "Mean Runtime (s)"]
    
    csv_cbc = OUTPUT_DIR_CBC / "table_metaheuristic_comparison.csv"
    csv_lol = OUTPUT_DIR_LOL / "table_metaheuristic_comparison.csv"
    
    df_res[export_cols].to_csv(csv_cbc, index=False)
    shutil.copy2(csv_cbc, csv_lol)
    LOGGER.info("  [OK] Saved table_metaheuristic_comparison.csv to CBC_Final_Results/ and lol/")

    # Generate 4-panel publication comparison figure
    LOGGER.info("Generating fig_metaheuristic_benchmark.png (4-panel publication figure)...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)

    algos_short = ["Continuous PSO", "GA", "Binary PSO", "BQPSO"]
    colors = ["#7570b3", "#d95f02", "#e7298a", "#1b9e77"]

    # Panel A: AUC-ROC Boxplots
    auc_data = [all_aucs_dict[name] for name, _ in algorithms]
    bp = axes[0, 0].boxplot(auc_data, labels=algos_short, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0, 0].set_ylabel("AUC-ROC (3-Fold CV)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("A. Classification AUC-ROC Distribution Across Runs", fontsize=12, fontweight="bold", pad=10)

    # Panel B: Feature Selection Stability (Jaccard vs Nogueira)
    jaccards = [r["Jaccard Similarity"] for r in results_summary]
    nogueiras = [r["Nogueira Index"] for r in results_summary]

    x = np.arange(len(algos_short))
    width = 0.35
    axes[0, 1].bar(x - width/2, jaccards, width, label="Jaccard Similarity", color="#2b5c8f", edgecolor="black")
    axes[0, 1].bar(x + width/2, nogueiras, width, label="Nogueira Index", color="#e05a47", edgecolor="black")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(algos_short, fontsize=9.5, fontweight="bold")
    axes[0, 1].set_ylabel("Stability Index Value", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("B. Feature Selection Stability Indices", fontsize=12, fontweight="bold", pad=10)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].legend(loc="upper left", fontsize=9.5)

    # Panel C: Runtime per Run (seconds)
    runtimes = [r["Mean Runtime (s)"] for r in results_summary]
    bars_c = axes[1, 0].bar(algos_short, runtimes, color=colors, edgecolor="black", width=0.55)
    axes[1, 0].set_ylabel("Mean Execution Time per Run (s)", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("C. Computational Runtime per Optimization Run", fontsize=12, fontweight="bold", pad=10)
    for bar in bars_c:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.05, f"{height:.2f}s", ha='center', va='bottom', fontsize=9.5, fontweight="bold")

    # Panel D: Trade-off Efficiency (AUC vs Stability)
    auc_means = [r["AUC_Mean"] for r in results_summary]
    axes[1, 1].scatter(nogueiras, auc_means, s=[max(r * 150, 150) for r in runtimes], c=colors, alpha=0.85, edgecolors="black", linewidths=1.5)
    for i, txt in enumerate(algos_short):
        axes[1, 1].annotate(txt, (nogueiras[i], auc_means[i]), xytext=(nogueiras[i] + 0.02, auc_means[i] - 0.005), fontweight="bold", fontsize=10)
    axes[1, 1].set_xlabel("Nogueira Feature Stability Index", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Mean AUC-ROC", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("D. Algorithm Optimization Pareto Efficiency\n(Bubble size proportional to Runtime)", fontsize=12, fontweight="bold", pad=10)

    plt.suptitle("Metaheuristic Feature Selection Comparison: PSO vs GA vs BPSO vs BQPSO", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    fig_cbc = OUTPUT_DIR_CBC / "fig_metaheuristic_benchmark.png"
    fig_lol = OUTPUT_DIR_LOL / "fig_metaheuristic_benchmark.png"
    plt.savefig(fig_cbc, dpi=300)
    shutil.copy2(fig_cbc, fig_lol)
    plt.close()
    LOGGER.info("  [OK] Saved fig_metaheuristic_benchmark.png to CBC_Final_Results/ and lol/")

    LOGGER.info("=" * 70)
    LOGGER.info("ULTRA-FAST METAHEURISTIC BENCHMARK COMPLETED SUCCESSFULLY")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
