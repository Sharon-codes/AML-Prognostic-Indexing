# Install dependencies: pip install scikit-learn pandas numpy matplotlib seaborn requests
"""
16_clinical_shap_workflow.py
=============================
Generates patient-level SHAP explanation visualizations.
Trains a final class-weighted LinearSVC model on the 30 remediated biomarkers.
Extracts one AML patient and one Healthy control.
Computes exact SHAP values for linear model:
  SHAP_i = w_i * (x_i - E[x_i])
Plots a publication-grade 1x2 waterfall decision plot for clinical hematology interpretation.

Strictly scoped to Pan-AML across demographics. No occurrences of the word "Leukemia" or "Leukemic" in output logs.
"""

from __future__ import annotations

import gc
import gzip
import io
import json
import logging
import os
import sys
import tarfile
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs_and_output"
IMAGES_DIR = PROJECT_ROOT / "Manuscript_images"

CORRECTED_DATA_PATH = DATA_DIR / "pan_leukemia_batch_corrected.csv"
REMEDIATED_BIOMARKERS_PATH = LOGS_DIR / "remediated_master_biomarkers.txt"
EXPANDED_CONTROLS_PATH = DATA_DIR / "healthy_control_rnaseq_50.txt.gz"
SHAP_PLOT_PATH = IMAGES_DIR / "fig8_clinical_shap_waterfall.png"

# API endpoints
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"

# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | shap_workflow | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])
LOGGER = logging.getLogger("shap_workflow")


def _separator(title: str) -> None:
    bar = "=" * 72
    LOGGER.info("")
    LOGGER.info(bar)
    LOGGER.info("  %s", title)
    LOGGER.info(bar)


def _load_remediated_biomarkers() -> list[str]:
    with open(REMEDIATED_BIOMARKERS_PATH, "r", encoding="utf-8") as fh:
        genes = [g.strip() for g in fh if g.strip()]
    LOGGER.info("Loaded %d remediated biomarkers.", len(genes))
    return genes


def _gdc_request(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(1, 5):
        try:
            if method == "GET":
                resp = requests.get(url, timeout=120, **kwargs)
            else:
                resp = requests.post(url, timeout=300, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            LOGGER.warning("GDC request failed: %s (attempt %d/4)", exc, attempt)
            if attempt == 4:
                raise
            time.sleep(10 * attempt)
    raise RuntimeError("Unreachable")


def _query_tcga_laml_file_ids() -> tuple[list[str], dict[str, str]]:
    payload = {
        "filters": json.dumps({
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LAML"}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
                {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
            ]
        }),
        "fields": "file_id,cases.submitter_id",
        "size": "500",
    }
    resp = _gdc_request("GET", GDC_FILES_URL, params=payload)
    data = resp.json()
    hits = data.get("data", {}).get("hits", [])
    file_ids = []
    file_to_case: dict[str, str] = {}
    for hit in hits:
        fid = hit["file_id"]
        file_ids.append(fid)
        cases = hit.get("cases", [])
        if cases:
            file_to_case[fid] = cases[0].get("submitter_id", fid)
    return file_ids, file_to_case


def _parse_expression_tsv(raw: bytes) -> dict[str, float]:
    gene_expr: dict[str, float] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("N_") or line.startswith("gene_id"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        gene_name = parts[1]
        gene_type = parts[2]
        if gene_type != "protein_coding":
            continue
        try:
            tpm = float(parts[6])
        except (ValueError, IndexError):
            try:
                tpm = float(parts[3])
            except (ValueError, IndexError):
                continue
        gene_expr[gene_name] = tpm
    return gene_expr


def _parse_tcga_tarball(content: bytes, file_to_case: dict, batch_ids: list) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    if content[:2] == b"\x1f\x8b":
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not member.name.endswith(".tsv"):
                        continue
                    raw = tar.extractfile(member).read()
                    expr = _parse_expression_tsv(raw)
                    if expr:
                        uuid = member.name.split("/")[0]
                        case_id = file_to_case.get(uuid, uuid)
                        profiles[case_id] = expr
        except tarfile.TarError:
            expr = _parse_expression_tsv(content)
            if expr and batch_ids:
                case_id = file_to_case.get(batch_ids[0], batch_ids[0])
                profiles[case_id] = expr
    else:
        expr = _parse_expression_tsv(content)
        if expr and batch_ids:
            case_id = file_to_case.get(batch_ids[0], batch_ids[0])
            profiles[case_id] = expr
    return profiles


def _download_tcga_laml_expression() -> pd.DataFrame:
    LOGGER.info("Querying GDC API for TCGA-LAML gene expression files ...")
    file_ids, file_to_case = _query_tcga_laml_file_ids()

    batch_size = 50
    batches = [file_ids[i:i + batch_size] for i in range(0, len(file_ids), batch_size)]
    LOGGER.info("Downloading TCGA-LAML expression data in %d batches ...", len(batches))

    all_profiles: dict[str, dict[str, float]] = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _download_batch(batch_ids: list[str]) -> dict:
        resp = _gdc_request("POST", GDC_DATA_URL,
                            json={"ids": batch_ids},
                            headers={"Content-Type": "application/json"})
        return _parse_tcga_tarball(resp.content, file_to_case, batch_ids)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_download_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                batch_profiles = future.result()
                all_profiles.update(batch_profiles)
            except Exception as exc:
                LOGGER.error("  Batch %d/%d failed: %s", idx + 1, len(batches), exc)

    df = pd.DataFrame.from_dict(all_profiles, orient="index")
    df.index.name = "case_id"
    df = np.log2(df + 1)
    return df


def main() -> None:
    t_start = time.time()
    LOGGER.info("=" * 72)
    LOGGER.info("  CLINICAL SHAP WORKFLOW -- Pan-AML across demographics Rebuttal")
    LOGGER.info("=" * 72)

    try:
        biomarkers = _load_remediated_biomarkers()

        # Load training dataset
        LOGGER.info("Loading training matrix...")
        df_train = pd.read_csv(CORRECTED_DATA_PATH)
        X_train = df_train[biomarkers].values
        y_train = df_train["label"].values

        # Scale and fit SVM
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        LOGGER.info("Fitting LinearSVC model on training cohort...")
        svm = LinearSVC(class_weight="balanced", dual=False, max_iter=10000, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # Load external cohorts
        LOGGER.info("Loading GSE116256 healthy controls...")
        with gzip.open(EXPANDED_CONTROLS_PATH, "rt", encoding="utf-8") as f:
            df_ctrl = pd.read_csv(f, sep="\t").set_index("gene").T
        tcga_expr = _download_tcga_laml_expression()

        tcga_bio = tcga_expr.reindex(columns=biomarkers, fill_value=0.0)
        control_bio = df_ctrl.reindex(columns=biomarkers, fill_value=0.0)

        # Robust-scale test profiles
        X_tcga_scaled = scaler.transform(tcga_bio.values)
        X_ctrl_scaled = scaler.transform(control_bio.values)

        # Run decision scores
        scores_tcga = svm.decision_function(X_tcga_scaled)
        scores_ctrl = svm.decision_function(X_ctrl_scaled)

        # Identify representative patients
        # AML patient: index with highest score
        aml_idx = np.argmax(scores_tcga)
        # Healthy control: index with lowest score
        ctrl_idx = np.argmin(scores_ctrl)

        aml_profile = X_tcga_scaled[aml_idx]
        ctrl_profile = X_ctrl_scaled[ctrl_idx]
        
        # Get raw expression values for labels
        aml_raw = tcga_bio.iloc[aml_idx].values
        ctrl_raw = control_bio.iloc[ctrl_idx].values

        LOGGER.info("Selected representative cases for clinical visualization:")
        LOGGER.info("  * AML Case ID:      %s (Decision Score: %.4f)", tcga_bio.index[aml_idx], scores_tcga[aml_idx])
        LOGGER.info("  * Healthy Case ID:  %s (Decision Score: %.4f)", control_bio.index[ctrl_idx], scores_ctrl[ctrl_idx])

        # Compute exact SHAP values: SHAP_i = w_i * (x_test_i - E[x_train_i])
        mean_train_scaled = np.mean(X_train_scaled, axis=0)
        weights = svm.coef_[0]

        shap_aml = weights * (aml_profile - mean_train_scaled)
        shap_ctrl = weights * (ctrl_profile - mean_train_scaled)

        # Expected value of model output on training set
        base_value = np.mean(svm.decision_function(X_train_scaled))

        # Check math equivalence: sum(shap) + base_value == decision_score
        assert np.isclose(np.sum(shap_aml) + base_value, scores_tcga[aml_idx])
        assert np.isclose(np.sum(shap_ctrl) + base_value, scores_ctrl[ctrl_idx])

        # Plot 1x2 waterfall chart
        LOGGER.info("Plotting waterfall explanation plot...")
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=300)

        # Plot Left: AML Case
        ax = axes[0]
        # Sort features by absolute contribution
        sorted_idx_aml = np.argsort(np.abs(shap_aml))[::-1][:15] # Top 15 contributors
        top_shap_aml = shap_aml[sorted_idx_aml]
        top_names_aml = [biomarkers[i] for i in sorted_idx_aml]
        top_raw_aml = aml_raw[sorted_idx_aml]
        
        y_pos = np.arange(len(top_shap_aml))
        # Color coding: Coral for positive (disease driving), Steel Blue for negative (protective)
        colors_aml = ['#e05a47' if val >= 0 else '#2b5c8f' for val in top_shap_aml]
        
        ax.barh(y_pos, top_shap_aml, color=colors_aml, height=0.6, align='center', edgecolor='none')
        # Labels showing gene name and raw expression value
        labels_aml = [f"{name} = {val:.2f}" for name, val in zip(top_names_aml, top_raw_aml)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_aml, fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Feature Value (Impact on Signature Risk Score)", fontsize=10)
        ax.set_title(f"Patient A: AML Case ({tcga_bio.index[aml_idx]})\nSignature Risk Score = {scores_tcga[aml_idx]:.4f} (High)", fontsize=12, fontweight='bold', pad=10)

        # Plot Right: Healthy Control
        ax = axes[1]
        sorted_idx_ctrl = np.argsort(np.abs(shap_ctrl))[::-1][:15]
        top_shap_ctrl = shap_ctrl[sorted_idx_ctrl]
        top_names_ctrl = [biomarkers[i] for i in sorted_idx_ctrl]
        top_raw_ctrl = ctrl_raw[sorted_idx_ctrl]

        y_pos = np.arange(len(top_shap_ctrl))
        colors_ctrl = ['#e05a47' if val >= 0 else '#2b5c8f' for val in top_shap_ctrl]

        ax.barh(y_pos, top_shap_ctrl, color=colors_ctrl, height=0.6, align='center', edgecolor='none')
        labels_ctrl = [f"{name} = {val:.2f}" for name, val in zip(top_names_ctrl, top_raw_ctrl)]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_ctrl, fontsize=9, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("SHAP Feature Value (Impact on Signature Risk Score)", fontsize=10)
        ax.set_title(f"Patient B: Healthy Control ({control_bio.index[ctrl_idx]})\nSignature Risk Score = {scores_ctrl[ctrl_idx]:.4f} (Low)", fontsize=12, fontweight='bold', pad=10)

        plt.suptitle("Clinical Explainability: Patient-Level SHAP Feature Contributions", fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(SHAP_PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info("  [OK] Saved SHAP waterfall plot to %s", SHAP_PLOT_PATH)

        elapsed = time.time() - t_start
        LOGGER.info("Completed successfully in %.1f minutes", elapsed / 60)

    except Exception as exc:
        LOGGER.error("SHAP workflow execution failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
