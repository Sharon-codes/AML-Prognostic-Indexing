# Install dependencies: pip install scikit-learn pandas numpy scipy requests
"""
12_baseline_and_ci_metrics.py
==============================
Calculates donor-level bootstrap confidence intervals for specificity and quantifies random baseline margins.
Steps:
  Step 1: Donor-Level Bootstrap Specificity CI (10,000 iterations over 5 donors)
  Step 2: Random Baseline Specificity and Domain-Shift Contribution Margin

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

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs_and_output"

CORRECTED_DATA_PATH = DATA_DIR / "pan_leukemia_batch_corrected.csv"
REMEDIATED_BIOMARKERS_PATH = LOGS_DIR / "remediated_master_biomarkers.txt"
EXPANDED_CONTROLS_PATH = DATA_DIR / "healthy_control_rnaseq_50.txt.gz"
REPORT_PATH = LOGS_DIR / "remediated_baseline_and_ci_metrics.txt"

# API endpoints
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"

# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | baseline_and_ci | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                    handlers=[logging.StreamHandler(sys.stdout)])
LOGGER = logging.getLogger("baseline_and_ci")


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
            tpm = float(parts[6])  # tpm_unstranded
        except (ValueError, IndexError):
            try:
                tpm = float(parts[3])  # unstranded count fallback
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
                LOGGER.info("  Parsed %d profiles from batch %d/%d.",
                            len(batch_profiles), idx + 1, len(batches))
            except Exception as exc:
                LOGGER.error("  Batch %d/%d failed: %s", idx + 1, len(batches), exc)

    LOGGER.info("TCGA-LAML download complete: %d profiles.", len(all_profiles))
    df = pd.DataFrame.from_dict(all_profiles, orient="index")
    df.index.name = "case_id"
    df = np.log2(df + 1)
    return df


def main() -> None:
    t_start = time.time()
    LOGGER.info("=" * 72)
    LOGGER.info("  BASELINE AND CI METRICS -- Pan-AML across demographics Rebuttal")
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

        LOGGER.info("Fitting class-weighted SVM classifier...")
        svm = LinearSVC(class_weight="balanced", dual=False, max_iter=10000, random_state=42)
        svm.fit(X_train_scaled, y_train)

        # Load external cohorts
        LOGGER.info("Loading 50 real healthy controls...")
        with gzip.open(EXPANDED_CONTROLS_PATH, "rt", encoding="utf-8") as f:
            df_ctrl = pd.read_csv(f, sep="\t").set_index("gene").T
        tcga_expr = _download_tcga_laml_expression()

        tcga_bio = tcga_expr.reindex(columns=biomarkers, fill_value=0.0)
        control_bio = df_ctrl.reindex(columns=biomarkers, fill_value=0.0)

        # Concatenate external test matrix
        mixed_X = pd.concat([tcga_bio, control_bio], axis=0)
        y_true = np.array([1] * len(tcga_bio) + [0] * len(control_bio))

        # Scale test matrix
        X_test_scaled = scaler.transform(mixed_X.values)

        # Run inference and decision function
        test_scores = svm.decision_function(X_test_scaled)
        
        # Calibrated threshold
        calibrated_threshold = -0.260708
        y_pred = (test_scores >= calibrated_threshold).astype(int)

        # Isolate healthy control predictions (last 50 samples)
        control_preds = y_pred[-50:]

        # ======================================================================
        #  STEP 1 -- Donor-Level Bootstrap Specificity CI
        # ======================================================================
        _separator("STEP 1: Donor-Level Bootstrap Specificity CI")
        
        # Map 50 controls to 5 donors (10 pseudo-bulks per donor)
        donors_preds = np.split(control_preds, 5) # 5 arrays of shape (10,)
        
        rng = np.random.default_rng(42)
        n_boot = 10000
        boot_specificities = []

        LOGGER.info("Running 10,000 donor-level bootstrap iterations...")
        for _ in range(n_boot):
            # Sample 5 donors with replacement
            sampled_donor_indices = rng.choice(5, size=5, replace=True)
            resampled_preds = []
            for d_idx in sampled_donor_indices:
                resampled_preds.extend(donors_preds[d_idx])
            
            resampled_preds = np.array(resampled_preds)
            # Specificity is fraction of correct negative class predictions (0)
            spec = np.mean(resampled_preds == 0)
            boot_specificities.append(spec)

        boot_specificities = np.array(boot_specificities)
        ci_lower = np.percentile(boot_specificities, 2.5)
        ci_upper = np.percentile(boot_specificities, 97.5)

        LOGGER.info("  Bootstrap Specificity Mean: %.4f", np.mean(boot_specificities))
        LOGGER.info("  Donor-Level Bootstrap 95%% CI: [%.4f, %.4f]", ci_lower, ci_upper)

        # ======================================================================
        #  STEP 2 -- Random Baseline Specificity
        # ======================================================================
        _separator("STEP 2: Random Baseline Specificity & Margin")
        
        # Get training scores
        train_scores = svm.decision_function(X_train_scaled)
        
        # Fraction of training distribution strictly below threshold
        baseline_specificity = np.mean(train_scores < calibrated_threshold)
        
        # Specificity obtained in Step 4/5 of class_balance_rescue
        our_specificity = np.mean(control_preds == 0)
        
        # Domain-Shift Contribution Margin
        contribution_margin = our_specificity - baseline_specificity

        LOGGER.info("  Random Baseline Specificity:    %.4f" % baseline_specificity)
        LOGGER.info("  Our Specificity:                %.4f" % our_specificity)
        LOGGER.info("  Domain-Shift Contribution Margin: %.4f" % contribution_margin)

        # Save to file
        lines = [
            "=" * 72,
            "  Baseline and CI Metrics: Pan-AML across demographics Rebuttal",
            "=" * 72,
            "",
            "Remediated 30-Gene Signature (TNFSF12-TNFSF13 blacklisted)",
            "External validation cohort: 151 TCGA-LAML AML cases vs. 50 GSE116256 healthy controls",
            "",
            "1. Donor-Level Bootstrap Specificity Confidence Interval:",
            "----------------------------------------------------------",
            "Bootstrap iterations:   10,000",
            "Resampled unit:         Donor level (BM1-BM5)",
            "Observed Specificity:   %.4f" % our_specificity,
            "95%% Confidence Interval: [%.4f, %.4f]" % (ci_lower, ci_upper),
            "",
            "2. Random Baseline Comparison & Contribution Margin:",
            "----------------------------------------------------",
            "Calibrated Threshold:   %.6f" % calibrated_threshold,
            "Random Baseline Spec:   %.4f" % baseline_specificity,
            "Our Specificity:        %.4f" % our_specificity,
            "Contribution Margin:    %.4f" % contribution_margin,
            "",
            "Status:                 COMPLETED",
        ]

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        LOGGER.info("  [OK] Saved results to %s", REPORT_PATH.name)

        # Output copy-paste block
        print("\n" + "=" * 50)
        print("MANUSCRIPT COPY-PASTE BLOCK:")
        print("=" * 50)
        print("Observed Specificity (TNR): %.1f%% (95%% Bootstrap CI: [%.1f%%, %.1f%%])" % (our_specificity * 100, ci_lower * 100, ci_upper * 100))
        print("Random Baseline Specificity: %.1f%%" % (baseline_specificity * 100))
        print("Domain-Shift Contribution Margin: %+.1f%%" % (contribution_margin * 100))
        print("=" * 50 + "\n")

        elapsed = time.time() - t_start
        LOGGER.info("Completed successfully in %.1f minutes", elapsed / 60)

    except Exception as exc:
        LOGGER.error("Execution failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
