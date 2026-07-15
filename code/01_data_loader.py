"""Download, annotate, and preprocess GEO GSE13159 transcriptomic data.

The public GEO series contains Affymetrix probe-level expression measurements.
This module maps probes to gene symbols using the corresponding GPL annotation,
summarises duplicate probes by median expression, and derives a conservative
binary phenotype: healthy/control samples are 0 and AML/ALL/leukaemia samples
are 1.  Samples whose phenotype cannot be established from GEO metadata are
excluded rather than guessed.
"""

from __future__ import annotations

import gzip
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import GEOparse
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)
GEO_ACCESSION = "GSE13159"


def _download_series_soft_https(accession: str, data_dir: Path) -> Path:
    """Download a GEO SOFT archive through NCBI HTTPS when FTP is unavailable.

    Recent Windows and enterprise network configurations commonly block the
    legacy FTP URL constructed by older GEOparse releases.  NCBI publishes the
    identical archive through HTTPS, so this fallback preserves the requested
    GEOparse-based parsing workflow while making retrieval reliable.
    """
    series_prefix = f"{accession[:-3]}nnn"
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_prefix}/{accession}/soft/"
        f"{accession}_family.soft.gz"
    )
    destination = data_dir / f"{accession}_family.soft.gz"
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    LOGGER.info("Retrying GEO archive through HTTPS: %s", url)
    try:
        request = Request(url, headers={"User-Agent": "LeukemiaQuantumPipeline/1.0"})
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if partial.stat().st_size == 0:
            raise OSError("HTTPS response contained no data.")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    LOGGER.info("HTTPS GEO archive download complete (%.1f MB).", destination.stat().st_size / 1024**2)
    return destination


def _metadata_text(metadata: dict[str, list[str] | str]) -> str:
    """Flatten a GEO metadata mapping into lower-case searchable text."""
    values: list[str] = []
    for value in metadata.values():
        values.extend(value if isinstance(value, list) else [str(value)])
    return " ".join(values).lower()


def _derive_binary_label(metadata: dict[str, list[str] | str]) -> int | None:
    """Return 0 for controls, 1 for AML/ALL/leukaemia, otherwise ``None``.

    We target the 'leukemia class:' characteristic specifically to avoid
    false positive control matches triggered by description/protocol words
    like 'control' (for example, 'Poly-A control transcripts').
    """
    all_vals: list[str] = []
    for value in metadata.values():
        all_vals.extend(value if isinstance(value, list) else [str(value)])
        
    class_text = None
    for val in all_vals:
        val_str = str(val).strip()
        if val_str.lower().startswith("leukemia class:"):
            class_text = val_str.lower()
            break
            
    if class_text is None:
        for key, val in metadata.items():
            if "characteristics" in key.lower():
                val_list = val if isinstance(val, list) else [str(val)]
                for v in val_list:
                    if str(v).lower().startswith("leukemia class:"):
                        class_text = str(v).lower()
                        break
                if class_text:
                    break
                    
    if class_text is None:
        text = " ".join(all_vals).lower()
        if "non-leukemia" in text or "healthy" in text:
            return 0
        case_pattern = r"\b(aml|all|cll|cml|mds|leukemia|leukaemia|t-all|b-all|myeloid|lymphoblastic)\b"
        if re.search(case_pattern, text):
            return 1
        return None

    if "non-leukemia" in class_text or "healthy" in class_text:
        return 0
        
    case_terms = ("aml", "all", "cll", "cml", "mds", "leukemia", "leukaemia", "t-all", "b-all")
    if any(term in class_text for term in case_terms):
        return 1
        
    return None


def _first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Find a column by exact, case-insensitive candidate name."""
    lookup = {str(column).strip().upper(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.upper() in lookup:
            return lookup[candidate.upper()]
    return None


def _gene_symbol_map(gse: GEOparse.GSE) -> dict[str, str]:
    """Construct a platform probe-ID to first valid gene-symbol mapping."""
    mapping: dict[str, str] = {}
    for platform_id, gpl in gse.gpls.items():
        table = gpl.table.copy()
        probe_column = _first_column(table.columns, ("ID", "ID_REF", "PROBE_ID"))
        gene_column = _first_column(
            table.columns, ("GENE SYMBOL", "GENE_SYMBOL", "GENE ASSIGNMENT", "SYMBOL")
        )
        if probe_column is None or gene_column is None:
            LOGGER.warning("GPL %s lacks usable probe/gene annotation columns; skipping it.", platform_id)
            continue
        for probe, raw_symbol in zip(table[probe_column], table[gene_column], strict=False):
            if pd.isna(raw_symbol):
                continue
            # GPL tables may contain symbols such as "TP53 /// WRAP53" or
            # descriptions following " // ".  Retain the first HGNC-like item.
            symbol = re.split(r"\s*(?:///|//|;|,|\|)\s*", str(raw_symbol))[0].strip()
            if symbol and symbol not in {"---", "NA", "nan"}:
                mapping[str(probe).strip()] = symbol
    if not mapping:
        raise RuntimeError("No usable probe-to-gene annotations were found in the GEO platform tables.")
    return mapping


def _expression_column(table: pd.DataFrame) -> str:
    """Select the expression value column from an individual GSM table."""
    value_column = _first_column(table.columns, ("VALUE", "SIGNAL", "RMA"))
    if value_column:
        return value_column
    numeric = [column for column in table.columns if pd.api.types.is_numeric_dtype(table[column])]
    if not numeric:
        raise ValueError("No numeric expression column is present in the GSM table.")
    return str(numeric[-1])


def _find_local_geo_file(data_dir: Path, pattern: str) -> Path | None:
    """Return a single locally supplied GEO artefact, when present."""
    matches = [Path(path) for path in glob(str(data_dir / pattern))]
    if len(matches) > 1:
        raise RuntimeError(f"More than one file matches {pattern!r} in {data_dir}: {matches}")
    return matches[0] if matches else None


def _parse_gpl_fast(soft_path: Path) -> dict[str, str]:
    """Parse the Platform (GPL) table from the SOFT family file rapidly,

    without loading the entire family metadata or GSMs.
    """
    LOGGER.info("Fast-parsing Platform (GPL) annotation table from SOFT file: %s", soft_path)
    gpl_lines = []
    in_table = False
    with gzip.open(soft_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            elif line.startswith("!platform_table_end"):
                break
            if in_table:
                gpl_lines.append(line.strip())
                
    if not gpl_lines:
        raise RuntimeError("No platform table found in the SOFT family file.")
        
    header = gpl_lines[0].split("\t")
    probe_col_idx = None
    gene_col_idx = None
    
    for cand in ("ID", "ID_REF", "PROBE_ID"):
        if cand in header:
            probe_col_idx = header.index(cand)
            break
            
    header_upper = [col.upper() for col in header]
    for cand in ("GENE SYMBOL", "GENE_SYMBOL", "GENE ASSIGNMENT", "SYMBOL"):
        if cand in header_upper:
            gene_col_idx = header_upper.index(cand)
            break
            
    if probe_col_idx is None or gene_col_idx is None:
        raise RuntimeError(f"Could not find probe/gene symbol columns in GPL header: {header}")
        
    probe_to_gene = {}
    for line in gpl_lines[1:]:
        tokens = line.split("\t")
        if len(tokens) <= max(probe_col_idx, gene_col_idx):
            continue
        probe_id = tokens[probe_col_idx].strip().strip('"')
        raw_symbol = tokens[gene_col_idx].strip()
        if not raw_symbol:
            continue
        symbol = re.split(r"\s*(?:///|//|;|,|\|)\s*", raw_symbol)[0].strip()
        if symbol and symbol not in {"---", "NA", "nan"}:
            probe_to_gene[probe_id] = symbol
            
    return probe_to_gene


def _preprocess_series_matrix_fast(
    matrix_path: Path, probe_to_gene: dict[str, str]
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Read a GEO Series Matrix line-by-line using streaming decompression

    and temporary file filtering to avoid memory crashes on pandas read_csv.
    """
    LOGGER.info("Streaming and filtering local GEO Series Matrix: %s", matrix_path)
    temp_filtered_file = matrix_path.parent / "temp_filtered_matrix.txt"
    sample_ids = None
    sample_meta_dict = {}

    with gzip.open(matrix_path, "rt", encoding="utf-8") as f_in, open(temp_filtered_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [tok.strip().strip('"') for tok in line.strip().split("\t")[1:]]
                sample_meta_dict = {sid: [] for sid in sample_ids}
                f_out.write(line)
                continue
                
            if line.startswith("!Sample_"):
                if sample_ids is not None:
                    tokens = line.strip().split("\t")
                    vals = tokens[1:]
                    if len(vals) == len(sample_ids):
                        for j, val in enumerate(vals):
                            val_cleaned = val.strip().strip('"')
                            if val_cleaned:
                                sample_meta_dict[sample_ids[j]].append(val_cleaned)
                f_out.write(line)
                continue
                
            if line.startswith("!"):
                f_out.write(line)
                continue
                
            stripped = line.strip()
            if not stripped:
                continue
                
            tab_idx = line.find("\t")
            if tab_idx == -1:
                continue
                
            first_token = line[:tab_idx].strip().strip('"')
            if first_token == "ID_REF":
                if sample_ids is None:
                    sample_ids = [tok.strip().strip('"') for tok in line.strip().split("\t")[1:]]
                    sample_meta_dict = {sid: [] for sid in sample_ids}
                f_out.write(line)
                continue
                
            if sample_ids is None:
                continue
                
            probe_id = first_token
            gene_symbol = probe_to_gene.get(probe_id)
            if gene_symbol is not None:
                f_out.write(f'"{gene_symbol}"' + line[tab_idx:])

    LOGGER.info("Matrix filtered to temporary file on disk. Reading into Pandas...")
    try:
        probe_by_sample = pd.read_csv(
            temp_filtered_file,
            sep="\t",
            comment="!",
            index_col=0,
            low_memory=False,
        ).astype(np.float32)
    finally:
        if temp_filtered_file.exists():
            temp_filtered_file.unlink()

    LOGGER.info("Collapsing duplicate probes by median...")
    expression_df = probe_by_sample.groupby(level=0, sort=False).median().T
    del probe_by_sample
    expression_df.index = expression_df.index.astype(str).str.strip()

    labels: dict[str, int] = {}
    excluded = 0
    for sample_id in expression_df.index:
        meta_list = sample_meta_dict.get(sample_id, [])
        fake_metadata = {"characteristics": meta_list}
        label = _derive_binary_label(fake_metadata)
        if label is not None:
            labels[sample_id] = label
        else:
            excluded += 1

    if len(labels) < 10 or len(set(labels.values())) < 2:
        raise RuntimeError(
            "Too few labelled samples were found in the metadata. "
            "Inspect the clinical phenotype rules."
        )

    expression_df = expression_df.loc[list(labels)]
    y = pd.Series(labels, index=expression_df.index, name="label", dtype=np.int8)
    return expression_df, y, excluded


def load_and_preprocess(base_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Download GSE13159 and return scaled gene expression and binary labels.

    Parameters
    ----------
    base_dir:
        Root of the project scaffold (for example ``D:\\Leukemia_Quantum_Pipeline``).

    Returns
    -------
    (X, y):
        ``X`` has samples as rows and gene symbols as columns. ``y`` is a
        same-index Series holding 0 (control) or 1 (AML/ALL/leukaemia).
    """
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    local_soft = _find_local_geo_file(data_dir, f"{GEO_ACCESSION}*family.soft.gz")
    local_matrix = _find_local_geo_file(data_dir, f"{GEO_ACCESSION}*series_matrix*.txt.gz")
    use_local_matrix = local_soft is not None and local_matrix is not None
    if use_local_matrix:
        LOGGER.info("Using locally supplied SOFT metadata and Series Matrix; no GEO download is required.")
        probe_to_gene = _parse_gpl_fast(local_soft)
        expression_df, y, excluded = _preprocess_series_matrix_fast(local_matrix, probe_to_gene)
    else:
        LOGGER.info("Downloading/loading %s from GEO into %s", GEO_ACCESSION, data_dir)
        try:
            gse = GEOparse.get_GEO(geo=GEO_ACCESSION, destdir=str(data_dir), silent=True)
        except Exception as ftp_exc:  # GEOparse versions using FTP can fail on modern networks.
            try:
                soft_file = _download_series_soft_https(GEO_ACCESSION, data_dir)
                gse = GEOparse.get_GEO(filepath=str(soft_file), silent=True)
            except Exception as https_exc:
                raise RuntimeError(
                    f"Unable to retrieve {GEO_ACCESSION}; FTP error: {ftp_exc}; HTTPS fallback error: {https_exc}"
                ) from https_exc

        probe_to_gene = _gene_symbol_map(gse)
        sample_series: dict[str, pd.Series] = {}
        labels: dict[str, int] = {}
        excluded = 0
        for sample_id, gsm in gse.gsms.items():
            label = _derive_binary_label(gsm.metadata)
            if label is None:
                excluded += 1
                continue
            table = gsm.table.copy()
            probe_column = _first_column(table.columns, ("ID_REF", "ID", "PROBE_ID"))
            if probe_column is None:
                LOGGER.warning("Skipping %s because its expression table has no probe-ID column.", sample_id)
                continue
            try:
                value_column = _expression_column(table)
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", sample_id, exc)
                continue
            frame = pd.DataFrame(
                {
                    "gene": table[probe_column].astype(str).str.strip().map(probe_to_gene),
                    "value": pd.to_numeric(table[value_column], errors="coerce"),
                }
            ).dropna(subset=["gene"])
            expression = frame.groupby("gene", sort=False)["value"].median()
            sample_series[sample_id] = expression
            labels[sample_id] = label
        if len(sample_series) < 10 or len(set(labels.values())) < 2:
            raise RuntimeError(
                "Too few labelled samples were extracted. Inspect GEO metadata and update the phenotype rules."
            )
        expression_df = pd.DataFrame.from_dict(sample_series, orient="index")
        expression_df.index.name = "sample_id"
        expression_df = expression_df.dropna(axis=1, how="all")
        y = pd.Series(labels, index=expression_df.index, name="label", dtype=np.int8)

    expression_df = expression_df.dropna(axis=1, how="all")
    # Median imputation is fitted feature-wise over samples and preserves all genes.
    imputed = SimpleImputer(strategy="median").fit_transform(expression_df)
    scaled = StandardScaler().fit_transform(imputed)
    X = pd.DataFrame(scaled, index=expression_df.index, columns=expression_df.columns, dtype=np.float32)
    y = y.loc[X.index]

    output = X.copy()
    output.insert(0, "label", y)
    output.to_csv(data_dir / "processed_expression.csv", index=True)
    LOGGER.info(
        "Preprocessing complete: %d labelled samples (%d controls, %d leukaemia) and %d genes; %d samples excluded.",
        len(X), int((y == 0).sum()), int((y == 1).sum()), X.shape[1], excluded,
    )
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    load_and_preprocess(Path(r"D:\Leukemia_Quantum_Pipeline"))
