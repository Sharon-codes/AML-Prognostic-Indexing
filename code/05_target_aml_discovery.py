# pip install matplotlib-venn networkx scipy requests pandas numpy scikit-learn
"""Pediatric TARGET-AML Biomarker Discovery & Validation Pipeline.

This script parses target metadata and clinical outcomes, downloads and streams
gene expression data from GDC, runs Binary Quantum-Behaved PSO feature selection,
compares pediatric and adult biomarkers, and validates findings using
differential expression analysis (Volcano plot) and STRING PPI network topology.
"""

import os
import sys
import json
import gzip
import csv
import time
import io
import logging
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib_venn as mv
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(r"D:\Leukemia_Quantum_Pipeline\logs_and_output\pipeline.log"), encoding="utf-8")
    ]
)
LOGGER = logging.getLogger("target_aml_discovery")

# Paths and Constants
PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs_and_output"
IMAGES_DIR = PROJECT_ROOT / "images"

METADATA_PATH = DATA_DIR / "target_metadata.json"
CLINICAL_PATH = DATA_DIR / "target_clinical.gz"
PROCESSED_CSV_PATH = DATA_DIR / "processed_target_aml.csv"
PROCESSED_RAW_PATH = DATA_DIR / "processed_target_aml_raw.csv"
ADULT_GENES_PATH = LOGS_DIR / "selected_genes.txt"
PEDIATRIC_GENES_PATH = LOGS_DIR / "pediatric_selected_genes.txt"
OVERLAP_SETS_PATH = LOGS_DIR / "biomarker_overlap_sets.txt"
PPI_METRICS_PATH = LOGS_DIR / "ppi_network_metrics.txt"

# GDC API endpoints
GDC_DOWNLOAD_URL = "https://api.gdc.cancer.gov/data"


def load_bqpso_selector():
    """Dynamically load BinaryQuantumPSO from 02_bqpso_selector.py."""
    code_dir = PROJECT_ROOT / "code"
    if str(code_dir) not in sys.path:
        sys.path.append(str(code_dir))
    module = __import__("02_bqpso_selector")
    return module.BinaryQuantumPSO


def parse_clinical_outcomes() -> dict[str, str]:
    """Parse target_clinical.gz and map cases.submitter_id to demographic.vital_status."""
    LOGGER.info("Parsing clinical vital statuses from %s...", CLINICAL_PATH.name)
    patient_vitals = {}
    
    with tarfile.open(CLINICAL_PATH, "r:gz") as tar:
        f = tar.extractfile("clinical.tsv")
        if not f:
            raise FileNotFoundError("clinical.tsv not found inside target_clinical.gz")
            
        header_line = f.readline().decode('utf-8')
        headers = header_line.strip().split('\t')
        
        case_idx = headers.index('cases.submitter_id')
        vital_idx = headers.index('demographic.vital_status')
        
        for line in f:
            row = line.decode('utf-8').strip('\n').split('\t')
            if len(row) > max(case_idx, vital_idx):
                case_id = row[case_idx].strip()
                vital = row[vital_idx].strip()
                if case_id and case_id != "'--" and vital in ['Alive', 'Dead']:
                    patient_vitals[case_id] = vital
                    
    LOGGER.info("Successfully mapped outcomes for %d pediatric cases.", len(patient_vitals))
    return patient_vitals


def parse_metadata_mapping(patient_vitals: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Parse target_metadata.json to map file IDs to case submitter IDs."""
    LOGGER.info("Parsing metadata mappings from %s...", METADATA_PATH.name)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    file_to_case = {}
    valid_file_ids = []
    
    for entry in meta:
        file_id = entry['file_id']
        entities = entry.get('associated_entities', [])
        for ent in entities:
            entity_id = ent.get('entity_submitter_id')
            if entity_id:
                # Extract the 3-part case ID prefix (e.g. TARGET-20-PANTIV)
                case_id = "-".join(entity_id.split("-")[:3])
                if case_id in patient_vitals:
                    file_to_case[file_id] = case_id
                    valid_file_ids.append(file_id)
                    break # Use first valid mapping aliquot
                    
    LOGGER.info("Found %d GDC expression files mapping to labeled cases.", len(valid_file_ids))
    return file_to_case, valid_file_ids


def download_gdc_batch(file_ids: list[str]) -> bytes:
    """Download a compressed tarball containing a batch of GDC files."""
    headers = {"Content-Type": "application/json"}
    data = {"ids": file_ids}
    
    for attempt in range(4):
        try:
            response = requests.post(GDC_DOWNLOAD_URL, json=data, stream=True, timeout=180)
            if response.status_code == 200:
                return response.content
            else:
                LOGGER.warning("GDC bulk API returned status %d. Attempt %d/4.", response.status_code, attempt + 1)
        except Exception as e:
            LOGGER.warning("GDC bulk API request failed with exception: %s. Attempt %d/4.", e, attempt + 1)
        time.sleep(5 + 2 * attempt)
        
    raise RuntimeError(f"Unable to download batch of {len(file_ids)} files from GDC API after multiple attempts.")


def parse_gdc_tarball(
    content: bytes, file_to_case: dict[str, str], gene_data_dict: dict[str, dict[str, float]]
) -> int:
    """Decompress and parse GDC TSV files in-memory to extract protein-coding expressions."""
    parsed_count = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Check for star gene counts file in directory structure
            if member.name.endswith(".tsv") and "/" in member.name:
                file_uuid = member.name.split("/")[0]
                case_id = file_to_case.get(file_uuid)
                if not case_id or case_id in gene_data_dict:
                    continue
                
                f_obj = tar.extractfile(member)
                if f_obj:
                    gene_counts = {}
                    for line in f_obj:
                        decoded = line.decode('utf-8').strip()
                        if not decoded or decoded.startswith("#") or decoded.startswith("gene_id"):
                            continue
                        tokens = decoded.split("\t")
                        if len(tokens) >= 4:
                            gene_name = tokens[1]
                            gene_type = tokens[2]
                            unstranded_count = tokens[3]
                            
                            if gene_type == "protein_coding":
                                try:
                                    count = float(unstranded_count)
                                except ValueError:
                                    count = 0.0
                                # Collapse duplicates by summing counts (standard RNA-seq practice)
                                gene_counts[gene_name] = gene_counts.get(gene_name, 0.0) + count
                                
                    if gene_counts:
                        gene_data_dict[case_id] = gene_counts
                        parsed_count += 1
    return parsed_count


def stream_and_process_dataset(file_to_case: dict[str, str], valid_file_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download files in concurrent batches and compile raw and scaled dataframes."""
    LOGGER.info("Starting concurrent bulk download and streaming parsing...")
    batch_size = 200
    batches = [valid_file_ids[i:i + batch_size] for i in range(0, len(valid_file_ids), batch_size)]
    
    gene_data_dict = {}
    
    def process_batch_worker(batch_idx: int, batch_ids: list[str]) -> int:
        LOGGER.info("Processing GDC download batch %d/%d (%d files)...", batch_idx + 1, len(batches), len(batch_ids))
        content = download_gdc_batch(batch_ids)
        parsed = parse_gdc_tarball(content, file_to_case, gene_data_dict)
        LOGGER.info("Parsed %d expression profiles from batch %d.", parsed, batch_idx + 1)
        return parsed

    start_time = time.time()
    # 5 workers balances concurrency and GDC API load safely
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_batch_worker, idx, batch): idx for idx, batch in enumerate(batches)}
        for future in as_completed(futures):
            future.result() # Propagate errors if any worker fails
            
    LOGGER.info("Data download and extraction complete in %.2f minutes.", (time.time() - start_time) / 60)
    LOGGER.info("Creating pandas matrices for %d unique cases...", len(gene_data_dict))
    
    raw_df = pd.DataFrame.from_dict(gene_data_dict, orient="index")
    raw_df.index.name = "sample_id"
    raw_df = raw_df.dropna(axis=1, how="all")
    
    # Impute missing counts and apply standard scaling
    LOGGER.info("Imputing missing values and scaling expression matrix...")
    imputer = SimpleImputer(strategy="median")
    raw_imputed = pd.DataFrame(imputer.fit_transform(raw_df), index=raw_df.index, columns=raw_df.columns)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_imputed)
    scaled_df = pd.DataFrame(scaled_data, index=raw_imputed.index, columns=raw_imputed.columns, dtype=np.float32)
    
    return raw_imputed, scaled_df


def run_pediatric_bqpso(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Execute BQPSO feature selection on the pediatric dataset."""
    LOGGER.info("Initializing BQPSO biomarker selector on TARGET-AML (50 particles, 100 epochs)...")
    BinaryQuantumPSO = load_bqpso_selector()
    
    selector = BinaryQuantumPSO(
        n_particles=50,
        n_epochs=100,
        min_features=15,
        max_features=30,
        candidate_features=1000,
        random_state=42
    )
    
    # Run BQPSO
    X_reduced = selector.fit_select(X, y, LOGS_DIR)
    selected_genes = list(X_reduced.columns)
    
    # Save selected genes list
    with open(PEDIATRIC_GENES_PATH, "w") as f:
        f.write("\n".join(selected_genes) + "\n")
        
    LOGGER.info("BQPSO selection complete! Selected %d biomarkers: %s", len(selected_genes), selected_genes)
    return selected_genes


def compute_cross_demographic_overlap(pediatric_genes: list[str]):
    """Compare pediatric and adult biomarkers and save sets + Venn diagram."""
    LOGGER.info("Analyzing overlap between pediatric and adult biomarker signatures...")
    
    # Load adult biomarkers
    if not ADULT_GENES_PATH.exists():
        raise FileNotFoundError(f"Adult biomarkers list not found at {ADULT_GENES_PATH}")
        
    with open(ADULT_GENES_PATH, "r") as f:
        adult_genes = [line.strip() for line in f if line.strip()]
        
    adult_set = set(adult_genes)
    pediatric_set = set(pediatric_genes)
    
    universal = adult_set.intersection(pediatric_set)
    adult_only = adult_set - pediatric_set
    pediatric_only = pediatric_set - adult_set
    
    # Save overlap sets text
    with open(OVERLAP_SETS_PATH, "w") as f:
        f.write("=== Universal Biomarkers (Intersection) ===\n")
        f.write("\n".join(sorted(universal)) + "\n\n")
        f.write("=== Adult-Specific Biomarkers (GSE13159 only) ===\n")
        f.write("\n".join(sorted(adult_only)) + "\n\n")
        f.write("=== Pediatric-Specific Biomarkers (TARGET-AML only) ===\n")
        f.write("\n".join(sorted(pediatric_only)) + "\n")
        
    LOGGER.info("Overlap computed: %d universal, %d adult-only, %d pediatric-only biomarkers.",
                len(universal), len(adult_only), len(pediatric_only))
                
    # Plot Venn Diagram
    plt.figure(figsize=(7, 5), dpi=300)
    mv.venn2(
        [adult_set, pediatric_set],
        set_labels=('Adult cohort (GSE13159)', 'Pediatric cohort (TARGET-AML)'),
        set_colors=('#1f77b4', '#d62728'),
        alpha=0.65
    )
    plt.title("Biomarker overlap across Clinical Cohorts", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "pediatric_adult_venn.png", dpi=300)
    plt.close()
    LOGGER.info("Saved overlap Venn diagram to images/pediatric_adult_venn.png")


def run_differential_expression_analysis(raw_imputed: pd.DataFrame, y: pd.Series, pediatric_genes: list[str]):
    """Perform differential expression analysis and generate Volcano Plot novelty proof."""
    LOGGER.info("Performing Differential Expression Analysis (DEA) on raw counts...")
    
    group_dead = raw_imputed.loc[y == 1]
    group_alive = raw_imputed.loc[y == 0]
    
    # Calculate Log2 Fold-Change (using pseudo-count of 1.0 to handle zeros safely)
    mean_dead = group_dead.mean(axis=0)
    mean_alive = group_alive.mean(axis=0)
    log2fc = np.log2(mean_dead + 1.0) - np.log2(mean_alive + 1.0)
    
    # Calculate p-values using Welch's t-test (independent two-sample unequal variance)
    _, p_values = stats.ttest_ind(group_dead.to_numpy(), group_alive.to_numpy(), axis=0, equal_var=False)
    p_values = np.nan_to_num(p_values, nan=1.0)
    p_values[p_values <= 0.0] = 1e-300 # Prevent log10 overflow
    neg_log_p = -np.log10(p_values)
    
    # Compile Volcano Plot
    plt.figure(figsize=(9, 7), dpi=300)
    
    # Background genes
    plt.scatter(log2fc, neg_log_p, color="lightgrey", alpha=0.5, s=8, label="All background genes")
    
    # Overlay selected pediatric biomarkers
    sel_log2fc = log2fc.loc[pediatric_genes]
    sel_neg_log_p = neg_log_p[raw_imputed.columns.get_indexer(pediatric_genes)]
    
    plt.scatter(
        sel_log2fc, sel_neg_log_p,
        color="crimson", alpha=0.9, s=40, edgecolor="black", linewidth=0.5,
        label="BQPSO selected biomarkers"
    )
    
    # Label the biomarkers
    for gene, x, y_val in zip(pediatric_genes, sel_log2fc, sel_neg_log_p):
        plt.annotate(
            gene, (x, y_val),
            textcoords="offset points", xytext=(0, 4),
            ha='center', fontsize=6, fontweight='bold', color='darkred'
        )
        
    plt.axhline(-np.log10(0.05), color="blue", linestyle="--", alpha=0.4, label="p = 0.05 threshold")
    plt.axvline(0.0, color="grey", linestyle="-", alpha=0.3)
    
    plt.xlabel("Log2 Fold Change (Dead vs. Alive)", fontsize=11, fontweight='bold')
    plt.ylabel("-Log10 p-value (t-test)", fontsize=11, fontweight='bold')
    plt.title("Pediatric Volcano Plot: BQPSO Biomarkers vs. Background Genes", fontsize=12, fontweight='bold', pad=15)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "volcano_novelty_proof.png", dpi=300)
    plt.close()
    LOGGER.info("Saved volcano plot novelty proof to images/volcano_novelty_proof.png")


def validate_biological_importance(pediatric_genes: list[str]):
    """Query STRING API and build/analyze topological NetworkX interaction graph."""
    LOGGER.info("Retrieving protein-protein interactions from STRING database API...")
    
    # Join with newlines
    genes_payload = "\n".join(pediatric_genes)
    params = {
        "identifiers": genes_payload,
        "species": 9606
    }
    
    try:
        response = requests.get("https://string-db.org/api/json/network", params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"STRING API returned HTTP error {response.status_code}")
        interactions = response.json()
    except Exception as e:
        LOGGER.error("Failed to retrieve network edges from STRING API: %s. Creating empty PPI network.", e)
        interactions = []
        
    LOGGER.info("STRING API returned %d interaction edges.", len(interactions))
    
    # Build Graph
    G = nx.Graph()
    G.add_nodes_from(pediatric_genes) # Ensure isolated nodes are represented
    
    for inter in interactions:
        p1 = inter.get("preferredName_A")
        p2 = inter.get("preferredName_B")
        score = inter.get("score")
        if p1 in G and p2 in G:
            G.add_edge(p1, p2, weight=score)
            
    # Calculate Centrality
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Find hub genes
    top_hub_degree = max(degree_cent, key=degree_cent.get) if degree_cent else "None"
    top_hub_betweenness = max(betweenness_cent, key=betweenness_cent.get) if betweenness_cent else "None"
    
    # Save network metrics
    with open(PPI_METRICS_PATH, "w") as f:
        f.write("Gene\tDegree Centrality\tBetweenness Centrality\n")
        for gene in sorted(pediatric_genes):
            f.write(f"{gene}\t{degree_cent[gene]:.6f}\t{betweenness_cent[gene]:.6f}\n")
        f.write("\n=== Hub Analysis ===\n")
        f.write(f"Top Hub Gene (Degree): {top_hub_degree} (Centrality: {degree_cent.get(top_hub_degree, 0.0):.6f})\n")
        f.write(f"Top Bottleneck Gene (Betweenness): {top_hub_betweenness} (Centrality: {betweenness_cent.get(top_hub_betweenness, 0.0):.6f})\n")
        
    LOGGER.info("Saved network centrality metrics to %s.", PPI_METRICS_PATH.name)
    
    # Plot network
    plt.figure(figsize=(11, 9), dpi=300)
    pos = nx.spring_layout(G, k=0.35, seed=42)
    
    # Scale node sizes based on Degree Centrality (between 150 and 1500)
    node_sizes = [150 + 1500 * degree_cent[node] for node in G.nodes()]
    
    # Color nodes: Hub (gold), Bottleneck (orange), Intersect (orange/gold if both, else skyblue)
    node_colors = []
    for node in G.nodes():
        if node == top_hub_degree and node == top_hub_betweenness:
            node_colors.append("gold")
        elif node == top_hub_degree:
            node_colors.append("gold")
        elif node == top_hub_betweenness:
            node_colors.append("darkorange")
        else:
            node_colors.append("skyblue")
            
    # Draw Graph
    edges = G.edges(data=True)
    edge_widths = [0.5 + 3.0 * edge[2]['weight'] for edge in edges] if edges else []
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors="black", linewidths=0.6)
    if edges:
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="grey", alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")
    
    plt.title(
        f"STRING PPI Network of TARGET-AML Biomarkers\n(Node sizes scaled by Degree Centrality; Hub: {top_hub_degree}; Bottleneck: {top_hub_betweenness})",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ppi_hub_network.png", dpi=300)
    plt.close()
    LOGGER.info("Saved network topological graph to images/ppi_hub_network.png")


def main():
    """Execute the pediatric TARGET-AML discovery pipeline end-to-end."""
    LOGGER.info("=" * 72)
    LOGGER.info("Starting TARGET-AML pediatric biomarker discovery & validation pipeline")
    LOGGER.info("=" * 72)
    
    try:
        # Load datasets from cache if they already exist, else download/process them
        if PROCESSED_CSV_PATH.exists() and PROCESSED_RAW_PATH.exists():
            LOGGER.info("Loading existing processed and raw datasets from cache...")
            processed_df = pd.read_csv(PROCESSED_CSV_PATH, index_col=0)
            y = processed_df["label"].astype(np.int8)
            scaled_df = processed_df.drop(columns=["label"]).astype(np.float32)
            
            raw_df = pd.read_csv(PROCESSED_RAW_PATH, index_col=0)
            raw_imputed = raw_df.drop(columns=["label"]).astype(np.float32)
        else:
            # Step 2: Parse outcomes (needs to happen first to allow filtering)
            patient_vitals = parse_clinical_outcomes()
            
            # Step 1: Parse metadata and stream files
            file_to_case, valid_file_ids = parse_metadata_mapping(patient_vitals)
            raw_imputed, scaled_df = stream_and_process_dataset(file_to_case, valid_file_ids)
            
            # Add labels and save matrices
            y = pd.Series({cid: 1 if patient_vitals[cid] == 'Dead' else 0 for cid in scaled_df.index}, name="label", dtype=np.int8)
            
            # Save scaled
            processed_df = scaled_df.copy()
            processed_df.insert(0, "label", y)
            processed_df.to_csv(PROCESSED_CSV_PATH, index=True)
            
            # Save raw
            raw_df = raw_imputed.copy()
            raw_df.insert(0, "label", y)
            raw_df.to_csv(PROCESSED_RAW_PATH, index=True)
            LOGGER.info("Saved processed scaled and raw matrices to disk.")
            
        # Step 3: Run BQPSO selection or load from cache if it exists
        if PEDIATRIC_GENES_PATH.exists():
            LOGGER.info("Found existing selected pediatric genes list. Loading from cache...")
            with open(PEDIATRIC_GENES_PATH, "r") as f:
                pediatric_genes = [line.strip() for line in f if line.strip()]
        else:
            pediatric_genes = run_pediatric_bqpso(scaled_df, y)
            
        # Step 4: Overlap and Venn diagram
        compute_cross_demographic_overlap(pediatric_genes)
        
        # Step 5: Volcano Plot novelty proof
        run_differential_expression_analysis(raw_imputed, y, pediatric_genes)
        
        # Step 6: STRING API & Network centrality analysis
        validate_biological_importance(pediatric_genes)
        
        LOGGER.info("=" * 72)
        LOGGER.info("TARGET-AML pediatric discovery pipeline executed successfully!")
        LOGGER.info("=" * 72)
        
    except Exception as exc:
        LOGGER.exception("Pipeline execution failed with exception: %s", exc)
        raise exc


if __name__ == "__main__":
    main()
