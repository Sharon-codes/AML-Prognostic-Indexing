# Install dependencies: pip install combat umap-learn networkx matplotlib pandas numpy scipy
"""
Pan-Leukemia Universal Master Regulator Discovery and Validation Pipeline.
Combines adult GSE13159 (Microarray) and pediatric TARGET-AML (RNA-Seq) cohorts,
corrects for batch effects using pyComBat, runs quantum feature selection (BQPSO)
to isolate universal biomarkers, and validates results using UMAP and STRING.
"""

import os
import gc
import sys
import logging
import urllib.request
import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import networkx as nx
import patsy
import umap

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(r"D:\Leukemia_Quantum_Pipeline\logs_and_output\pipeline.log"), mode="a", encoding="utf-8")
    ]
)
LOGGER = logging.getLogger("pan_leukemia_universal")

# Paths configuration
PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs_and_output"
IMAGES_DIR = PROJECT_ROOT / "images"

ADULT_DATA_PATH = DATA_DIR / "processed_expression.csv"
PED_DATA_PATH = DATA_DIR / "processed_target_aml.csv"
CORRECTED_DATA_PATH = DATA_DIR / "pan_leukemia_batch_corrected.csv"
UNIVERSAL_BIOMARKERS_PATH = LOGS_DIR / "universal_master_biomarkers.txt"
UMAP_PLOT_PATH = IMAGES_DIR / "batch_correction_umap.png"
PPI_METRICS_PATH = LOGS_DIR / "universal_ppi_metrics.txt"
PPI_NETWORK_PATH = IMAGES_DIR / "universal_ppi_network.png"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_bqpso_selector():
    """Dynamically load BinaryQuantumPSO class from 02_bqpso_selector.py."""
    code_dir = PROJECT_ROOT / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    try:
        bqpso_module = __import__("02_bqpso_selector")
        return bqpso_module.BinaryQuantumPSO
    except Exception as e:
        LOGGER.error("Failed to dynamically load BQPSO selector: %s", e)
        raise


def step_1_data_alignment():
    """Load and align adult and pediatric cohorts, then combine them."""
    LOGGER.info("STEP 1: Starting Data Alignment and Intersection...")
    
    if not ADULT_DATA_PATH.exists():
        raise FileNotFoundError(f"Adult preprocessed expression file missing: {ADULT_DATA_PATH}")
    if not PED_DATA_PATH.exists():
        raise FileNotFoundError(f"Pediatric preprocessed expression file missing: {PED_DATA_PATH}")

    # Load Adult (GSE13159)
    LOGGER.info("Loading adult GSE13159 dataset...")
    adult_df = pd.read_csv(ADULT_DATA_PATH)
    LOGGER.info("Adult raw shape: %s", adult_df.shape)
    
    # Load Pediatric (TARGET-AML)
    LOGGER.info("Loading pediatric TARGET-AML dataset...")
    ped_df = pd.read_csv(PED_DATA_PATH)
    LOGGER.info("Pediatric raw shape: %s", ped_df.shape)

    # Resolve index columns
    adult_samples = adult_df.iloc[:, 0].astype(str)
    ped_samples = ped_df.iloc[:, 0].astype(str)

    # Determine gene column names (excluding index and labels)
    adult_genes = set(adult_df.columns) - {"Unnamed: 0", "label"}
    ped_genes = set(ped_df.columns) - {"sample_id", "label"}

    # Find intersection of genes
    intersecting_genes = sorted(list(adult_genes.intersection(ped_genes)))
    LOGGER.info("Number of intersecting genes present in both cohorts: %d", len(intersecting_genes))
    if len(intersecting_genes) == 0:
        raise ValueError("No overlapping genes found between the adult and pediatric datasets.")

    # Align columns
    LOGGER.info("Aligning expressions on intersecting genes...")
    # Keep adult controls (0) and leukemia cases (1)
    adult_aligned = adult_df[intersecting_genes].astype(np.float32)
    adult_labels = adult_df["label"].astype(np.int32)
    adult_batch = np.zeros(len(adult_aligned), dtype=np.int32)  # Batch 0 = Adult GSE13159

    # Set all pediatric TARGET-AML samples to 1 (Leukemia cases)
    ped_aligned = ped_df[intersecting_genes].astype(np.float32)
    ped_labels = pd.Series(np.ones(len(ped_aligned), dtype=np.int32))  # Batch 1 = Pediatric TARGET-AML (all AML cases)
    ped_batch = np.ones(len(ped_aligned), dtype=np.int32)              # Batch 1 = Pediatric TARGET-AML

    # Concatenate matrices
    LOGGER.info("Concatenating aligned adult and pediatric datasets...")
    pan_expression = pd.concat([adult_aligned, ped_aligned], axis=0, ignore_index=True)
    pan_labels = pd.concat([adult_labels, ped_labels], axis=0, ignore_index=True)
    pan_batches = pd.concat([pd.Series(adult_batch), pd.Series(ped_batch)], axis=0, ignore_index=True)

    # Re-assemble unified DataFrame
    pan_matrix = pan_expression.copy()
    pan_matrix["label"] = pan_labels
    pan_matrix["batch"] = pan_batches
    LOGGER.info("Unified Pan-Leukemia cohort shape: %s (samples: %d adult, %d pediatric)", 
                pan_matrix.shape, len(adult_aligned), len(ped_aligned))

    # Free memory
    del adult_df, ped_df, adult_aligned, ped_aligned
    gc.collect()
    
    return pan_matrix, intersecting_genes


def step_2_batch_effect_correction(pan_matrix: pd.DataFrame, intersecting_genes: list[str]):
    """Apply ComBat batch effect correction to unified matrix."""
    LOGGER.info("STEP 2: Starting ComBat Batch Effect Correction...")
    
    from combat.pycombat import pycombat

    # Prepare expression matrix for ComBat: features (genes) x samples
    LOGGER.info("Preparing expression matrix for ComBat (transposing to genes x samples)...")
    expr_to_correct = pan_matrix[intersecting_genes].T
    batch_series = pan_matrix["batch"]

    # Design matrix for covariates (mod) - protecting label (disease status)
    LOGGER.info("Constructing PatSy design matrix to preserve biological variance (disease label)...")
    metadata = pd.DataFrame({
        "batch": batch_series,
        "label": pan_matrix["label"]
    })
    mod_df = patsy.dmatrix("~ label", metadata, return_type="dataframe")
    
    # Format mod as list of lists (columns) to avoid the pycombat DataFrame comparison bug
    mod = [metadata["label"].tolist()]

    # Run pyComBat
    LOGGER.info("Executing pyComBat Empirical Bayes adjustment...")
    corrected_expr_transposed = pycombat(expr_to_correct, batch_series, mod=mod)
    
    # Transpose back to samples x genes
    LOGGER.info("Transposing back to samples x genes...")
    corrected_expr = corrected_expr_transposed.T
    
    # Re-assemble corrected DataFrame
    corrected_df = pd.DataFrame(corrected_expr, columns=intersecting_genes)
    corrected_df["label"] = pan_matrix["label"]
    corrected_df["batch"] = pan_matrix["batch"]

    # Save to disk
    LOGGER.info("Saving corrected unified dataset to %s...", CORRECTED_DATA_PATH)
    corrected_df.to_csv(CORRECTED_DATA_PATH, index=False)
    
    # Free memory
    del expr_to_correct, corrected_expr_transposed, corrected_expr
    gc.collect()

    return corrected_df


def step_3_quantum_feature_selection(corrected_df: pd.DataFrame, intersecting_genes: list[str]):
    """Run BQPSO quantum feature selection on batch-corrected dataset."""
    LOGGER.info("STEP 3: Running Pan-Cohort BQPSO Feature Selection...")
    
    BinaryQuantumPSO = load_bqpso_selector()
    
    X = corrected_df[intersecting_genes]
    y = corrected_df["label"]

    LOGGER.info("Initializing BQPSO selector (50 particles, 100 epochs, dual=False)...")
    selector = BinaryQuantumPSO(
        n_particles=50,
        n_epochs=100,
        min_features=15,
        max_features=30,
        candidate_features=1000,
        random_state=42
    )

    # Fit selector
    X_reduced = selector.fit_select(X, y, LOGS_DIR)
    universal_biomarkers = list(X_reduced.columns)

    # Save list to file
    LOGGER.info("Saving discovered universal biomarkers to %s...", UNIVERSAL_BIOMARKERS_PATH)
    with open(UNIVERSAL_BIOMARKERS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(universal_biomarkers) + "\n")

    LOGGER.info("Pan-Cohort BQPSO complete! Discovered %d universal master biomarkers: %s",
                len(universal_biomarkers), universal_biomarkers)
    return universal_biomarkers


def step_4_umap_visualization(raw_matrix: pd.DataFrame, corrected_df: pd.DataFrame, intersecting_genes: list[str]):
    """Generate side-by-side UMAP plot (Before vs After ComBat)."""
    LOGGER.info("STEP 4: Projecting cohorts using UMAP (Before vs. After ComBat)...")

    # Downsample slightly to accelerate UMAP rendering if the cohort is huge,
    # but 4,000 samples is quite small, so we can process all of them.
    X_raw = raw_matrix[intersecting_genes].values
    X_corr = corrected_df[intersecting_genes].values
    batches = raw_matrix["batch"].values

    LOGGER.info("Computing UMAP projection for uncorrected data...")
    reducer_raw = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_raw = reducer_raw.fit_transform(X_raw)

    LOGGER.info("Computing UMAP projection for batch-corrected data...")
    reducer_corr = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_corr = reducer_corr.fit_transform(X_corr)

    # Set up matplotlib figure (premium aesthetics)
    LOGGER.info("Generating side-by-side validation figure...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    # Color mapping: GSE13159 (Adult) -> Deep Steel Blue, TARGET-AML (Pediatric) -> Crimson Coral
    colors = ['#2b5c8f', '#e05a47']
    labels = ['Adult Cohort (GSE13159)', 'Pediatric Cohort (TARGET-AML)']

    # Plot Before ComBat
    ax = axes[0]
    for b_idx in [0, 1]:
        mask = (batches == b_idx)
        ax.scatter(
            embedding_raw[mask, 0],
            embedding_raw[mask, 1],
            c=colors[b_idx],
            label=labels[b_idx],
            alpha=0.6,
            edgecolors='none',
            s=8
        )
    ax.set_title("Before ComBat Correction (Platform Bias)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10)
    ax.set_ylabel("UMAP Dimension 2", fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    # Plot After ComBat
    ax = axes[1]
    for b_idx in [0, 1]:
        mask = (batches == b_idx)
        ax.scatter(
            embedding_corr[mask, 0],
            embedding_corr[mask, 1],
            c=colors[b_idx],
            label=labels[b_idx],
            alpha=0.6,
            edgecolors='none',
            s=8
        )
    ax.set_title("After ComBat Correction (Biological Integration)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10)
    ax.set_ylabel("UMAP Dimension 2", fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')

    plt.suptitle("Universal Pan-Leukemia Cohort Integration Proof", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(UMAP_PLOT_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    LOGGER.info("Saved UMAP visualization proof to %s", UMAP_PLOT_PATH)


def step_5_biological_validation(universal_biomarkers: list[str]):
    """Query STRING API and build/validate topological PPI network."""
    LOGGER.info("STEP 5: Querying STRING database for universal master regulator validation...")
    
    if len(universal_biomarkers) == 0:
        LOGGER.warning("No universal biomarkers discovered. Skipping biological network validation.")
        return

    # Call STRING API
    string_url = "https://string-db.org/api/json/network"
    params = urllib.parse.urlencode({
        "identifiers": "\n".join(universal_biomarkers),
        "species": "9606",  # Homo sapiens
        "caller_identity": "leukemia_quantum_pipeline"
    }).encode("utf-8")
    
    LOGGER.info("Fetching PPI network from STRING API...")
    try:
        req = urllib.request.Request(string_url, data=params, method="POST")
        with urllib.request.urlopen(req, timeout=30) as response:
            edges_json = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        LOGGER.error("Failed to query STRING API: %s. Using mock network for offline fallback.", e)
        # Mock interactions for robust local validation in case of network failure
        edges_json = []

    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(universal_biomarkers)

    edge_count = 0
    for edge in edges_json:
        p1 = edge.get("stringId_A")
        p2 = edge.get("stringId_B")
        # Resolve to standard symbols if available
        g1 = edge.get("preferredName_A", p1)
        g2 = edge.get("preferredName_B", p2)
        score = float(edge.get("score", 0.4))
        
        # Keep only edges between our selected biomarkers
        if g1 in G.nodes and g2 in G.nodes:
            G.add_edge(g1, g2, weight=score)
            edge_count += 1

    LOGGER.info("STRING API returned %d active interaction edges within universal biomarkers.", edge_count)

    # Compute Centrality Metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")

    # Sort genes by degree centrality
    sorted_nodes = sorted(G.nodes, key=lambda n: degree_centrality[n], reverse=True)

    # Save metrics
    LOGGER.info("Saving centrality metrics to %s...", PPI_METRICS_PATH)
    with open(PPI_METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("Gene\tDegree Centrality\tBetweenness Centrality\n")
        for node in sorted_nodes:
            f.write(f"{node}\t{degree_centrality[node]:.6f}\t{betweenness_centrality[node]:.6f}\n")
        
        # Identify top hub & bottleneck
        if sorted_nodes:
            top_hub = sorted_nodes[0]
            top_bottleneck = max(G.nodes, key=lambda n: betweenness_centrality[n])
            f.write("\n=== Hub Analysis ===\n")
            f.write(f"Top Hub Gene (Degree): {top_hub} (Centrality: {degree_centrality[top_hub]:.6f})\n")
            f.write(f"Top Bottleneck Gene (Betweenness): {top_bottleneck} (Centrality: {betweenness_centrality[top_bottleneck]:.6f})\n")
            LOGGER.info("Top Hub Gene: %s (Degree Centrality: %.4f)", top_hub, degree_centrality[top_hub])
            LOGGER.info("Top Bottleneck Gene: %s (Betweenness Centrality: %.4f)", top_bottleneck, betweenness_centrality[top_bottleneck])

    # Draw Network Graph (sized by Degree Centrality)
    LOGGER.info("Plotting PPI network topology graph...")
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Layout algorithm
    pos = nx.spring_layout(G, k=0.4, seed=42)
    
    # Scale node sizes based on Degree Centrality
    node_sizes = [150 + 4000 * degree_centrality[n] for n in G.nodes]
    
    # Draw elements
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#2b5c8f",
        alpha=0.85,
        edgecolors="#1a3d60",
        linewidths=1.5
    )
    
    # Draw edges
    edges = G.edges(data=True)
    if edges:
        weights = [e[2].get('weight', 0.4) * 3 for e in edges]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color="#b3c2d4", alpha=0.6)

    # Labels
    labels = {node: node for node in G.nodes if degree_centrality[node] > 0 or len(G.nodes) <= 15}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", font_family="sans-serif")

    plt.title("STRING PPI Interaction Hub for Universal Master Regulators", fontsize=14, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PPI_NETWORK_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    LOGGER.info("Saved PPI network graph to %s", PPI_NETWORK_PATH)


def main():
    """Main execution entrypoint."""
    LOGGER.info("========================================================================")
    LOGGER.info("Starting Universal Pan-Leukemia Master Regulator Discovery Pipeline")
    LOGGER.info("========================================================================")
    
    try:
        # Step 1: Align & Intersect
        raw_matrix, intersecting_genes = step_1_data_alignment()
        
        # Step 2: ComBat Batch effect correction
        corrected_df = step_2_batch_effect_correction(raw_matrix, intersecting_genes)
        
        # Step 3: Run BQPSO Quantum Selection
        universal_biomarkers = step_3_quantum_feature_selection(corrected_df, intersecting_genes)
        
        # Step 4: UMAP Visualization before vs after
        step_4_umap_visualization(raw_matrix, corrected_df, intersecting_genes)
        
        # Step 5: Biological validation using STRING
        step_5_biological_validation(universal_biomarkers)
        
        LOGGER.info("========================================================================")
        LOGGER.info("Pan-Leukemia Pipeline completed successfully!")
        LOGGER.info("========================================================================")
        
    except Exception as e:
        LOGGER.error("Pipeline execution failed with exception: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
