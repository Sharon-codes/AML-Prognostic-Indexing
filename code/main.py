"""Master script for the GSE13159 BQPSO + 1D-CNN leukaemia pipeline."""

from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(r"D:\Leukemia_Quantum_Pipeline")
CODE_DIR = PROJECT_ROOT / "code"


def _load_module(filename: str, module_name: str) -> ModuleType:
    """Load numbered pipeline modules without invalid Python identifier imports."""
    path = CODE_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _configure_logging() -> None:
    log_path = PROJECT_ROOT / "logs_and_output" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def main() -> None:
    """Execute retrieval/preprocessing, BQPSO selection, then CNN training."""
    for directory in (PROJECT_ROOT / "data", CODE_DIR, PROJECT_ROOT / "logs_and_output", PROJECT_ROOT / "images"):
        directory.mkdir(parents=True, exist_ok=True)
    _configure_logging()
    logger = logging.getLogger("pipeline")
    started = datetime.now()
    logger.info("=" * 72)
    logger.info("Starting early-stage leukaemia biomarker discovery pipeline")
    logger.info("Project root: %s", PROJECT_ROOT)
    try:
        data_loader = _load_module("01_data_loader.py", "data_loader")
        selector_module = _load_module("02_bqpso_selector.py", "bqpso_selector")
        cnn_module = _load_module("03_1d_cnn_classifier.py", "cnn_classifier")

        logger.info("STEP 1/3: GEO download, clinical label extraction, and preprocessing")
        X, y = data_loader.load_and_preprocess(PROJECT_ROOT)
        logger.info("STEP 2/3: BQPSO biomarker selection (50 particles x 100 epochs, 5-fold SVM CV)")
        selector = selector_module.BinaryQuantumPSO(n_particles=50, n_epochs=100, min_features=15, max_features=30)
        X_reduced = selector.fit_select(X, y, PROJECT_ROOT / "logs_and_output")
        logger.info("STEP 3/3: 1D-CNN training and held-out evaluation using %d selected genes", X_reduced.shape[1])
        metrics = cnn_module.train_and_evaluate(X_reduced, y, PROJECT_ROOT)
        logger.info("Pipeline completed successfully in %s", datetime.now() - started)
        logger.info("Final held-out metrics: %s", metrics)
    except Exception:
        logger.exception("Pipeline failed. See this terminal output and logs_and_output/pipeline.log for details.")
        raise


if __name__ == "__main__":
    main()
