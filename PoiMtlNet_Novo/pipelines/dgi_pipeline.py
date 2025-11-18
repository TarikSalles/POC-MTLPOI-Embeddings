"""
DGI Pipeline - End-to-end pipeline for processing states with DGI embeddings.

This pipeline executes three sequential stages for each state:
1. Preprocessing: Create graph structure and intermediate data
2. Embedding: Train DGI model and generate embeddings
3. Input Generation: Create training inputs for next POI and category prediction

Usage:
    python pipelines/dgi_pipeline.py
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
from argparse import Namespace

from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.dgi.dgi import create_embedding
from embeddings.dgi.preprocess import preprocess_dgi
from etl.create_input import create_input

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Edit this list to add/remove states
# ============================================================================

STATES_TO_PROCESS = [
    {
        'name': 'Alabama',
        'shapefile': Resources.TL_AL,
        'cta_file': None,
    },
    {
        'name': 'Arizona',
        'shapefile': Resources.TL_AZ,
        'cta_file': None,
    },
    {
        'name': 'Georgia',
        'shapefile': Resources.TL_GA,
        'cta_file': None,
    },
    {
        'name': 'Florida',
        'shapefile': Resources.TL_FL,
        'cta_file': None,
    },
    {
        'name': 'California',
        'shapefile': Resources.TL_CA,
        'cta_file': None,
    },
    {
        'name': 'Texas',
        'shapefile': Resources.TL_TX,
        'cta_file': None,
    },
]

# DGI Training Parameters
DGI_CONFIG = {
    'dim': InputsConfig.EMBEDDING_DIM,  # Embedding dimension
    'lr': 0.01,  # Learning rate
    'gamma': 1.0,  # Learning rate decay
    'epochs': 70,  # Number of training epochs
    'max_norm': 0.9,  # Gradient clipping max norm
    'device': 'auto',  # 'auto', 'mps', 'cuda', or 'cpu'
}


# ============================================================================
# Pipeline Implementation - No need to edit below this line
# ============================================================================


def get_device(device_preference: str = 'auto') -> str:
    """Auto-detect or validate device."""
    if device_preference == 'auto':
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    return device_preference


def preprocess_stage(state_name: str, shapefile: Path, cta_file: Optional[Path] = None) -> bool:
    """
    Stage 1: Preprocess state data.

    Args:
        state_name: Name of the state
        shapefile: Path to shapefile
        cta_file: Optional path to CTA file

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[STAGE 1/3] Preprocessing: {state_name}")
    logger.info(f"  Shapefile: {shapefile}")

    try:
        preprocess_dgi(
            city=state_name,
            city_shapefile=str(shapefile),
            cta_file=str(cta_file) if cta_file else None
        )
        logger.info(f"  ✓ Preprocessing completed for {state_name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Preprocessing failed for {state_name}: {e}", exc_info=True)
        return False


def embedding_stage(state_name: str, config: dict) -> bool:
    """
    Stage 2: Create DGI embeddings.

    Args:
        state_name: Name of the state
        config: DGI configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[STAGE 2/3] Creating embeddings: {state_name}")
    logger.info(f"  Device: {config['device']}, Dim: {config['dim']}, Epochs: {config['epochs']}")

    try:
        args = Namespace(
            dim=config['dim'],
            lr=config['lr'],
            gamma=config['gamma'],
            epoch=config['epochs'],
            device=config['device'],
            max_norm=config['max_norm']
        )

        create_embedding(city=state_name, args=args)
        logger.info(f"  ✓ Embeddings created for {state_name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Embedding creation failed for {state_name}: {e}", exc_info=True)
        return False


def input_generation_stage(state_name: str) -> bool:
    """
    Stage 3: Generate training inputs.

    Args:
        state_name: Name of the state

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"[STAGE 3/3] Generating inputs: {state_name}")

    try:
        create_input(state=state_name, embedd_eng=EmbeddingEngine.DGI)
        logger.info(f"  ✓ Inputs generated for {state_name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Input generation failed for {state_name}: {e}", exc_info=True)
        return False


def process_single_state(state_config: dict, dgi_config: dict) -> bool:
    """
    Process a single state through all three stages.

    Args:
        state_config: State configuration dictionary
        dgi_config: DGI training configuration

    Returns:
        True if all stages succeeded, False otherwise
    """
    state_name = state_config['name']
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Processing state: {state_name}")
    logger.info(f"{'=' * 80}\n")

    start_time = datetime.now()

    # Stage 1: Preprocessing
    if not preprocess_stage(
            state_name=state_name,
            shapefile=state_config['shapefile'],
            cta_file=state_config.get('cta_file')
    ):
        logger.error(f"Pipeline aborted for {state_name} at preprocessing stage")
        return False

    # Stage 2: Embedding
    if not embedding_stage(state_name, dgi_config):
        logger.error(f"Pipeline aborted for {state_name} at embedding stage")
        return False

    # Stage 3: Input Generation
    if not input_generation_stage(state_name):
        logger.error(f"Pipeline aborted for {state_name} at input generation stage")
        return False

    # Success!
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"\n{'=' * 80}")
    logger.info(f"✓ Successfully completed all stages for {state_name}")
    logger.info(f"Total time: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    logger.info(f"{'=' * 80}\n")

    return True


def run_pipeline(states: list, dgi_config: dict) -> dict:
    """
    Run the pipeline for multiple states sequentially.

    Args:
        states: List of state configuration dictionaries
        dgi_config: DGI training configuration

    Returns:
        Dictionary with success/failure status for each state
    """
    logger.info(f"\n{'#' * 80}")
    logger.info(f"DGI PIPELINE - Processing {len(states)} state(s)")
    logger.info(f"{'#' * 80}\n")

    # Auto-detect device if needed
    device = get_device(dgi_config['device'])
    dgi_config = {**dgi_config, 'device': device}
    logger.info(f"Using device: {device}")
    logger.info(f"DGI Config: dim={dgi_config['dim']}, epochs={dgi_config['epochs']}, lr={dgi_config['lr']}\n")

    pipeline_start = datetime.now()
    results = {}

    for i, state_config in enumerate(states, 1):
        state_name = state_config['name']
        logger.info(f"\n[{i}/{len(states)}] Starting: {state_name}")
        success = process_single_state(state_config, dgi_config)
        results[state_name] = success

    # Summary
    pipeline_end = datetime.now()
    total_duration = (pipeline_end - pipeline_start).total_seconds()

    logger.info(f"\n{'#' * 80}")
    logger.info(f"PIPELINE SUMMARY")
    logger.info(f"{'#' * 80}")
    logger.info(f"Total states processed: {len(states)}")
    logger.info(f"Successful: {sum(results.values())}")
    logger.info(f"Failed: {len(states) - sum(results.values())}")
    logger.info(f"Total time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
    logger.info(f"\nResults:")
    for state_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {state_name}: {status}")
    logger.info(f"{'#' * 80}\n")

    return results


if __name__ == '__main__':
    # Validate configuration
    if not STATES_TO_PROCESS:
        logger.error("No states configured! Please add states to STATES_TO_PROCESS list.")
        exit(1)

    # Run the pipeline
    results = run_pipeline(STATES_TO_PROCESS, DGI_CONFIG)

    # Exit with error code if any state failed
    if not all(results.values()):
        exit(1)
