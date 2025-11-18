import logging
import random
import time
from typing import List, Tuple, Optional

import joblib

from configs.paths import IoPaths, RESULTS_ROOT, EmbeddingEngine
from configs.category_config import CfgCategoryTraining, CfgCategoryHyperparams
from etl.category.fold import load_data, create_folds
from train.category.cross_validation import run_cv
from common.ml_history.metrics import MLHistory
from common.ml_history.utils.dataset import DatasetHistory


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_category_model(
    state: str,
    embedd_engine: EmbeddingEngine,
    epochs: int = CfgCategoryTraining.EPOCHS,
    batch_size: int = CfgCategoryTraining.BATCH_SIZE,
    learning_rate: float = CfgCategoryHyperparams.LR,
    save_folds: bool = False,
    folds_chkpt: Optional[str] = None
) -> dict:
    """
    Train Category prediction model for a specific state and embedding engine.

    Args:
        state: State name (e.g., "florida", "california")
        embedd_engine: Embedding engine to use (DGI, HGI, HMRM)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_folds: Whether to save folds to disk
        folds_chkpt: Path to checkpoint file for folds

    Returns:
        Dictionary with training results
    """
    logger.info(f"{'='*80}")
    logger.info(f"Starting Category training: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}")

    # Get paths
    data_input = IoPaths.get_category(state, embedd_engine)
    output_dir = IoPaths.get_results_dir(state, embedd_engine)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Creating folds
    logger.info("Creating folds...")
    folds = None
    if folds_chkpt is not None:
        folds_chkpt_path = output_dir / folds_chkpt
        logger.info(f"Loading folds from checkpoint: {folds_chkpt_path}")
        with open(folds_chkpt_path, 'rb') as f:
            folds = joblib.load(f)
    else:
        X, y = load_data(str(data_input))
        folds = create_folds(
            X,
            y,
            n_splits=CfgCategoryTraining.K_FOLDS,
            batch_size=batch_size,
            seed=random.randint(1, 10000),
        )
        if save_folds:
            folds_pth = output_dir / 'folds'
            folds_pth.mkdir(parents=True, exist_ok=True)
            fold_file = folds_pth / (time.strftime("%Y%m%d_%H%M") + "_folds.pkl")
            logger.info(f"Saving folds to: {fold_file}")
            with open(fold_file, 'wb') as f:
                joblib.dump(folds, f)

    # Creating history
    history: MLHistory = MLHistory(
        model_name="Category",
        model_type="Single-Task",
        tasks='category',
        num_folds=CfgCategoryTraining.K_FOLDS,
        datasets={
            DatasetHistory(
                raw_data=str(data_input),
                folds_signature=folds_chkpt or None,
                description="POI Category Classification",
            )
        }
    )

    # Running cross-validation
    logger.info(f"Starting cross-validation training...")
    with history.context() as history:
        results = run_cv(history, folds)

    history.display.end_training()

    # Save results
    logger.info(f"Saving results to: {output_dir}")
    history.storage.save(path=str(output_dir))

    logger.info(f"Completed Category training: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    # Define configurations to train: [(state, embedding_engine), ...]
    TRAINING_CONFIGS: List[Tuple[str, EmbeddingEngine]] = [
        # ("florida", EmbeddingEngine.DGI),
        # ("florida", EmbeddingEngine.HGI),
        # ("florida", EmbeddingEngine.HMRM),
        # ("alabama", EmbeddingEngine.DGI),
        # ("arizona", EmbeddingEngine.DGI),
        # ("georgia", EmbeddingEngine.DGI),
        ("florida", EmbeddingEngine.DGI),
        # ("california", EmbeddingEngine.DGI),
        # ("texas", EmbeddingEngine.DGI),
    ]

    logger.info(f"Starting Category training pipeline")
    logger.info(f"Total configurations to train: {len(TRAINING_CONFIGS)}")
    logger.info(f"Configurations: {[(s, e.value) for s, e in TRAINING_CONFIGS]}\n")

    # Train each configuration sequentially
    all_results = {}
    for idx, (state, embedd_engine) in enumerate(TRAINING_CONFIGS, 1):
        logger.info(f"Training configuration {idx}/{len(TRAINING_CONFIGS)}")

        try:
            results = train_category_model(
                state=state,
                embedd_engine=embedd_engine,
                epochs=CfgCategoryTraining.EPOCHS,
                batch_size=CfgCategoryTraining.BATCH_SIZE,
                learning_rate=CfgCategoryHyperparams.LR,
                save_folds=False,
                folds_chkpt=None
            )
            all_results[f"{state}_{embedd_engine.value}"] = results
        except Exception as e:
            logger.error(f"Failed to train {state} with {embedd_engine.value}: {str(e)}", exc_info=True)
            continue

    logger.info(f"\n{'='*80}")
    logger.info(f"All training completed!")
    logger.info(f"Successfully trained: {len(all_results)}/{len(TRAINING_CONFIGS)} configurations")
    logger.info(f"{'='*80}")
