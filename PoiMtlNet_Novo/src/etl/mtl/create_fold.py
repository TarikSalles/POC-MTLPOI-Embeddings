import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import gc

from src.configs.globals import DEVICE, CATEGORIES_MAP
from src.configs.model import MTLModelConfig, InputsConfig
from src.configs.paths import EmbeddingEngine, IoPaths

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InputData:
    """Class for organizing model input data"""
    dataloader: DataLoader
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class SuperInputData:
    """Class for organizing all splits of data"""
    train: InputData
    val: InputData


class POIDataset(Dataset):
    """Dataset for POI data with memory pinning for faster data transfer"""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        # Ensure tensors are on CPU for efficient loading
        self.features = features.cpu()
        self.targets = targets.cpu()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x': self.features[idx],
            'y': self.targets[idx]
        }


# Worker initialization function to set environment variables and improve GPU utilization
def worker_init_fn(worker_id):
    """Initialize each worker with optimized settings"""
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    # Set affinity if on Linux to bind workers to specific CPU cores
    try:
        import os
        import psutil
        process = psutil.Process()
        # Get available cores and distribute evenly
        cores = list(range(psutil.cpu_count(logical=False)))
        if cores:
            core_id = worker_id % len(cores)
            process.cpu_affinity([cores[core_id]])
    except (ImportError, AttributeError):
        pass


def map_categories(y: Union[pd.DataFrame, pd.Series]) -> Union[int, np.ndarray, pd.Series]:
    """Map categories using the CATEGORIES_MAP"""
    inv_categories = {v: k for k, v in CATEGORIES_MAP.items()}

    try:
        if isinstance(y, pd.DataFrame):
            # Vectorized operation is faster than list comprehension
            return y.applymap(lambda x: inv_categories.get(x, 0)).values
        elif isinstance(y, pd.Series):
            return y.map(inv_categories)
        elif isinstance(y, str):
            return inv_categories.get(y, 0)
        else:
            raise ValueError("Unsupported type for category mapping")
    except KeyError as e:
        logger.error(f"Unknown category encountered: {e}")
        raise


def convert_to_tensors(x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series],
                       task_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert pandas objects to PyTorch tensors with appropriate shapes"""
    # Convert to contiguous array first for better memory layout
    x_values = np.ascontiguousarray(x.values, dtype=np.float32)
    y_values = np.ascontiguousarray(y.values, dtype=np.int64)

    # Create tensors
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.long)

    # Reshape based on task type
    if task_type == 'next':
        x_tensor = x_tensor.view(-1, 9, InputsConfig.EMBEDDING_DIM)
    else:  # category
        x_tensor = x_tensor.view(-1, 1, InputsConfig.EMBEDDING_DIM)

    return x_tensor, y_tensor


def create_dataloader(
        x: torch.Tensor,
        y: torch.Tensor,
        indices: Optional[np.ndarray] = None,
        batch_size: int = 64,
        shuffle: bool = True,
        prefetch_factor: int = 5
) -> DataLoader:
    """Create DataLoader with optimal settings for performance"""
    if indices is not None:
        x_subset = x[indices]
        y_subset = y[indices]
    else:
        x_subset = x
        y_subset = y

    # Optimize worker count based on system resources
    num_workers = min(8, os.cpu_count() or 1)

    # Create dataset
    dataset = POIDataset(x_subset, y_subset)

    # Create dataloader with performance optimizations
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory=True,
        pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
    )

def create_folds(
        path_next_input: str,
        path_category_input: str,
        k_splits: int = 5,
        batch_size: int = MTLModelConfig.BATCH_SIZE,
        target_column_start: int = InputsConfig.EMBEDDING_DIM*InputsConfig.SLIDE_WINDOW,
        random_state: int = 42,
        save_folder: Optional[str] = None
) -> tuple[dict[int, dict[str, SuperInputData]], str]:
    """Process POI data and create k-fold cross-validation splits

    Args:
        path_next_input: Path to the next POI input CSV
        path_category_input: Path to the category POI input CSV
        k_splits: Number of folds for cross-validation
        batch_size: Batch size for DataLoaders
        target_column_start: Index where target columns start in next POI data
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with fold indices mapping to task data
    """
    # Set seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Process Next POI data
    logger.info("Loading Next POI data...")
    df_next = pd.read_csv(path_next_input)
    df_next = df_next.dropna(subset=['next_category'])
    logger.info(f"Next POI data shape: {df_next.shape}")

    # Get unique user IDs
    df_next['userid'] = df_next['userid'].astype(int)
    userids = df_next['userid'].unique()
    logger.info(f"Number of unique users: {len(userids)}")

    # Extract target columns
    target_cols = df_next.columns[target_column_start:-1]

    # Create features and targets for next POI
    x_next = df_next.drop(target_cols, axis=1).drop('userid', axis=1)
    x_next = df_next.drop(columns=['next_category', 'userid'], errors='ignore')

    # se ainda houver algo não-numérico, filtra por tipo numérico
    x_next = x_next.select_dtypes(include=[np.number]).astype(np.float32)

    y_next = df_next['next_category']
    y_next = map_categories(y_next)
    logger.info(f"Next POI features shape: {x_next.shape}, targets shape: {y_next.shape}")

    # Process Category POI data
    logger.info("Loading Category POI data...")
    df_category = pd.read_csv(path_category_input)
    logger.info(f"Category POI data shape: {df_category.shape}")

    # Get unique place IDs
    places_ids = df_category['placeid'].unique().astype(int)
    logger.info(f"Number of unique places: {len(places_ids)}")

    # Set place ID as index and create features/targets
    df_category = df_category.set_index('placeid')
    x_category = df_category.drop('category', axis=1)
    y_category = df_category['category']
    y_category = map_categories(y_category)
    logger.info(f"Category POI features shape: {x_category.shape}, targets shape: {y_category.shape}")


    # Setup StratifiedKFold for users
    next_skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=random_state)

    # Setup StratifiedKFold for places
    place_skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=random_state)

    fold_results: Dict[int, Dict[str, SuperInputData]] = {}

    # Process each fold using stratified splits
    fold_idx = 0

    # Convert to tensors
    x_next_tensor, y_next_tensor = convert_to_tensors(x_next, y_next, 'next')
    x_category_tensor, y_category_tensor = convert_to_tensors(x_category, y_category, 'category')

    for (train_next_idx, test_next_idx), (train_place_idx, test_place_idx) in zip(
            next_skf.split(x_next, y_next),
            place_skf.split(x_category, y_category)):

        logger.info(f"Processing fold {fold_idx + 1}/{k_splits}")

        # Create dataloaders
        next_train_loader = create_dataloader(
            x_next_tensor, y_next_tensor, train_next_idx, batch_size)
        next_val_loader = create_dataloader(
            x_next_tensor, y_next_tensor, test_next_idx, batch_size)

        category_train_loader = create_dataloader(
            x_category_tensor, y_category_tensor, train_place_idx, batch_size)
        category_val_loader = create_dataloader(
            x_category_tensor, y_category_tensor, test_place_idx, batch_size)

        # Create input data objects
        next_data = SuperInputData(
            train=InputData(
                dataloader=next_train_loader,
                x=x_next_tensor[train_next_idx],
                y=y_next_tensor[train_next_idx]
            ),
            val=InputData(
                dataloader=next_val_loader,
                x=x_next_tensor[test_next_idx],
                y=y_next_tensor[test_next_idx]
            )
        )

        category_data = SuperInputData(
            train=InputData(
                dataloader=category_train_loader,
                x=x_category_tensor[train_place_idx],
                y=y_category_tensor[train_place_idx]
            ),
            val=InputData(
                dataloader=category_val_loader,
                x=x_category_tensor[test_place_idx],
                y=y_category_tensor[test_place_idx]
            )
        )

        # Store results for this fold
        fold_results[fold_idx] = {
            'next': next_data,
            'category': category_data
        }

        fold_idx += 1

        # Manually run garbage collection after each fold to free memory
        gc.collect()

    folds_save_path = time.strftime("%Y%m%d_%H%M") + "_folds.pkl"
    if save_folder:
        with open(os.path.join(save_folder, folds_save_path), 'wb') as f:
            joblib.dump(fold_results, f)
        logger.info(f"Folds saved to {save_folder}")

    return fold_results, folds_save_path

def restore_folds(path: str) -> Dict[int, Dict[str, SuperInputData]]:
    """Restore folds from a saved file"""
    with open(path, 'rb') as f:
        fold_results = joblib.load(f)
    logger.info(f"Folds restored from {path}")
    return fold_results


# Example usage
def main():
    """Example usage of the create_folds function"""
    path_nextpoi_input = "path/to/nextpoi_input.csv"
    path_categorypoi_input = "path/to/categorypoi_input.csv"

    fold_results = create_folds(
        "Florida",
        EmbeddingEngine.DGI,
        k_splits=5,
    )

    logger.info(f"Created {len(fold_results)} folds successfully")


if __name__ == "__main__":
    main()
