import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from common.poi_dataset import POIDataset
from configs.globals import CATEGORIES_MAP
from configs.model import InputsConfig
from configs.category_config import CfgCategoryTraining


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    inv_map = {v: k for k, v in CATEGORIES_MAP.items()}
    df['label'] = df['category'].map(inv_map)
    df.drop(columns=['category'], inplace=True)
    feature_cols = df.columns[:InputsConfig.EMBEDDING_DIM]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y


def create_folds(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        seed: int,
        batch_size: int = CfgCategoryTraining.BATCH_SIZE,
) -> list[tuple[DataLoader, DataLoader]]:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = POIDataset(X_train, y_train)
        val_ds = POIDataset(X_val, y_val)

        num_workers = min(8, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        folds.append((train_loader, val_loader))
    return folds
