import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from common.poi_dataset import POIDataset
from configs.globals import CATEGORIES_MAP
from configs.model import InputsConfig
from model.next.head.configs.next_config import CfgNextTraining, CfgNextModel


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    inv_map = {v: k for k, v in CATEGORIES_MAP.items()}
    df['label'] = df['next_category'].map(inv_map)
    df.drop(columns=['userid', 'next_category'], inplace=True)
    df.dropna(subset=['label'], inplace=True) # TODO: THIS IS DUE A ERROR IN THE CREATION OF THE EMBEDD DGI FIX IT
    feature_cols = df.columns[0 : CfgNextModel.INPUT_DIM * InputsConfig.SLIDE_WINDOW]
    X = df[feature_cols].values
    y = df['label'].astype(int).values
    return X, y


def create_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    batch_size: int = CfgNextTraining.BATCH_SIZE,
) -> list[tuple[DataLoader, DataLoader]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # reshape for your 9×embedding inputs
        X_tr = X_tr.reshape(-1, InputsConfig.SLIDE_WINDOW, CfgNextModel.INPUT_DIM)
        X_val = X_val.reshape(-1, InputsConfig.SLIDE_WINDOW, CfgNextModel.INPUT_DIM)

        train_ds = POIDataset(X_tr, y_tr)
        val_ds   = POIDataset(X_val, y_val)

        # ——— Balanced class weights ———
        classes = np.unique(y_tr)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_tr
        )
        # map back to each sample
        weight_per_class = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_per_class[label] for label in y_tr], dtype=np.float32)

        # Convert to tensor
        sample_weights = torch.from_numpy(sample_weights)

        # Seed the sampler for reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator
        )

        num_workers = min(8, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            # sampler=sampler,       # no shuffle when using sampler
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