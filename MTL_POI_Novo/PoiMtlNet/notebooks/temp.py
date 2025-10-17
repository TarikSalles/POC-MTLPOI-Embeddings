import os
from collections import defaultdict

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import json
from configs.globals import DEVICE, CATEGORIES_MAP
from configs.model import MTLModelConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from multiprocessing import freeze_support

# --- wrap into a Dataset ---
class POIDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class CategoryClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int = 107,
            hidden_dim: int = 128,
            num_classes: int = 7,
            dropout: float = 0.3,
    ):
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_dim)

        # First-order features
        self.first_order = nn.Linear(input_dim, hidden_dim)

        # Second-order interactions
        self.interaction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # Combine and classify
        self.combine = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.bn_input(x)

        # First-order path
        first = self.first_order(x)

        # Second-order path
        second = self.interaction(x)

        # Combine
        combined = torch.cat([first, second], dim=1)
        features = self.combine(combined)

        return self.out(features)


def main():
    # --- your pre-processing steps ---
    df = pd.read_csv('/Users/vitor/Desktop/mestrado/ingred/data/output/florida_new/pre-processing/category-input.csv')
    categories_inv = {v: k for k, v in CATEGORIES_MAP.items()}
    df['y'] = df['category'].map(categories_inv)
    df.drop(columns=['placeid', 'category'], inplace=True)
    feature_cols = df.columns[0:107]
    df['x'] = df[feature_cols].values.tolist()

    # prepare raw lists
    X = np.array(df['x'].tolist())
    y = np.array(df['y'].tolist())

    # --- create 5‐fold splits ---
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        # slice out this fold's train/val
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # build Datasets
        train_ds = POIDataset(X_train, y_train)
        val_ds = POIDataset(X_val, y_val)

        # and DataLoaders
        train_loader = DataLoader(train_ds, batch_size=2**9, shuffle=True,
                                  num_workers=min(8, os.cpu_count() or 1),
                                  prefetch_factor=3,
                                  pin_memory=False,
                                  pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
                                  persistent_workers=True,

                                  )
        val_loader = DataLoader(val_ds, batch_size=2**9, shuffle=False,
                                num_workers=min(8, os.cpu_count() or 1),
                                prefetch_factor=3,
                                persistent_workers=True,
                                pin_memory=False,
                                pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
                                )

        folds.append((train_loader, val_loader))

        print(f"Fold {fold}: train={len(train_ds)}  val={len(val_ds)}")

    metrics_history = {}
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        fold_metrics = defaultdict(list)

        model = CategoryClassifier()
        model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=150,
            steps_per_epoch=len(train_loader)
        )

        best_val_acc = 0.0
        epoch_progress = tqdm(range(150), desc=f"Fold {fold_idx}")

        for epoch in epoch_progress:
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)
                x_batch = x_batch.contiguous()
                y_batch = y_batch.contiguous()

                optimizer.zero_grad()
                out_a = model(x_batch)
                loss = criterion(out_a, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                _, predicted = torch.max(out_a, 1)
                correct = (predicted == y_batch).sum().item()
                total = y_batch.size(0)

                train_loss += loss.item() * total
                train_acc += correct
                train_total += total

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_acc / train_total

            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    x_batch, y_batch = batch  # Unpacking properly
                    x_batch = x_batch.to(DEVICE, non_blocking=True)
                    y_batch = y_batch.to(DEVICE, non_blocking=True)

                    out_a = model(x_batch)
                    loss = criterion(out_a, y_batch)

                    _, predicted = torch.max(out_a, 1)
                    correct = (predicted == y_batch).sum().item()
                    total = y_batch.size(0)

                    val_loss += loss.item() * total
                    val_acc += correct
                    val_total += total

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_acc / val_total

            # Track best validation accuracy
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc

            fold_metrics['train_loss'].append(epoch_train_loss)
            fold_metrics['train_acc'].append(epoch_train_acc)
            fold_metrics['val_loss'].append(epoch_val_loss)
            fold_metrics['val_acc'].append(epoch_val_acc)

            epoch_progress.set_postfix({
                'tr_loss': f"{epoch_train_loss:.4f}",
                'tr_acc': f"{epoch_train_acc:.4f}",
                'vl_loss': f"{epoch_val_loss:.4f}",
                'vl_acc': f"{epoch_val_acc:.4f}"
            })

        metrics_history[fold_idx] = dict(fold_metrics)

        model.eval()

        with torch.no_grad():
            predicted = []
            ground_truth = []
            for batch in val_loader:
                x_batch, y_batch = batch  # Unpacking properly
                x_batch = x_batch.to(DEVICE, non_blocking=True)
                y_batch = y_batch.to(DEVICE, non_blocking=True)

                out_a = model(x_batch)

                _, pred = torch.max(out_a, 1)
                predicted.append(pred.cpu().numpy())
                ground_truth.append(y_batch.cpu().numpy())

            report = classification_report(
                np.concatenate(ground_truth),
                np.concatenate(predicted),
                output_dict=True,
                zero_division=0
            )
            print(json.dumps(report, indent=4))

        print(f"Fold {fold_idx} - Best Val Acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()