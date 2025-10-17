from typing import Union

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
                best_state: dict = None
             ) -> Union[str, dict]:
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truths.append(y_batch.numpy())

    report = classification_report(
        np.concatenate(truths),
        np.concatenate(preds),
        output_dict=True,
        zero_division=0,
    )
    return report