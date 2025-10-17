import json

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from configs.globals import DEVICE, CATEGORIES_MAP
from configs.model import InputsConfig

from model.next.head.configs.next_config import CfgNextModel, CfgNextHyperparams, CfgNextTraining
from model.next.head.engine.evaluation import evaluate
from model.next.head.engine.trainer import train
from model.next.head.modeling.next_head import NextHeadSingle
from utils.calc_flops.calculate_model_flops import calculate_model_flops
from utils.ml_history.metrics import MLHistory, FlopsMetrics
from utils.ml_history.parms.neural import NeuralParams
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, 
                 gamma: float = 2.0, 
                 alpha: torch.Tensor = None, 
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        gamma: focusing parameter >= 0
        alpha: tensor of shape (num_classes,) giving per-class weighting
        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (batch, C) raw outputs
        targets: (batch,)  long tensor of class indices
        """
        # 1) get log-probs and probs
        log_probs = F.log_softmax(logits, dim=1)        # (B, C)
        probs     = torch.exp(log_probs)                # (B, C)

        # 2) gather the log-prob and prob for the true class
        idx = targets.view(-1, 1)
        log_pt = log_probs.gather(1, idx).view(-1)      # (B,)
        pt     = probs.gather(1, idx).view(-1)          # (B,)

        # 3) if alpha (class-balancing) is given, apply it
        if self.alpha is not None:
            # make sure alpha is on the right device & dtype
            alpha = self.alpha.to(logits.device).float()
            at = alpha.gather(0, targets)              # (B,)
            log_pt = log_pt * at

        # 4) focal loss formula: FL = - (1 - pt)^γ * log_pt
        focal_term = (1 - pt).pow(self.gamma)
        loss = -focal_term * log_pt

        # 5) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        history.display.start_fold()
        model = NextHeadSingle(
            embed_dim=CfgNextModel.INPUT_DIM,
            num_classes=CfgNextModel.NUM_CLASSES,
            num_heads=CfgNextModel.NUM_HEADS,
            seq_length=CfgNextModel.MAX_SEQ_LENGTH,
            num_layers=CfgNextModel.NUM_LAYERS,
            dropout=CfgNextModel.DROPOUT,
        ).to(DEVICE)

        y_all = np.concatenate([y.numpy() for _, y in train_loader])
        cls = np.arange(CfgNextModel.NUM_CLASSES)
        weights = compute_class_weight('balanced', classes=cls, y=y_all)
        alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            weight=alpha,
        )

        # criterion = FocalLoss(gamma=2.0, alpha=alpha, reduction='mean')

        optimizer = AdamW(
            model.parameters(),
            lr=CfgNextHyperparams.LR,
            weight_decay=CfgNextHyperparams.WEIGHT_DECAY,

        )

        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=CfgNextHyperparams.MAX_LR,
            epochs=CfgNextTraining.EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        # optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=CfgNextTraining.EPOCHS)

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=CfgNextTraining.BATCH_SIZE,
                num_epochs=CfgNextTraining.EPOCHS,
                learning_rate=CfgNextHyperparams.LR,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'next': criterion.__class__.__name__},
                criterion_state={
                    'next': criterion.state_dict()
                }

            )
        )

        # Calculate FLOPs
        sample = next(iter(train_loader))[0].to(DEVICE)
        result = calculate_model_flops(model,
                                       sample_input=sample,
                                       print_report=True,
                                       units='K'
                                       )
        history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            history.get_curr_fold(),
            DEVICE,
            timeout=InputsConfig.TIMEOUT_TEST,
            target_cutoff=InputsConfig.NEXT_TARGET,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.get_curr_fold().to('next').best_model)

        history.get_curr_fold().to('next').add_report(report)
        history.step()
        history.display.end_fold()
