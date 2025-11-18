import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import DataLoader

from configs.globals import DEVICE
from configs.model import InputsConfig

from configs.category_config import CfgCategoryModel, CfgCategoryHyperparams, CfgCategoryTraining
from model.category.category_head_enhanced import CategoryHeadGated, CategoryHeadResidual, CategoryHeadEnsemble, \
    CategoryHeadAttentionPooling
from train.category.evaluation import evaluate
from train.category.trainer import train
from model.category.CategoryHeadTransformer import CategoryHeadTransformer
from configs.next_config import CfgNextModel
from common.calc_flops.calculate_model_flops import calculate_model_flops
from common.ml_history.metrics import MLHistory, FlopsMetrics
from common.ml_history.parms.neural import NeuralParams


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        history.display.start_fold()
        model = CategoryHeadEnsemble(
            input_dim=CfgCategoryModel.INPUT_DIM,
            num_classes=CfgCategoryModel.NUM_CLASSES,
        ).to(DEVICE)

        y_all = np.concatenate([y.numpy() for _, y in train_loader])
        cls = np.arange(CfgNextModel.NUM_CLASSES)
        weights = compute_class_weight('balanced', classes=cls, y=y_all)
        alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            # weight=alpha
        )

        optimizer = AdamW(
            model.parameters(),
            lr=CfgCategoryHyperparams.LR,
            weight_decay=CfgCategoryHyperparams.WEIGHT_DECAY,
        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=CfgCategoryHyperparams.MAX_LR,
            epochs=CfgCategoryTraining.EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        # optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        # scheduler = CosineAnnealingLR(optimizer, T_max=CfgCategoryTraining.EPOCHS)

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=CfgCategoryTraining.BATCH_SIZE,
                num_epochs=CfgCategoryTraining.EPOCHS,
                learning_rate=CfgCategoryHyperparams.LR,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'category': criterion.__class__.__name__},
                criterion_state={
                    'category': criterion.state_dict()
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
            DEVICE,
            history=history.get_curr_fold(),
            timeout=InputsConfig.TIMEOUT_TEST,
            target_cutoff=InputsConfig.CATEGORY_TARGET,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.get_curr_fold().to('category').best_model)

        history.get_curr_fold().to('category').add_report(report)
        history.step()
        history.display.end_fold()