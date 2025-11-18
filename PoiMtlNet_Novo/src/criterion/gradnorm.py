import torch
from torch import nn

from configs.globals import DEVICE


class GradNormLoss:
    def __init__(self, model, alpha=1.5):
        self.model = model
        self.alpha = alpha
        self.task_weights = nn.Parameter(torch.ones(2, device=DEVICE, requires_grad=True))  # 2 tasks

    def compute_loss(self, loss1, loss2, t, L0):
        # Compute weighted losses
        weighted_loss1 = self.task_weights[0] * loss1
        weighted_loss2 = self.task_weights[1] * loss2
        total_loss = weighted_loss1 + weighted_loss2

        # Backpropagate to compute gradients of shared parameters
        total_loss.backward(retain_graph=True)

        # Compute gradient norms for each task
        W = list(self.model.shared_layers.parameters())  # shared layers
        G1 = torch.autograd.grad(weighted_loss1, W, retain_graph=True, create_graph=True)
        G2 = torch.autograd.grad(weighted_loss2, W, retain_graph=True, create_graph=True)

        G1_norm = torch.norm(torch.stack([g.norm() for g in G1]))
        G2_norm = torch.norm(torch.stack([g.norm() for g in G2]))

        G_avg = (G1_norm + G2_norm) / 2

        # Compute relative inverse training rates
        L_ratio_1 = loss1.item() / L0[0]
        L_ratio_2 = loss2.item() / L0[1]
        inverse_train_rate = torch.tensor([L_ratio_1, L_ratio_2],device=DEVICE)
        target = G_avg * (inverse_train_rate ** self.alpha)
        target = target.detach()

        # Compute GradNorm loss
        grad_norm_loss = nn.functional.l1_loss(torch.stack([G1_norm, G2_norm]), target)

        # Backward on task weights
        self.task_weights.grad = None  # zero previous grad
        grad_norm_loss.backward()

        return total_loss