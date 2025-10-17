import torch
from torch import nn


class CategoryHeadSingle(nn.Module):
    """
    Multi-layer perceptron for category classification.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            prev_dim = h
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)
