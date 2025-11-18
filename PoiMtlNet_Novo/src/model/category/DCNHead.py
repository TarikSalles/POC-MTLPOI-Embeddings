import torch
from torch import nn


class DCNHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, cross_layers=2, dropout=0.):
        super().__init__()
        # Cross network
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True)  # w_k^T x + b_k
            for _ in range(cross_layers)
        ])
        # Deep network
        prev = input_dim
        deep = []
        for h in hidden_dims:
            deep += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep)
        self.classifier = nn.Linear(prev + input_dim, num_classes)

    def forward(self, x):
        x0 = x
        x_cross = x
        for layer in self.cross_layers:
            # compute scalar = w_k^T x_cross + b_k
            scalar = layer(x_cross)
            x_cross = x0 * scalar + x_cross
        x_deep = self.deep(x)
        return self.classifier(torch.cat([x_cross, x_deep], dim=-1))