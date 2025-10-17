import torch
from torch import nn


class SEHead(nn.Module):
    def __init__(self, input_dim, reduction=8, hidden_dims=(128,64), num_classes=7):
        super().__init__()
        # Squeeze
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        # Deep head
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        # channel‐wise gating
        s = torch.relu(self.fc1(x))
        g = torch.sigmoid(self.fc2(s))
        x = x * g
        # classification
        x = self.deep(x)
        return self.classifier(x)