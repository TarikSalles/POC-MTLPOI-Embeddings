import torch
from torch import nn

class CategoryHeadMTL(nn.Module):
    """
    Multi-path architecture that processes the embedding through different pathways
    and combines them. More expressive than single MLP without artificial tokenization.

    Benefits:
    - Multiple complementary views of the same embedding
    - Different paths can specialize (e.g., local vs global patterns)
    - More robust than single pathway
    - Still respects the holistic nature of the embedding

    Example usage:
        model = CategoryHeadEnsemble(
            input_dim=256,
            hidden_dim=128,
            num_paths=3,
            num_classes=7,
            dropout=0.2
        )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_paths: int = 3,
            num_classes: int = 7,
            dropout: float = 0.5,
    ):
        super().__init__()
        self.num_paths = num_paths

        # Multiple parallel paths with different architectures
        self.paths = nn.ModuleList([
            self._make_path(input_dim, hidden_dim, depth=i+2, dropout=dropout)
            for i in range(num_paths)
        ])

        # Combine paths
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * num_paths, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _make_path(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Module:
        """Create a single path with specified depth."""
        layers = []
        prev_dim = input_dim

        for i in range(depth):
            # Vary hidden sizes across depth
            current_hidden = hidden_dim if i == depth - 1 else hidden_dim // (2 ** (depth - i - 1))
            current_hidden = max(current_hidden, hidden_dim)

            layers.extend([
                nn.Linear(prev_dim, current_hidden),
                nn.LayerNorm(current_hidden),
                nn.GELU(),
                nn.Dropout(dropout) if i < depth - 1 else nn.Identity(),
            ])
            prev_dim = current_hidden

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X is [B,1,input_dim] reshaped to [B,input_dim]
        x = x.squeeze(1)

        # Process through all paths
        path_outputs = [path(x) for path in self.paths]

        # Concatenate and combine
        combined = torch.cat(path_outputs, dim=-1)
        return self.combiner(combined)