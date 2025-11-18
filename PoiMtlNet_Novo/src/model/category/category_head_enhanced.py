import torch
from torch import nn
import torch.nn.functional as F


# Model
class CategoryHeadResidual(nn.Module):
    """
    MLP with residual connections for deeper, more stable training.
    More appropriate for embeddings than Transformer-based approaches.

    Benefits:
    - Residual connections allow gradient flow through deep networks
    - More parameters without vanishing gradients
    - Better feature reuse compared to vanilla MLP

    Example usage:
        model = CategoryHeadResidual(
            input_dim=256,
            hidden_dims=(128, 64, 32),
            num_classes=7,
            dropout=0.2
        )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection to first hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            current_dim = hidden_dims[i]
            next_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else current_dim

            self.blocks.append(
                ResidualBlock(
                    dim=current_dim,
                    hidden_dim=current_dim * 2,  # Expansion factor
                    output_dim=next_dim,
                    dropout=dropout
                )
            )

        # Final classifier
        final_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Single residual block with expansion."""

    def __init__(self, dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.needs_projection = (dim != output_dim)

        # Main path
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

        # Shortcut projection if dims change
        if self.needs_projection:
            self.shortcut = nn.Linear(dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.shortcut(x)

# Model
class CategoryHeadGated(nn.Module):
    """
    MLP with gating mechanism to dynamically emphasize important features.

    Benefits:
    - Gates allow model to focus on relevant parts of the embedding
    - More interpretable than attention (can inspect gate values)
    - Computationally cheaper than Transformer
    - Suitable for embeddings where different dimensions may have varying importance

    Example usage:
        model = CategoryHeadGated(
            input_dim=256,
            hidden_dims=(128, 64),
            num_classes=7,
            dropout=0.2
        )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float = 0.2,
    ):
        super().__init__()

        # Feature gating on input
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # Main pathway
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for h in hidden_dims:
            self.layers.append(
                GatedLayer(prev_dim, h, dropout)
            )
            prev_dim = h

        # Classifier
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate input features
        gate = self.input_gate(x)
        x = x * gate

        # Process through gated layers
        for layer in self.layers:
            x = layer(x)

        return self.classifier(x)


class GatedLayer(nn.Module):
    """Single gated layer with GLU-style gating."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()

        # Main transformation
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Gate
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transform(x)
        g = self.gate(x)
        return self.dropout(self.activation(h * g))

# Model
class CategoryHeadEnsemble(nn.Module):
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
        # Process through all paths
        path_outputs = [path(x) for path in self.paths]

        # Concatenate and combine
        combined = torch.cat(path_outputs, dim=-1)
        return self.combiner(combined)

# Model
class CategoryHeadAttentionPooling(nn.Module):
    """
    MLP with attention-based pooling over feature dimensions.
    Lighter than Transformer but still benefits from learned attention.

    Benefits:
    - Learns which parts of embedding are important (interpretable weights)
    - No artificial tokenization
    - Single attention layer (much lighter than multi-head multi-layer Transformer)
    - Works directly on the continuous embedding representation

    Example usage:
        model = CategoryHeadAttentionPooling(
            input_dim=256,
            hidden_dim=128,
            num_classes=7,
            dropout=0.2
        )
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_classes: int = 7,
            dropout: float = 0.5,
    ):
        super().__init__()

        # Project embedding to hidden space
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Attention mechanism to weight features
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Final processing
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]

        # Project to hidden space
        h = self.proj(x)  # [B, hidden_dim]

        # Compute attention weights (can be used for interpretation)
        # Reshape to treat each dimension as an element
        h_expanded = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # For each sample, compute attention over hidden dimensions
        # This is a simplified version - treating dims as sequence
        att_scores = self.attention_weights(h)  # [B, 1]
        att_weights = torch.sigmoid(att_scores)  # [B, 1]

        # Weight the features
        weighted_h = h * att_weights  # [B, hidden_dim]

        # Classify
        return self.classifier(weighted_h)
