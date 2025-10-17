import torch
from torch import nn

class CategoryHead(nn.Module):
    """
    Split embedding into L tokens, run through a small Transformer encoder,
    pool and project to num_classes.
    """
    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 2,
        token_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        num_classes: int = 7,
    ):
        super().__init__()
        assert num_tokens * token_dim == input_dim, \
            f"need num_tokens*token_dim==input_dim ({num_tokens}×{token_dim}≠{input_dim})"

        self.num_tokens = num_tokens
        self.token_dim   = token_dim

        # 1) Project flat embedding into exactly (L*D_token) dims
        self.token_proj = nn.Linear(input_dim, num_tokens * token_dim)

        # 2) Learned positional embeddings [1, L, D_token]
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, token_dim))

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            # norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim)
        )

        # 4) Final classifier on the pooled token representation
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X is [B,1,input_dim] reshaped to [B,input_dim]
        x = x.squeeze(1)
        # x: [B, input_dim]
        B, D = x.shape

        # a) Project and reshape -> [B, L, token_dim]
        tokens = self.token_proj(x)              # [B, L*token_dim]
        tokens = tokens.view(B,                 # ← reshape into tokens
                             self.num_tokens,
                             self.token_dim)

        # b) Add position embeddings (auto-broadcast on batch dim)
        tokens = tokens + self.pos_emb          # [B, L, token_dim]

        # c) Transformer encoder over the pseudo-tokens
        encoded = self.transformer(tokens)       # [B, L, token_dim]

        # d) Pool (mean over tokens) then classify
        pooled = encoded.mean(dim=1)             # [B, token_dim]
        return self.classifier(pooled)          # [B, num_classes]