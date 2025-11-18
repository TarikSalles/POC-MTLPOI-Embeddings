import torch
import torch.nn as nn
import torch.nn.functional as F


class NextHeadLSTM(nn.Module):
    """
    LSTM-based next-POI predictor.
    More efficient than Transformer for short sequences.

    Benefits:
    - Designed specifically for sequential data
    - Implicit temporal modeling (no need for positional encoding)
    - Fewer parameters than Transformer
    - Better for sequences < 20 steps
    - Naturally handles variable-length sequences

    Example:
        model = NextHeadLSTM(
            embed_dim=256,
            hidden_dim=256,
            num_classes=7,
            num_layers=2,
            dropout=0.3
        )
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM processes the sequence
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_length, embed_dim)
        """
        # Detect padding (all zeros)
        padding_mask = (x.abs().sum(dim=-1) == 0)  # (batch, seq)

        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)  # output: (batch, seq, hidden*directions)

        # Use last non-padding hidden state
        # Get sequence lengths
        seq_lengths = (~padding_mask).sum(dim=1)  # (batch,)

        # Gather last valid output for each sequence
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(output[i, last_idx])

        last_output = torch.stack(last_outputs)  # (batch, hidden*directions)

        return self.classifier(last_output)


class NextHeadGRU(nn.Module):
    """
    GRU-based next-POI predictor.
    Lighter than LSTM, often performs similarly for short sequences.

    Benefits:
    - 25% fewer parameters than LSTM
    - Faster training and inference
    - Simpler gating mechanism
    - Good for short sequences where long-term memory is less critical

    Example:
        model = NextHeadGRU(
            embed_dim=256,
            hidden_dim=256,
            num_classes=7,
            num_layers=2,
            dropout=0.3
        )
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_length, embed_dim)
        """
        padding_mask = (x.abs().sum(dim=-1) == 0)
        seq_lengths = (~padding_mask).sum(dim=1)

        output, h_n = self.gru(x)

        # Use last valid output
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(output[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


class NextHeadTemporalCNN(nn.Module):
    """
    Temporal CNN for next-POI prediction.
    Uses causal convolutions to capture local temporal patterns.

    Benefits:
    - Captures local temporal patterns (e.g., "gym → smoothie bar")
    - Parallel computation (faster than RNNs)
    - Good for sequences with strong local dependencies
    - Receptive field grows exponentially with layers

    Example:
        model = NextHeadTemporalCNN(
            embed_dim=256,
            hidden_channels=128,
            num_classes=7,
            num_layers=4,
            kernel_size=3,
            dropout=0.2
        )
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_channels: int = 128,
        num_classes: int = 7,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Temporal convolution blocks
        self.convs = nn.ModuleList()
        in_channels = embed_dim

        for i in range(num_layers):
            out_channels = hidden_channels
            padding = (kernel_size - 1)  # Causal padding

            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),  # BatchNorm1d for Conv1d (channels dim)
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels

        # Global pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_length, embed_dim)
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_length)

        # Apply causal convolutions
        for conv in self.convs:
            x = conv(x)
            # Remove future context (causal)
            x = x[:, :, :x.size(2) - (conv[0].padding[0])]

        return self.classifier(x)


class NextHeadHybrid(nn.Module):
    """
    Hybrid GRU + Self-Attention for next-POI prediction.
    Combines strengths of RNNs and Transformers.

    Benefits:
    - GRU captures sequential dependencies efficiently
    - Self-attention focuses on important past visits
    - Best of both worlds for short sequences
    - More interpretable (can visualize attention weights)

    Example:
        model = NextHeadHybrid(
            embed_dim=256,
            hidden_dim=256,
            num_classes=7,
            num_heads=4,
            dropout=0.3
        )
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 4,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # GRU for sequential processing
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )

        # Self-attention to focus on important visits
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_length, embed_dim)
        """
        # Detect padding
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # GRU processes sequence
        gru_out, _ = self.gru(x)  # (batch, seq, hidden)
        gru_out = self.norm1(gru_out)

        # Self-attention over GRU outputs
        attn_out, attn_weights = self.attention(
            gru_out, gru_out, gru_out,
            key_padding_mask=padding_mask,
            need_weights=True
        )

        # Residual connection
        out = self.norm2(gru_out + attn_out)

        # Use last valid timestep
        seq_lengths = (~padding_mask).sum(dim=1)
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(out[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


class NextHeadTransformerOptimized(nn.Module):
    """
    Optimized Transformer for short sequences.
    Improvements over vanilla Transformer:
    - Learned positional embeddings (better for short sequences)
    - Temporal decay in attention (recent visits matter more)
    - Efficient pooling strategy

    Example:
        model = NextHeadTransformerOptimized(
            embed_dim=256,
            num_classes=7,
            num_heads=8,
            num_layers=2,
            seq_length=9,
            dropout=0.3
        )
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 2,
        seq_length: int = 9,
        dropout: float = 0.3,
        use_temporal_decay: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.use_temporal_decay = use_temporal_decay

        # LEARNED positional embeddings (better for short sequences)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Temporal decay weights (recent visits are more important)
        if use_temporal_decay:
            # Exponential decay: [e^-4, e^-3, ..., e^-1, e^0]
            decay = torch.exp(-torch.arange(seq_length - 1, -1, -1).float() * 0.5)
            self.register_buffer('temporal_decay', decay)

        # Classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_length, embed_dim)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # Padding mask
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Add learned positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        # Causal mask (can't see future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        x = self.norm(x)

        # Smart pooling: weight by temporal decay (recent = important)
        if self.use_temporal_decay:
            # Apply temporal decay
            decay_weights = self.temporal_decay[:seq_len].unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
            weighted_x = x * decay_weights

            # Mask padding
            mask = (~padding_mask).unsqueeze(-1).float()
            weighted_x = weighted_x * mask

            # Sum and normalize
            pooled = weighted_x.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            # Use last valid output
            seq_lengths = (~padding_mask).sum(dim=1)
            last_outputs = []
            for i in range(batch_size):
                last_idx = seq_lengths[i] - 1
                last_outputs.append(x[i, last_idx])
            pooled = torch.stack(last_outputs)

        return self.classifier(pooled)