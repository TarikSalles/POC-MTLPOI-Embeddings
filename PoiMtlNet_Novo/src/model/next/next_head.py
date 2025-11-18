import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=5000, dropout=0.1):
        """
        Args:
            embed_dim: The dimensionality of the embedding space.
            max_seq_length: The maximum sequence length to handle.
            dropout: Dropout rate applied to positional encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings once
        pe = torch.zeros(max_seq_length, embed_dim)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_seq_length, embed_dim)

        self.register_buffer('positional_encoding', pe)
        self.eps = 1e-12

    def forward(self, x, padding_mask=None):
        """
        x: (batch_size, seq_length, embed_dim)
        padding_mask: (batch_size, seq_length), True for padding positions
        """
        seq_length = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_length, :]

        if padding_mask is not None:
            # zero out both x and pos_enc where padding_mask == True
            mask = (~padding_mask).unsqueeze(-1).float() + self.eps
            x = x * mask
            pos_enc = pos_enc * mask

        x = x + pos_enc
        return self.dropout(x)


class NextHeadSingle(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.35):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length

        # Positional encoding that can zero out padding
        self.pe = PositionalEncoding(embed_dim, max_seq_length=seq_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Project each time step down to a single logit for pooling
        self.sequence_reduction = nn.Linear(embed_dim, 1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, _ = x.size()
        device = x.device

        padding_mask = (x.abs().sum(dim=-1) == 0)  # (batch, seq), True for pad

        x = self.pe(x, padding_mask)

        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
            diagonal=1
        )

        x = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        x = self.layer_norm(x)

        attn_logits = self.sequence_reduction(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(padding_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0).unsqueeze(-1)  # (batch, seq, 1)

        pooled = torch.sum(x * attn_weights, dim=1)  # (batch, embed_dim)

        out = self.classifier(pooled)
        return out
