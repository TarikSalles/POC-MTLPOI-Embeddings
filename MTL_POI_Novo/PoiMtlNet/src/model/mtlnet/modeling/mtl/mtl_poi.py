from typing import Iterator, Tuple, List
import torch
from torch import nn

from configs.globals import DEVICE

from configs.model import InputsConfig
from src.model.mtlnet.modeling.category_head import CategoryHead
from src.model.mtlnet.modeling.next_head import NextHead


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and Dropout for deep shared layers."""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual sublayer
        residual = x
        out = self.norm1(x)
        out = self.activation(self.layer1(out))
        out = self.dropout1(out) + residual

        # Second residual sublayer
        residual = out
        out = self.norm2(out)
        out = self.activation(self.layer2(out))
        out = self.dropout2(out) + residual

        return out


class FiLMLayer(nn.Module):
    """
    Feature‐Wise Linear Modulation: gamma * x + beta,
    where gamma/beta come from a small task embedding.
    """

    def __init__(self, emb_dim: int, layer_size: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, layer_size)
        self.beta = nn.Linear(emb_dim, layer_size)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        # x: [batch, ... , layer_size]
        # task_emb: [batch, emb_dim]
        gamma = self.gamma(task_emb)  # [batch, layer_size]
        beta = self.beta(task_emb)  # [batch, layer_size]

        # unsqueeze so gamma/beta can broadcast along any extra dims of x
        # e.g. if x is [batch, seq_len, layer_size] we need [batch, 1, layer_size]
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


class MTLnet(nn.Module):
    def __init__(
        self,
        feature_size: int,
        shared_layer_size: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        seq_length: int,
        num_shared_layers: int,
        encoder_layer_size: int = 256,
        num_encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        shared_dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Task‐specific encoders
        self.category_encoder = self._build_encoder(
            in_size=feature_size,
            hidden_size=encoder_layer_size,
            num_layers=num_encoder_layers,
            out_size=shared_layer_size,
            dropout=encoder_dropout,
        )
        self.next_encoder = self._build_encoder(
            in_size=feature_size,
            hidden_size=encoder_layer_size,
            num_layers=num_encoder_layers,
            out_size=shared_layer_size,
            dropout=encoder_dropout,
        )

        # Task‐ID embedding + FiLM block for shared layers
        self.task_embedding = nn.Embedding(2, shared_layer_size)
        self.film = FiLMLayer(emb_dim=shared_layer_size, layer_size=shared_layer_size)

        # Shared processing layers (post‐FiLM)
        self.shared_layers = self._build_shared_layers(
            layer_size=shared_layer_size,
            num_blocks=num_shared_layers,
            dropout=shared_dropout,
        )

        # Task heads
        self.category_poi = CategoryHead(
            input_dim=shared_layer_size,
            num_classes=num_classes,
            dropout=0.1,
        )
        self.next_poi = NextHead(
            shared_layer_size, num_classes, num_heads, seq_length, num_layers,
            dropout=0.1
        )

    def _build_encoder(
            self,
            in_size: int,
            hidden_size: int,
            num_layers: int,
            out_size: int,
            dropout: float,
    ) -> nn.Sequential:
        layers = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            ]
        layers += [
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
            nn.LayerNorm(out_size),
        ]
        return nn.Sequential(*layers)

    def _build_shared_layers(
            self,
            layer_size: int,
            num_blocks: int,
            dropout: float,
    ) -> nn.Sequential:
        layers = [
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.LayerNorm(layer_size),
            nn.Dropout(dropout),
        ]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(layer_size, dropout))
        return nn.Sequential(*layers)

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs  # ([B, 1, feature_size], [B, seq_len, feature_size])

        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)  # (batch_size, seq_len)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)  # zero out all-pad tokens

        # Task‐specific encoding
        enc_cat = self.category_encoder(category_input)  # [batch, shared_size]
        enc_next = self.next_encoder(next_input)  # [batch, shared_size]

        # Build task‐ID vectors
        b_cat = enc_cat.size(0)
        b_next = enc_next.size(0)
        id_cat = torch.zeros(b_cat, dtype=torch.long, device=enc_cat.device)
        id_next = torch.ones(b_next, dtype=torch.long, device=enc_next.device)

        emb_cat = self.task_embedding(id_cat)
        emb_next = self.task_embedding(id_next)

        # FiLM modulation
        mod_cat = self.film(enc_cat, emb_cat)
        mod_next = self.film(enc_next, emb_next)

        # Shared processing
        shared_cat = self.shared_layers(mod_cat)
        shared_next = self.shared_layers(mod_next)

        # Heads
        # Cat in: [batch, 1, shared_size] Cat out: [batch, num_classes]
        out_cat = self.category_poi(shared_cat).view(-1, self.num_classes)
        # Next in: [batch, seq_len, shared_size] Next out: [batch, seq_len, num_classes]
        out_next = self.next_poi(shared_next)

        return out_cat, out_next

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "shared_layers" in name
               or "task_embedding" in name
               or "film" in name
        )

    def task_specific_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if any(
            key in name
            for key in (
                "category_encoder",
                "next_encoder",
                "category_poi",
                "next_poi",
            )
        )
        )