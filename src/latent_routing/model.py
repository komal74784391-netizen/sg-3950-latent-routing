import torch
import torch.nn as nn

from .router import LatentRouter


class LatentRoutingModel(nn.Module):
    """Small mixture-of-experts backbone with latent routing."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_experts: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True) for _ in range(num_layers)]
        )
        self.router = LatentRouter(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size))
             for _ in range(num_experts)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=attention_mask)
        weights = self.router(hidden)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=2)
        mixed = torch.einsum("bte,bteh->bth", weights, expert_outputs)
        return self.norm(hidden + mixed)
