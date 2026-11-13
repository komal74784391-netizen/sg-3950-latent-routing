import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentRouter(nn.Module):
    """Token-level latent router with temperature-controlled sparsity."""

    def __init__(self, hidden_size: int, num_experts: int, temperature: float = 0.7):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.projection = nn.Linear(hidden_size, num_experts, bias=False)
        self.latent_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.projection(hidden_states * torch.exp(self.latent_scale))
        logits = logits / max(self.temperature, 1e-6)
        weights = F.softmax(logits, dim=-1)
        return weights

    def entropy(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = self.forward(hidden_states)
        return -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, temperature={self.temperature}"
