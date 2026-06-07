# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Mixture of Experts Feed-Forward Network (MoEFFN).

This module implements a token-choice MoE layer where each token is routed
to top-k experts via a learned router. A load-balancing auxiliary loss
encourages uniform expert utilization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """
    Mixture of Experts FFN with top-k routing and load balancing loss.

    Each expert is a small MLP: Linear -> GELU -> Dropout -> Linear -> Dropout.
    The router selects top-k experts per token and combines their outputs
    weighted by normalized router probabilities.

    The load balancing loss (aux_loss) is computed as:
        L_aux = num_experts * sum_i(f_i * P_i)
    where f_i is the fraction of tokens dispatched to expert i, and P_i is
    the mean router probability for expert i. This encourages all experts
    to receive roughly equal numbers of tokens.

    Attributes:
        aux_loss (torch.Tensor or None): The load balancing loss from the
            most recent forward pass. Set to None before the first forward.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_ratio: float = 0.5,
        out_features: int = None,
        drop: float = 0.0,
    ):
        """
        Args:
            dim: Input feature dimension.
            num_experts: Number of expert networks.
            top_k: Number of experts to activate per token.
            expert_hidden_ratio: Hidden dimension ratio relative to dim
                (expert hidden = int(dim * expert_hidden_ratio)).
            out_features: Output dimension. Defaults to dim.
            drop: Dropout probability applied after each linear in the expert.
        """
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_features = out_features or dim
        hidden_dim = int(dim * expert_hidden_ratio)

        # Router: produces a probability distribution over experts for each token
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Expert networks — each is a small two-layer MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, self.out_features),
                nn.Dropout(drop),
            )
            for _ in range(num_experts)
        ])

        # Holds the auxiliary load-balancing loss from the last forward pass
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with top-k expert routing.

        Args:
            x: Input tensor of shape (B, S, dim).

        Returns:
            Output tensor of shape (B, S, out_features).
        """
        B, S, C = x.shape

        # Flatten batch and sequence dimensions for token-level routing
        x_flat = x.view(B * S, C)  # (B*S, dim)

        # ---- Router ----
        router_logits = self.router(x_flat)            # (B*S, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (B*S, num_experts)

        # Select top-k experts per token
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # topk_probs:   (B*S, top_k)
        # topk_indices: (B*S, top_k)

        # Re-normalize the selected probabilities so they sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # ---- Load balancing auxiliary loss ----
        # f_i: fraction of tokens dispatched to expert i
        expert_mask = F.one_hot(topk_indices, self.num_experts).float()  # (B*S, top_k, num_experts)
        density_per_token = expert_mask.sum(dim=1) / self.top_k           # (B*S, num_experts)
        f_i = density_per_token.mean(dim=0)                                # (num_experts,)

        # P_i: mean router probability for expert i
        P_i = router_probs.mean(dim=0)  # (num_experts,)

        # L_aux = E * sum_i(f_i * P_i)
        self.aux_loss = self.num_experts * (f_i * P_i).sum()

        # ---- Expert computation ----
        output = torch.zeros(B * S, self.out_features, device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            # Find all (token_index, topk_slot) pairs where expert i was selected
            idx_i, k_i = torch.where(topk_indices == i)
            if idx_i.numel() > 0:
                expert_input = x_flat[idx_i]                     # tokens routed to expert i
                expert_output = self.experts[i](expert_input)    # (num_tokens, out_features)
                weight = topk_probs[idx_i, k_i].unsqueeze(-1)    # (num_tokens, 1)
                output[idx_i] += expert_output * weight           # accumulate weighted output

        # Restore original shape
        output = output.view(B, S, self.out_features)
        return output
