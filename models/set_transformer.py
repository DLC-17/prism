"""Relational Set Transformer Backbone with 4 SAB layers and PMA_1 global pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 32
        assert self.d_k == 32, f"Expected d_k = 32, got {self.d_k}"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, N_q, _ = Q.shape
        _, N_k, _ = K.shape

        q = self.W_q(Q).view(B, N_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, N_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        return self.W_o(out)


class SetAttentionBlock(nn.Module):
    """SAB(X) = LayerNorm(X' + FFN(X')) where X' = LayerNorm(X + MHA(X, X, X))"""

    def __init__(self, d_model: int = 256, num_heads: int = 8, d_ffn: int = 1024):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Pre-LN / Post-LN with residual connection
        X_prime = self.ln1(X + self.mha(X, X, X))
        out = self.ln2(X_prime + self.ffn(X_prime))
        return out


class PMA1(nn.Module):
    """Pooling by Multihead Attention with 1 learned seed vector S in R^{1 x 1 x d_model}.

    Yields a permutation-invariant pooled global representation h_global in R^{B x d_model}.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, H_field: torch.Tensor) -> torch.Tensor:
        B = H_field.shape[0]
        S = self.seed.expand(B, -1, -1)  # (B, 1, d_model)
        pooled = self.mha(Q=S, K=H_field, V=H_field)  # (B, 1, d_model)
        return self.ln(pooled.squeeze(1))  # (B, d_model)


class RelationalSetTransformerBackbone(nn.Module):
    """4-layer SAB backbone with zero absolute positional encodings on bench and hand.

    Preserves strict permutation equivariance over entity set.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ffn: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [SetAttentionBlock(d_model=d_model, num_heads=num_heads, d_ffn=d_ffn) for _ in range(num_layers)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Args:

            X: Matrix of shape (B, N, d_model)

        Returns:
            H_field: Matrix of shape (B, N, d_model)
        """
        H = X
        for layer in self.layers:
            H = layer(H)
        return H
