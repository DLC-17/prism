"""Tests for Relational Set Transformer permutation equivariance and PMA1 invariance."""

import torch
from models.set_transformer import RelationalSetTransformerBackbone, PMA1


def test_set_transformer_permutation_equivariance():
    torch.manual_seed(42)
    B, N, d_model = 2, 64, 256
    backbone = RelationalSetTransformerBackbone(
        d_model=d_model,
        num_layers=4,
        num_heads=8,
        d_ffn=1024,
    )
    backbone.eval()

    X = torch.randn(B, N, d_model)

    # Forward pass on original set
    with torch.no_grad():
        H_orig = backbone(X)

    # Permute bench indices (e.g. swap index 1 and index 4)
    perm = list(range(N))
    perm[1], perm[4] = perm[4], perm[1]
    X_perm = X[:, perm, :]

    with torch.no_grad():
        H_perm = backbone(X_perm)

    # Equivariance guarantee: H_perm must equal H_orig permuted!
    H_orig_permuted = H_orig[:, perm, :]

    max_diff = torch.max(torch.abs(H_perm - H_orig_permuted)).item()
    assert max_diff < 1e-4, f"Permutation equivariance violated! Max diff: {max_diff}"


def test_pma1_permutation_invariance():
    torch.manual_seed(42)
    B, N, d_model = 2, 64, 256
    pma1 = PMA1(d_model=d_model, num_heads=8)
    pma1.eval()

    H_field = torch.randn(B, N, d_model)

    with torch.no_grad():
        h_orig = pma1(H_field)

    # Arbitrary random permutation of entities
    rand_perm = torch.randperm(N)
    H_perm = H_field[:, rand_perm, :]

    with torch.no_grad():
        h_perm = pma1(H_perm)

    # Invariance guarantee: h_perm must equal h_orig!
    max_diff = torch.max(torch.abs(h_orig - h_perm)).item()
    assert max_diff < 1e-4, f"PMA1 permutation invariance violated! Max diff: {max_diff}"
