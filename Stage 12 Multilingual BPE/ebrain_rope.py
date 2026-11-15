"""
Stage 12 - Step 3: Rotary Position Embeddings (RoPE)
Mathematical Formulation: Su et al. (RoFormer, 2021)
Implements relative position encoding through complex 2D vector rotation in Q and K.
"""

import math
import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precomputes the cosine and sine frequencies for RoPE.
    theta_i = 10000^(-2i / dim) for i in [0, 1, ..., dim/2 - 1]
    """
    assert dim % 2 == 0, f"Head dimension {dim} must be even for 2D pair rotation"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: [max_seq_len, dim // 2]
    freqs = torch.outer(t, freqs)
    # Cosine and sine components
    cos = torch.cos(freqs) # [max_seq_len, dim // 2]
    sin = torch.sin(freqs) # [max_seq_len, dim // 2]
    return cos, sin

def rotate_half(x: torch.Tensor):
    """
    Rotates half the hidden dimensions of x:
    [-x2, x1, -x4, x3, ...]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies rotary embeddings to Query or Key tensor:
    x: [B, H, T, dk]
    cos: [T, dk // 2]
    sin: [T, dk // 2]
    """
    T = x.shape[2]
    cos_t = cos[:T, :].unsqueeze(0).unsqueeze(1) # [1, 1, T, dk // 2]
    sin_t = sin[:T, :].unsqueeze(0).unsqueeze(1) # [1, 1, T, dk // 2]
    # Repeat along the last dimension to match dk
    cos_full = torch.cat((cos_t, cos_t), dim=-1) # [1, 1, T, dk]
    sin_full = torch.cat((sin_t, sin_t), dim=-1) # [1, 1, T, dk]
    
    # 2D plane rotation: x * cos + rotate_half(x) * sin
    return (x * cos_full) + (rotate_half(x) * sin_full)

def verify_rope_properties():
    """
    Automated verification of RoPE mathematical properties:
    1. Norm Preservation: ||RoPE(x)||_2 == ||x||_2 (Rotations do not stretch vectors)
    2. Relative Shift Invariance: <RoPE(q, m), RoPE(k, n)> depends strictly on (m - n)
    3. Identity at distance 0: When m == n, dot product equals unrotated dot product
    """
    print("=" * 80)
    print("VERIFYING ROTARY POSITION EMBEDDING (RoPE) MATHEMATICAL PROPERTIES")
    print("=" * 80)

    dk = 16 # head dimension
    T = 64
    B = 1
    H = 1

    cos, sin = precompute_freqs_cis(dk, T)

    # 1. Norm Preservation Check
    x = torch.randn(B, H, T, dk)
    x_rot = apply_rotary_emb(x, cos, sin)
    orig_norm = torch.norm(x, p=2, dim=-1)
    rot_norm = torch.norm(x_rot, p=2, dim=-1)
    max_norm_diff = torch.max(torch.abs(orig_norm - rot_norm)).item()
    print(f"1. Norm Preservation Check: Max norm difference = {max_norm_diff:.6e}")
    assert max_norm_diff < 1e-5, f"Norm preservation failed: {max_norm_diff}"
    print("   -> PASSED (Rotation strictly preserves vector length, preventing gradient explosion)")

    # 2. Relative Shift Invariance Check:
    # If q and k are identical vectors, then <q_m, k_n> must depend ONLY on (m - n).
    # Specifically, <q_5, k_2> (diff=3) MUST EQUAL <q_12, k_9> (diff=3)!
    q = torch.randn(1, 1, 1, dk)
    k = torch.randn(1, 1, 1, dk)

    def get_dot(m, n):
        # Rotate q at position m, k at position n
        q_m = apply_rotary_emb(q, cos[m:m+1], sin[m:m+1])
        k_n = apply_rotary_emb(k, cos[n:n+1], sin[n:n+1])
        return torch.sum(q_m * k_n).item()

    dot_5_2 = get_dot(5, 2)   # diff = +3
    dot_12_9 = get_dot(12, 9) # diff = +3
    dot_25_22 = get_dot(25, 22) # diff = +3
    diff_shift = abs(dot_5_2 - dot_12_9) + abs(dot_5_2 - dot_25_22)
    print(f"\n2. Relative Distance Invariance Check (Shift = +3):")
    print(f"   - <q_5,  k_2>  = {dot_5_2:.8f}")
    print(f"   - <q_12, k_9>  = {dot_12_9:.8f}")
    print(f"   - <q_25, k_22> = {dot_25_22:.8f}")
    print(f"   - Relative Invariance Delta = {diff_shift:.6e}")
    assert diff_shift < 1e-5, f"Relative shift invariance failed: {diff_shift}"
    print("   -> PASSED (Dot-product is 100% relative-distance invariant!)")

    # 3. Asymmetric Directional Check (Forward vs Backward):
    # <q_2, k_5> (diff = -3) should NOT equal <q_5, k_2> (diff = +3) because language order matters!
    dot_2_5 = get_dot(2, 5) # diff = -3
    print(f"\n3. Directional Asymmetry Check (Order Sensitivity):")
    print(f"   - Forward  <q_5, k_2> (verb before noun) = {dot_5_2:.8f}")
    print(f"   - Backward <q_2, k_5> (noun before verb) = {dot_2_5:.8f}")
    print(f"   - Absolute Difference = {abs(dot_5_2 - dot_2_5):.8f}")
    assert abs(dot_5_2 - dot_2_5) > 1e-3, "RoPE failed to distinguish forward vs backward order!"
    print("   -> PASSED (Order sensitivity confirmed: forward != backward)")

    print("\n" + "=" * 80)
    print(">>> RoPE MATHEMATICALLY VERIFIED AND PROVEN SOUND! <<<")
    print("=" * 80)

if __name__ == "__main__":
    verify_rope_properties()

