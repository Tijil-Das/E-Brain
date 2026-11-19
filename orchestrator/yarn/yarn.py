"""
yarn.py — Real YaRN RoPE context extension (Peng et al., 2023)
===============================================================
Implements per-frequency ramp interpolation and attention temperature correction.
This is NOT NTK-aware scaling. Both components are required; using either alone
is not YaRN.

Reference: "YaRN: Efficient Context Window Extension of Large Language Models"
           Peng et al., 2023 — https://arxiv.org/abs/2309.00071

Public API
----------
  yarn_inv_freq(d_head, base, scale, beta, alpha, T_base) -> Tensor[d_head/2]
  yarn_attn_factor(scale) -> float
  build_yarn_cache(T, d_head, base, scale, device, beta, alpha, T_base)
      -> (cos[T, d_head], sin[T, d_head], attn_factor: float)
  yarn_rope_upgrade(model, T_ext, T_base, base, beta, alpha, device)
      -> replaces model's cos/sin buffers and stores attn_factor in model

Usage in MHA forward (after replacing the cache):
    q = q * self.yarn_attn_factor   # temperature correction BEFORE SDPA
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

Extension schedule (Phase 2 context extension):
    Stage 1: T_base=T_native -> T_ext=8192   (~4-5% of main steps, LR=LR_max/10)
    Stage 2: T_base=8192     -> T_ext=32768  (~4-5% of main steps, LR=LR_max/10)
    Eval each stage: loss by position band + needle test at T_ext.
    If T=32768 OOMs on T4: stop at T=16384 and document.
"""

import math
from typing import Tuple

import torch


# ── Core equations ────────────────────────────────────────────────────────────

def _correction_dim(n_rotations: float, d_head: int, base: float, T_base: int) -> float:
    """
    Returns the head-dimension index i at which the RoPE wavelength equals
    T_base / (2π × n_rotations).

    For i < correction_dim(beta)  → high-freq dimension → no interpolation needed.
    For i > correction_dim(alpha) → low-freq dimension  → full interpolation needed.

    Derivation: wavelength_i = 2π / theta_i = 2π × base^(2i/d)
    Set wavelength = T_base / (2π × n_rot):
        base^(2i/d) = T_base / (2π × n_rot)^2
        2i/d × ln(base) = ln(T_base / (2π × n_rot))
        i = d × ln(T_base / (2π × n_rot)) / (2 × ln(base))
    """
    return (d_head * math.log(T_base / (2.0 * math.pi * n_rotations))) / (
        2.0 * math.log(base)
    )


def yarn_inv_freq(
    d_head: int,
    base: float,
    scale: float,
    beta: float = 32.0,
    alpha: float = 1.0,
    T_base: int = 2048,
) -> torch.Tensor:
    """
    Compute YaRN-modified inverse frequencies for one attention head.

    Args:
        d_head:  head dimension (must be even)
        base:    RoPE base frequency (e.g. 10000)
        scale:   T_ext / T_base  (e.g. 32768/2048 = 16)
        beta:    high-freq factor — dimensions above correction_dim(beta): no interpolation
                 Typical: 32 (wavelength fits many times in T_base → safe to leave alone)
        alpha:   low-freq factor — dimensions below correction_dim(alpha): full interpolation
                 Typical: 1  (wavelength exceeds T_ext → must interpolate position)
        T_base:  original maximum context length

    Returns:
        Tensor of shape [d_head // 2] with modified inverse frequencies.

    Mathematical summary:
        inv_freq_orig[i]   = base^(-2i/d)                  (original RoPE)
        inv_freq_interp[i] = inv_freq_orig[i] / scale       (position interpolation)
        ramp[i]            = clamp((i-low_dim)/(high_dim-low_dim), 0, 1)
        inv_freq_yarn[i]   = ramp[i]*inv_freq_orig[i] + (1-ramp[i])*inv_freq_interp[i]
        → ramp=1 (high freq): original; ramp=0 (low freq): interpolated
    """
    n = d_head // 2
    inv_freq_orig   = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))

    # Ramp boundaries (dimension indices)
    low_dim  = max(0,     math.floor(_correction_dim(beta,  d_head, base, T_base)))
    high_dim = min(n - 1, math.ceil( _correction_dim(alpha, d_head, base, T_base)))

    if low_dim >= high_dim:
        # Degenerate: all dimensions on one side of the ramp
        # Treat everything below high_dim as fully interpolated
        ramp = (torch.arange(n, dtype=torch.float32) >= high_dim).float()
    else:
        ramp = ((torch.arange(n, dtype=torch.float32) - low_dim) / (high_dim - low_dim)
                ).clamp(0.0, 1.0)

    inv_freq_interp = inv_freq_orig / scale

    # ramp=1 → high-freq → keep original; ramp=0 → low-freq → interpolate
    return ramp * inv_freq_orig + (1.0 - ramp) * inv_freq_interp


def yarn_attn_factor(scale: float) -> float:
    """
    YaRN attention temperature correction factor (Peng et al. 2023, Appendix B).

    Applied by scaling query vectors BEFORE SDPA:
        q_scaled = q * yarn_attn_factor(scale)
        out = F.scaled_dot_product_attention(q_scaled, k, v, is_causal=True)

    This changes the effective attention denominator from sqrt(d_head) to
    sqrt(d_head × t) where t = 0.1 × ln(scale) + 1 ≥ 1.

    For scale=16 (2K→32K): t ≈ 1.277, factor ≈ 0.885
    For scale=4  (2K→8K):  t ≈ 1.139, factor ≈ 0.937
    For scale=1  (no ext): t = 1.0,   factor = 1.0  (identity)
    """
    t = 0.1 * math.log(scale) + 1.0
    return math.sqrt(1.0 / t)


# ── Cache builder ─────────────────────────────────────────────────────────────

def build_yarn_cache(
    T: int,
    d_head: int,
    base: float,
    scale: float,
    device: torch.device,
    beta: float = 32.0,
    alpha: float = 1.0,
    T_base: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Build RoPE cos/sin cache compatible with the production train.py RoPE scheme.

    The production scheme (from train.py build_rope_cache):
        freqs  = outer(pos, theta)       → [T, d_head/2]
        cos    = cat([cos, cos], dim=-1) → [T, d_head]    (duplicated halves)
        sin    = cat([sin, sin], dim=-1)

    Returns:
        cos:         [T, d_head] float32
        sin:         [T, d_head] float32
        attn_factor: float  (multiply onto q before SDPA)
    """
    inv_freq = yarn_inv_freq(d_head, base, scale, beta, alpha, T_base).to(device)
    pos      = torch.arange(T, dtype=torch.float32, device=device)
    freqs    = torch.outer(pos, inv_freq)                     # [T, d_head/2]
    cos      = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, d_head]
    sin      = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    factor   = yarn_attn_factor(scale)
    return cos, sin, factor


# ── Model upgrade helper ──────────────────────────────────────────────────────

def yarn_rope_upgrade(
    model: "torch.nn.Module",
    T_ext: int,
    T_base: int,
    base: float = 10_000.0,
    beta: float = 32.0,
    alpha: float = 1.0,
    device: "torch.device | None" = None,
) -> float:
    """
    Replace a production EBrain model's RoPE buffers with YaRN buffers
    for context extension T_base → T_ext.

    Modifies `model.cos` and `model.sin` in-place (registered buffers).
    Returns the attn_factor that must be applied to q in MHA forward.

    After calling this function, MHA.forward must be patched to scale q:
        q = q * model.yarn_attn_factor
    OR the model's MHA class must accept and apply the factor.

    Usage:
        factor = yarn_rope_upgrade(model, T_ext=32768, T_base=2048)
        model.yarn_attn_factor = factor   # store for MHA to read
    """
    dev    = device or next(model.parameters()).device
    scale  = T_ext / T_base

    # Infer d_head from model buffer shape: cos is [T, d_head]
    d_head = model.cos.shape[-1]

    cos, sin, factor = build_yarn_cache(
        T=T_ext, d_head=d_head, base=base, scale=scale,
        device=dev, beta=beta, alpha=alpha, T_base=T_base,
    )
    # Replace buffers — same dtype as original (float32 in production)
    model.cos = cos.to(model.cos.dtype)
    model.sin = sin.to(model.sin.dtype)
    model.yarn_attn_factor = factor

    return factor


# ── Diagnostics ───────────────────────────────────────────────────────────────

def yarn_diagnostics(
    d_head: int,
    base: float,
    scale: float,
    beta: float = 32.0,
    alpha: float = 1.0,
    T_base: int = 2048,
) -> str:
    """
    Returns a human-readable summary of the YaRN ramp for debugging.
    Prints which dimensions are fully interpolated, partially mixed, and unchanged.
    """
    n = d_head // 2
    low_dim  = max(0,     math.floor(_correction_dim(beta,  d_head, base, T_base)))
    high_dim = min(n - 1, math.ceil( _correction_dim(alpha, d_head, base, T_base)))
    t_factor = 0.1 * math.log(scale) + 1.0
    af       = math.sqrt(1.0 / t_factor)

    lines = [
        f"YaRN config: d_head={d_head} base={base} scale={scale:.1f} "
        f"(T_base={T_base} → T_ext={int(T_base*scale)})",
        f"  alpha={alpha} beta={beta}",
        f"  ramp dims: 0..{low_dim} = fully interpolated (÷{scale:.0f})",
        f"             {low_dim}..{high_dim} = linear mix",
        f"             {high_dim}..{n-1} = original (no interp)",
        f"  attn temperature t: {t_factor:.4f}",
        f"  attn_factor (×q):   {af:.4f}  (q logits dampened by {(1-af)*100:.1f}%)",
    ]
    return "\n".join(lines)
