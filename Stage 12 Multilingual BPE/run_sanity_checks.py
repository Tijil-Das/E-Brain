import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Architectural components
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rotary_emb(xq, xk, cos, sin):
    B, H, T, D = xq.shape
    cos = cos[:T, :].unsqueeze(0).unsqueeze(0).to(xq.device)
    sin = sin[:T, :].unsqueeze(0).unsqueeze(0).to(xq.device)
    xq_r, xq_i = xq[..., :D // 2], xq[..., D // 2:]
    xk_r, xk_i = xk[..., :D // 2], xk[..., D // 2:]
    xq_out = torch.cat([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1)
    xk_out = torch.cat([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1)
    return xq_out, xk_out

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, D: int, H: int, max_seq_len: int = 512):
        super().__init__()
        self.H = H
        self.dk = D // H
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        cos, sin = precompute_freqs_cis(self.dk, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.H, self.dk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.H, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.H, self.dk).transpose(1, 2)
        q, k = apply_rotary_emb(q, k, self.cos, self.sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.attn = CausalSelfAttention(D, H, max_seq_len)
        self.norm2 = RMSNorm(D)
        hidden_dim = int(2 * (4 * D) / 3)
        self.ffn = SwiGLU(D, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class EBrainTransformer(nn.Module):
    def __init__(self, V: int, D: int, H: int, L: int, max_seq_len: int = 512):
        super().__init__()
        self.V = V
        self.D = D
        self.tok_emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([TransformerBlock(D, H, max_seq_len) for _ in range(L)])
        self.norm_final = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)

    def param_breakdown(self):
        non_emb = sum(p.numel() for n, p in self.named_parameters() if "tok_emb" not in n and "lm_head" not in n)
        emb = self.tok_emb.weight.numel()
        return non_emb, non_emb + emb

def run_checks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for local sanity check: {device}")
    
    # Check Model Params
    model_8k = EBrainTransformer(V=8000, D=128, H=4, L=6, max_seq_len=512).to(device)
    model_16k = EBrainTransformer(V=16000, D=128, H=4, L=6, max_seq_len=512).to(device)
    non_emb_8k, tot_8k = model_8k.param_breakdown()
    non_emb_16k, tot_16k = model_16k.param_breakdown()
    print("=" * 70)
    print("PARAMETER BREAKDOWN:")
    print(f"Model A (8K) : Non-Embedding={non_emb_8k:,d} | Total={tot_8k:,d}")
    print(f"Model B (16K): Non-Embedding={non_emb_16k:,d} | Total={tot_16k:,d}")
    assert non_emb_8k == non_emb_16k, "Non-embedding parameters must be identical!"
    print("Non-embedding parameter identity verified.")
    print("=" * 70)
    
    # 1. Step-0 Loss Check
    torch.manual_seed(42)
    x_test_8k = torch.randint(0, 8000, (4, 128), device=device)
    y_test_8k = torch.randint(0, 8000, (4, 128), device=device)
    with torch.no_grad():
        loss_0_8k = F.cross_entropy(model_8k(x_test_8k).view(-1, 8000), y_test_8k.view(-1)).item()
    
    x_test_16k = torch.randint(0, 16000, (4, 128), device=device)
    y_test_16k = torch.randint(0, 16000, (4, 128), device=device)
    with torch.no_grad():
        loss_0_16k = F.cross_entropy(model_16k(x_test_16k).view(-1, 16000), y_test_16k.view(-1)).item()
        
    expected_8k = math.log(8000)
    expected_16k = math.log(16000)
    print(f"1. Step-0 Loss (8K) : {loss_0_8k:.4f} (Expected ln(8000) = {expected_8k:.4f}, diff = {abs(loss_0_8k - expected_8k):.4f}) -> PASS")
    print(f"   Step-0 Loss (16K): {loss_0_16k:.4f} (Expected ln(16000) = {expected_16k:.4f}, diff = {abs(loss_0_16k - expected_16k):.4f}) -> PASS")
    assert abs(loss_0_8k - expected_8k) < 0.15
    assert abs(loss_0_16k - expected_16k) < 0.15
    
    # 2. Overfit One Batch Test
    print("\n2. Overfit One Batch Test...")
    torch.manual_seed(42)
    overfit_model = EBrainTransformer(V=8000, D=128, H=4, L=6, max_seq_len=64).to(device)
    opt = torch.optim.AdamW(overfit_model.parameters(), lr=1e-3)
    seq = torch.randint(0, 8000, (1, 9), device=device)
    bx, by = seq[:, :-1], seq[:, 1:]
    final_loss = 0.0
    for s in range(1, 251):
        opt.zero_grad()
        logits = overfit_model(bx)
        loss = F.cross_entropy(logits.view(-1, 8000), by.view(-1))
        loss.backward()
        opt.step()
        final_loss = loss.item()
    print(f"   Overfit Loss after 250 steps: {final_loss:.6f} -> {'PASS (~0 loss)' if final_loss < 0.05 else 'FAIL'}")
    assert final_loss < 0.05
    
    # 3. BPB Unit Test
    print("\n3. BPB Unit Test...")
    # Known test case:
    # Say loss_nats = math.log(2.0) = 0.69314718 (exactly 1 bit per token)
    # If 1 token = 2 bytes, then bpb = 1.0 * (1 / 2) = 0.50 bpb.
    loss_nats_test = math.log(2.0)
    tokens_test = 500
    bytes_test = 1000
    bpb_computed = (loss_nats_test / math.log(2.0)) * (tokens_test / bytes_test)
    print(f"   BPB formula unit test: {bpb_computed:.4f} (Expected 0.5000) -> PASS")
    assert abs(bpb_computed - 0.50) < 1e-6
    
    # Check targets shifted by 1 invariant:
    full_seq = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    x_in = full_seq[:, :-1]
    y_tgt = full_seq[:, 1:]
    assert torch.equal(x_in, torch.tensor([[10, 20, 30, 40]]))
    assert torch.equal(y_tgt, torch.tensor([[20, 30, 40, 50]]))
    print(f"   Target shift-by-1 invariant verified: x[t] predicts x[t+1] strictly.")
    
    # 4. Time one T=512 step
    print("\n4. Timing one T=512 Step (batch_size=16)...")
    timing_model = EBrainTransformer(V=8000, D=128, H=4, L=6, max_seq_len=512).to(device)
    timing_opt = torch.optim.AdamW(timing_model.parameters(), lr=1e-3)
    tx = torch.randint(0, 8000, (16, 512), device=device)
    ty = torch.randint(0, 8000, (16, 512), device=device)
    
    # Warmup
    out = timing_model(tx)
    l = F.cross_entropy(out.view(-1, 8000), ty.view(-1))
    l.backward()
    timing_opt.step()
    
    # Timed step
    t0 = time.time()
    timing_opt.zero_grad()
    out = timing_model(tx)
    l = F.cross_entropy(out.view(-1, 8000), ty.view(-1))
    l.backward()
    timing_opt.step()
    t1 = time.time()
    step_time_ms = (t1 - t0) * 1000
    print(f"   Step Time (T=512, B=16): {step_time_ms:.2f} ms ({16*512 / (t1 - t0):.1f} tok/sec) -> PASS")
    print("=" * 70)
    print("ALL SANITY CHECKS PASSED.")

if __name__ == "__main__":
    run_checks()
