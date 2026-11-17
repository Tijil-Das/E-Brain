import os
import sys
import time
import math
import json
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_VAULT = os.path.join(BASE_DIR, "data_v3", "test_vault")
VAL_VAULT = os.path.join(BASE_DIR, "data_v3", "val_vault")
LOCK_PATH = os.path.join(BASE_DIR, "data_v3", "stage2_certified_final_results.json")
TOK16K_PATH = os.path.join(BASE_DIR, "ebrain_tokenizer_16k.json")
CKPT_PATH = os.path.join(BASE_DIR, "kaggle_stage7_output", "checkpoint_best.pt")
OUT_JSON = os.path.join(BASE_DIR, "kaggle_stage7_output", "stage7_test_evaluation.json")

LANGUAGES = ["en", "es", "ru", "fr", "de", "ja", "zh", "hi", "bn", "ar"]

print("=" * 80)
print("STAGE 7: OFFICIAL FINAL TEST EVALUATION (ONE-SHOT SEALED TEST)")
print("=" * 80)

# Step 1: Verify Checkpoint Hash
with open(CKPT_PATH, "rb") as f:
    ckpt_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Checkpoint Best SHA-256: {ckpt_hash}")
expected_ckpt_hash = "06f66851349919b37a70602e4c1fab24b64375286dc7e946e2cdfb600ee7894f"
assert ckpt_hash == expected_ckpt_hash, f"Checkpoint hash mismatch! Expected {expected_ckpt_hash}"

# Step 2: Verify Tokenizer Hash
with open(TOK16K_PATH, "rb") as f:
    tok_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
print(f"16K Tokenizer SHA-256: {tok_hash}")

# Step 3: Verify Sealed Test Vault Hashes against Stage 2 Locks
with open(LOCK_PATH, "r", encoding="utf-8") as f:
    locks = json.load(f)

print("\nVerifying bitwise integrity of sealed test files...")
test_hashes = {}
for lang in LANGUAGES:
    fp = os.path.join(TEST_VAULT, f"test_{lang}.txt")
    with open(fp, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
        h = hashlib.sha256(data).hexdigest()
    expected = locks[lang]["sha256"]["test"]
    assert h == expected, f"Hash mismatch for {lang}! Actual {h} != Expected {expected}"
    test_hashes[lang] = h
    print(f"  [{lang}] SHA-256 VERIFIED: {h[:16]}... (matches Stage 2 lock)")

print("All 10/10 test files bitwise certified matching sealed Stage 2 lock!\n")

# Model Architecture Definition (exact match to kaggle_stage7_pretraining.py)
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
    return torch.cos(freqs), torch.sin(freqs)

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
    def __init__(self, V: int = 16000, D: int = 256, H: int = 8, L: int = 6, max_seq_len: int = 512):
        super().__init__()
        self.V = V
        self.D = D
        self.tok_emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([TransformerBlock(D, H, max_seq_len) for _ in range(L)])
        self.norm_final = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_final(x)
        return self.lm_head(x)

# Load Tokenizer & Model
tok16k = Tokenizer.from_file(TOK16K_PATH)
model = EBrainTransformer().eval()
loaded = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(loaded["model_state"])

# Set optimal CPU thread count
torch.set_num_threads(2)

T = 512
results = {}

print("Beginning one-time evaluation on sealed test vault...")
print(f"{'Lang':<6} | {'Bytes':<10} | {'Tokens':<10} | {'Chunks':<8} | {'Nats':<8} | {'Test BPB':<10} | {'Elapsed':<8}")
print("-" * 75)

total_eval_start = time.time()

for lang in LANGUAGES:
    fp = os.path.join(TEST_VAULT, f"test_{lang}.txt")
    with open(fp, "rb") as f:
        raw_b = f.read()
    raw_text = raw_b.decode("utf-8", errors="ignore")
    toks = tok16k.encode(raw_text).ids
    total_bytes = len(raw_b)
    
    num_chunks = (len(toks) - 1) // T
    tot_loss = 0.0
    tot_toks = 0
    t0 = time.time()
    
    with torch.inference_mode():
        for i in range(num_chunks):
            chunk = toks[i * T : (i + 1) * T + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1), reduction="sum")
            tot_loss += loss.item()
            tot_toks += y.numel()
            
    mean_nats = tot_loss / max(1, tot_toks)
    # Standard project BPB formula: (mean_nats / ln(2)) * (tot_toks / total_bytes)
    bpb = (mean_nats / math.log(2.0)) * (tot_toks / max(1, total_bytes))
    elapsed = time.time() - t0
    
    results[lang] = {
        "bytes": total_bytes,
        "tokens": len(toks),
        "evaluated_tokens": tot_toks,
        "chunks": num_chunks,
        "mean_nats": round(mean_nats, 4),
        "test_bpb": round(bpb, 4),
        "sha256": test_hashes[lang],
        "eval_seconds": round(elapsed, 2)
    }
    
    print(f"{lang:<6} | {total_bytes:<10,d} | {len(toks):<10,d} | {num_chunks:<8} | {mean_nats:<8.4f} | {bpb:<10.4f} | {elapsed:<6.1f}s", flush=True)

mean_test_bpb = sum(r["test_bpb"] for r in results.values()) / len(results)
total_elapsed = time.time() - total_eval_start

print("-" * 75)
print(f"OVERALL MEAN TEST BPB: {mean_test_bpb:.4f} bpb (Total time: {total_elapsed:.1f}s)")
print("=" * 80)

final_payload = {
    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "checkpoint_hash": ckpt_hash,
    "checkpoint_step": 22990,
    "tokenizer_hash": tok_hash,
    "mean_test_bpb": round(mean_test_bpb, 4),
    "languages": results
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_payload, f, indent=2)

print(f"Results recorded in {OUT_JSON}")
