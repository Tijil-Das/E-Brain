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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Correct Kaggle mount paths (uses /kaggle/input/datasets/owner/slug/)
KAGGLE_DATASETS = "/kaggle/input/datasets/tijildas"
DATA_DIR  = os.path.join(KAGGLE_DATASETS, "ebrain-multilingual-v3-certified")
CKPT_DIR  = os.path.join(KAGGLE_DATASETS, "ebrain-stage7-checkpoint")

TEST_VAULT = os.path.join(DATA_DIR, "test_vault")
LOCK_PATH  = os.path.join(DATA_DIR, "stage2_certified_final_results.json")
TOK16K_PATH= os.path.join(DATA_DIR, "ebrain_tokenizer_16k.json")
CKPT_PATH  = os.path.join(CKPT_DIR, "checkpoint_best.pt")
OUT_JSON   = "/kaggle/working/stage7_test_evaluation.json"

LANGUAGES = ["en", "es", "ru", "fr", "de", "ja", "zh", "hi", "bn", "ar"]

print("=" * 80)
print("STAGE 7: OFFICIAL SEALED TEST EVALUATION — KAGGLE GPU")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# --- Checkpoint SHA-256 ---
with open(CKPT_PATH, "rb") as f:
    ckpt_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Checkpoint Best SHA-256: {ckpt_hash}")
expected_ckpt = "06f66851349919b37a70602e4c1fab24b64375286dc7e946e2cdfb600ee7894f"
assert ckpt_hash == expected_ckpt, f"CHECKPOINT HASH MISMATCH: {ckpt_hash}"
print("Checkpoint hash VERIFIED.", flush=True)

# --- Tokenizer SHA-256 ---
with open(TOK16K_PATH, "rb") as f:
    tok_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
print(f"16K Tokenizer SHA-256: {tok_hash}", flush=True)

# --- Test vault bitwise verification against Stage 2 locks ---
with open(LOCK_PATH, "r", encoding="utf-8") as f:
    locks = json.load(f)

print("\nVerifying sealed test file hashes against Stage 2 locks...")
test_hashes = {}
for lang in LANGUAGES:
    fp = os.path.join(TEST_VAULT, f"test_{lang}.txt")
    with open(fp, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
        h = hashlib.sha256(data).hexdigest()
    expected = locks[lang]["sha256"]["test"]
    assert h == expected, f"HASH MISMATCH [{lang}]: {h} != {expected}"
    test_hashes[lang] = h
    print(f"  [{lang}] VERIFIED: {h[:16]}...", flush=True)
print("ALL 10/10 test files verified against Stage 2 sealed locks.\n")

# --- Model Architecture (identical to kaggle_stage7_pretraining.py) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(xq, xk, cos, sin):
    B, H, T, D = xq.shape
    cos = cos[:T].unsqueeze(0).unsqueeze(0).to(xq.device)
    sin = sin[:T].unsqueeze(0).unsqueeze(0).to(xq.device)
    xq_r, xq_i = xq[..., :D//2], xq[..., D//2:]
    xk_r, xk_i = xk[..., :D//2], xk[..., D//2:]
    return (torch.cat([xq_r*cos - xq_i*sin, xq_r*sin + xq_i*cos], dim=-1),
            torch.cat([xk_r*cos - xk_i*sin, xk_r*sin + xk_i*cos], dim=-1))

class SwiGLU(nn.Module):
    def __init__(self, in_f, hid_f):
        super().__init__()
        self.w1 = nn.Linear(in_f, hid_f, bias=False)
        self.w2 = nn.Linear(hid_f, in_f, bias=False)
        self.w3 = nn.Linear(in_f, hid_f, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, max_seq_len=512):
        super().__init__()
        self.H = H; self.dk = D // H
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
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))

class TransformerBlock(nn.Module):
    def __init__(self, D, H, max_seq_len=512):
        super().__init__()
        self.norm1 = RMSNorm(D); self.attn = CausalSelfAttention(D, H, max_seq_len)
        self.norm2 = RMSNorm(D); self.ffn = SwiGLU(D, int(2 * (4 * D) / 3))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class EBrainTransformer(nn.Module):
    def __init__(self, V=16000, D=256, H=8, L=6, max_seq_len=512):
        super().__init__()
        self.V = V; self.D = D
        self.tok_emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([TransformerBlock(D, H, max_seq_len) for _ in range(L)])
        self.norm_final = RMSNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight
    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm_final(x))

# --- Load model ---
tok16k = Tokenizer.from_file(TOK16K_PATH)
model = EBrainTransformer().to(device).eval()
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
print(f"Model loaded. Step: {ckpt.get('step', 22990)}", flush=True)

# --- Evaluate ---
T = 512
B = 32  # larger batch on GPU
results = {}

print(f"\n{'Lang':<6} | {'Bytes':>10} | {'Tokens':>10} | {'Chunks':>7} | {'Nats':>8} | {'Test BPB':>10} | {'Time':>6}")
print("-" * 72)

eval_start = time.time()

for lang in LANGUAGES:
    fp = os.path.join(TEST_VAULT, f"test_{lang}.txt")
    with open(fp, "rb") as f:
        raw_b = f.read()
    toks = tok16k.encode(raw_b.decode("utf-8", errors="ignore")).ids
    total_bytes = len(raw_b)
    num_chunks = (len(toks) - 1) // T
    tot_loss = 0.0
    tot_toks = 0
    t0 = time.time()

    with torch.inference_mode():
        for s in range(0, num_chunks, B):
            e = min(s + B, num_chunks)
            batch = [toks[i*T:(i+1)*T+1] for i in range(s, e)]
            bt = torch.tensor(batch, dtype=torch.long, device=device)
            x = bt[:, :-1].contiguous()
            y = bt[:, 1:].contiguous()
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1), reduction="sum")
            tot_loss += loss.item()
            tot_toks += y.numel()

    mean_nats = tot_loss / max(1, tot_toks)
    bpb = (mean_nats / math.log(2.0)) * (tot_toks / max(1, total_bytes))
    elapsed = time.time() - t0

    results[lang] = {
        "bytes": total_bytes, "tokens": len(toks), "chunks": num_chunks,
        "mean_nats": round(mean_nats, 4), "test_bpb": round(bpb, 4),
        "sha256": test_hashes[lang], "eval_seconds": round(elapsed, 2)
    }
    print(f"{lang:<6} | {total_bytes:>10,} | {len(toks):>10,} | {num_chunks:>7} | {mean_nats:>8.4f} | {bpb:>10.4f} | {elapsed:>5.1f}s", flush=True)

mean_bpb = sum(r["test_bpb"] for r in results.values()) / len(results)
total_time = time.time() - eval_start
print("-" * 72)
print(f"MEAN TEST BPB: {mean_bpb:.4f}  |  Total GPU time: {total_time:.1f}s")
print("=" * 80)

payload = {
    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "checkpoint_hash": ckpt_hash,
    "checkpoint_step": ckpt.get("step", 22990),
    "tokenizer_hash": tok_hash,
    "mean_test_bpb": round(mean_bpb, 4),
    "languages": results
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Saved → {OUT_JSON}", flush=True)
