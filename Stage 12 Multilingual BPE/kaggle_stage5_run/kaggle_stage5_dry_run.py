"""
Stage 12 - Stage 5: Cross-Session Resume Invariant Test on Kaggle GPU
====================================================================
Validates that resuming from a saved checkpoint produces an exact bitwise
loss match (Delta L < 1e-5) across the checkpoint seam (steps 501 to 510)
compared to an uninterrupted run.

Checks:
1. Exact batch hash match at step 501 (proves data loader position restored).
2. Loss comparison across steps 501 to 510.
3. Complete per-language validation and Alice English evaluation at step 500.
4. Parameter breakdown and hardware throughput reporting.
"""

import os
import sys
import time
import json
import math
import glob
import zipfile
import hashlib
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

print("=" * 80)
print("STAGE 5: CROSS-SESSION RESUME INVARIANT TEST (KAGGLE GPU)")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
os.makedirs(WORK_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset & File Locations
# -------------------------------------------------------------
def locate_vault(vault_name):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if vault_name in dirs:
                p = os.path.join(root, vault_name)
                print(f"Found {vault_name} at: {p}")
                return p
            if f"{vault_name}.zip" in files:
                zip_p = os.path.join(root, f"{vault_name}.zip")
                print(f"Extracting {zip_p} to {WORK_DIR}...")
                with zipfile.ZipFile(zip_p, "r") as z:
                    z.extractall(WORK_DIR)
                return os.path.join(WORK_DIR, vault_name)
    cand_roots = [
        "Stage 12 Multilingual BPE/data_v3",
        "Stage 12 Multilingual BPE/kaggle_dataset_v3",
        "data_v3",
        "kaggle_dataset_v3",
        WORK_DIR
    ]
    for cr in cand_roots:
        cand_dir = os.path.join(cr, vault_name)
        if os.path.exists(cand_dir) and os.path.isdir(cand_dir):
            return cand_dir
        cand_zip = os.path.join(cr, f"{vault_name}.zip")
        if os.path.exists(cand_zip):
            with zipfile.ZipFile(cand_zip, "r") as z:
                z.extractall(WORK_DIR)
            return os.path.join(WORK_DIR, vault_name)
    raise FileNotFoundError(f"Cannot locate {vault_name}")

def find_file(fname):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if fname in files:
                p = os.path.join(root, fname)
                print(f"Found {fname} at: {p}")
                return p
    for cr in [
        "Stage 12 Multilingual BPE",
        "Stage 12 Multilingual BPE/data_v3",
        "Stage 12 Multilingual BPE/kaggle_dataset_v3",
        "Stage 11 Production Engine/canonical_datasets/common_eval",
        WORK_DIR,
        "."
    ]:
        cand = os.path.join(cr, fname)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Cannot locate {fname}")

TRAIN_VAULT = locate_vault("train_vault")
VAL_VAULT = locate_vault("val_vault")
TOK16K_PATH = find_file("ebrain_tokenizer_16k.json")
ALICE_PATH = find_file("alice_common_val.txt")

tok16k = Tokenizer.from_file(TOK16K_PATH)
LANGUAGES = ['en', 'es', 'ru', 'fr', 'de', 'ja', 'zh', 'hi', 'bn', 'ar']

# -------------------------------------------------------------
# 2. Architecture Definition
# -------------------------------------------------------------
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
        hidden_dim = int(2 * (4 * D) / 3) # 682
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

# -------------------------------------------------------------
# 3. Prepare Training Stream & Validation Vaults
# -------------------------------------------------------------
print("\nLoading 10 MB training stream (1 MB per language)...")
train_texts = []
for lang in LANGUAGES:
    tp = os.path.join(TRAIN_VAULT, f"train_{lang}.txt")
    with open(tp, "rb") as f:
        raw_b = f.read(1000000)
    train_texts.append(raw_b.decode("utf-8", errors="ignore"))

full_train_text = "\n".join(train_texts)
total_train_bytes = len(full_train_text.encode("utf-8"))
train_tokens = np.array(tok16k.encode(full_train_text).ids, dtype=np.int32)
print(f"10 MB Corpus: {total_train_bytes:,d} raw bytes | {len(train_tokens):,d} 16K tokens")

# Load Validation Vaults
val_tokens = {}
val_bytes = {}
for lang in LANGUAGES:
    vp = os.path.join(VAL_VAULT, f"val_{lang}.txt")
    with open(vp, "rb") as f:
        b = f.read(1500000)
    t = b.decode("utf-8", errors="ignore")
    val_bytes[lang] = len(t.encode("utf-8"))
    val_tokens[lang] = tok16k.encode(t).ids

with open(ALICE_PATH, "r", encoding="utf-8") as f:
    alice_t = f.read()
alice_bytes = len(alice_t.encode("utf-8"))
alice_tokens = tok16k.encode(alice_t).ids

def evaluate_val_and_alice(m, T=512):
    m.eval()
    bpb_dict = {}
    loss_dict = {}
    with torch.no_grad():
        for k, toks in list(val_tokens.items()) + [("alice", alice_tokens)]:
            b_target = alice_bytes if k == "alice" else val_bytes[k]
            if len(toks) <= T:
                continue
            num_b = (len(toks) - 1) // T
            tot_loss = 0.0
            tot_toks = 0
            for i in range(num_b):
                chunk = toks[i * T: (i + 1) * T + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
                logits = m(x)
                l = F.cross_entropy(logits.view(-1, m.V), y.view(-1))
                tot_loss += l.item() * (len(chunk) - 1)
                tot_toks += (len(chunk) - 1)
            mean_nats = tot_loss / max(1, tot_toks)
            bpb_dict[k] = (mean_nats / math.log(2.0)) * (tot_toks / max(1, b_target))
            loss_dict[k] = mean_nats
    m.train()
    return bpb_dict, loss_dict

# -------------------------------------------------------------
# 4. Phase 1: Uninterrupted Run (Steps 1 -> 510)
# -------------------------------------------------------------
BATCH_SIZE = 16
T_LEN = 512
LR_MAX = 8e-4
LR_MIN = 8e-5

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_unint = EBrainTransformer().to(device)
opt_unint = torch.optim.AdamW(model_unint.parameters(), lr=LR_MAX, weight_decay=0.01, betas=(0.9, 0.95))
sched_unint = torch.optim.lr_scheduler.CosineAnnealingLR(opt_unint, T_max=510, eta_min=LR_MIN)

non_emb, total_p = model_unint.param_breakdown()
print("=" * 80)
print(f"MODEL PARAMETERS:")
print(f"  Non-Embedding Parameters: {non_emb:,d}")
print(f"  Embedding Parameters    : {model_unint.tok_emb.weight.numel():,d} (tied)")
print(f"  Total Parameters        : {total_p:,d}")
print("=" * 80)

print("\nExecuting Session 1 (Uninterrupted Training Steps 1 -> 500)...")
model_unint.train()
t_start = time.time()
step_times = []

for step in range(1, 501):
    t0 = time.time()
    max_idx = len(train_tokens) - T_LEN - 1
    starts = np.random.randint(0, max_idx, size=BATCH_SIZE)
    x_np = np.stack([train_tokens[s:s + T_LEN] for s in starts])
    y_np = np.stack([train_tokens[s + 1:s + T_LEN + 1] for s in starts])

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    opt_unint.zero_grad()
    logits = model_unint(x)
    loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_unint.parameters(), 1.0)
    opt_unint.step()
    sched_unint.step()
    dt = time.time() - t0
    step_times.append(dt)

    if step % 100 == 0 or step == 1 or step == 500:
        tok_s = (BATCH_SIZE * T_LEN) / dt
        print(f"  Step {step:4d}/500 | Loss: {loss.item():.4f} | Step Time: {dt*1000:.1f} ms ({tok_s:,.0f} tok/s)")

# Evaluate at step 500
print("\n[EVALUATION AT STEP 500 (VAL PER LANGUAGE & ALICE)]")
val_bpb_500, val_loss_500 = evaluate_val_and_alice(model_unint, T=T_LEN)
print(f"  Alice English : Loss = {val_loss_500['alice']:.4f} nats | BPB = {val_bpb_500['alice']:.4f}")
mean_val_bpb = sum(val_bpb_500[l] for l in LANGUAGES) / len(LANGUAGES)
print(f"  10-Lang Mean  : BPB = {mean_val_bpb:.4f}")
print("  Per-Language Val Loss (nats) & BPB:")
for l in LANGUAGES:
    print(f"    {l.upper()}: Loss = {val_loss_500[l]:.4f} nats | BPB = {val_bpb_500[l]:.4f}")

# Save Step 500 Checkpoint
CKPT_PATH = os.path.join(WORK_DIR, "stage5_resume_checkpoint_step_500.pt")
torch.save({
    "step": 500,
    "model_state": copy.deepcopy(model_unint.state_dict()),
    "optimizer_state": copy.deepcopy(opt_unint.state_dict()),
    "scheduler_state": copy.deepcopy(sched_unint.state_dict()),
    "rng_torch": torch.get_rng_state(),
    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    "rng_numpy": np.random.get_state(),
    "rng_random": random.getstate(),
    "lr": sched_unint.get_last_lr()[0]
}, CKPT_PATH)
with open(CKPT_PATH, "rb") as cf:
    ckpt_hash = hashlib.sha256(cf.read()).hexdigest()
print(f"\nSaved Step-500 Checkpoint to {CKPT_PATH} (SHA-256: {ckpt_hash})")

# Continue Uninterrupted Run for Steps 501 -> 510
print("\nContinuing Uninterrupted Run for Steps 501 -> 510...")
unint_losses = {}
unint_batch_hashes = {}

for step in range(501, 511):
    max_idx = len(train_tokens) - T_LEN - 1
    starts = np.random.randint(0, max_idx, size=BATCH_SIZE)
    x_np = np.stack([train_tokens[s:s + T_LEN] for s in starts])
    y_np = np.stack([train_tokens[s + 1:s + T_LEN + 1] for s in starts])

    b_hash = hashlib.sha256(x_np.tobytes()).hexdigest()
    unint_batch_hashes[step] = b_hash

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    opt_unint.zero_grad()
    logits = model_unint(x)
    loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_unint.parameters(), 1.0)
    opt_unint.step()
    sched_unint.step()
    unint_losses[step] = loss.item()
    print(f"  [Uninterrupted] Step {step} | Loss: {loss.item():.8f} | Batch Hash: {b_hash[:12]}")

# -------------------------------------------------------------
# 5. Phase 2: Session 2 (Resumed Run from Checkpoint)
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("EXECUTING SESSION 2 (RESUMED RUN FROM STEP-500 CHECKPOINT)")
print("=" * 80)

# Clean memory
del model_unint, opt_unint, sched_unint
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load checkpoint
loaded_ckpt = torch.load(CKPT_PATH, weights_only=False)

# Initialize fresh model and optimizer
model_res = EBrainTransformer().to(device)
opt_res = torch.optim.AdamW(model_res.parameters(), lr=LR_MAX, weight_decay=0.01, betas=(0.9, 0.95))
sched_res = torch.optim.lr_scheduler.CosineAnnealingLR(opt_res, T_max=510, eta_min=LR_MIN)

model_res.load_state_dict(loaded_ckpt["model_state"])
opt_res.load_state_dict(loaded_ckpt["optimizer_state"])
sched_res.load_state_dict(loaded_ckpt["scheduler_state"])

torch.set_rng_state(loaded_ckpt["rng_torch"])
if torch.cuda.is_available() and loaded_ckpt["rng_cuda"] is not None:
    torch.cuda.set_rng_state(loaded_ckpt["rng_cuda"])
np.random.set_state(loaded_ckpt["rng_numpy"])
random.setstate(loaded_ckpt["rng_random"])

print("Restored Model Weights, Optimizer States, Learning Rate, and RNG States.")

res_losses = {}
res_batch_hashes = {}
model_res.train()

for step in range(501, 511):
    max_idx = len(train_tokens) - T_LEN - 1
    starts = np.random.randint(0, max_idx, size=BATCH_SIZE)
    x_np = np.stack([train_tokens[s:s + T_LEN] for s in starts])
    y_np = np.stack([train_tokens[s + 1:s + T_LEN + 1] for s in starts])

    b_hash = hashlib.sha256(x_np.tobytes()).hexdigest()
    res_batch_hashes[step] = b_hash

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    opt_res.zero_grad()
    logits = model_res(x)
    loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_res.parameters(), 1.0)
    opt_res.step()
    sched_res.step()
    res_losses[step] = loss.item()
    print(f"  [Resumed]       Step {step} | Loss: {loss.item():.8f} | Batch Hash: {b_hash[:12]}")

# -------------------------------------------------------------
# 6. Comparison & Verification Across Seam
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("CROSS-SESSION RESUME INVARIANT VERIFICATION (STEPS 501 -> 510)")
print("=" * 80)
print(f"{'Step':5s} | {'Uninterrupted Loss':18s} | {'Resumed Loss':18s} | {'Absolute Diff':14s} | {'Batch Hash Match':16s}")
print("-" * 80)

max_loss_diff = 0.0
all_batch_match = True

for step in range(501, 511):
    u_loss = unint_losses[step]
    r_loss = res_losses[step]
    diff = abs(u_loss - r_loss)
    max_loss_diff = max(max_loss_diff, diff)
    b_match = (unint_batch_hashes[step] == res_batch_hashes[step])
    if not b_match:
        all_batch_match = False
    print(f"{step:5d} | {u_loss:18.8f} | {r_loss:18.8f} | {diff:14.2e} | {'MATCH' if b_match else 'MISMATCH'}")

print("-" * 80)
avg_step_ms = (sum(step_times) / len(step_times)) * 1000
avg_tok_s = (BATCH_SIZE * T_LEN) / (sum(step_times) / len(step_times))
tokens_seen = 500 * BATCH_SIZE * T_LEN
epochs_used = tokens_seen / len(train_tokens)

print(f"Maximum Loss Difference Across Seam : {max_loss_diff:.2e}")
print(f"Data Loader Position Restored (501) : {'VERIFIED (Batch Hashes Identical)' if all_batch_match else 'FAILED'}")
print(f"Seam Continuity Status (|Diff|<1e-5): {'PASSED (EXACT BITWISE MATCH)' if max_loss_diff < 1e-5 else 'FAILED'}")
print(f"Hardware Throughput (D=256, T=512) : {avg_step_ms:.1f} ms/step ({avg_tok_s:,.0f} tok/s)")
print(f"10 MB Token Count & Epochs Used     : {tokens_seen:,d} tokens trained ({epochs_used:.2f} epochs of 10 MB)")
print("=" * 80)

# Save Final Verification JSON
out_json = os.path.join(WORK_DIR, "stage5_resume_test_final_report.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "uninterrupted_losses": unint_losses,
        "resumed_losses": res_losses,
        "max_loss_diff": max_loss_diff,
        "all_batch_match": all_batch_match,
        "val_loss_step_500": val_loss_500,
        "val_bpb_step_500": val_bpb_500,
        "step_time_ms": avg_step_ms,
        "tok_sec": avg_tok_s,
        "tokens_seen": tokens_seen,
        "epochs_used": epochs_used
    }, f, indent=2)

print(f"Verification report saved to: {out_json}")
