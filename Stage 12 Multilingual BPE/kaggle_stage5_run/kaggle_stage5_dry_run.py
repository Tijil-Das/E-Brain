"""
Stage 12 - Stage 5: 10 MB Modern Multilingual Transformer Dry Run (Kaggle GPU)
================================================================================
Architecture:
- V=16,000 (Adopted 16K BPE, SHA-256: 41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf)
- D=256, H=8, L=6, T=512 (RoPE, SwiGLU, RMSNorm, Weight Tying)
- Parameters:
  - Non-Embedding: 4,718,848
  - Embedding (Tied): 4,096,000
  - Total: 8,814,848 (~8.81M)

Chinchilla Scaling Reference:
- 20 tokens/param = ~176M tokens.
- Certified 250 MB corpus = ~80M tokens -> ~2.2 epochs full pretraining.

10 MB Dry Run:
- 1 MB per language across all 10 languages (~3.2M tokens total).
- 500 steps (B=16, T=512 = 8,192 tok/step = ~4.1M tokens total).
- Checkpoints at step 250 and 500.
- Tracks Alice English and 10-language validation BPB at each checkpoint.
- Saves full state dicts (model, optimizer, scaler, RNG, step, LR, data_pos).
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

print("=" * 80)
print("STAGE 5: 10 MB MODERN MULTILINGUAL TRANSFORMER DRY RUN (KAGGLE GPU)")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: CUDA not detected. Kaggle GPU accelerator recommended.")

# -------------------------------------------------------------
# 1. Dataset & File Locations (Recursive Search)
# -------------------------------------------------------------
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
os.makedirs(WORK_DIR, exist_ok=True)

if os.path.exists("/kaggle/input"):
    print("Inspecting /kaggle/input:", os.listdir("/kaggle/input"))
    for item in os.listdir("/kaggle/input"):
        item_p = os.path.join("/kaggle/input", item)
        if os.path.isdir(item_p):
            print(f"  Folder {item}: {os.listdir(item_p)[:8]}")

def locate_vault(vault_name):
    # 1. Search recursively in /kaggle/input
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if vault_name in dirs:
                p = os.path.join(root, vault_name)
                print(f"Found {vault_name} directory at: {p}")
                return p
            if f"{vault_name}.zip" in files:
                zip_p = os.path.join(root, f"{vault_name}.zip")
                print(f"Found {vault_name}.zip at: {zip_p}, unzipping to {WORK_DIR}...")
                with zipfile.ZipFile(zip_p, "r") as z:
                    z.extractall(WORK_DIR)
                return os.path.join(WORK_DIR, vault_name)
    # 2. Search local candidate locations
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
    raise FileNotFoundError(f"Cannot locate {vault_name} in /kaggle/input or local paths.")

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

with open(TOK16K_PATH, "rb") as f:
    tok16k_hash = hashlib.sha256(f.read()).hexdigest()
print(f"16K Tokenizer Path: {TOK16K_PATH} (SHA-256: {tok16k_hash})")
tok16k = Tokenizer.from_file(TOK16K_PATH)

LANGUAGES = ['en', 'es', 'ru', 'fr', 'de', 'ja', 'zh', 'hi', 'bn', 'ar']

# -------------------------------------------------------------
# 2. Modern Transformer Architecture (D=256, H=8, L=6)
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

model = EBrainTransformer().to(device)
non_emb, total_p = model.param_breakdown()
print("=" * 80)
print(f"STAGE 5 MODEL ARCHITECTURE (V=16,000, D=256, H=8, L=6, T=512):")
print(f"  Non-Embedding Parameters: {non_emb:,d} (6 layers x 786k + final norm)")
print(f"  Embedding Parameters    : {model.tok_emb.weight.numel():,d} (16,000 x 256, tied)")
print(f"  Total Model Parameters  : {total_p:,d} (~8.81M)")
print("=" * 80)

# -------------------------------------------------------------
# 3. Prepare 10 MB Dry Run Stream & Equalized Val
# -------------------------------------------------------------
print("\nPreparing 10 MB Dry Run Training Stream (1 MB per language)...")
train_texts = []
for lang in LANGUAGES:
    tp = os.path.join(TRAIN_VAULT, f"train_{lang}.txt")
    with open(tp, "rb") as f:
        raw_b = f.read(1000000) # exactly 1 MB
    train_texts.append(raw_b.decode("utf-8", errors="ignore"))

full_train_text = "\n".join(train_texts)
total_train_bytes = len(full_train_text.encode("utf-8"))
train_tokens = np.array(tok16k.encode(full_train_text).ids, dtype=np.int32)
print(f"10 MB Dry Run: {total_train_bytes:,d} raw bytes | {len(train_tokens):,d} 16K tokens")

# Validation streams (1.5 MB equalized per language)
print("Loading validation vaults...")
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
print(f"Alice English Validation: {alice_bytes:,d} bytes | {len(alice_tokens):,d} 16K tokens")

def evaluate_model(m, tokens_dict, bytes_dict, T=512):
    m.eval()
    bpb_dict = {}
    with torch.no_grad():
        for k, toks in tokens_dict.items():
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
                loss = F.cross_entropy(logits.view(-1, m.V), y.view(-1))
                tot_loss += loss.item() * (len(chunk) - 1)
                tot_toks += (len(chunk) - 1)
            mean_nats = tot_loss / max(1, tot_toks)
            raw_b = bytes_dict[k]
            bpb_dict[k] = (mean_nats / math.log(2.0)) * (tot_toks / max(1, raw_b))
    return bpb_dict

# -------------------------------------------------------------
# 4. Training Loop with Periodic State Checkpointing
# -------------------------------------------------------------
TOTAL_STEPS = 500
BATCH_SIZE = 16
T_LEN = 512
LR_MAX = 8e-4
LR_MIN = 8e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=LR_MIN)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

print(f"\nStarting 10 MB Dry Run Training ({TOTAL_STEPS} steps, B={BATCH_SIZE}, T={T_LEN})...")
model.train()
t0 = time.time()
training_log = []
eval_checkpoints = {}

for step in range(1, TOTAL_STEPS + 1):
    step_t0 = time.time()
    max_idx = len(train_tokens) - T_LEN - 1
    starts = np.random.randint(0, max_idx, size=BATCH_SIZE)
    x_np = np.stack([train_tokens[s:s + T_LEN] for s in starts])
    y_np = np.stack([train_tokens[s + 1:s + T_LEN + 1] for s in starts])

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 16000), y.view(-1))

    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError(f"CRITICAL: NaN/Inf detected at step {step}!")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    step_time_ms = (time.time() - step_t0) * 1000
    tok_per_sec = (BATCH_SIZE * T_LEN) / max(1e-5, (time.time() - step_t0))

    if step % 50 == 0 or step == 1 or step == TOTAL_STEPS:
        curr_lr = scheduler.get_last_lr()[0]
        print(f"  Step {step:4d}/{TOTAL_STEPS} | Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | Step Time: {step_time_ms:.1f} ms ({tok_per_sec:,.0f} tok/s)")
        training_log.append({
            "step": step,
            "loss": loss.item(),
            "lr": curr_lr,
            "step_time_ms": step_time_ms,
            "tok_sec": tok_per_sec
        })

    # Checkpoint & Validation Evaluation
    if step % 250 == 0 or step == TOTAL_STEPS:
        print(f"\n[CHECKPOINT EVALUATION AT STEP {step}]")
        val_bpb = evaluate_model(model, val_tokens, val_bytes, T=T_LEN)
        alice_eval = evaluate_model(model, {"alice": alice_tokens}, {"alice": alice_bytes}, T=T_LEN)
        alice_bpb = alice_eval["alice"]
        mean_val_bpb = sum(val_bpb.values()) / len(val_bpb)

        print(f"  Alice English BPB       : {alice_bpb:.4f}")
        print(f"  10-Lang Mean Val BPB   : {mean_val_bpb:.4f}")
        for l in LANGUAGES:
            print(f"    {l.upper()}: {val_bpb[l]:.4f} bpb")

        eval_checkpoints[f"step_{step}"] = {
            "alice_bpb": alice_bpb,
            "mean_val_bpb": mean_val_bpb,
            "per_lang_bpb": val_bpb
        }

        # Save Checkpoint Artifact
        ckpt_path = os.path.join(WORK_DIR, f"stage5_checkpoint_step_{step}.pt")
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_random": random.getstate(),
            "lr": scheduler.get_last_lr()[0],
            "total_params": total_p,
            "non_emb_params": non_emb
        }, ckpt_path)

        with open(ckpt_path, "rb") as cf:
            ckpt_hash = hashlib.sha256(cf.read()).hexdigest()
        print(f"  Saved Checkpoint: {ckpt_path} (Size: {os.path.getsize(ckpt_path)/(1024*1024):.1f} MB | SHA-256: {ckpt_hash})")
        model.train()

# -------------------------------------------------------------
# 5. Summary & Stage 5 Dry Run Report
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("STAGE 5 DRY RUN RESULTS SUMMARY")
print("=" * 80)
loss_init = training_log[0]["loss"]
loss_final = training_log[-1]["loss"]
print(f"Initial Step-1 Loss : {loss_init:.4f}")
print(f"Final Step-500 Loss : {loss_final:.4f} (Delta: {loss_init - loss_final:.4f})")

# Alice Trend
alice_250 = eval_checkpoints["step_250"]["alice_bpb"]
alice_500 = eval_checkpoints["step_500"]["alice_bpb"]
print(f"Alice BPB Progression : Step 250 = {alice_250:.4f} -> Step 500 = {alice_500:.4f} (Delta: {alice_250 - alice_500:.4f})")
alice_falling = (alice_500 < alice_250)
print(f"Alice BPB Monotonically Falling: {'YES (Undertraining confirmed)' if alice_falling else 'FLAT/RISING (Domain shift / potential mismatch)'}")

out_report = os.path.join(WORK_DIR, "stage5_dry_run_report.json")
with open(out_report, "w", encoding="utf-8") as f:
    json.dump({
        "model_architecture": {
            "V": 16000, "D": 256, "H": 8, "L": 6, "T": 512,
            "non_emb_params": non_emb, "total_params": total_p
        },
        "training_log": training_log,
        "eval_checkpoints": eval_checkpoints,
        "loss_progression": {"init": loss_init, "final": loss_final},
        "alice_progression": {"step_250": alice_250, "step_500": alice_500, "falling": alice_falling}
    }, f, indent=2)

print(f"Dry run report written to: {out_report}")
print("=" * 80)
