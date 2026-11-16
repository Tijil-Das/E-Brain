"""
Stage 12 - Stage 6a: 100 MB Multilingual Pretraining Dress Rehearsal (Kaggle GPU)
===================================================================================
Standards & Pass Criteria:
1. Architecture: Scaled Modern Transformer (V=16,000, D=256, H=8, L=6, T=512)
   - Parameters: Non-Embedding = 4,718,848 | Tied Embedding = 4,096,000 | Total = 8,814,848
   - RoPE, SwiGLU (dim=682, exactly 12*D^2 = 786,432 per block), RMSNorm, Weight Tying.
2. Data Budget: 100 MB certified multilingual slice (10 MB per language across all 10 languages).
   - ~32.1M tokens total at 16K BPE compression.
3. Training & Throughput:
   - AMP FP16 enabled with GradScaler.
   - Pure GPU forward/backward step time profiled separately from eval and checkpointing.
4. Monitoring & Early Stopping:
   - Checkpoints every 1,000 steps (Steps 1000, 2000, 3000, 4000, 5000).
   - Evaluates all 10 equalized 1.5 MB validation vaults + Alice English at each checkpoint.
   - Tracks Train Loss, Val BPB per language, Mean Val BPB, Alice BPB, and Generalization Gap.
   - Early stops if mean validation BPB turns upward (overfitting beyond 1 epoch).
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
print("STAGE 6a: 100 MB MULTILINGUAL PRETRAINING DRESS REHEARSAL (KAGGLE GPU)")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: CUDA not detected. Kaggle GPU accelerator recommended.")

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
os.makedirs(WORK_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Recursive Dataset & File Locations
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
with open(TOK16K_PATH, "rb") as f:
    tok_hash = hashlib.sha256(f.read()).hexdigest()
print(f"16K Tokenizer SHA-256: {tok_hash}")

LANGUAGES = ['en', 'es', 'ru', 'fr', 'de', 'ja', 'zh', 'hi', 'bn', 'ar']

# -------------------------------------------------------------
# 2. Modern Transformer Architecture (D=256, H=8, L=6, T=512)
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
        hidden_dim = int(2 * (4 * D) / 3) # 682 -> exactly 12*D^2 params per block
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
print(f"STAGE 6a MODEL PARAMETERS:")
print(f"  Non-Embedding Parameters: {non_emb:,d} (12*D^2 per block = 786,432 * 6 + 256)")
print(f"  Tied Embedding Parameters: {model.tok_emb.weight.numel():,d} (16,000 * 256)")
print(f"  Total Model Parameters  : {total_p:,d} (~8.81M)")
print("=" * 80)

# -------------------------------------------------------------
# 3. Load 100 MB Pretraining Slice & Validation Vaults
# -------------------------------------------------------------
print("\nLoading 100 MB Multilingual Training Stream (10 MB per language)...")
train_texts = []
for lang in LANGUAGES:
    tp = os.path.join(TRAIN_VAULT, f"train_{lang}.txt")
    with open(tp, "rb") as f:
        raw_b = f.read(10000000) # exactly 10 MB per language
    train_texts.append(raw_b.decode("utf-8", errors="ignore"))

full_train_text = "\n".join(train_texts)
total_train_bytes = len(full_train_text.encode("utf-8"))
print(f"Tokenizing {total_train_bytes:,d} raw bytes with 16K BPE...")
train_tokens = np.array(tok16k.encode(full_train_text).ids, dtype=np.int32)
print(f"100 MB Corpus: {total_train_bytes:,d} bytes | {len(train_tokens):,d} 16K tokens ({total_train_bytes/len(train_tokens):.2f} B/tok)")

# Validation streams (1.5 MB equalized per language)
print("\nLoading validation vaults (1.5 MB per language)...")
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
print(f"Alice English Validation: {alice_bytes:,d} bytes | {len(alice_tokens):,d} tokens")

def evaluate_all(m, T=512):
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
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
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
# 4. Training Engine: Stage 6a (AMP FP16, Separated Throughput)
# -------------------------------------------------------------
TOTAL_STEPS = 5000
CHECKPOINT_INTERVAL = 1000
BATCH_SIZE = 16
T_LEN = 512
LR_MAX = 8e-4
LR_MIN = 8e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=LR_MIN)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

print(f"\nStarting Stage 6a Pretraining ({TOTAL_STEPS} steps, B={BATCH_SIZE}, T={T_LEN})...")
print(f"Total tokens per step: {BATCH_SIZE * T_LEN:,d}")
print(f"Evaluation cadence: every {CHECKPOINT_INTERVAL} steps\n")

model.train()
t_total_start = time.time()
pure_gpu_times = []
checkpoints_log = []
rolling_train_loss = 0.0
best_mean_bpb = float("inf")

for step in range(1, TOTAL_STEPS + 1):
    # Data sampling on CPU
    max_idx = len(train_tokens) - T_LEN - 1
    starts = np.random.randint(0, max_idx, size=BATCH_SIZE)
    x_np = np.stack([train_tokens[s:s + T_LEN] for s in starts])
    y_np = np.stack([train_tokens[s + 1:s + T_LEN + 1] for s in starts])

    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    # Pure GPU forward + backward + step timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step_start = time.time()

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
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

    if device.type == "cuda":
        torch.cuda.synchronize()
    step_duration = time.time() - t_step_start
    pure_gpu_times.append(step_duration)
    rolling_train_loss += loss.item()

    if step % 100 == 0 or step == 1:
        avg_gpu_ms = (sum(pure_gpu_times[-100:]) / len(pure_gpu_times[-100:])) * 1000
        pure_tok_s = (BATCH_SIZE * T_LEN) / (avg_gpu_ms / 1000)
        curr_lr = scheduler.get_last_lr()[0]
        avg_loss = rolling_train_loss / (100 if step >= 100 else 1)
        print(f"  Step {step:4d}/{TOTAL_STEPS} | Train Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | GPU Step: {avg_gpu_ms:.1f} ms ({pure_tok_s:,.0f} tok/s)")
        rolling_train_loss = 0.0

    # Checkpoint Evaluation
    if step % CHECKPOINT_INTERVAL == 0 or step == TOTAL_STEPS:
        t_eval_start = time.time()
        print(f"\n{'='*75}")
        print(f"CHECKPOINT EVALUATION AT STEP {step:d} / {TOTAL_STEPS:d}")
        print(f"{'='*75}")

        val_bpb, val_loss = evaluate_all(model, T=T_LEN)
        eval_duration = time.time() - t_eval_start

        mean_val_bpb = sum(val_bpb[l] for l in LANGUAGES) / len(LANGUAGES)
        mean_val_loss = sum(val_loss[l] for l in LANGUAGES) / len(LANGUAGES)
        alice_bpb = val_bpb["alice"]

        # Approximate train BPB for generalization gap
        # train_bpb = (train_loss_nats / ln 2) * (tokens / bytes)
        avg_b_per_tok = total_train_bytes / len(train_tokens)
        train_bpb = (loss.item() / math.log(2.0)) * (1.0 / avg_b_per_tok)
        val_train_gap = mean_val_bpb - train_bpb

        # Throughput summary
        recent_gpu_ms = (sum(pure_gpu_times[-CHECKPOINT_INTERVAL:]) / len(pure_gpu_times[-CHECKPOINT_INTERVAL:])) * 1000
        recent_tok_s = (BATCH_SIZE * T_LEN) / (recent_gpu_ms / 1000)
        tokens_trained_so_far = step * BATCH_SIZE * T_LEN
        epochs_so_far = tokens_trained_so_far / len(train_tokens)

        print(f"  Training Progress  : {tokens_trained_so_far:,d} tokens trained ({epochs_so_far:.2f} epochs of 100 MB)")
        print(f"  Throughput (Pure)  : {recent_gpu_ms:.1f} ms/step ({recent_tok_s:,.0f} tok/sec) [Eval took: {eval_duration:.1f} s]")
        print(f"  Train Loss (nats)  : {loss.item():.4f} (Train BPB: {train_bpb:.4f})")
        print(f"  Mean Val BPB       : {mean_val_bpb:.4f}")
        print(f"  Val/Train Gap      : {val_train_gap:+.4f} BPB")
        print(f"  Alice English BPB  : {alice_bpb:.4f}")
        print("\n  Per-Language Val BPB:")
        for l in LANGUAGES:
            print(f"    {l.upper():2s}: {val_bpb[l]:.4f} bpb (Loss: {val_loss[l]:.4f} nats)")

        ckpt_info = {
            "step": step,
            "train_loss": loss.item(),
            "train_bpb": train_bpb,
            "mean_val_bpb": mean_val_bpb,
            "val_train_gap": val_train_gap,
            "alice_bpb": alice_bpb,
            "gpu_ms_per_step": recent_gpu_ms,
            "pure_tokens_per_sec": recent_tok_s,
            "eval_time_sec": eval_duration,
            "tokens_trained": tokens_trained_so_far,
            "epochs_trained": epochs_so_far,
            "per_lang_bpb": val_bpb,
            "per_lang_loss": val_loss
        }
        checkpoints_log.append(ckpt_info)

        # Save Checkpoint
        ckpt_fname = os.path.join(WORK_DIR, f"stage6a_checkpoint_step_{step}.pt")
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "lr": scheduler.get_last_lr()[0],
            "total_params": total_p,
            "non_emb_params": non_emb
        }, ckpt_fname)
        print(f"  Checkpoint saved: {ckpt_fname}")

        # Early stopping check: if val turns upward significantly
        if mean_val_bpb < best_mean_bpb:
            best_mean_bpb = mean_val_bpb
        elif mean_val_bpb > best_mean_bpb + 0.05 and step >= 3000:
            print(f"\n[EARLY STOPPING TRIGGERED] Mean Val BPB turned upward: {mean_val_bpb:.4f} > Best {best_mean_bpb:.4f} + 0.05")
            break

        print(f"{'='*75}\n")

# -------------------------------------------------------------
# 5. Save Final Stage 6a Report
# -------------------------------------------------------------
final_report_path = os.path.join(WORK_DIR, "stage6a_100mb_results.json")
with open(final_report_path, "w", encoding="utf-8") as f:
    json.dump({
        "stage": "Stage 6a: 100 MB Pretraining Dress Rehearsal",
        "parameters": {
            "non_embedding": non_emb,
            "tied_embedding": model.tok_emb.weight.numel(),
            "total": total_p,
            "mlp_hidden_dim": int(2 * (4 * 256) / 3),
            "mlp_budget_formula": "12 * D^2"
        },
        "checkpoints": checkpoints_log,
        "final_summary": {
            "best_mean_val_bpb": best_mean_bpb,
            "final_val_train_gap": checkpoints_log[-1]["val_train_gap"],
            "final_alice_bpb": checkpoints_log[-1]["alice_bpb"],
            "pure_gpu_throughput": checkpoints_log[-1]["pure_tokens_per_sec"]
        }
    }, f, indent=2)

print(f"\nStage 6a Final Report written to: {final_report_path}")
print("STAGE 6a COMPLETE.")
print("=" * 80)
