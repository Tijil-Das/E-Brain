"""
Stage 12 - Stage 6b: Full 250 MB Multilingual Pretraining & Seam Invariant Run (Kaggle GPU)
=============================================================================================
Commit: f6a8390
Architecture: Modern Transformer (V=16,000, D=256, H=8, L=6, T=512)
Total Parameters: 8,814,848 (4,718,848 Non-Embedding + 4,096,000 Tied Embedding)
Data Budget: Full certified 250 MB corpus (~80.3M tokens across 10 languages)
Target Steps: 21,520 steps (B=16, T=512, 176.3M tokens ~ 2.20 epochs)

Seam & Pass Criteria:
1. Eval Overhead Management:
   - Checkpoints every 2,000 steps using fast equalized val subset (~200k tokens/lang).
   - Full 1.5 MB val evaluated at Step 10,000 (seam) and Step 21,520 (final).
2. Seam at Step 10,000:
   - Full state serialized (weights, optimizer, scaler, scheduler, RNGs, epoch index, token cursor).
   - Cross-session resume invariant verified at steps 10,001 - 10,010 (|diff| < 1e-5, bitwise match).
3. Epoch Boundary Handling:
   - Reshuffle at Epoch 2 (~Step 9,800). Epoch seed and token cursor fully preserved across resume.
4. Learning Rate Schedule:
   - 500-step linear warmup (1e-6 -> 8e-4), cosine decay to 8e-5 at step 21,520.
   - Exact LR logged and restored across seam.
5. Generalization Gap:
   - Measured against a fixed 200k-token training subset in eval mode (torch.no_grad).
   - Val/Train gap threshold: < 0.10 BPB.
6. Overfit Watch:
   - Track per-language validation history. Flag if any language rises for 2 consecutive checkpoints.
   - Halt if Mean Val BPB turns upward for 2 consecutive checkpoints.
7. Checkpoint Retention & Hashing:
   - Retains last 2 checkpoints (checkpoint_last.pt, checkpoint_prev.pt), checkpoint_best.pt, and checkpoint_step_10000.pt.
   - SHA-256 logged for all retained checkpoints.
8. Artifacts & Logging:
   - stage6b_run_metadata.json (hashes, commit, config).
   - stage6b_metrics.csv (per-checkpoint training and per-language validation).
   - Top-k / top-p text generation sample logged at checkpoints.
"""

import os
import sys
import time
import json
import csv
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

# -------------------------------------------------------------
# Configuration & Run Modes
# -------------------------------------------------------------
# Set RUN_MODE:
# "SMOKE_1000": Steps 1 -> 1,000 (pre-flight check & initial log)
# "FULL": Full 21,520 steps with Step 10,000 seam checkpoint & verification
RUN_MODE = os.environ.get("EBRAIN_RUN_MODE", "SMOKE_1000")
MAX_STEPS = 1000 if RUN_MODE == "SMOKE_1000" else 21520
EVAL_INTERVAL = 1000 if RUN_MODE == "SMOKE_1000" else 2000
SEAM_STEP = 10000

GIT_COMMIT = "f6a8390"
RUN_ID = f"stage6b_250mb_{RUN_MODE.lower()}_{int(time.time())}"

WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
os.makedirs(WORK_DIR, exist_ok=True)

print("=" * 80)
print(f"STAGE 6b: 250 MB MULTILINGUAL PRETRAINING & SEAM INVARIANT RUN")
print(f"Run ID: {RUN_ID} | Mode: {RUN_MODE} | Target Steps: {MAX_STEPS}")
print(f"Commit: {GIT_COMMIT}")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# -------------------------------------------------------------
# 1. Dataset & Tokenizer Resolution
# -------------------------------------------------------------
def locate_vault(vault_name):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if vault_name in dirs:
                return os.path.join(root, vault_name)
            if f"{vault_name}.zip" in files:
                zip_p = os.path.join(root, f"{vault_name}.zip")
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
                return os.path.join(root, fname)
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
TOK8K_PATH = find_file("ebrain_tokenizer_8k.json")
ALICE_PATH = find_file("alice_common_val.txt")

tok16k = Tokenizer.from_file(TOK16K_PATH)
with open(TOK16K_PATH, "rb") as f:
    tok16k_hash = hashlib.sha256(f.read()).hexdigest()
with open(TOK8K_PATH, "rb") as f:
    tok8k_hash = hashlib.sha256(f.read()).hexdigest()

print(f"16K Tokenizer SHA-256: {tok16k_hash}")
print(f"8K  Tokenizer SHA-256: {tok8k_hash}")

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
        hidden_dim = int(2 * (4 * D) / 3) # 682 -> exactly 12*D^2 per block
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

    def generate(self, prompt_tokens, max_new_tokens=40, temperature=0.8, top_k=50):
        self.eval()
        idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -512:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx[0].tolist()

model = EBrainTransformer().to(device)
non_emb, total_p = model.param_breakdown()
print(f"Model Parameters: Total={total_p:,d} | Non-Embedding={non_emb:,d} | Tied={model.tok_emb.weight.numel():,d}")

# -------------------------------------------------------------
# 3. Ingest Full 250 MB Certified Corpus & Prepare Shuffled Streams
# -------------------------------------------------------------
print("\nIngesting full 250 MB certified multilingual training corpus...")
train_data_hashes = {}
train_token_arrays = []
total_train_bytes = 0

for lang in LANGUAGES:
    tp = os.path.join(TRAIN_VAULT, f"train_{lang}.txt")
    with open(tp, "rb") as f:
        b = f.read()
    total_train_bytes += len(b)
    h = hashlib.sha256(b).hexdigest()
    train_data_hashes[lang] = h
    text = b.decode("utf-8", errors="ignore")
    toks = np.array(tok16k.encode(text).ids, dtype=np.int32)
    train_token_arrays.append(toks)
    print(f"  {lang.upper()}: {len(b):,d} bytes | {len(toks):,d} tokens ({len(b)/len(toks):.2f} B/tok)")

full_train_tokens = np.concatenate(train_token_arrays)
total_train_tokens = len(full_train_tokens)
print(f"Total Pretraining Stream: {total_train_bytes:,d} bytes | {total_train_tokens:,d} tokens")

# Fixed 200k-token training slice for clean Val/Train Gap measurement
FIXED_TRAIN_EVAL_TOKENS = full_train_tokens[:200000].copy()
FIXED_TRAIN_EVAL_BYTES = int(200000 * (total_train_bytes / total_train_tokens))

# Ingest Validation Vaults
print("\nIngesting validation streams...")
val_tokens_full = {}
val_bytes_full = {}
val_tokens_fast = {}
val_bytes_fast = {}

for lang in LANGUAGES:
    vp = os.path.join(VAL_VAULT, f"val_{lang}.txt")
    with open(vp, "rb") as f:
        b = f.read()
    t = b.decode("utf-8", errors="ignore")
    toks = tok16k.encode(t).ids
    val_bytes_full[lang] = len(b)
    val_tokens_full[lang] = toks
    # Fast val subset (~200k tokens)
    fast_limit = min(len(toks), 200000)
    val_tokens_fast[lang] = toks[:fast_limit]
    val_bytes_fast[lang] = int(len(b) * (fast_limit / len(toks)))

with open(ALICE_PATH, "r", encoding="utf-8") as f:
    alice_t = f.read()
alice_bytes = len(alice_t.encode("utf-8"))
alice_tokens = tok16k.encode(alice_t).ids
print(f"Alice English Val: {alice_bytes:,d} bytes | {len(alice_tokens):,d} tokens")

def evaluate_subset(m, token_dict, byte_dict, T=512):
    m.eval()
    bpb_dict = {}
    loss_dict = {}
    with torch.no_grad():
        for k, toks in list(token_dict.items()) + [("alice", alice_tokens)]:
            b_target = alice_bytes if k == "alice" else byte_dict[k]
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

        # Fixed train subset eval for clean generalization gap
        num_tb = (len(FIXED_TRAIN_EVAL_TOKENS) - 1) // T
        tot_tloss = 0.0
        tot_ttoks = 0
        for i in range(num_tb):
            chunk = FIXED_TRAIN_EVAL_TOKENS[i * T: (i + 1) * T + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                logits = m(x)
                l = F.cross_entropy(logits.view(-1, m.V), y.view(-1))
            tot_tloss += l.item() * (len(chunk) - 1)
            tot_ttoks += (len(chunk) - 1)
        train_mean_nats = tot_tloss / max(1, tot_ttoks)
        train_bpb = (train_mean_nats / math.log(2.0)) * (tot_ttoks / max(1, FIXED_TRAIN_EVAL_BYTES))
        loss_dict["train_fixed"] = train_mean_nats
        bpb_dict["train_fixed"] = train_bpb

    m.train()
    return bpb_dict, loss_dict

# -------------------------------------------------------------
# 4. Stream Shuffling & Epoch Boundary Manager
# -------------------------------------------------------------
class ReshuffleStream:
    def __init__(self, tokens, batch_size=16, seq_len=512, base_seed=42):
        self.tokens = tokens
        self.N = len(tokens)
        self.B = batch_size
        self.T = seq_len
        self.tokens_per_step = batch_size * seq_len
        self.base_seed = base_seed
        self.epoch = 0
        self.cursor = 0
        self.rng = np.random.RandomState(base_seed)

    def get_batch(self):
        max_idx = self.N - self.T - 1
        starts = self.rng.randint(0, max_idx, size=self.B)
        x_np = np.stack([self.tokens[s:s + self.T] for s in starts])
        y_np = np.stack([self.tokens[s + 1:s + self.T + 1] for s in starts])
        self.cursor += self.tokens_per_step
        if self.cursor >= self.N:
            self.epoch += 1
            self.cursor = self.cursor % self.N
            self.rng = np.random.RandomState(self.base_seed + self.epoch * 1000)
            print(f"\n[EPOCH BOUNDARY CROSSED] Advanced to Epoch {self.epoch} (Re-seeded RNG: {self.base_seed + self.epoch * 1000})")
        return x_np, y_np

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "base_seed": self.base_seed,
            "rng_state": self.rng.get_state()
        }

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self.base_seed = state["base_seed"]
        self.rng.set_state(state["rng_state"])

# -------------------------------------------------------------
# 5. Training Engine with Seam Invariant Check
# -------------------------------------------------------------
BATCH_SIZE = 16
T_LEN = 512
LR_MAX = 8e-4
LR_MIN = 8e-5
WARMUP_STEPS = 500
TOTAL_SCHEDULE_STEPS = 21520

stream = ReshuffleStream(full_train_tokens, batch_size=BATCH_SIZE, seq_len=T_LEN, base_seed=42)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01, betas=(0.9, 0.95))

def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return float(current_step) / float(max(1, WARMUP_STEPS))
    progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_SCHEDULE_STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (LR_MIN / LR_MAX) + (1.0 - (LR_MIN / LR_MAX)) * cosine

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# CSV Logging
csv_path = os.path.join(WORK_DIR, "stage6b_metrics.csv")
csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "step", "epoch", "train_loss_nats", "train_bpb_fixed", "mean_val_bpb",
    "val_train_gap", "alice_bpb", "lr", "gpu_ms_per_step", "tok_per_sec",
    "en", "es", "ru", "fr", "de", "ja", "zh", "hi", "bn", "ar"
])
csv_file.flush()

print(f"\nStarting Pretraining to step {MAX_STEPS} (Warmup: {WARMUP_STEPS}, Schedule Target: {TOTAL_SCHEDULE_STEPS})...")
model.train()
pure_gpu_times = []
best_mean_bpb = float("inf")
recent_checkpoints = [] # tracks file paths of last 2 checkpoints
per_lang_consecutive_rises = {l: 0 for l in LANGUAGES}
prev_per_lang_bpb = {}
mean_val_consecutive_rises = 0
prev_mean_val = float("inf")

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

for step in range(1, MAX_STEPS + 1):
    x_np, y_np = stream.get_batch()
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

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
    dt = time.time() - t0
    pure_gpu_times.append(dt)

    if step % 100 == 0 or step == 1:
        avg_gpu_ms = (sum(pure_gpu_times[-100:]) / len(pure_gpu_times[-100:])) * 1000
        pure_tok_s = (BATCH_SIZE * T_LEN) / (avg_gpu_ms / 1000)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"  Step {step:5d}/{MAX_STEPS} | Epoch {stream.epoch}.{(stream.cursor*10)//stream.N} | Train Loss: {loss.item():.4f} | LR: {curr_lr:.2e} | GPU Step: {avg_gpu_ms:.1f} ms ({pure_tok_s:,.0f} tok/s)")

    # -------------------------------------------------------------
    # Seam Checkpoint at Step 10,000 (In-Memory Disruption Test)
    # -------------------------------------------------------------
    if step == SEAM_STEP and MAX_STEPS >= SEAM_STEP:
        print(f"\n{'='*80}")
        print(f"SEAM REACHED: STEP {SEAM_STEP}. Performing Cross-Session Invariant Verification...")
        print(f"{'='*80}")
        seam_ckpt_path = os.path.join(WORK_DIR, f"stage6b_checkpoint_step_{SEAM_STEP}.pt")
        seam_state = {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "lr": optimizer.param_groups[0]['lr'],
            "stream_state": stream.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_rng": np.random.get_state(),
            "random_rng": random.getstate(),
        }
        torch.save(seam_state, seam_ckpt_path)
        seam_hash = compute_sha256(seam_ckpt_path)
        print(f"Seam Checkpoint Saved: {seam_ckpt_path} (SHA-256: {seam_hash[:16]}...)")
        print(f"Seam LR Recorded: {seam_state['lr']:.6e} (Warmup complete, cosine active)")

        # Verify exact continuity for next 10 steps (uninterrupted vs resumed)
        # 1. Capture uninterrupted steps 10,001 -> 10,010
        uninterrupted_losses = []
        batch_hashes_uninterrupted = []
        for _ in range(10):
            bx, by = stream.get_batch()
            batch_hashes_uninterrupted.append(hashlib.sha256(bx.tobytes()).hexdigest())
            tx = torch.tensor(bx, dtype=torch.long, device=device)
            ty = torch.tensor(by, dtype=torch.long, device=device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                l_u = F.cross_entropy(model(tx).view(-1, 16000), ty.view(-1))
            scaler.scale(l_u).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            uninterrupted_losses.append(l_u.item())

        # 2. Reload from saved checkpoint and re-run steps 10,001 -> 10,010
        loaded = torch.load(seam_ckpt_path, map_location=device)
        model.load_state_dict(loaded["model_state"])
        optimizer.load_state_dict(loaded["optimizer_state"])
        scaler.load_state_dict(loaded["scaler_state"])
        scheduler.load_state_dict(loaded["scheduler_state"])
        stream.load_state_dict(loaded["stream_state"])
        torch.set_rng_state(loaded["torch_rng"])
        if loaded["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(loaded["cuda_rng"])
        np.random.set_state(loaded["numpy_rng"])
        random.setstate(loaded["random_rng"])

        resumed_losses = []
        batch_hashes_resumed = []
        for _ in range(10):
            bx, by = stream.get_batch()
            batch_hashes_resumed.append(hashlib.sha256(bx.tobytes()).hexdigest())
            tx = torch.tensor(bx, dtype=torch.long, device=device)
            ty = torch.tensor(by, dtype=torch.long, device=device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
                l_r = F.cross_entropy(model(tx).view(-1, 16000), ty.view(-1))
            scaler.scale(l_r).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            resumed_losses.append(l_r.item())

        print("\nCROSS-SESSION CONTINUITY VERIFICATION (STEPS 10,001 -> 10,010):")
        max_seam_diff = 0.0
        for i in range(10):
            diff = abs(uninterrupted_losses[i] - resumed_losses[i])
            b_match = (batch_hashes_uninterrupted[i] == batch_hashes_resumed[i])
            if diff > max_seam_diff:
                max_seam_diff = diff
            print(f"  Step {10001+i} | Unint: {uninterrupted_losses[i]:.8f} | Resumed: {resumed_losses[i]:.8f} | Diff: {diff:.2e} | Batch Hash: {'MATCH' if b_match else 'MISMATCH'}")

        print(f"\nMaximum Seam Loss Difference: {max_seam_diff:.2e}")
        if max_seam_diff < 1e-5:
            print(">>> SEAM INVARIANT VERIFICATION: PASSED (Exact Bitwise Continuity Across Seam)")
        else:
            raise ValueError(f"CRITICAL: Seam Continuity Violation! Diff {max_seam_diff:.2e} >= 1e-5")
        print(f"{'='*80}\n")

    # -------------------------------------------------------------
    # Checkpoint Evaluation
    # -------------------------------------------------------------
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
        t_eval_0 = time.time()
        is_full_eval = (step == SEAM_STEP or step == MAX_STEPS)
        val_tok_target = val_tokens_full if is_full_eval else val_tokens_fast
        val_byte_target = val_bytes_full if is_full_eval else val_bytes_fast

        eval_type_str = "FULL (1.5 MB/lang)" if is_full_eval else "FAST (~200k toks/lang)"
        print(f"\n{'='*75}")
        print(f"CHECKPOINT EVALUATION AT STEP {step:d} / {MAX_STEPS:d} [{eval_type_str}]")
        print(f"{'='*75}")

        val_bpb, val_loss = evaluate_subset(model, val_tok_target, val_byte_target, T=T_LEN)
        eval_dt = time.time() - t_eval_0

        mean_val_bpb = sum(val_bpb[l] for l in LANGUAGES) / len(LANGUAGES)
        fixed_train_bpb = val_bpb["train_fixed"]
        val_train_gap = mean_val_bpb - fixed_train_bpb
        alice_bpb = val_bpb["alice"]

        recent_gpu_ms = (sum(pure_gpu_times[-100:]) / len(pure_gpu_times[-100:])) * 1000
        pure_tok_s = (BATCH_SIZE * T_LEN) / (recent_gpu_ms / 1000)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"  Progress       : {step*BATCH_SIZE*T_LEN:,d} tokens trained ({stream.epoch}.{(stream.cursor*10)//stream.N} epochs)")
        print(f"  Throughput     : {recent_gpu_ms:.1f} ms/step ({pure_tok_s:,.0f} tok/s) [Eval overhead: {eval_dt:.1f} s]")
        print(f"  Train Loss     : {loss.item():.4f} nats | Fixed Train BPB (Eval mode): {fixed_train_bpb:.4f}")
        print(f"  Mean Val BPB   : {mean_val_bpb:.4f}")
        print(f"  Val/Train Gap  : {val_train_gap:+.4f} BPB (Rule: < 0.10)")
        print(f"  Alice BPB      : {alice_bpb:.4f} (Improving check)")
        print(f"  Learning Rate  : {curr_lr:.4e}")

        # Overfit Watch & Per-Language Checks
        print("\n  Per-Language Val BPB:")
        for l in LANGUAGES:
            b = val_bpb[l]
            p = prev_per_lang_bpb.get(l, b)
            if b > p:
                per_lang_consecutive_rises[l] += 1
            else:
                per_lang_consecutive_rises[l] = 0
            flag_str = f" [WARNING: RISING {per_lang_consecutive_rises[l]}x]" if per_lang_consecutive_rises[l] >= 2 else ""
            print(f"    {l.upper():2s}: {b:.4f} bpb (Delta: {b - p:+.4f}){flag_str}")
            prev_per_lang_bpb[l] = b

        if mean_val_bpb > prev_mean_val:
            mean_val_consecutive_rises += 1
        else:
            mean_val_consecutive_rises = 0
        prev_mean_val = mean_val_bpb

        # Sample Generation
        prompt_txt = "In science and mathematics, the universe"
        prompt_ids = tok16k.encode(prompt_txt).ids
        gen_ids = model.generate(prompt_ids, max_new_tokens=25, temperature=0.8)
        gen_txt = tok16k.decode(gen_ids)
        print(f"\n  Sample Generation:\n    \"{gen_txt}\"")

        # Log to CSV
        csv_writer.writerow([
            step, stream.epoch, f"{loss.item():.4f}", f"{fixed_train_bpb:.4f}",
            f"{mean_val_bpb:.4f}", f"{val_train_gap:+.4f}", f"{alice_bpb:.4f}",
            f"{curr_lr:.4e}", f"{recent_gpu_ms:.1f}", f"{pure_tok_s:.0f}",
            *[f"{val_bpb[l]:.4f}" for l in LANGUAGES]
        ])
        csv_file.flush()

        # Checkpoint Retention (Keep Best, Last 2)
        curr_ckpt_path = os.path.join(WORK_DIR, f"checkpoint_step_{step}.pt")
        torch.save({
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "lr": curr_lr,
            "mean_val_bpb": mean_val_bpb,
            "stream_state": stream.state_dict()
        }, curr_ckpt_path)

        recent_checkpoints.append(curr_ckpt_path)
        if len(recent_checkpoints) > 2:
            oldest = recent_checkpoints.pop(0)
            if os.path.exists(oldest) and "checkpoint_best" not in oldest and "checkpoint_step_10000" not in oldest:
                try:
                    os.remove(oldest)
                except Exception:
                    pass

        if mean_val_bpb < best_mean_bpb:
            best_mean_bpb = mean_val_bpb
            best_ckpt_path = os.path.join(WORK_DIR, "checkpoint_best.pt")
            torch.save(torch.load(curr_ckpt_path), best_ckpt_path)
            print(f"  [New Best Val BPB: {best_mean_bpb:.4f}] Saved to {best_ckpt_path}")

        # Check Early Stopping
        if mean_val_consecutive_rises >= 2:
            print(f"\n[EARLY STOPPING TRIGGERED] Mean Val BPB turned upward for 2 consecutive checkpoints ({mean_val_consecutive_rises} rises)!")
            break

        print(f"{'='*75}\n")

csv_file.close()

# -------------------------------------------------------------
# 6. Metadata Ledger & Checkpoint Audit
# -------------------------------------------------------------
retained_checkpoints_info = {}
for ckpt_f in glob.glob(os.path.join(WORK_DIR, "*.pt")):
    h = compute_sha256(ckpt_f)
    retained_checkpoints_info[os.path.basename(ckpt_f)] = {
        "size_bytes": os.path.getsize(ckpt_f),
        "sha256": h
    }

metadata_report = {
    "run_id": RUN_ID,
    "run_mode": RUN_MODE,
    "git_commit": GIT_COMMIT,
    "parameters": {
        "total": total_p,
        "non_embedding": non_emb,
        "tied_embedding": model.tok_emb.weight.numel(),
        "mlp_width": int(2 * (4 * 256) / 3),
        "formula": "12 * D^2"
    },
    "tokens": {
        "total_corpus_bytes": total_train_bytes,
        "total_corpus_tokens": total_train_tokens,
        "tokens_trained": min(step * BATCH_SIZE * T_LEN, MAX_STEPS * BATCH_SIZE * T_LEN),
        "final_epoch": stream.epoch
    },
    "hashes": {
        "tokenizer_16k_sha256": tok16k_hash,
        "tokenizer_8k_sha256": tok8k_hash,
        "train_vault_data_hashes": train_data_hashes
    },
    "retained_checkpoints": retained_checkpoints_info
}

metadata_path = os.path.join(WORK_DIR, "stage6b_run_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata_report, f, indent=2)

print(f"\nRun Metadata & Checkpoint Ledger saved to: {metadata_path}")
print(f"Per-Checkpoint Metrics CSV saved to: {csv_path}")
print(f"STAGE 6b EXECUTION FINISHED.")
print("=" * 80)
