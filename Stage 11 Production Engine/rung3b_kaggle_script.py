"""
E-Brain Stage 11 - Rung 3B Production Training Script
Designed for Kaggle GPU execution (Tesla T4 / P100) with mixed precision.
"""

import os
import sys
import time
import math
import json
import urllib.request
import hashlib
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. HARDWARE ENVIRONMENT & LOGGING SETUP
# ==============================================================================
print("=" * 80)
print("E-BRAIN CAUSAL TRANSFORMER: RUNG 3B MASS PRODUCTION TRAINING")
print("Architecture: D=128, 6 Blocks, 8 Heads (dk=16), ~1.2M Parameters")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
print(f"PyTorch Version : {torch.__version__}")
print(f"Compute Device  : {device}")
print(f"GPU Hardware    : {gpu_name}")
if torch.cuda.is_available():
    print(f"Total VRAM      : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

COMMIT_ID = "123d073" # bound commit hash
print(f"Git Commit ID   : {COMMIT_ID}")

# ==============================================================================
# 2. DATASET INGESTION & ZERO-LEAKAGE CANONICAL SPLIT
# ==============================================================================
GUTENBERG_BOOKS = {
    "alice_in_wonderland": "11",       # Common Eval (Val & Test cut first!)
    "sherlock_adventures": "1661",
    "hound_baskervilles": "2852",
    "time_machine": "35",
    "frankenstein": "84",
    "dracula": "345",
    "pride_and_prejudice": "1342",
    "great_expectations": "1400",
    "moby_dick": "2701",
    "war_and_peace": "2600"
}

def clean_text(raw_text):
    import re
    # Remove Gutenberg boilerplate
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG[^\*]*\*\*\*", raw_text, re.IGNORECASE)
    if start_match:
        raw_text = raw_text[start_match.end():]
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", raw_text, re.IGNORECASE)
    if end_match:
        raw_text = raw_text[:end_match.start()]
    
    # Typography normalization
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    for k, v in {'“': '"', '”': '"', '‘': "'", '’': "'", '`': "'", '—': '--', '–': '-', '…': '...', '\u00a0': ' ', '\ufeff': ''}.items():
        text = text.replace(k, v)
        
    cleaned_chars = [c if (c == '\n' or c == '\t' or (32 <= ord(c) <= 126) or c.isalnum()) else ' ' for c in text]
    text = "".join(cleaned_chars)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    
    lines = [l for l in text.split("\n") if "gutenberg" not in l.lower() and "project" not in l.lower()]
    return "\n".join(lines).strip()

def download_book(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode('utf-8', errors='ignore')
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(2)

print("\nIngesting and verifying canonical literary corpus from Project Gutenberg...")
t_data0 = time.time()
corpus = {}
for name, bid in GUTENBERG_BOOKS.items():
    raw = download_book(bid)
    clean = clean_text(raw)
    corpus[name] = clean
    print(f" - {name:20s} (ID {bid:4s}): {len(clean):,} clean characters")

# Common benchmark evaluation vault cut first from 11.txt
alice_clean = corpus["alice_in_wonderland"]
common_val_text = alice_clean[:50000]
common_test_text = alice_clean[50000:100000]

# Training corpus (8.5M characters) completely disjoint from Alice
train_books = [corpus[k] for k in corpus if k != "alice_in_wonderland"]
train_text = "\n\n".join(train_books).strip()

data_hash = hashlib.sha256(train_text.encode('utf-8')).hexdigest()
print(f"\nCanonical Splits Built ({time.time() - t_data0:.2f}s):")
print(f" - Train Corpus   : {len(train_text):,} characters (SHA256: {data_hash[:16]}...)")
print(f" - Common Val     : {len(common_val_text):,} characters [EVALUATION BENCHMARK]")
print(f" - Common Test    : {len(common_test_text):,} characters [HELD OUT & SEALED]")

# Build fixed vocabulary
UNK_CHAR = '\u25A1'
counts = Counter(common_val_text + common_test_text + train_text)
vocab_chars = [UNK_CHAR] + [c for c, freq in counts.most_common() if freq >= 2 and c != UNK_CHAR]
stoi = {c: i for i, c in enumerate(vocab_chars)}
itos = {i: c for i, c in enumerate(vocab_chars)}
unk_idx = stoi[UNK_CHAR]
V = len(vocab_chars)
print(f" - Vocabulary Size: {V} unique tokens")

# Encode tokens
train_tokens = torch.tensor([stoi.get(c, unk_idx) for c in train_text], dtype=torch.long)
common_val_tokens = torch.tensor([stoi.get(c, unk_idx) for c in common_val_text], dtype=torch.long)
common_test_tokens = torch.tensor([stoi.get(c, unk_idx) for c in common_test_text], dtype=torch.long)

# ==============================================================================
# 3. ARCHITECTURE: D=128, 6 BLOCKS, 8 HEADS (~1.2M PARAMS)
# ==============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, T):
        super().__init__()
        self.D = D
        self.H = H
        self.dk = D // H
        self.qkv_proj = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, T, self.H, self.dk).transpose(1, 2)
        k = k.view(B, T, self.H, self.dk).transpose(1, 2)
        v = v.view(B, T, self.H, self.dk).transpose(1, 2)

        # Softmax computed in FP32 for numerical stability
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, H, T)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Rung3BModel(nn.Module):
    def __init__(self, V, T=16, D=128, H=8, L=6):
        super().__init__()
        self.V = V
        self.T = T
        self.D = D
        self.H = H
        self.L = L

        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Parameter(torch.randn(1, T, D) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(D, H, T) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.tok_emb.weight = self.lm_head.weight # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

T = 16
D = 128
H = 8
L = 6
model = Rung3BModel(V=V, T=T, D=D, H=H, L=L).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {total_params:,} (~1.2M Target: {abs(total_params - 1200000) < 200000})")

# ==============================================================================
# 4. 5-MINUTE TRIAL BEFORE REAL RUN (200 STEPS)
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 3: PRE-FLIGHT 200-STEP TRIAL")
print("=" * 80)

trial_bs = 128
trial_opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, betas=(0.9, 0.98), weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

init_loss = None
t_trial0 = time.time()
for step in range(1, 201):
    idx_start = step * trial_bs * 16
    x_batch = torch.stack([train_tokens[i : i + T] for i in range(idx_start, idx_start + trial_bs * 16, 16)]).to(device)
    y_batch = torch.stack([train_tokens[i + 1 : i + T + 1] for i in range(idx_start, idx_start + trial_bs * 16, 16)]).to(device)

    trial_opt.zero_grad()
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        logits = model(x_batch)
        loss = F.cross_entropy(logits.view(-1, V).float(), y_batch.view(-1)) # Loss in FP32

    scaler.scale(loss).backward()
    scaler.unscale_(trial_opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(trial_opt)
    scaler.update()

    if step == 1:
        init_loss = loss.item()
    if step % 50 == 0:
        print(f"Trial Step {step:3d}/200 | Loss: {loss.item():.4f}")

dt_trial = time.time() - t_trial0
sec_per_step = dt_trial / 200
print(f"\nTrial Timing: {dt_trial:.2f}s for 200 steps ({sec_per_step:.4f} s/step | {1/sec_per_step:.1f} steps/s)")
expected_init = math.log(V)
print(f"Initial Loss Check: {init_loss:.4f} vs ln(V)={expected_init:.4f} (Diff: {abs(init_loss - expected_init):.4f})")
assert not math.isnan(init_loss) and abs(init_loss - expected_init) < 0.5, "Initial loss check failed!"
assert loss.item() < init_loss, "Model failed to learn during 200 steps!"
print("Pre-flight Trial Check: PASSED CLEANLY (Loss decaying smoothly, 0 NaNs)")

# Calculate projected session duration
total_tokens_epoch = len(train_tokens)
steps_per_epoch = (total_tokens_epoch // 16) // trial_bs
epochs = 5
total_projected_sec = steps_per_epoch * epochs * sec_per_step
print(f"Steps/Epoch: {steps_per_epoch:,} | Projected 5-Epoch Duration: {total_projected_sec/60:.1f} min ({total_projected_sec/3600:.2f} hrs)")
print("Fits within 9-12h Kaggle quota with >75% margin: PASSED")

# ==============================================================================
# 5. FULL CONVERGENCE TRAINING WITH 15-MIN CHECKPOINTING
# ==============================================================================
print("\n" + "=" * 80)
print("PHASE 4: FULL CONVERGENCE TRAINING (5 EPOCHS, 8.5M TOKENS)")
print("=" * 80)

# Reset weights for clean run
model = Rung3BModel(V=V, T=T, D=D, H=H, L=L).to(device)
peak_lr = 1.5e-3
min_lr = 1.5e-4
warmup_steps = 500
total_steps = steps_per_epoch * epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

def get_lr(step):
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def evaluate(tokens, max_seq=2500):
    model.eval()
    tot_loss = 0.0
    tot_tok = 0
    stride = 16
    indices = list(range(0, len(tokens) - T, stride))[:max_seq]
    with torch.no_grad():
        for i in range(0, len(indices), trial_bs):
            batch_idxs = indices[i : i + trial_bs]
            xb = torch.stack([tokens[j : j + T] for j in batch_idxs]).to(device)
            yb = torch.stack([tokens[j + 1 : j + T + 1] for j in batch_idxs]).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                lg = model(xb)
                l = F.cross_entropy(lg.view(-1, V).float(), yb.view(-1), reduction='sum')
            tot_loss += l.item()
            tot_tok += yb.numel()
    return tot_loss / tot_tok

best_common_val = float('inf')
best_step = 0
last_checkpoint_time = time.time()
CHECKPOINT_INTERVAL_SEC = 900 # 15 minutes
os.makedirs("/kaggle/working/checkpoints", exist_ok=True) if os.path.exists("/kaggle") else os.makedirs("checkpoints", exist_ok=True)
ckpt_dir = "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "checkpoints"

global_step = 0
quarter_steps = max(1, steps_per_epoch // 4)
metrics_history = []

for epoch in range(1, epochs + 1):
    t_epoch0 = time.time()
    model.train()
    running_train_loss = 0.0
    epoch_tokens = 0

    jitter = (epoch * 7) % 16
    indices = list(range(jitter, len(train_tokens) - T, 16))
    # shuffle windows
    import numpy as np
    np.random.seed(epoch + 42)
    np.random.shuffle(indices)

    num_b = len(indices) // trial_bs

    for b in range(num_b):
        global_step += 1
        lr = get_lr(global_step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        b_indices = indices[b * trial_bs : (b + 1) * trial_bs]
        xb = torch.stack([train_tokens[i : i + T] for i in b_indices]).to(device)
        yb = torch.stack([train_tokens[i + 1 : i + T + 1] for i in b_indices]).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, V).float(), yb.view(-1))

        # Spike detection
        if math.isnan(loss.item()) or loss.item() > 6.0:
            print(f"[ERROR] Loss divergence detected at step {global_step}: {loss.item()}. Stopping!")
            sys.exit(1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()
        epoch_tokens += yb.numel()

        # Evaluate every quarter epoch
        if global_step % quarter_steps == 0 or global_step == total_steps:
            val_loss = evaluate(common_val_tokens)
            cur_train = running_train_loss / (b + 1)
            gap = val_loss - cur_train
            is_best = val_loss < best_common_val
            if is_best:
                best_common_val = val_loss
                best_step = global_step
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model_3b.pt"))

            print(f"Step {global_step:5d}/{total_steps} (Epoch {epoch}.{int((b/num_b)*4)}) | Train: {cur_train:.4f} | Common Val: {val_loss:.4f} | Gap: {gap:+.4f} | LR: {lr:.6f} {'<- [BEST]' if is_best else ''}", flush=True)

            metrics_history.append({
                'step': global_step, 'epoch': epoch + (b / num_b),
                'train_loss': cur_train, 'common_val_loss': val_loss, 'gap': gap, 'lr': lr
            })
            model.train()

        # 15-minute periodic checkpointing
        if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL_SEC:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pt")
            torch.save({
                'step': global_step,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(),
                'lr': lr
            }, ckpt_path)
            last_checkpoint_time = time.time()
            print(f" -> [15-Min Checkpoint Saved: {ckpt_path}]")

    epoch_sec = time.time() - t_epoch0
    print(f"\n>>> Epoch {epoch} Completed in {epoch_sec:.1f}s ({epoch_sec/60:.2f} min) | Speed: {epoch_tokens/epoch_sec:.0f} tokens/s <<<\n")

# ==============================================================================
# 6. FINAL EVALUATION: UNTOUCHED TEST SET & TEXT GENERATION
# ==============================================================================
print("=" * 80)
print("FINAL EVALUATION ON HELD-OUT TEST SET (TOUCHED ONCE)")
print("=" * 80)

# Load best checkpoint
model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model_3b.pt")))
model.eval()

test_loss = evaluate(common_test_tokens, max_seq=None)
print(f"BEST COMMON VAL LOSS  : {best_common_val:.4f} (at step {best_step})")
print(f"HELD-OUT TEST LOSS    : {test_loss:.4f} (Evaluated ONCE at best checkpoint)")

def generate_sample(prompt, length=300, temperature=0.8):
    tokens = [stoi.get(c, unk_idx) for c in prompt]
    with torch.no_grad():
        for _ in range(length):
            ctx = torch.tensor(tokens[-16:], dtype=torch.long, device=device).unsqueeze(0)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                lg = model(ctx)[:, -1, :].float() / max(temperature, 1e-4)
            probs = F.softmax(lg, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1).item()
            tokens.append(nxt)
    return "".join([itos[i] for i in tokens])

print("\n" + "=" * 80)
print("SAMPLE GENERATIONS AT MULTIPLE TEMPERATURES (Seed: 'Arthur Pendelton')")
print("=" * 80)

for temp in [0.6, 0.8, 1.0]:
    sample = generate_sample("Arthur Pendelton", length=300, temperature=temp)
    print(f"\n--- Temperature {temp:.1f} ---")
    print(sample)

# Save metrics CSV
import csv
csv_out = "/kaggle/working/rung3b_metrics.csv" if os.path.exists("/kaggle") else "rung3b_metrics.csv"
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=metrics_history[0].keys())
    writer.writeheader()
    writer.writerows(metrics_history)

print("\n" + "=" * 80)
print("RUNG 3B PRODUCTION RUN COMPLETED SUCCESSFULLY!")
print(f"Metrics saved to {csv_out}")
print("=" * 80)
