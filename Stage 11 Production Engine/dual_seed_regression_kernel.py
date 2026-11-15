"""
E-Brain Gate 2: Kaggle GPU Dual-Seed Regression Runner
=====================================================
Uses attached Kaggle Dataset: tijildas/ebrain-canonical-datasets-v1
Runs dual-seed benchmark reproduction (Seed 42 and Seed 123) of the canonical
character-level baseline on Tesla T4 GPU.
Evaluates strictly on canonical common_val_tokens.npy (50,000 characters from Alice in Wonderland).
Target Band: [1.4957, 1.5357] (Target: 1.5157 ± 0.02).
Sealed test set is strictly untouched.
"""

import os
import sys
import time
import math
import json
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("E-BRAIN GATE 2: KAGGLE GPU DUAL-SEED REGRESSION RUNNER")
print("Target Band: [1.4957, 1.5357] (Target: 1.5157 ± 0.02)")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Locate dataset in /kaggle/input
input_dir = "/kaggle/input/ebrain-canonical-datasets-v1"
train_tokens_path = os.path.join(input_dir, "train_tokens.npy")
val_tokens_path = os.path.join(input_dir, "common_val_tokens.npy")

# If inside zip
if not os.path.exists(train_tokens_path):
    zip_path = os.path.join(input_dir, "canonical_data.zip")
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("/kaggle/working/data")
        train_tokens_path = "/kaggle/working/data/train_tokens.npy"
        val_tokens_path = "/kaggle/working/data/common_val_tokens.npy"
    else:
        # Search anywhere in /kaggle/input
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f == "train_tokens.npy":
                    train_tokens_path = os.path.join(root, f)
                elif f == "common_val_tokens.npy":
                    val_tokens_path = os.path.join(root, f)

print(f"Loading train tokens from {train_tokens_path}...")
train_tokens = torch.tensor(np.load(train_tokens_path), dtype=torch.long)
print(f"Loading val tokens from {val_tokens_path}...")
common_val_tokens = torch.tensor(np.load(val_tokens_path), dtype=torch.long)

print(f"Canonical Datasets Loaded:")
print(f" - Train Tokens     : {len(train_tokens):,d} (tier3_8mb)")
print(f" - Common Val Tokens: {len(common_val_tokens):,d} (common_eval/Alice)")
print(f" - Sealed Test Set  : UNTOUCHED & SEALED")

V = 110 # Vocabulary size
T = 16  # Context window
D = 128 # Embedding dim
H = 8   # Heads
L = 6   # Layers
batch_size = 128
epochs = 5
steps_per_epoch = (len(train_tokens) // T) // batch_size
total_steps = steps_per_epoch * epochs
print(f"Architecture: V={V}, T={T}, D={D}, H={H}, L={L} | Batch={batch_size} | Steps/Epoch={steps_per_epoch:,d} | Total Steps={total_steps:,d}")

# Rung 3B Exact Architecture
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
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.H, self.dk).transpose(1, 2)
        k = k.view(B, T, self.H, self.dk).transpose(1, 2)
        v = v.view(B, T, self.H, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        probs = F.softmax(scores, dim=-1)
        out = (probs @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, H, T)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D, bias=False),
            nn.GELU(),
            nn.Linear(4 * D, D, bias=False)
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
        return self.lm_head(x)

def evaluate_val(model, tokens, max_seq=2500):
    model.eval()
    tot_loss = 0.0
    tot_tok = 0
    stride = 16
    indices = list(range(0, len(tokens) - T, stride))[:max_seq]
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idxs = indices[i : i + batch_size]
            xb = torch.stack([tokens[j : j + T] for j in batch_idxs]).to(device)
            yb = torch.stack([tokens[j + 1 : j + T + 1] for j in batch_idxs]).to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                lg = model(xb)
                l = F.cross_entropy(lg.view(-1, V).float(), yb.view(-1), reduction='sum')
            tot_loss += l.item()
            tot_tok += yb.numel()
    return tot_loss / tot_tok

def train_seed(seed):
    print("\n" + "=" * 80)
    print(f">>> STARTING TRAINING RUN: SEED {seed} <<<")
    print("=" * 80)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = Rung3BModel(V=V, T=T, D=D, H=H, L=L).to(device)
    peak_lr = 1.5e-3
    min_lr = 1.5e-4
    warmup_steps = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    def get_lr(step):
        if step < warmup_steps:
            return peak_lr * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        
    best_val_loss = float("inf")
    best_step = 0
    epoch_1_train_loss = None
    final_train_loss = None
    
    t0 = time.time()
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        jitter = (epoch * 7) % 16
        indices = list(range(jitter, len(train_tokens) - T, 16))
        np.random.seed(epoch + seed)
        np.random.shuffle(indices)
        num_batches = len(indices) // batch_size
        
        running_train_loss = 0.0
        
        for b in range(num_batches):
            global_step += 1
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
                
            batch_idxs = indices[b * batch_size : (b + 1) * batch_size]
            xb = torch.stack([train_tokens[i : i + T] for i in batch_idxs]).to(device)
            yb = torch.stack([train_tokens[i + 1 : i + T + 1] for i in batch_idxs]).to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, V).float(), yb.view(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_train_loss += loss.item()
            
            # Evaluate every quarter epoch
            quarter_step = max(1, steps_per_epoch // 4)
            if global_step % quarter_step == 0 or global_step == total_steps:
                val_loss = evaluate_val(model, common_val_tokens)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = global_step
                tok_s = (global_step * batch_size * T) / (time.time() - t0)
                print(f"[Seed {seed}] Step {global_step:5d}/{total_steps} (Epoch {global_step/steps_per_epoch:.2f}) | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f} | Speed: {tok_s:.0f} tok/s")
                model.train()
                
        avg_epoch_loss = running_train_loss / num_batches
        if epoch == 1:
            epoch_1_train_loss = avg_epoch_loss
        if epoch == epochs:
            final_train_loss = avg_epoch_loss
            
    target = 1.5157
    diff = abs(best_val_loss - target)
    passed = (1.4957 <= best_val_loss <= 1.5357)
    
    print(f"\n[Seed {seed} Complete]")
    print(f" -> Epoch 1 Train Loss : {epoch_1_train_loss:.4f}")
    print(f" -> Final Train Loss   : {final_train_loss:.4f}")
    print(f" -> Best Common Val    : {best_val_loss:.4f} (at step {best_step})")
    print(f" -> Target Diff        : {diff:.4f}")
    print(f" -> Band [1.4957, 1.5357]: {'PASSED' if passed else 'FAILED'}")
    
    return {
        "seed": seed,
        "epoch_1_train_loss": epoch_1_train_loss,
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "diff_from_target": diff,
        "passed": passed
    }

# Run both seeds
res_42 = train_seed(42)
res_123 = train_seed(123)

summary = {
    "target": 1.5157,
    "allowed_band": [1.4957, 1.5357],
    "seed_42": res_42,
    "seed_123": res_123,
    "both_passed": res_42["passed"] and res_123["passed"]
}

print("\n" + "=" * 80)
print("FINAL GATE 2 REGRESSION REPORT (DUAL-SEED):")
print("=" * 80)
print(f"Target Baseline : 1.5157")
print(f"Allowed Band    : [1.4957, 1.5357] (Target ± 0.02)")
print(f"Seed 42 Result  : Best Val = {res_42['best_val_loss']:.4f} (Step {res_42['best_step']}) | Ep1 Train: {res_42['epoch_1_train_loss']:.4f} | Final Train: {res_42['final_train_loss']:.4f} | Status = {'PASSED' if res_42['passed'] else 'FAILED'}")
print(f"Seed 123 Result : Best Val = {res_123['best_val_loss']:.4f} (Step {res_123['best_step']}) | Ep1 Train: {res_123['epoch_1_train_loss']:.4f} | Final Train: {res_123['final_train_loss']:.4f} | Status = {'PASSED' if res_123['passed'] else 'FAILED'}")
print(f"Overall Gate 2  : {'PASSED' if summary['both_passed'] else 'FAILED'}")
print("=" * 80)

out_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
with open(os.path.join(out_dir, "regression_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved regression_summary.json successfully.")
