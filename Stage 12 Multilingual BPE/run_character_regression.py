"""
E-Brain Gate 2: Character Regression Benchmark Runner
====================================================
Runs benchmark reproduction of the canonical character-level baseline.
Evaluates strictly on canonical common_val_tokens.npy (50,000 characters from Alice in Wonderland).
Target Band: [1.4957, 1.5357] (Target: 1.5157 ± 0.02).
Sealed test set is strictly untouched.
Supports --seed (e.g. 42 or 123).
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def run_regression(seed=42):
    print("=" * 80)
    print(f"E-BRAIN GATE 2: CHARACTER REGRESSION BENCHMARK (SEED: {seed})")
    print("Target Band: [1.4957, 1.5357] (Target: 1.5157 ± 0.02)")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_dir)
    canon_dir = os.path.join(root_dir, "Stage 11 Production Engine", "canonical_datasets")

    train_tokens_path = os.path.join(canon_dir, "tier3_8mb", "train_tokens.npy")
    val_tokens_path = os.path.join(canon_dir, "common_eval", "common_val_tokens.npy")

    train_tokens = torch.tensor(np.load(train_tokens_path), dtype=torch.long)
    val_tokens = torch.tensor(np.load(val_tokens_path), dtype=torch.long)

    V = 112 # Vocabulary size (covers max token id 111)
    T = 16
    D = 128
    H = 8
    L = 6

    print(f"Dataset: Train={len(train_tokens):,d} | Common Val={len(val_tokens):,d} | Sealed Test Set: UNTOUCHED")
    model = Rung3BModel(V=V, T=T, D=D, H=H, L=L).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized: V={V}, T={T}, D={D}, H={H}, L={L} | Parameters: {params:,d}")

    batch_size = 128
    epochs = 5
    steps_per_epoch = (len(train_tokens) // T) // batch_size
    total_steps = steps_per_epoch * epochs

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

    def evaluate_exact(tokens_tensor, max_seq=2500):
        model.eval()
        tot_loss = 0.0
        tot_tok = 0
        stride = 16
        indices = list(range(0, len(tokens_tensor) - T, stride))[:max_seq]
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_idxs = indices[i : i + batch_size]
                xb = torch.stack([tokens_tensor[j : j + T] for j in batch_idxs]).to(device)
                yb = torch.stack([tokens_tensor[j + 1 : j + T + 1] for j in batch_idxs]).to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    lg = model(xb)
                    l = F.cross_entropy(lg.view(-1, V).float(), yb.view(-1), reduction='sum')
                tot_loss += l.item()
                tot_tok += yb.numel()
        return tot_loss / tot_tok

    init_val = evaluate_exact(val_tokens)
    print(f"\n[Step 0 Sanity] Initial Val Loss: {init_val:.4f} (ln(110) = {math.log(110):.4f})")

    best_val_loss = float("inf")
    best_step = 0
    epoch_1_train_loss = None
    final_train_loss = None

    t0 = time.time()
    global_step = 0
    quarter_step = max(1, steps_per_epoch // 4)

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

            if global_step % quarter_step == 0 or global_step == total_steps:
                val_loss = evaluate_exact(val_tokens)
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

    print("\n" + "=" * 80)
    print(f"GATE 2 REGRESSION REPORT (SEED: {seed}):")
    print(f" -> Initial Step 0 Val Loss : {init_val:.4f}")
    print(f" -> Epoch 1 Avg Train Loss  : {epoch_1_train_loss:.4f}")
    print(f" -> Final Epoch 5 Train Loss: {final_train_loss:.4f}")
    print(f" -> Best Common Val Loss    : {best_val_loss:.4f} (at step {best_step})")
    print(f" -> Target Value            : {target:.4f}")
    print(f" -> Allowed Band            : [1.4957, 1.5357] (Target ± 0.02)")
    print(f" -> Absolute Difference     : {diff:.4f}")
    print(f" -> Status                  : {'PASSED' if passed else 'FAILED'}")
    print("=" * 80)
    return {
        "seed": seed,
        "init_val": init_val,
        "epoch_1_train_loss": epoch_1_train_loss,
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "diff": diff,
        "passed": passed
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_regression(seed=args.seed)
