import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ebrain_torch import EBrainTransformer

def run_rung1_experiment(seed=42, epochs=30, lr=0.003, batch_size=32):
    print("==========================================================================")
    print(f"RUNG 1: ADAM + PRE-LAYERNORM CONVERGENCE (SEED {seed})")
    print("==========================================================================")

    with open("Text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    full_text = raw_text + raw_text
    split_idx = int(len(full_text) * 0.90)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    chars = []
    for c in full_text:
        if c not in chars:
            chars.append(c)

    V = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    train_tokens = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val_tokens = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)

    T = 16
    D = 32
    n_heads = 4
    H_mlp = 100
    L = 2
    stride = 8

    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = EBrainTransformer(V=V, T=T, D=D, n_heads=n_heads, H_mlp=H_mlp, L=L, dropout=0.0, use_layernorm=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    total_steps = ((len(train_tokens) - T) // (stride * batch_size)) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    # Prepare val windows
    val_windows = []
    val_targets = []
    for i in range(0, len(val_tokens) - T, stride):
        val_windows.append(val_tokens[i : i + T])
        val_targets.append(val_tokens[i + 1 : i + T + 1])
    val_x = torch.stack(val_windows).to(device)
    val_y = torch.stack(val_targets).to(device)

    best_val_pos8_15 = float("inf")
    best_val_all = float("inf")
    best_epoch = 1
    train_loss_at_best = float("inf")

    train_limit = len(train_tokens) - T - 1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        jitter_offset = epoch % stride
        window_indices = list(range(jitter_offset, train_limit, stride))

        epoch_train_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(window_indices), batch_size):
            b_indices = window_indices[b_start : b_start + batch_size]
            b_x = torch.stack([train_tokens[idx : idx + T] for idx in b_indices]).to(device)
            b_y = torch.stack([train_tokens[idx + 1 : idx + T + 1] for idx in b_indices]).to(device)

            optimizer.zero_grad()
            logits, loss = model(b_x, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            n_batches += 1

        avg_train = epoch_train_loss / n_batches

        # Eval
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_x)
            loss_all = F.cross_entropy(val_logits.view(-1, V), val_y.view(-1)).item()
            val_logits_half = val_logits[:, 8:, :].contiguous()
            val_y_half = val_y[:, 8:].contiguous()
            loss_pos8_15 = F.cross_entropy(val_logits_half.view(-1, V), val_y_half.view(-1)).item()

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        is_best = loss_pos8_15 < best_val_pos8_15
        if is_best:
            best_val_pos8_15 = loss_pos8_15
            best_val_all = loss_all
            best_epoch = epoch
            train_loss_at_best = avg_train
            torch.save(model.state_dict(), f"rung1_best_seed_{seed}.pt")

        print(f"Epoch {epoch:2d}/{epochs} | Train: {avg_train:.4f} | Val (All): {loss_all:.4f} | Val (8..15): {loss_pos8_15:.4f} | LR: {cur_lr:.6f} | {elapsed:.1f}s {'<- [BEST]' if is_best else ''}", flush=True)

    print("--------------------------------------------------------------------------")
    print(f"RUNG 1 (Seed {seed}) COMPLETE!")
    print(f" - Best Epoch:                {best_epoch}")
    print(f" - Best Val Loss (Pos 8..15): {best_val_pos8_15:.4f}")
    print(f" - Val Loss (All Pos 0..15):  {best_val_all:.4f}")
    print(f" - Train Loss at Best Epoch:  {train_loss_at_best:.4f}")
    print(f" - Baseline Benchmark:        1.7505")
    gap = 1.7505 - best_val_pos8_15
    print(f" - Improvement over Baseline: {gap:+.4f} ({'BEATS BASELINE [PASS]' if gap > 0 else 'CHECK'})")
    print("==========================================================================\n", flush=True)
    return best_val_pos8_15, train_loss_at_best

if __name__ == "__main__":
    v1, t1 = run_rung1_experiment(seed=42)
    v2, t2 = run_rung1_experiment(seed=123)
    avg_val = (v1 + v2) / 2.0
    print(f">>> RUNG 1 FINAL MULTI-SEED SUMMARY: Seed 42 = {v1:.4f} | Seed 123 = {v2:.4f} | Mean = {avg_val:.4f} <<<")
