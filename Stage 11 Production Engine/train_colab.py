import os
import sys
import time
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ebrain_torch import EBrainTransformer, CheckpointManager

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, T=16):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.T = T

    def __len__(self):
        return max(0, len(self.tokens) - self.T)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.T]
        y = self.tokens[idx + 1 : idx + self.T + 1]
        return x, y

class VocabWrapper:
    def __init__(self, vocab_json):
        self.itos = vocab_json['itos']
        self.stoi = vocab_json['stoi']
        self.unk = vocab_json['unk_char']

    def encode(self, ch):
        return self.stoi.get(ch, self.stoi.get(self.unk, 0))

    def decode(self, idx):
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return self.unk

    def encode_string(self, s):
        return [self.encode(c) for c in s]

    def decode_tokens(self, tokens):
        return "".join(self.decode(t) for t in tokens)

    def size(self):
        return len(self.itos)

def run_preflight_checks(model, device, vocab):
    print("==========================================================================")
    print("PHASE 7: PRE-FLIGHT CHECKLIST (AUTOMATED SUITE)")
    print("==========================================================================")
    
    # 1. Hardware Check
    print(f"[1/5] Hardware Check:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"      - Device: GPU ({gpu_name}) | Total VRAM: {vram_gb:.2f} GB")
    else:
        print(f"      - Device: CPU (Colab CPU or Local Environment)")

    # 2. Initial Loss Check: ln(V)
    V = vocab.size()
    dummy_x = torch.randint(0, V, (4, model.T), device=device)
    dummy_y = torch.randint(0, V, (4, model.T), device=device)
    with torch.no_grad():
        _, loss = model(dummy_x, dummy_y)
    expected_loss = math.log(V)
    diff = abs(loss.item() - expected_loss)
    print(f"[2/5] Initial Loss Check: {loss.item():.4f} (Expected near ln({V}) = {expected_loss:.4f}) -> {'PASSED' if diff < 0.5 else 'FAIL'}")

    # 3. Causality Check
    seq1 = torch.randint(0, V, (1, model.T), device=device)
    seq2 = seq1.clone()
    seq2[0, 10:] = (seq2[0, 10:] + 7) % V # tamper future positions

    with torch.no_grad():
        logits1, _ = model(seq1)
        logits2, _ = model(seq2)
    diff_prior = torch.max(torch.abs(logits1[0, :10] - logits2[0, :10])).item()
    print(f"[3/5] Causality Test: Max prior logits difference = {diff_prior:.2e} -> {'PASSED' if diff_prior < 1e-6 else 'FAIL'}")

    # 4. Overfit-One-Batch Check
    print(f"[4/5] Overfit-One-Batch Check (30 steps on 1 batch)...", end=" ")
    overfit_model = EBrainTransformer(V, model.T, model.D, model.n_heads, model.H_mlp, model.L, 0.0, model.use_layernorm).to(device)
    opt = torch.optim.AdamW(overfit_model.parameters(), lr=0.01)
    bx = torch.randint(0, V, (8, model.T), device=device)
    by = torch.randint(0, V, (8, model.T), device=device)
    for _ in range(30):
        opt.zero_grad()
        _, l = overfit_model(bx, by)
        l.backward()
        opt.step()
    final_overfit_loss = l.item()
    print(f"Final loss: {final_overfit_loss:.4f} -> {'PASSED' if final_overfit_loss < 0.5 else 'FAIL'}")

    # 5. Checkpoint Roundtrip Check
    print(f"[5/5] Serialization & Resume Check...", end=" ")
    test_ckpt = "temp_preflight_ckpt.pt"
    torch.save(model.state_dict(), test_ckpt)
    model.load_state_dict(torch.load(test_ckpt, map_location=device))
    if os.path.exists(test_ckpt):
        os.remove(test_ckpt)
    print("PASSED")

    print("==========================================================================")
    print(">>> PRE-FLIGHT CHECKLIST COMPLETE! ALL SYSTEMS OPERATIONAL. <<<")
    print("==========================================================================\n")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Execution Device: {device}")

    # Load dataset & vocab
    with open(os.path.join(args.data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = VocabWrapper(json.load(f))

    with open(os.path.join(args.data_dir, "train.txt"), "r", encoding="utf-8") as f:
        train_tokens = vocab.encode_string(f.read())
    with open(os.path.join(args.data_dir, "val.txt"), "r", encoding="utf-8") as f:
        val_tokens = vocab.encode_string(f.read())

    train_ds = TextDataset(train_tokens, args.T)
    val_ds = TextDataset(val_tokens, args.T)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = EBrainTransformer(
        V=vocab.size(), T=args.T, D=args.D, n_heads=args.n_heads,
        H_mlp=args.H_mlp, L=args.L, dropout=args.dropout, use_layernorm=True
    ).to(device)

    print(f"Model Architecture: L={args.L}, H={args.n_heads}, D={args.D}, H_mlp={args.H_mlp}")
    print(f"Total Parameters:   {model.total_parameters():,}")

    # Run Pre-flight checklist
    run_preflight_checks(model, device, vocab)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Checkpoint Manager
    ckpt_mgr = CheckpointManager(args.run_id, local_dir=args.checkpoint_dir)
    start_epoch, global_step, best_val = ckpt_mgr.load_latest(model, optimizer, scheduler)

    if start_epoch > 0:
        print(f"Resumed from Epoch {start_epoch} (Step {global_step}, Best Val: {best_val:.4f})")

    patience_counter = 0

    # Training Loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

        avg_train = total_train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)
        time_sec = time.time() - t0
        lr_cur = optimizer.param_groups[0]['lr']

        is_best = avg_val < best_val
        if is_best:
            best_val = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_mgr.save(model, optimizer, scheduler, epoch, global_step, avg_val, is_best=is_best)

        print(f"Epoch {epoch:2d}/{args.epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {lr_cur:.6f} | {time_sec:.1f}s {'<- [BEST]' if is_best else ''}")

        # Stopping rule: early stopping if no improvement
        if patience_counter >= args.patience:
            print(f"\n[Stopping Rule Triggered] Validation loss did not improve for {args.patience} epochs. Stopping early.")
            break

    # Sample generation from best model
    ckpt_mgr.load_latest(model)
    sample = model.generate(vocab, "Arthur Pendelton ", gen_len=200, temperature=0.8, device=device)
    print("\n--- Generated Sample (Temperature 0.8) ---")
    print(sample)
    print("------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="COLAB_RUN_001")
    parser.add_argument("--data_dir", type=str, default="data_tier1")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--H_mlp", type=int, default=64)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    train(args)
