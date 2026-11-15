import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage10BBlock(nn.Module):
    """
    Exact mirror of Stage 10B Transformer Block (No LayerNorm, No Wo matrix, Direct Head Concat).
    Total parameters per block: 3*4*32*8 + 32*100 + 100 + 100*32 + 32 = 9,604
    """
    def __init__(self, D=32, n_heads=4, H_mlp=100, T=16):
        super().__init__()
        self.D = D
        self.n_heads = n_heads
        self.dk = D // n_heads
        self.T = T

        # Individual Q, K, V heads matching Stage 10B [4][32][8]
        self.Wq = nn.Parameter(torch.empty(n_heads, D, self.dk))
        self.Wk = nn.Parameter(torch.empty(n_heads, D, self.dk))
        self.Wv = nn.Parameter(torch.empty(n_heads, D, self.dk))

        # Feed-forward
        self.W1 = nn.Parameter(torch.empty(D, H_mlp))
        self.b1 = nn.Parameter(torch.zeros(H_mlp))
        self.W2 = nn.Parameter(torch.empty(H_mlp, D))
        self.b2 = nn.Parameter(torch.zeros(D))

        # Initialize with uniform(-0.05, 0.05) matching Java init
        scale = 0.05
        nn.init.uniform_(self.Wq, -scale, scale)
        nn.init.uniform_(self.Wk, -scale, scale)
        nn.init.uniform_(self.Wv, -scale, scale)
        nn.init.uniform_(self.W1, -scale, scale)
        nn.init.uniform_(self.W2, -scale, scale)

        # Causal mask: upper triangle set to -1e9
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        sqrt_dk = math.sqrt(self.dk)

        # Multi-head attention
        # x: [B, T, D], Wq: [H, D, dk] -> Q: [B, H, T, dk]
        Q = torch.einsum('btd,hdk->bhtk', x, self.Wq)
        K = torch.einsum('btd,hdk->bhtk', x, self.Wk)
        V = torch.einsum('btd,hdk->bhtk', x, self.Wv)

        # Scores: [B, H, T, T]
        scores = torch.einsum('bhtk,bhik->bhti', Q, K) / sqrt_dk
        scores = scores.masked_fill(self.causal_mask[:T, :T], -1e9)
        A = F.softmax(scores, dim=-1)

        # Blend: [B, H, T, dk] -> concat heads to [B, T, D]
        head_out = torch.einsum('bhti,bhik->bhtk', A, V)
        att_out = head_out.permute(0, 2, 1, 3).contiguous().view(B, T, D)

        # Residual 1: Y = X + att_out
        Y = x + att_out

        # Feed-Forward: W1 -> tanh -> W2
        h_mlp = torch.tanh(Y @ self.W1 + self.b1)
        ff_out = h_mlp @ self.W2 + self.b2

        # Residual 2: Z = Y + ff_out
        Z = Y + ff_out
        return Z


class Stage10BModel(nn.Module):
    """
    Exact Stage 10B Model with C, P, 2 Blocks, Wout, bout.
    Total Parameters: 24,205
    """
    def __init__(self, V, T=16, D=32, n_heads=4, H_mlp=100, L=2):
        super().__init__()
        self.V = V
        self.T = T
        self.D = D
        self.L = L

        self.C = nn.Parameter(torch.empty(V, D))
        self.P = nn.Parameter(torch.empty(T, D))

        self.blocks = nn.ModuleList([
            Stage10BBlock(D, n_heads, H_mlp, T) for _ in range(L)
        ])

        self.Wout = nn.Parameter(torch.empty(D, V))
        self.bout = nn.Parameter(torch.zeros(V))

        scale = 0.05
        nn.init.uniform_(self.C, -scale, scale)
        nn.init.uniform_(self.P, -scale, scale)
        nn.init.uniform_(self.Wout, -scale, scale)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.C[idx] + self.P[pos] # [B, T, D]

        for block in self.blocks:
            x = block(x)

        logits = x @ self.Wout + self.bout # [B, T, V]
        return logits


def run_rung0():
    print("==========================================================================")
    print("RUNG 0: REPRODUCE JAVA RESULT IN PYTORCH (STAGE 10B EXACT MIRROR)")
    print("Goal: Match Java Stage 10B (~1.75 validation loss) on identical 170K text")
    print("==========================================================================")

    # 1. Load and prepare identical 2x text
    with open("Text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    full_text = raw_text + raw_text
    split_idx = int(len(full_text) * 0.90)
    train_text = full_text[:split_idx]
    val_text = full_text[split_idx:]

    # Build character vocabulary matching Java
    chars = []
    for c in full_text:
        if c not in chars:
            chars.append(c)

    V = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary Size: {V}")
    print(f"Train characters: {len(train_text):,} | Val characters: {len(val_text):,}")

    train_tokens = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val_tokens = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)

    # 2. Hyperparameters
    T = 16
    D = 32
    n_heads = 4
    H_mlp = 100
    L = 2
    epochs = 30
    start_lr = 0.020
    lr_decay = 0.95
    stride = 8

    # Fix seed for reproducible baseline
    torch.manual_seed(42)
    device = torch.device("cpu") # match CPU execution
    model = Stage10BModel(V, T, D, n_heads, H_mlp, L).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (Exact match to Java 24,205: {total_params == 24205})")

    # Optimizer: SGD matching Java update
    # In Java: w = w - lr * grad
    current_lr = start_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr)

    # Prepare validation windows
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

    # Pre-slice training windows for speed
    train_limit = len(train_tokens) - T - 1

    print("\nStarting 30-Epoch PyTorch FP32 Training...")
    print("--------------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # Update learning rate: lr = start_lr * (0.95^(epoch-1))
        current_lr = start_lr * (lr_decay ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        jitter_offset = epoch % stride
        epoch_train_loss = 0.0
        n_batches = 0

        # Process in batches for efficient vectorization while keeping same effective SGD steps
        # Using batch_size=16 with standard learning rate scaling
        # To match online SGD identically: batch_size=1 (or mini-batch equivalent)
        # Let's vectorize windows with batch_size=32 for 20x faster CPU execution
        window_indices = list(range(jitter_offset, train_limit, stride))
        batch_size = 32

        for b_start in range(0, len(window_indices), batch_size):
            b_indices = window_indices[b_start : b_start + batch_size]
            b_x = torch.stack([train_tokens[idx : idx + T] for idx in b_indices]).to(device)
            b_y = torch.stack([train_tokens[idx + 1 : idx + T + 1] for idx in b_indices]).to(device)

            optimizer.zero_grad()
            logits = model(b_x) # [B, T, V]
            # Cross entropy averaged over T positions, but summed over batch B
            # to match the cumulative update magnitude of online SGD (batch_size=1)
            loss_sum = F.cross_entropy(logits.view(-1, V), b_y.view(-1), reduction='sum') / T
            loss_sum.backward()
            optimizer.step()

            # For reporting, track average per-character loss
            loss_mean = loss_sum.item() / len(b_indices)
            epoch_train_loss += loss_mean
            n_batches += 1

        avg_train = epoch_train_loss / n_batches

        # Evaluate Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x) # [N_val, T, V]
            loss_all = F.cross_entropy(val_logits.view(-1, V), val_y.view(-1)).item()
            val_logits_half = val_logits[:, 8:, :].contiguous()
            val_y_half = val_y[:, 8:].contiguous()
            loss_pos8_15 = F.cross_entropy(val_logits_half.view(-1, V), val_y_half.view(-1)).item()

        elapsed = time.time() - t0
        is_best = loss_pos8_15 < best_val_pos8_15
        if is_best:
            best_val_pos8_15 = loss_pos8_15
            best_val_all = loss_all
            best_epoch = epoch
            train_loss_at_best = avg_train
            torch.save(model.state_dict(), "rung0_best_model.pt")

        print(f"Epoch {epoch:2d}/30 | Train: {avg_train:.4f} | Val (All): {loss_all:.4f} | Val (8..15): {loss_pos8_15:.4f} | LR: {current_lr:.5f} | {elapsed:.1f}s {'<- [BEST]' if is_best else ''}", flush=True)

    print("--------------------------------------------------------------------------")
    print(f"RUNG 0 COMPLETE!")
    print(f" - Best Epoch:                {best_epoch}")
    print(f" - Best Val Loss (Pos 8..15): {best_val_pos8_15:.4f}")
    print(f" - Val Loss (All Pos 0..15):  {best_val_all:.4f}")
    print(f" - Train Loss at Best Epoch:  {train_loss_at_best:.4f}")
    print(f" - Java Target Benchmark:     1.7505")
    gap = abs(best_val_pos8_15 - 1.7505)
    print(f" - Difference vs. Java Target: {gap:.4f} ({'PASSED' if gap < 0.05 else 'INVESTIGATE'})")
    print("==========================================================================\n")
    return best_val_pos8_15

if __name__ == "__main__":
    run_rung0()
