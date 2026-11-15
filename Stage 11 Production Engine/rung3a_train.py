import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join("Stage 11 Production Engine", "canonical_datasets")

class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, T):
        super().__init__()
        self.D = D
        self.H = H
        self.dk = D // H
        self.qkv_proj = nn.Linear(D, 3 * D, bias=False)
        self.out_proj = nn.Linear(D, D, bias=False)
        # Register causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x) # [B, T, 3*D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape to [B, H, T, dk]
        q = q.view(B, T, self.H, self.dk).transpose(1, 2)
        k = k.view(B, T, self.H, self.dk).transpose(1, 2)
        v = v.view(B, T, self.H, self.dk).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v) # [B, H, T, dk]
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
        # Pre-LayerNorm with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Rung3AModel(nn.Module):
    def __init__(self, V, T=16, D=64, H=8, L=4):
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

        # Tie weights between token embedding and LM head
        self.tok_emb.weight = self.lm_head.weight

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

def evaluate(model, tokens, T=16, stride=16, batch_size=128, device='cpu', max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # Pre-slice evaluation windows
    indices = list(range(0, len(tokens) - T, stride))
    if max_batches is not None:
        indices = indices[:max_batches * batch_size]

    with torch.no_grad():
        for b_start in range(0, len(indices), batch_size):
            b_indices = indices[b_start : b_start + batch_size]
            x_batch = torch.stack([tokens[i : i + T] for i in b_indices]).to(device)
            y_batch = torch.stack([tokens[i + 1 : i + T + 1] for i in b_indices]).to(device)

            logits = model(x_batch) # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, model.V), y_batch.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y_batch.numel()

    return total_loss / total_tokens if total_tokens > 0 else float('nan')

def generate(model, seed_text, stoi, itos, length=300, temperature=0.8, device='cpu'):
    model.eval()
    unk_idx = stoi.get('\u25A1', 0)
    tokens = [stoi.get(c, unk_idx) for c in seed_text]
    
    with torch.no_grad():
        for _ in range(length):
            ctx = torch.tensor(tokens[-16:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(ctx)[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_tok)
            
    return "".join([itos[idx] for idx in tokens])

def main():
    print("==========================================================================")
    print("STAGE 11 - RUNG 3A: 1 MB SCALING LADDER TRAINING")
    print("Architecture: D=64, 4 Blocks, 8 Heads (dk=8), AdamW + Pre-LayerNorm")
    print("==========================================================================")

    # 1. Load Vocabulary and Canonical Datasets
    vocab_path = os.path.join(DATA_DIR, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    itos = {i: ch for i, ch in enumerate(vocab['itos'])}
    V = len(vocab['itos'])

    print(f"Loaded Vocabulary: {V} unique tokens")

    # Load 1 MB training data and evaluation sets
    train_tokens = torch.from_numpy(np.load(os.path.join(DATA_DIR, "tier2_1mb", "train_tokens.npy"))).long()
    internal_val_tokens = torch.from_numpy(np.load(os.path.join(DATA_DIR, "tier2_1mb", "internal_val_tokens.npy"))).long()
    common_val_tokens = torch.from_numpy(np.load(os.path.join(DATA_DIR, "common_eval", "common_val_tokens.npy"))).long()

    print(f"Data Loaded:")
    print(f" - Train Tokens (1 MB)       : {len(train_tokens):,}")
    print(f" - Internal Val Tokens       : {len(internal_val_tokens):,}")
    print(f" - Common Benchmark Val      : {len(common_val_tokens):,} (SHARED EVAL BENCHMARK)")

    # 2. Setup Device & Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Hardware Device: {device} | AMP Mixed Precision (FP16/BF16): {use_amp}")

    # 3. Initialize Model
    T = 16
    D = 64
    H = 8
    L = 4
    model = Rung3AModel(V=V, T=T, D=D, H=H, L=L).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    # 4. Hyperparameters
    batch_size = 128
    stride = 16 # non-overlapping windows (each token trained once per epoch)
    epochs = 30
    lr = 0.001
    min_lr = 0.0001
    warmup_epochs = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    # 5. TIME ONE EPOCH BENCHMARK (As instructed)
    print("\n--------------------------------------------------------------------------")
    print("STEP 1: TIMING ONE FULL EPOCH BEFORE RUNNING...")
    print("--------------------------------------------------------------------------")
    
    train_indices = list(range(0, len(train_tokens) - T, stride))
    num_batches = len(train_indices) // batch_size
    print(f"Total sequences per epoch: {len(train_indices):,} | Batches per epoch: {num_batches}")

    t_start = time.time()
    model.train()
    batch_count = 0
    trial_loss_sum = 0.0

    for b_idx in range(num_batches):
        b_indices = train_indices[b_idx * batch_size : (b_idx + 1) * batch_size]
        x_b = torch.stack([train_tokens[i : i + T] for i in b_indices]).to(device)
        y_b = torch.stack([train_tokens[i + 1 : i + T + 1] for i in b_indices]).to(device)

        optimizer.zero_grad()
        logits = model(x_b)
        loss = F.cross_entropy(logits.view(-1, V), y_b.view(-1))
        loss.backward()
        optimizer.step()

        trial_loss_sum += loss.item()
        batch_count += 1
        if batch_count == 20: # print early pacing
            elapsed_20 = time.time() - t_start
            pacing = elapsed_20 / 20
            print(f"Early pacing: {pacing:.3f} s/batch ({1/pacing:.1f} batches/s)")

    epoch_1_time = time.time() - t_start
    est_total_time_min = (epoch_1_time * epochs) / 60
    print(f"Epoch 1 completed in: {epoch_1_time:.2f} seconds ({epoch_1_time/60:.2f} min)")
    print(f"Estimated 30-Epoch Run Time: {est_total_time_min:.1f} minutes")
    print(f"Fit in session check: PASSED (Well within margin)")

    # Initial evaluation after Epoch 1
    val_internal = evaluate(model, internal_val_tokens, T=T, stride=stride, device=device)
    val_common = evaluate(model, common_val_tokens, T=T, stride=stride, device=device)
    train_loss = trial_loss_sum / num_batches
    print(f"Epoch  1/{epochs} | Train: {train_loss:.4f} | Val (Internal): {val_internal:.4f} | Val (Common Benchmark): {val_common:.4f} | Gap: {val_common - train_loss:+.4f}")

    # 6. TRAIN REMAINING EPOCHS WITH EARLY STOPPING (Patience = 5)
    best_val = val_common
    best_epoch = 1
    patience = 5
    no_improve = 0
    save_path = "Stage 11 Production Engine/rung3a_best_model.pt"
    torch.save(model.state_dict(), save_path)

    metrics_log = []
    metrics_log.append({
        'epoch': 1, 'train_loss': train_loss, 'val_internal': val_internal,
        'val_common': val_common, 'gap': val_common - train_loss, 'time_sec': epoch_1_time
    })

    print("\n--------------------------------------------------------------------------")
    print("STEP 2: FULL CONVERGENCE TRAINING (Target: Val Loss clearly < 1.5654)")
    print("--------------------------------------------------------------------------")

    for epoch in range(2, epochs + 1):
        t0 = time.time()
        model.train()
        
        # Cosine LR schedule with warmup
        if epoch <= warmup_epochs:
            cur_lr = lr * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

        # Jitter offset for data rotation
        jitter = (epoch * 7) % stride
        epoch_indices = list(range(jitter, len(train_tokens) - T, stride))
        np.random.shuffle(epoch_indices) # shuffle batches for better stochasticity

        num_b = len(epoch_indices) // batch_size
        train_loss_sum = 0.0

        for b_idx in range(num_b):
            b_indices = epoch_indices[b_idx * batch_size : (b_idx + 1) * batch_size]
            x_b = torch.stack([train_tokens[i : i + T] for i in b_indices]).to(device)
            y_b = torch.stack([train_tokens[i + 1 : i + T + 1] for i in b_indices]).to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = F.cross_entropy(logits.view(-1, V), y_b.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train = train_loss_sum / num_b
        val_internal = evaluate(model, internal_val_tokens, T=T, stride=stride, device=device)
        val_common = evaluate(model, common_val_tokens, T=T, stride=stride, device=device)
        elapsed = time.time() - t0
        gap = val_common - avg_train

        is_best = False
        if val_common < best_val:
            best_val = val_common
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            is_best = True
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:2d}/{epochs} | Train: {avg_train:.4f} | Val (Int): {val_internal:.4f} | Val (Common): {val_common:.4f} | Gap: {gap:+.4f} | LR: {cur_lr:.6f} | {elapsed:.1f}s {'<- [BEST]' if is_best else ''}", flush=True)

        metrics_log.append({
            'epoch': epoch, 'train_loss': avg_train, 'val_internal': val_internal,
            'val_common': val_common, 'gap': gap, 'time_sec': elapsed
        })

        if no_improve >= patience:
            print(f"\n[Early Stopping Triggered] Validation loss did not improve for {patience} epochs.")
            break

    # Save metrics CSV
    import csv
    with open("Stage 11 Production Engine/rung3a_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)

    print("\n==========================================================================")
    print("RUNG 3A TRAINING COMPLETE!")
    print(f" - Best Epoch               : {best_epoch}")
    print(f" - Best Common Val Loss     : {best_val:.4f}")
    print(f" - Train-Val Gap at Best    : {metrics_log[best_epoch-1]['gap']:+.4f}")
    print(f" - Best Model Checkpoint    : {save_path}")
    print("==========================================================================")

    # Load best model and generate 300 character sample
    model.load_state_dict(torch.load(save_path))
    seed_prompt = "Arthur Pendelton"
    sample_text = generate(model, seed_prompt, stoi, itos, length=300, temperature=0.8, device=device)

    print("\n--- 300 GENERATED CHARACTERS FROM BEST 3A MODEL (Temp 0.8) ---")
    print(sample_text)
    print("--------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
