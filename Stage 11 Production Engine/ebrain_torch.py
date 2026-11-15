import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class EBrainCausalSelfAttention(nn.Module):
    """
    Handcrafted Multi-Head Causal Self-Attention.
    Direct 1:1 architectural mirror of the Java and NumPy implementations.
    """
    def __init__(self, D=32, n_heads=4, T=16, dropout=0.0):
        super().__init__()
        assert D % n_heads == 0, "D must be divisible by n_heads"
        self.D = D
        self.n_heads = n_heads
        self.dk = D // n_heads
        self.T = T

        # Individual Q, K, V projections matching our handcrafted matrices
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular 1s, upper triangular 0s)
        self.register_buffer("causal_mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, T, D = x.size()

        # Project and reshape into [B, n_heads, T, dk]
        q = self.Wq(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk) # [B, n_heads, T, T]
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Blend values
        out = attn_weights @ v # [B, n_heads, T, dk]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.resid_dropout(self.Wo(out))
        return out


class EBrainMLP(nn.Module):
    """
    Position-wise Feed-Forward Network: W1 -> tanh -> Dropout -> W2.
    """
    def __init__(self, D=32, H_mlp=64, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(D, H_mlp, bias=True)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(H_mlp, D, bias=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.W1(x))
        h = self.dropout(h)
        out = self.resid_dropout(self.W2(h))
        return out


class EBrainBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block:
    x = x + Attn(LN1(x))
    x = x + MLP(LN2(x))
    """
    def __init__(self, D=32, n_heads=4, H_mlp=64, T=16, dropout=0.0, use_layernorm=True):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.ln1 = nn.LayerNorm(D, eps=1e-5) if use_layernorm else nn.Identity()
        self.attn = EBrainCausalSelfAttention(D, n_heads, T, dropout)
        self.ln2 = nn.LayerNorm(D, eps=1e-5) if use_layernorm else nn.Identity()
        self.mlp = EBrainMLP(D, H_mlp, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EBrainTransformer(nn.Module):
    """
    Complete E-Brain Causal Transformer PyTorch Architecture for High-Speed GPU Training.
    """
    def __init__(self, V, T=16, D=32, n_heads=4, H_mlp=64, L=2, dropout=0.0, use_layernorm=True):
        super().__init__()
        self.V = V
        self.T = T
        self.D = D
        self.n_heads = n_heads
        self.H_mlp = H_mlp
        self.L = L
        self.use_layernorm = use_layernorm

        # Token & Learned Position Embeddings
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(T, D)
        self.emb_dropout = nn.Dropout(dropout)

        # Stacked Blocks
        self.blocks = nn.ModuleList([
            EBrainBlock(D, n_heads, H_mlp, T, dropout, use_layernorm)
            for _ in range(L)
        ])

        # Final Pre-LN and Head
        self.ln_f = nn.LayerNorm(D, eps=1e-5) if use_layernorm else nn.Identity()
        self.head = nn.Linear(D, V, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.T, f"Sequence length {T} exceeds maximum context length {self.T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x) # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, vocab, prompt, gen_len=200, temperature=0.8, device="cpu"):
        self.eval()
        tokens = [vocab.encode(c) for c in prompt]
        if len(tokens) < self.T:
            tokens = [vocab.encode(' ')] * (self.T - len(tokens)) + tokens
        idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        out_chars = list(prompt)
        for _ in range(gen_len):
            idx_cond = idx[:, -self.T:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            out_chars.append(vocab.decode(next_token.item()))

        return "".join(out_chars)


# =========================================================================
# CHECKPOINTING & RESUME MANAGER (GOOGLE DRIVE READY)
# =========================================================================
class CheckpointManager:
    """
    Manages persistent checkpoints locally and syncs to Google Drive.
    Supports seamless auto-resume after Colab session timeouts.
    """
    def __init__(self, run_id, local_dir="checkpoints", drive_dir=None):
        self.run_id = run_id
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)

        # Check if Google Drive is mounted
        if drive_dir is None and os.path.exists("/content/drive/MyDrive"):
            drive_dir = "/content/drive/MyDrive/EBrain_Checkpoints"
        self.drive_dir = drive_dir
        if self.drive_dir:
            os.makedirs(self.drive_dir, exist_ok=True)
            print(f"[CheckpointManager] Google Drive checkpointing active: {self.drive_dir}")

    def save(self, model, optimizer, scheduler, epoch, step, val_loss, is_best=False):
        checkpoint = {
            'run_id': self.run_id,
            'epoch': epoch,
            'step': step,
            'val_loss': val_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': {
                'V': model.V, 'T': model.T, 'D': model.D,
                'n_heads': model.n_heads, 'H_mlp': model.H_mlp,
                'L': model.L, 'use_layernorm': model.use_layernorm
            }
        }

        # Save latest checkpoint
        latest_file = os.path.join(self.local_dir, f"{self.run_id}_latest.pt")
        torch.save(checkpoint, latest_file)

        if is_best:
            best_file = os.path.join(self.local_dir, f"{self.run_id}_best.pt")
            torch.save(checkpoint, best_file)

        # Sync to Google Drive if active
        if self.drive_dir:
            import shutil
            shutil.copyfile(latest_file, os.path.join(self.drive_dir, f"{self.run_id}_latest.pt"))
            if is_best:
                shutil.copyfile(best_file, os.path.join(self.drive_dir, f"{self.run_id}_best.pt"))

    def load_latest(self, model, optimizer=None, scheduler=None):
        latest_file = os.path.join(self.local_dir, f"{self.run_id}_latest.pt")
        # Check Google Drive if local does not exist
        if not os.path.exists(latest_file) and self.drive_dir:
            drive_file = os.path.join(self.drive_dir, f"{self.run_id}_latest.pt")
            if os.path.exists(drive_file):
                import shutil
                shutil.copyfile(drive_file, latest_file)

        if not os.path.exists(latest_file):
            return 0, 0, float("inf") # Start from beginning

        print(f"[CheckpointManager] Resuming from existing checkpoint: {latest_file}")
        ckpt = torch.load(latest_file, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return ckpt['epoch'], ckpt['step'], ckpt['val_loss']
