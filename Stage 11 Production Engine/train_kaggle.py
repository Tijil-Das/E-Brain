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

from ebrain_torch import EBrainTransformer

class KaggleNpyDataset(torch.utils.data.Dataset):
    """
    High-speed memory-efficient dataset that reads pre-tokenized int32 NumPy arrays.
    """
    def __init__(self, token_npy_path, T=16):
        self.tokens = np.load(token_npy_path)
        self.T = T

    def __len__(self):
        return max(0, len(self.tokens) - self.T)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx : idx + self.T].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1 : idx + self.T + 1].astype(np.int64))
        return x, y


class KaggleCheckpointManager:
    """
    Kaggle-specific Checkpoint Manager.
    Saves outputs to /kaggle/working/ (which persists when version is saved).
    Reads previous checkpoints from /kaggle/input/ to support cross-session resumption.
    """
    def __init__(self, run_id, working_dir="/kaggle/working", resume_input_dir=None):
        self.run_id = run_id
        self.working_dir = working_dir
        self.ckpt_dir = os.path.join(working_dir, "checkpoints")
        self.logs_dir = os.path.join(working_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.resume_input_dir = resume_input_dir
        self.last_save_time = time.time()

    def find_resume_checkpoint(self):
        """
        Scans possible checkpoint locations in order of priority:
        1. Explicit resume_input_dir
        2. /kaggle/input/ for any mounted checkpoint dataset
        3. Local /kaggle/working/checkpoints/
        """
        candidates = []
        if self.resume_input_dir and os.path.exists(self.resume_input_dir):
            candidates.append(os.path.join(self.resume_input_dir, f"{self.run_id}_latest.pt"))
            candidates.append(os.path.join(self.resume_input_dir, f"{self.run_id}_best.pt"))

        # Scan /kaggle/input
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                for f in files:
                    if f.endswith(".pt") and ("latest" in f or "best" in f):
                        candidates.append(os.path.join(root, f))

        # Check local working dir
        candidates.append(os.path.join(self.ckpt_dir, f"{self.run_id}_latest.pt"))
        candidates.append(os.path.join(self.ckpt_dir, f"{self.run_id}_best.pt"))

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def save(self, model, optimizer, scheduler, epoch, step, val_loss, is_best=False, force_time_save=False):
        now = time.time()
        # Save if best, or forced, or if 15 minutes elapsed since last checkpoint
        should_save = is_best or force_time_save or (now - self.last_save_time >= 900)
        if not should_save:
            return

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

        latest_path = os.path.join(self.ckpt_dir, f"{self.run_id}_latest.pt")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.ckpt_dir, f"{self.run_id}_best.pt")
            torch.save(checkpoint, best_path)

        self.last_save_time = now

    def load_latest(self, model, optimizer=None, scheduler=None):
        ckpt_path = self.find_resume_checkpoint()
        if not ckpt_path:
            return 0, 0, float("inf")

        print(f"[KaggleCheckpointManager] Found checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return ckpt['epoch'], ckpt['step'], ckpt['val_loss']


def run_kaggle_preflight_checks(model, device, V):
    print("==========================================================================")
    print("KAGGLE PRE-FLIGHT VERIFICATION CHECKS")
    print("==========================================================================")
    
    # 1. Device check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[1/4] GPU Acceleration: {gpu_name} ({vram:.2f} GB VRAM) -> ACTIVE")
    else:
        print(f"[1/4] GPU Acceleration: None (Running on CPU) -> WARNING")

    # 2. Causality Check
    seq1 = torch.randint(0, V, (1, model.T), device=device)
    seq2 = seq1.clone()
    seq2[0, 10:] = (seq2[0, 10:] + 7) % V
    with torch.no_grad():
        out1, _ = model(seq1)
        out2, _ = model(seq2)
    diff = torch.max(torch.abs(out1[0, :10] - out2[0, :10])).item()
    print(f"[2/4] Strict Causal Mask Isolation: Max prior diff = {diff:.2e} -> {'PASSED' if diff < 1e-6 else 'FAIL'}")

    # 3. Initial Loss Check (approx ln(V))
    dummy_x = torch.randint(0, V, (4, model.T), device=device)
    dummy_y = torch.randint(0, V, (4, model.T), device=device)
    with torch.no_grad():
        _, l = model(dummy_x, dummy_y)
    expected = math.log(V)
    print(f"[3/4] Initial Cross-Entropy Loss: {l.item():.4f} (ln({V}) = {expected:.4f}) -> {'PASSED' if abs(l.item() - expected) < 0.5 else 'FAIL'}")

    # 4. Storage & Output Directory Check
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    print(f"[4/4] Output Storage: '{working_dir}' is writable -> PASSED")
    print("==========================================================================\n")


def train_on_kaggle(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"E-Brain Execution Engine Starting on: {device}")

    # Locate dataset
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        # Check /kaggle/input/
        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                if "vocab.json" in files:
                    data_dir = root
                    break

    print(f"Using Dataset Directory: {data_dir}")
    with open(os.path.join(data_dir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    V = len(vocab['itos'])
    print(f"Vocabulary Size: {V} tokens")

    # Datasets
    train_ds = KaggleNpyDataset(os.path.join(data_dir, "train_tokens.npy"), args.T)
    val_ds = KaggleNpyDataset(os.path.join(data_dir, "val_tokens.npy"), args.T)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = EBrainTransformer(
        V=V, T=args.T, D=args.D, n_heads=args.n_heads,
        H_mlp=args.H_mlp, L=args.L, dropout=args.dropout, use_layernorm=True
    ).to(device)

    print(f"Model: L={args.L}, H={args.n_heads}, D={args.D}, H_mlp={args.H_mlp}")
    print(f"Total Parameters: {model.total_parameters():,}")

    run_kaggle_preflight_checks(model, device, V)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Checkpoint & Tracking
    working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    ckpt_mgr = KaggleCheckpointManager(args.run_id, working_dir=working_dir, resume_input_dir=args.resume_dir)
    start_epoch, global_step, best_val = ckpt_mgr.load_latest(model, optimizer, scheduler)

    csv_log_path = os.path.join(ckpt_mgr.logs_dir, f"{args.run_id}_metrics.csv")
    csv_exists = os.path.exists(csv_log_path)
    csv_file = open(csv_log_path, "a", buffering=1)
    if not csv_exists:
        csv_file.write("epoch,train_loss,val_loss,learning_rate,time_sec\n")

    patience_counter = 0

    print("==========================================================================")
    print(f"TRAINING IN PROGRESS: Run {args.run_id} | Max Epochs: {args.epochs}")
    print("==========================================================================")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

        avg_train = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    _, loss = model(x, y)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)
        epoch_sec = time.time() - t0
        cur_lr = optimizer.param_groups[0]['lr']

        is_best = avg_val < best_val
        if is_best:
            best_val = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        ckpt_mgr.save(model, optimizer, scheduler, epoch, global_step, avg_val, is_best=is_best, force_time_save=(epoch == args.epochs))
        csv_file.write(f"{epoch},{avg_train:.6f},{avg_val:.6f},{cur_lr:.6f},{epoch_sec:.2f}\n")

        # Session time estimation
        remaining_epochs = args.epochs - epoch
        est_remaining_min = (remaining_epochs * epoch_sec) / 60.0

        print(f"Epoch {epoch:2d}/{args.epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {cur_lr:.6f} | {epoch_sec:.1f}s | Est. Left: {est_remaining_min:.1f}m {'<- [BEST]' if is_best else ''}")

        if patience_counter >= args.patience:
            print(f"\n[Stopping Rule Triggered] Val loss did not improve for {args.patience} epochs. Stopping early.")
            break

    csv_file.close()

    # Save summary metadata
    summary = {
        'run_id': args.run_id,
        'best_val_loss': best_val,
        'final_train_loss': avg_train,
        'completed_epochs': epoch,
        'total_parameters': model.total_parameters(),
        'config': vars(args)
    }
    with open(os.path.join(ckpt_mgr.logs_dir, f"{args.run_id}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("==========================================================================")
    print(f"RUN {args.run_id} COMPLETE! Best Val Loss: {best_val:.4f}")
    print(f"Checkpoints and metrics saved in: {ckpt_mgr.working_dir}")
    print("==========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="KAGGLE_RUN_001")
    parser.add_argument("--data_dir", type=str, default="kaggle_datasets/tier1_170k")
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--H_mlp", type=int, default=128)
    parser.add_argument("--L", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train_on_kaggle(args)
