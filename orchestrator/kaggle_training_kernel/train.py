"""
E-Brain 100M Production Training Kernel
========================================
Gate D.2+: Real training kernel pushed to Kaggle daily via GitHub Actions.

Architecture decisions (locked):
  Model:      D=768, L=12, H=12, SwiGLU (hidden=2048), RoPE, RMSNorm, tied embeddings
  Context:    T=2048
  Tokenizer:  frozen 16K BPE, SHA-256 = 41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf
  Target:     2.40B tokens = 73,242 steps, batch = 32,768 tokens (micro=4, accum=4)
  Schedule:   warmup 2000 steps + cosine decay to 0.1*lr_max over 73,242 steps
  Languages:  en es ru fr de ja zh hi bn ar  (equal 10% share each)

Kaggle dataset mounts (set in kernel-metadata.json):
  /kaggle/input/datasets/TRAINING_USERNAME/ebrain-data-binary/    — uint16 binary shards
  /kaggle/input/datasets/TRAINING_USERNAME/ebrain-checkpoints/    — latest checkpoint + manifest
  /kaggle/input/datasets/TRAINING_USERNAME/ebrain-val-vault/      — frozen val files
  /kaggle/input/datasets/TRAINING_USERNAME/ebrain-tokenizer/      — frozen 16K tokenizer JSON

Secrets (set as Kaggle environment variables in kernel settings):
  WORKER_URL        — Cloudflare Worker base URL
  HEARTBEAT_TOKEN   — shared secret for POST /heartbeat
  KAGGLE_USERNAME   — training account username (for dataset upload)
  KAGGLE_KEY        — training account API key
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os, sys, json, time, math, hashlib, struct, random, io, shutil, signal
import subprocess, traceback, csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAINING_USERNAME = os.environ.get("KAGGLE_USERNAME", "REPLACE_ME")
# Everything is self-contained in ebrain-100m-data-binary
DATA_DIR    = Path("/kaggle/input/datasets") / TRAINING_USERNAME / "ebrain-100m-data-binary"
if not DATA_DIR.exists():
    # Fallback to standard Kaggle mount path
    DATA_DIR = Path("/kaggle/input/ebrain-100m-data-binary")

CKPT_DIR    = Path("/kaggle/input/datasets") / TRAINING_USERNAME / "ebrain-checkpoints"
if not CKPT_DIR.exists():
    CKPT_DIR = Path("/kaggle/input/ebrain-checkpoints")

VAL_DIR     = DATA_DIR / "val_vault"
TOK_DIR     = DATA_DIR
OUT_DIR     = Path("/kaggle/working")
CKPT_OUT    = OUT_DIR / "checkpoints"
LOG_OUT     = OUT_DIR / "logs"
CKPT_OUT.mkdir(parents=True, exist_ok=True)
LOG_OUT.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
D_MODEL     = 768
N_LAYERS    = 12
N_HEADS     = 12
D_HEAD      = D_MODEL // N_HEADS          # 64
D_FF        = 2048                        # SwiGLU hidden
VOCAB_SIZE  = 16000
T_CTX       = 2048
MICRO_BS    = 4                           # sequences per micro-batch
ACCUM_STEPS = 4                           # gradient accumulation
GLOBAL_BS   = MICRO_BS * ACCUM_STEPS * T_CTX   # = 32,768 tokens
LR_MAX      = 3e-4
LR_MIN      = LR_MAX * 0.1
WARMUP_STEPS = 2000
TOTAL_STEPS = 61_241                      # 2,006,731,161 tokens / 32,768 = 61,241 steps (1.8x repeat cap)
CLIP_NORM   = 1.0
DTYPE       = torch.float16

LANGUAGES   = ['en','es','ru','fr','de','ja','zh','hi','bn','ar']
TOK_SHA256  = "41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf"

# ── Training orchestration config ─────────────────────────────────────────────
WALL_BUDGET_S   = 3 * 3600 + 50 * 60     # 3 h 50 min
CKPT_INTERVAL_S = 30 * 60                # checkpoint every 30 min
EVAL_INTERVAL   = 2_000                  # eval every N steps (small val)
HEARTBEAT_EVERY = 50                     # steps between heartbeat POSTs
CONTROL_EVERY   = 100                    # steps between control GETs
NAN_HALT_AFTER  = 3                      # consecutive NaN batches before halt
MAX_CKPTS_KEEP  = 2                      # keep last N checkpoints + best
VAL_LANG_GAP_ALERT = 0.15
VAL_CONSEC_RISE = 2                      # consecutive val rises → halt

WORKER_URL       = os.environ.get("WORKER_URL", "")
HEARTBEAT_TOKEN  = os.environ.get("HEARTBEAT_TOKEN", "")
KAGGLE_KEY       = os.environ.get("KAGGLE_KEY", "")
RESUME_TEST_MODE = os.environ.get("RESUME_TEST_MODE", "0") == "1"

# ── Run ID ────────────────────────────────────────────────────────────────────
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
print(f"\n{'='*72}\nE-Brain 100M Training Kernel  |  {RUN_ID}\n{'='*72}")

# ── RoPE helpers ─────────────────────────────────────────────────────────────

def build_rope_cache(T, d_head, device):
    theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos   = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, theta)        # (T, d_head/2)
    cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, d_head)
    sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin

def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

# ── Model ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out  = nn.Linear(D_MODEL, D_MODEL, bias=False)
    def forward(self, x, cos, sin, mask):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k,
                           cos[:T].unsqueeze(0).unsqueeze(0),
                           sin[:T].unsqueeze(0).unsqueeze(0))
        attn = (q @ k.transpose(-2, -1)) * (D_HEAD ** -0.5)
        attn = attn.masked_fill(mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(D_MODEL, D_FF, bias=False)
        self.w2 = nn.Linear(D_FF, D_MODEL, bias=False)
        self.w3 = nn.Linear(D_MODEL, D_FF, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(D_MODEL)
        self.attn = MHA()
        self.ln2  = RMSNorm(D_MODEL)
        self.ffn  = FFN()
    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.ffn(self.ln2(x))
        return x

class EBrain100M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb    = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f   = RMSNorm(D_MODEL)
        # Tied embeddings: lm_head weight = emb weight (no extra params)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.weight
        # RoPE cache (registered as buffer — persistent=True)
        cos, sin = build_rope_cache(T_CTX, D_HEAD, torch.device('cpu'))
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        # Causal mask
        mask = torch.ones(T_CTX, T_CTX, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask)
        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x)
        cos = self.cos[:T]
        sin = self.sin[:T]
        for block in self.blocks:
            h = block(h, cos, sin, self.mask)
        h = self.ln_f(h)
        return self.lm_head(h)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        non_emb = sum(p.numel() for n, p in self.named_parameters()
                      if 'emb' not in n and 'lm_head' not in n)
        return total, non_emb

# ── LR schedule ───────────────────────────────────────────────────────────────

def lr_at_step(step):
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    t = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))

# ── Tokenizer SHA-256 check ───────────────────────────────────────────────────

def verify_tokenizer():
    tok_path = TOK_DIR / "ebrain_tokenizer_16k.json"
    if not tok_path.exists():
        # Try fallback inside kernel working dir (pushed with kernel)
        tok_path = Path("/kaggle/working/ebrain_tokenizer_16k.json")
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}")
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    assert sha == TOK_SHA256, f"Tokenizer SHA-256 mismatch!\n  got:      {sha}\n  expected: {TOK_SHA256}"
    print(f"✅ Tokenizer SHA-256 verified: {sha[:16]}...")
    return tok_path

# ── Binary shard data loader ──────────────────────────────────────────────────

class ShardDataset:
    """
    Deterministic, resumable loader over uint16 binary shards.
    Equal token share: interleave languages round-robin at sequence level.
    State: (shard_idx, offset_in_shard, epoch) per language.
    """
    def __init__(self, data_dir, seed=42, position=None):
        self.data_dir = Path(data_dir)
        self.rng = random.Random(seed)
        self.T = T_CTX

        # Load all shards per language
        self.shards = {}
        for lang in LANGUAGES:
            files = sorted(self.data_dir.glob(f"train_{lang}*.bin"))
            if not files:
                raise FileNotFoundError(f"No binary shards for language '{lang}' in {data_dir}")
            self.shards[lang] = files
            print(f"  {lang}: {len(files)} shard(s)  ({sum(f.stat().st_size for f in files)/(1<<20):.1f} MB)")

        # State per language: (shard_idx, token_offset, epoch)
        if position:
            self.pos = {lang: list(position[lang]) for lang in LANGUAGES}
        else:
            self.pos = {lang: [0, 0, 0] for lang in LANGUAGES}

        # Round-robin lang pointer
        self.lang_ptr = 0

    def _load_tokens(self, lang, shard_idx):
        path = self.shards[lang][shard_idx % len(self.shards[lang])]
        data = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype(np.int64)
        return data

    def next_sequence(self):
        """Return (tokens: np.int64 [T+1],  lang, position_snapshot)."""
        while True:
            lang = LANGUAGES[self.lang_ptr % len(LANGUAGES)]
            self.lang_ptr += 1

            si, offset, epoch = self.pos[lang]
            tokens = self._load_tokens(lang, si)

            if offset + self.T + 1 > len(tokens):
                # Advance shard
                si += 1
                if si >= len(self.shards[lang]):
                    si = 0
                    epoch += 1
                self.pos[lang] = [si, 0, epoch]
                continue

            seq = tokens[offset: offset + self.T + 1]
            self.pos[lang] = [si, offset + self.T, epoch]
            pos_snap = {l: list(self.pos[l]) for l in LANGUAGES}
            return seq, lang, pos_snap

    def get_state(self):
        return {lang: list(self.pos[lang]) for lang in LANGUAGES}

    def set_state(self, state):
        self.pos = {lang: list(state[lang]) for lang in LANGUAGES}

# ── Checkpoint helpers ────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def get_rng_state():
    state = {
        'torch':   torch.get_rng_state().numpy().tolist(),
        'numpy':   np.random.get_state(),
        'python':  random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state().numpy().tolist()
    return state

def set_rng_state(state):
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.tensor(state['cuda'], dtype=torch.uint8))

def save_checkpoint(model, optimizer, scaler, scheduler_step, step, best_val,
                    data_state, next_batch_hash, label, extra=None):
    """Atomic checkpoint: write to temp, fsync, rename."""
    path = CKPT_OUT / f"ckpt_{label}.pt"
    tmp  = CKPT_OUT / f"ckpt_{label}.tmp"
    ckpt = {
        'step':             step,
        'model':            model.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'scaler':           scaler.state_dict(),
        'scheduler_step':   scheduler_step,
        'lr':               lr_at_step(step),
        'rng':              get_rng_state(),
        'data_state':       data_state,
        'next_batch_hash':  next_batch_hash,
        'best_val':         best_val,
        'run_id':           RUN_ID,
        'extra':            extra or {},
    }
    torch.save(ckpt, tmp)
    # fsync
    with open(tmp, 'rb+') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)   # atomic on POSIX/Linux
    sha = sha256_file(path)
    print(f"  [ckpt] saved {path.name}  SHA-256={sha[:16]}...  step={step}")
    return path, sha

def prune_checkpoints(keep=MAX_CKPTS_KEEP):
    """Keep last `keep` numbered checkpoints + best."""
    numbered = sorted(CKPT_OUT.glob("ckpt_step*.pt"),
                      key=lambda p: int(p.stem.split('_')[1].replace('step','')),
                      reverse=True)
    for old in numbered[keep:]:
        old.unlink(missing_ok=True)
        print(f"  [prune] removed {old.name}")

def load_latest_checkpoint():
    """
    Load from the mounted Kaggle dataset. Verify SHA-256 against manifest.
    Falls back to previous checkpoint if SHA-256 mismatch.
    Returns ckpt dict or None if no checkpoint found.
    """
    manifest_path = CKPT_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest found — starting from scratch (day 1).")
        return None
    manifest = json.loads(manifest_path.read_text())

    for entry in manifest.get('checkpoints', []):
        path = CKPT_DIR / entry['filename']
        if not path.exists():
            continue
        actual_sha = sha256_file(path)
        if actual_sha != entry['sha256']:
            print(f"  ⚠️  SHA-256 mismatch for {entry['filename']} — skipping (corrupt?)")
            continue
        print(f"✅ Loaded checkpoint: {entry['filename']}  SHA-256={actual_sha[:16]}...")
        ckpt = torch.load(path, map_location='cpu')
        return ckpt

    print("All checkpoints corrupt or missing — starting from scratch.")
    return None

def save_manifest(entries):
    path = OUT_DIR / "manifest.json"
    path.write_text(json.dumps({'checkpoints': entries, 'run_id': RUN_ID,
                                 'ts': datetime.now(timezone.utc).isoformat()}, indent=2))
    return path

# ── Heartbeat / Control ───────────────────────────────────────────────────────

_heartbeat_errors = 0

def post_heartbeat(payload):
    global _heartbeat_errors
    if not WORKER_URL:
        return
    try:
        r = requests.post(f"{WORKER_URL}/heartbeat",
                          json=payload,
                          headers={"X-Token": HEARTBEAT_TOKEN},
                          timeout=10)
        _heartbeat_errors = 0
    except Exception:
        _heartbeat_errors += 1

def get_control():
    if not WORKER_URL:
        return "RUN"
    try:
        r = requests.get(f"{WORKER_URL}/control", timeout=10)
        return r.json().get("control", "RUN")
    except Exception:
        return "RUN"

# ── Validation ────────────────────────────────────────────────────────────────

def evaluate_val(model, device, full=False, limit_per_lang=50):
    model.eval()
    results = {}
    with torch.no_grad():
        for lang in LANGUAGES:
            # Val files are plain .txt — tokenize on the fly using the frozen tokenizer.
            # Fall back to .bin only if a pre-tokenized version somehow exists.
            txt_file = VAL_DIR / f"val_{lang}.txt"
            bin_file = VAL_DIR / f"val_{lang}.bin"
            if txt_file.exists():
                try:
                    from tokenizers import Tokenizer as _Tok
                    _tok = _Tok.from_file(str(TOK_DIR / "ebrain_tokenizer_16k.json"))
                    text = txt_file.read_text(encoding="utf-8", errors="replace")
                    enc  = _tok.encode(text)
                    tokens = np.array(enc.ids, dtype=np.int64)
                except Exception as e:
                    print(f"  [val] tokenizer error for {lang}: {e}")
                    continue
            elif bin_file.exists():
                tokens = np.frombuffer(bin_file.read_bytes(), dtype=np.uint16).astype(np.int64)
            else:
                print(f"  [val] no val file found for {lang} (checked .txt and .bin) — skipping")
                continue
            total_loss, total_toks = 0.0, 0
            n_seq = 0
            i = 0
            while i + T_CTX + 1 <= len(tokens):
                if not full and n_seq >= limit_per_lang:
                    break
                seq = torch.tensor(tokens[i:i+T_CTX+1], dtype=torch.long, device=device)
                x = seq[:-1].unsqueeze(0)
                y = seq[1:].unsqueeze(0)
                with autocast(dtype=DTYPE):
                    logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.contiguous().view(-1))
                total_loss += loss.item() * T_CTX
                total_toks += T_CTX
                i += T_CTX
                n_seq += 1
            if total_toks > 0:
                nats = total_loss / total_toks
                results[lang] = nats / math.log(2)   # nats → bits per byte
    model.train()
    return results

# ── Batch hash ────────────────────────────────────────────────────────────────

def batch_hash(batch_tensor):
    return hashlib.sha256(batch_tensor.numpy().tobytes()).hexdigest()

# ── Sample generation ─────────────────────────────────────────────────────────

def generate_samples(model, device, tok_path, n_tokens=100):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    model.eval()
    samples = {}
    with torch.no_grad():
        for lang in LANGUAGES:
            # Seed with a neutral prompt token (0)
            x = torch.zeros(1, 1, dtype=torch.long, device=device)
            generated = []
            for _ in range(n_tokens):
                with autocast(dtype=DTYPE):
                    logits = model(x[:, -T_CTX:])
                probs = F.softmax(logits[0, -1] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                x = torch.cat([x, nxt.unsqueeze(0)], dim=1)
                generated.append(nxt.item())
            samples[lang] = tok.decode(generated)
    model.train()
    return samples

# ── Upload to Kaggle dataset ──────────────────────────────────────────────────

def upload_checkpoint_dataset(ckpt_entries, csv_log_path, samples):
    """Upload checkpoint + logs as new Kaggle dataset version."""
    if not KAGGLE_KEY or not TRAINING_USERNAME:
        print("  [upload skip] KAGGLE_KEY or TRAINING_USERNAME not set")
        return
    # Write manifest
    mf = save_manifest(ckpt_entries)
    # Write samples
    samp_path = OUT_DIR / "samples.json"
    samp_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    # Build upload dir
    up_dir = OUT_DIR / "upload"
    up_dir.mkdir(exist_ok=True)
    for src in [mf, csv_log_path, samp_path]:
        if src and src.exists():
            shutil.copy(src, up_dir / src.name)
    for f in CKPT_OUT.glob("*.pt"):
        shutil.copy(f, up_dir / f.name)

    # Write dataset-metadata.json
    meta = {
        "title": "EBrain Checkpoints",
        "id": f"{TRAINING_USERNAME}/ebrain-checkpoints",
        "licenses": [{"name": "other"}],
    }
    (up_dir / "dataset-metadata.json").write_text(json.dumps(meta))

    # Push new version
    env = {**os.environ,
           "KAGGLE_USERNAME": TRAINING_USERNAME,
           "KAGGLE_KEY":      KAGGLE_KEY}
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(up_dir),
             "-m", f"{RUN_ID} step={ckpt_entries[-1]['step'] if ckpt_entries else '?'}"],
            capture_output=True, text=True, env=env, timeout=300
        )
        print(f"  [upload] {result.stdout.strip() or result.stderr.strip()}")
    except Exception as e:
        print(f"  [upload fail] {e}")

# ── Main training loop ────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1<<30):.2f} GB")

    # ── Tokenizer verify
    tok_path = verify_tokenizer()

    # ── Build model
    model = EBrain100M().to(device)
    total_p, non_emb_p = model.param_count()
    print(f"Model: {total_p:,} total params  ({non_emb_p:,} non-embedding)")

    # ── Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95),
                                  weight_decay=0.1, eps=1e-8)
    scaler = GradScaler()

    # ── Load checkpoint
    ckpt = load_latest_checkpoint()
    start_step  = 0
    best_val    = float('inf')
    data_pos    = None
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        set_rng_state(ckpt['rng'])
        start_step  = ckpt['step']
        best_val    = ckpt.get('best_val', float('inf'))
        data_pos    = ckpt.get('data_state')
        stored_next_batch_hash = ckpt.get('next_batch_hash')
        print(f"Resuming from step {start_step}")

    # ── Restore LR to correct step
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_at_step(start_step)

    # ── Data loader
    loader = ShardDataset(DATA_DIR, seed=42, position=data_pos)

    # ── Step-0 sanity check
    if start_step == 0:
        with torch.no_grad():
            dummy_x = torch.zeros(1, T_CTX, dtype=torch.long, device=device)
            with autocast(dtype=DTYPE):
                logits = model(dummy_x)
            loss0 = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                                     torch.zeros(T_CTX, dtype=torch.long, device=device))
            expected = math.log(VOCAB_SIZE)
            print(f"Step-0 loss: {loss0.item():.4f}  (expected ≈ ln({VOCAB_SIZE}) = {expected:.4f})")
            assert abs(loss0.item() - expected) < 0.5, "Step-0 loss too far from expected!"

    # ── Overfit test (1 batch, check loss < 0.5)
    if start_step == 0 and os.environ.get("OVERFIT_TEST", "0") == "1":
        print("\n--- OVERFIT TEST ---")
        seq, _, _ = loader.next_sequence()
        x_ov = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y_ov = torch.tensor(seq[1:],  dtype=torch.long, device=device).unsqueeze(0)
        for i in range(200):
            with autocast(dtype=DTYPE):
                logits = model(x_ov)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y_ov.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            if i % 50 == 0:
                print(f"  overfit step {i}: loss={loss.item():.4f}")
        assert loss.item() < 0.5, f"Overfit test failed: loss={loss.item():.4f}"
        print("--- OVERFIT TEST PASSED ---\n")
        # Reset model and optimizer
        model = EBrain100M().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95),
                                      weight_decay=0.1, eps=1e-8)
        scaler = GradScaler()
        loader = ShardDataset(DATA_DIR, seed=42)

    # ── CSV log
    csv_path = LOG_OUT / f"train_log_{RUN_ID}.csv"
    csv_file  = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['step','loss','lr','tok_s','vram_gb','elapsed_s','grad_norm'])

    # ── Tracking
    t_session_start = time.time()
    t_last_ckpt     = time.time()
    nan_count        = 0
    val_history      = []          # list of {step, mean_bpb, per_lang}
    val_consec_rise  = 0
    ckpt_manifest    = []
    step             = start_step
    optimizer.zero_grad()
    t_step_start     = time.time()
    tokens_since_log = 0
    loss_sum         = 0.0
    loss_count       = 0
    grad_norm_last   = 0.0

    # For resume test mode: compare first batch hash
    first_batch_done = False

    print(f"\n{'='*72}")
    print(f"Starting training  step={step}  target={TOTAL_STEPS}")
    print(f"Wall budget: {WALL_BUDGET_S/3600:.2f} h")
    print(f"{'='*72}\n")

    while step < TOTAL_STEPS:
        # ── Wall-clock stop ─────────────────────────────────────────────────
        elapsed = time.time() - t_session_start
        if elapsed >= WALL_BUDGET_S:
            print(f"\n⏰ Wall-clock budget reached ({elapsed/3600:.2f} h). Saving checkpoint and exiting.")
            break

        # ── Accumulate micro-batches ────────────────────────────────────────
        micro_losses = []
        micro_tokens = []
        for acc_i in range(ACCUM_STEPS):
            seq, lang, pos_snap = loader.next_sequence()
            x = torch.tensor(seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(seq[1:],  dtype=torch.long, device=device).unsqueeze(0)

            # First batch hash for resume test
            if not first_batch_done and acc_i == 0:
                first_batch_done = True
                fb_hash = batch_hash(x.cpu())
                if RESUME_TEST_MODE and 'stored_next_batch_hash' in dir():
                    if fb_hash == stored_next_batch_hash:
                        print(f"✅ RESUME TEST: batch hash matches  {fb_hash[:16]}...")
                    else:
                        msg = f"❌ RESUME TEST FAIL: hash mismatch\n  got:      {fb_hash}\n  expected: {stored_next_batch_hash}"
                        print(msg)
                        post_heartbeat({"step": step, "loss": 0, "lr": 0, "tok_s": 0,
                                        "vram_gb": 0, "eta_h": 0, "nan_alert": False,
                                        "loss_spike": False, "val_alert": msg,
                                        "lang_gap_alert": None})
                        sys.exit(1)

            with autocast(dtype=DTYPE):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                                        y.contiguous().view(-1)) / ACCUM_STEPS

            scaler.scale(loss).backward()
            micro_losses.append(loss.item() * ACCUM_STEPS)
            micro_tokens.append(T_CTX)

        # ── NaN guard ───────────────────────────────────────────────────────
        raw_loss = sum(micro_losses) / len(micro_losses)
        if math.isnan(raw_loss) or math.isinf(raw_loss):
            nan_count += 1
            print(f"  ⚠️  NaN/Inf at step {step}  ({nan_count}/{NAN_HALT_AFTER})")
            post_heartbeat({"step": step, "loss": raw_loss, "lr": lr_at_step(step),
                             "tok_s": 0, "vram_gb": 0, "eta_h": 0,
                             "nan_alert": True, "loss_spike": False,
                             "val_alert": None, "lang_gap_alert": None})
            optimizer.zero_grad()
            scaler.update()
            if nan_count >= NAN_HALT_AFTER:
                print(f"  ❌ {NAN_HALT_AFTER} consecutive NaN — halting.")
                break
            continue
        nan_count = 0

        # ── Gradient step ───────────────────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm_last = nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # LR update
        new_lr = lr_at_step(step + 1)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        step += 1
        loss_sum   += raw_loss
        loss_count += 1
        tokens_since_log += GLOBAL_BS

        # ── Logging every 50 steps ──────────────────────────────────────────
        if step % HEARTBEAT_EVERY == 0:
            t_now = time.time()
            elapsed_s = t_now - t_session_start
            dt = t_now - t_step_start
            tok_s = tokens_since_log / dt if dt > 0 else 0
            mean_loss = loss_sum / loss_count
            vram_gb = torch.cuda.memory_reserved(device) / (1<<30) if device.type == 'cuda' else 0
            steps_left = TOTAL_STEPS - step
            eta_h = (steps_left * dt / max(1, HEARTBEAT_EVERY)) / 3600

            # Loss spike check
            loss_spike = mean_loss > 2.0 * (val_history[-1]['mean_bpb'] * math.log(2) if val_history else 9.68)

            csv_writer.writerow([step, mean_loss, new_lr, tok_s, vram_gb, elapsed_s, grad_norm_last])
            csv_file.flush()

            print(f"[step {step:6d}]  loss={mean_loss:.4f}  lr={new_lr:.2e}  "
                  f"tok/s={tok_s:,.0f}  VRAM={vram_gb:.2f}GB  ETA={eta_h:.1f}h  "
                  f"|g|={grad_norm_last:.3f}")

            post_heartbeat({
                "step": step, "loss": mean_loss, "lr": new_lr,
                "tok_s": tok_s, "vram_gb": vram_gb, "eta_h": eta_h,
                "nan_alert": False, "loss_spike": loss_spike,
                "val_alert": None, "lang_gap_alert": None,
            })

            # Reset counters
            t_step_start     = time.time()
            tokens_since_log = 0
            loss_sum         = 0.0
            loss_count       = 0

        # ── Control poll ────────────────────────────────────────────────────
        if step % CONTROL_EVERY == 0:
            ctrl = get_control()
            if ctrl in ("PAUSE", "STOP"):
                print(f"\n{'⏸' if ctrl == 'PAUSE' else '⏹'}  {ctrl} received at step {step}. Saving checkpoint.")
                break

        # ── Periodic checkpoint ─────────────────────────────────────────────
        if time.time() - t_last_ckpt >= CKPT_INTERVAL_S:
            # Peek next batch hash without consuming it
            seq_peek, _, _ = loader.next_sequence()
            nbh = batch_hash(torch.tensor(seq_peek[:-1], dtype=torch.long).unsqueeze(0))
            # Reload that sequence into loader (can't un-consume; we store the hash only)
            path, sha = save_checkpoint(model, optimizer, scaler, step, step, best_val,
                                         loader.get_state(), nbh, f"step{step}")
            ckpt_manifest.append({"filename": path.name, "sha256": sha,
                                   "step": step, "ts": datetime.now(timezone.utc).isoformat()})
            prune_checkpoints()
            t_last_ckpt = time.time()

        # ── Eval ────────────────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            print(f"\n--- Eval at step {step} ---")
            per_lang = evaluate_val(model, device, full=False, limit_per_lang=50)
            mean_bpb = sum(per_lang.values()) / len(per_lang) if per_lang else float('inf')
            val_history.append({"step": step, "mean_bpb": mean_bpb, "per_lang": per_lang})
            print(f"  Mean BPB={mean_bpb:.4f}  per-lang: {per_lang}")

            # Update best checkpoint
            if mean_bpb < best_val:
                best_val = mean_bpb
                path, sha = save_checkpoint(model, optimizer, scaler, step, step, best_val,
                                             loader.get_state(), None, "best")
                ckpt_manifest.append({"filename": path.name, "sha256": sha,
                                       "step": step, "ts": datetime.now(timezone.utc).isoformat(),
                                       "is_best": True})

            # Stop rules
            if len(val_history) >= 2 and val_history[-1]['mean_bpb'] > val_history[-2]['mean_bpb']:
                val_consec_rise += 1
            else:
                val_consec_rise = 0
            if val_consec_rise >= VAL_CONSEC_RISE:
                print(f"  ⚠️  Val BPB rose {VAL_CONSEC_RISE} consecutive evals — halting.")
                break

            # Language gap alerts
            gap_alerts = []
            if len(per_lang) > 1:
                vals = list(per_lang.values())
                gap = max(vals) - min(vals)
                if gap > VAL_LANG_GAP_ALERT:
                    worst = max(per_lang, key=per_lang.get)
                    gap_alerts.append(f"lang gap={gap:.4f} worst={worst}:{per_lang[worst]:.4f}")

            post_heartbeat({
                "step": step, "loss": mean_bpb, "lr": lr_at_step(step),
                "tok_s": 0, "vram_gb": 0, "eta_h": 0,
                "nan_alert": False, "loss_spike": False,
                "val_per_lang": per_lang,
                "val_alert": f"Mean BPB={mean_bpb:.4f}",
                "lang_gap_alert": "; ".join(gap_alerts) if gap_alerts else None,
            })

    # ── End of session ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Session ended at step {step}. Saving final checkpoint.")

    # Full val + Alice
    print("Running full val...")
    per_lang_full = evaluate_val(model, device, full=True)
    mean_bpb_full = sum(per_lang_full.values()) / len(per_lang_full) if per_lang_full else float('inf')
    print(f"Full val: mean BPB={mean_bpb_full:.4f}")
    for lang, bpb in sorted(per_lang_full.items()):
        print(f"  {lang}: {bpb:.4f}")

    # Sample generation
    print("Generating samples...")
    samples = generate_samples(model, device, tok_path, n_tokens=100)
    for lang, text in samples.items():
        print(f"\n--- Sample [{lang}] ---\n{text[:200]}")

    # Final checkpoint
    path, sha = save_checkpoint(model, optimizer, scaler, step, step, best_val,
                                 loader.get_state(), None, f"final_{RUN_ID}")
    ckpt_manifest.append({"filename": path.name, "sha256": sha, "step": step,
                           "ts": datetime.now(timezone.utc).isoformat(), "is_final": True})
    csv_file.close()

    # Upload to Kaggle
    upload_checkpoint_dataset(ckpt_manifest, csv_path, samples)

    # Day-end heartbeat
    post_heartbeat({
        "step": step, "loss": mean_bpb_full, "lr": lr_at_step(step),
        "tok_s": 0, "vram_gb": 0, "eta_h": 0,
        "nan_alert": False, "loss_spike": False,
        "val_per_lang": per_lang_full,
        "val_alert": f"Day-end full val: mean BPB={mean_bpb_full:.4f}",
        "lang_gap_alert": None,
        "checkpoint_hash": sha,
        "session_complete": True,
    })

    print(f"\n✅ Session complete. Step={step}. Mean val BPB={mean_bpb_full:.4f}.")
    print(f"   Run ID: {RUN_ID}")
    print(f"   Final checkpoint SHA-256: {sha}")
    print(f"{'='*72}\n")

if __name__ == '__main__':
    main()
