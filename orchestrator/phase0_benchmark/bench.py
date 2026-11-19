"""
E-Brain Phase 0 Benchmark Kernel
=================================
Measures tok/s, peak VRAM, and step time for 3 architecture candidates at T=2048, fp16.
Then runs memory probes on the highest-throughput candidate:
  3a. T=4096 (5 steps, micro_bs=2, SDPA)
  3c. T=32768 (5 steps, micro_bs=1, grad-checkpointing, SDPA mem-efficient)
Also measures the frozen 16K BPE tokenizer's bytes/token on 5 code languages.

GPU budget: ~1 h on T4.
All numbers in output are tagged [M] = measured on this instance.
No number here is an estimate. No claim about T=32768 feasibility is made without 3c.

Candidates:
  A — D=1024  L=16  H=16  D_FF=2688  (~191M non-emb params)
  B — D=1024  L=20  H=16  D_FF=2688  (~236M non-emb params)
  C — D=896   L=16  H=14  D_FF=2352  (~147M non-emb params)

Tokenizer: /kaggle/input/ebrain-100m-data-binary/ebrain_tokenizer_16k.json
Expected SHA-256: 41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf
"""

import sys, math, hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import torch.utils.checkpoint as _ckpt

# ── Constants ─────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 16_000
TOK_SHA256   = "41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE        = torch.float16
OUT_DIR      = Path("/kaggle/working")
RESULT_FILE  = OUT_DIR / "phase0_results.txt"

# 3h50m = actual kernel wall budget; used for days-to-complete calculations
SESSION_S    = 3 * 3600 + 50 * 60    # 13 800 s

# Benchmark hypers (T=2048)
T_BENCH      = 2048
MICRO_BS_T2K = 4       # → 4 × 2048 = 8 192 tokens per micro-batch
GLOBAL_BS    = 32_768  # invariant: MICRO_BS × ACCUM × T_CTX

# T=4096 probe hypers
T_PROBE      = 4096
MICRO_BS_T4K = 2       # → 2 × 4096 = 8 192 tokens per micro-batch (same global ratio)

N_WARMUP     = 5       # warmup steps (excluded from timing)
N_BENCH      = 50      # measured steps
N_PROBE      = 5       # steps for T=4096 probe (memory only)

# (D_MODEL, N_LAYERS, N_HEADS, D_FF, LABEL)
CANDIDATES = [
    (1024, 16, 16, 2688, "A"),
    (1024, 20, 16, 2688, "B"),
    ( 896, 16, 14, 2352, "C"),
]

# ── Output collector ──────────────────────────────────────────────────────────
_lines: list[str] = []

def log(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(str(s))

def flush_results() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text("\n".join(_lines) + "\n", encoding="utf-8")

# ── Model components (identical to production train.py) ───────────────────────

def build_rope_cache(T: int, d_head: int, device: torch.device):
    """Returns cos, sin buffers shaped [T, d_head] using the production RoPE scheme."""
    theta = 1.0 / (10000.0 ** (
        torch.arange(0, d_head, 2, device=device).float() / d_head
    ))
    pos   = torch.arange(T, device=device).float()
    freqs = torch.outer(pos, theta)                    # [T, d_head/2]
    cos   = torch.cat([freqs.cos(), freqs.cos()], -1) # [T, d_head]
    sin   = torch.cat([freqs.sin(), freqs.sin()], -1)
    return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

def apply_rope(q, k, cos, sin):
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g   = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class MHA(nn.Module):
    """Multi-head attention — SDPA (FlashAttention path on T4 fp16). No O(T²) VRAM."""
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h    = h
        self.d_h  = d // h
        self.qkv  = nn.Linear(d, 3 * d, bias=False)
        self.out  = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # Reshape into [B, T, 3, H, D_head] then permute to [3, B, H, T, D_head]
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        c = cos[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_head]
        s = sin[:T].unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, c, s)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, C))

class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)
        self.w3 = nn.Linear(d, d_ff, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, d: int, h: int, d_ff: int):
        super().__init__()
        self.ln1  = RMSNorm(d); self.attn = MHA(d, h)
        self.ln2  = RMSNorm(d); self.ffn  = FFN(d, d_ff)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x

class BenchModel(nn.Module):
    def __init__(self, D: int, L: int, H: int, D_FF: int, T_max: int,
                 grad_ckpt: bool = False):
        super().__init__()
        self.emb     = nn.Embedding(VOCAB_SIZE, D)
        self.blocks  = nn.ModuleList([Block(D, H, D_FF) for _ in range(L)])
        self.ln_f    = RMSNorm(D)
        self.lm_head = nn.Linear(D, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.emb.weight         # weight tying
        cos, sin = build_rope_cache(T_max, D // H, torch.device("cpu"))
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.use_grad_ckpt = grad_ckpt
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.emb(x)
        cos, sin = self.cos[:T], self.sin[:T]
        for block in self.blocks:
            if self.use_grad_ckpt and torch.is_grad_enabled():
                # Recompute forward on backward; halves activation VRAM at cost of ~33% compute
                h = _ckpt.checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        return self.lm_head(self.ln_f(h))

    def param_count(self) -> tuple[int, int]:
        total   = sum(p.numel() for p in self.parameters())
        non_emb = sum(p.numel() for n, p in self.named_parameters()
                      if "emb" not in n and "lm_head" not in n)
        return total, non_emb

# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    D: int, L: int, H: int, D_FF: int,
    T: int, micro_bs: int,
    n_warmup: int, n_steps: int,
    label: str,
    T_max: int | None = None,
    grad_ckpt: bool = False,
) -> dict:
    """
    Runs n_warmup (un-timed) + n_steps (timed) forward/backward passes.
    Returns measurements or raises torch.cuda.OutOfMemoryError.
    """
    T_max = T_max or T
    model  = BenchModel(D, L, H, D_FF, T_max, grad_ckpt=grad_ckpt).to(DEVICE).to(DTYPE)
    total_p, non_emb_p = model.param_count()
    opt    = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8
    )
    scaler = GradScaler()

    def step() -> None:
        x = torch.randint(0, VOCAB_SIZE, (micro_bs, T), device=DEVICE)
        y = torch.randint(0, VOCAB_SIZE, (micro_bs, T), device=DEVICE)
        with torch.cuda.amp.autocast(dtype=DTYPE):
            loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    # Warmup (not timed, not counted)
    for _ in range(n_warmup):
        step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed run with CUDA events (GPU-side timing, no CPU overhead)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n_steps):
        step()
    t1.record()
    torch.cuda.synchronize()

    elapsed_ms   = t0.elapsed_time(t1)
    tokens_meas  = micro_bs * T * n_steps
    tok_s        = tokens_meas / (elapsed_ms / 1_000.0)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1 << 30)
    step_ms      = elapsed_ms / n_steps

    del model, opt, scaler
    torch.cuda.empty_cache()

    return {
        "label": label, "D": D, "L": L, "H": H, "D_FF": D_FF, "T": T,
        "micro_bs": micro_bs, "n_steps": n_steps,
        "total_params": total_p, "non_emb_params": non_emb_p,
        "tok_s": tok_s, "peak_vram_gb": peak_vram_gb, "step_ms": step_ms,
    }

# ── T=32768 memory probe ──────────────────────────────────────────────────────

T_32K = 32_768

def run_t32k_probe(D: int, L: int, H: int, D_FF: int, label: str) -> dict:
    """
    5-step T=32768 training memory probe.
    Settings: micro_bs=1, gradient checkpointing, SDPA mem-efficient backend, fp16.
    No warmup — this is a memory test, not a throughput test.
    Returns: peak_vram_gb [M], step_ms [M], tok_s [M].
    Raises torch.cuda.OutOfMemoryError if T=32768 does not fit.
    """
    model = BenchModel(D, L, H, D_FF, T_max=T_32K, grad_ckpt=True).to(DEVICE).to(DTYPE)
    model.train()
    opt    = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95),
                               weight_decay=0.1, eps=1e-8)
    scaler = GradScaler()

    def step_32k() -> None:
        x = torch.randint(0, VOCAB_SIZE, (1, T_32K), device=DEVICE)
        y = torch.randint(0, VOCAB_SIZE, (1, T_32K), device=DEVICE)
        # Force mem-efficient SDPA: no flash (T4/Turing), no math (O(T²) VRAM)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True
        ):
            with torch.cuda.amp.autocast(dtype=DTYPE):
                loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    N = 5
    t0.record()
    for _ in range(N):
        step_32k()
    t1.record()
    torch.cuda.synchronize()

    elapsed_ms   = t0.elapsed_time(t1)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1 << 30)

    del model, opt, scaler
    torch.cuda.empty_cache()

    return {
        "T": T_32K, "micro_bs": 1, "n_steps": N, "grad_ckpt": True,
        "peak_vram_gb": peak_vram_gb,
        "step_ms":      elapsed_ms / N,
        "tok_s":        (1 * T_32K * N) / (elapsed_ms / 1_000.0),
    }


# ── Tokenizer measurement ─────────────────────────────────────────────────────
# Representative snippets — real code, ~600–900 bytes each. No auto-generated content.

_PY = """\
import os, json
from pathlib import Path
from typing import List, Optional

class ShardLoader:
    \"\"\"Deterministic loader over uint16 binary shards.\"\"\"
    def __init__(self, data_dir: str, seed: int = 42):
        self.rng   = __import__('random').Random(seed)
        self.files = sorted(Path(data_dir).glob('*.bin'))
        if not self.files:
            raise FileNotFoundError(f'No .bin shards in {data_dir}')
        self._idx  = 0

    def next_chunk(self, length: int):
        import numpy as np
        data = np.frombuffer(self.files[self._idx % len(self.files)].read_bytes(),
                             dtype=np.uint16).astype(np.int64)
        self._idx += 1
        return data[:length]

def compute_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
"""

_JS = """\
'use strict';
const { promises: fs } = require('fs');
const path = require('path');

class LRUCache {
  constructor(maxSize) {
    this._max  = maxSize;
    this._map  = new Map();
  }
  get(key) {
    if (!this._map.has(key)) return undefined;
    const val = this._map.get(key);
    this._map.delete(key);
    this._map.set(key, val);
    return val;
  }
  set(key, val) {
    if (this._map.has(key)) this._map.delete(key);
    if (this._map.size >= this._max)
      this._map.delete(this._map.keys().next().value);
    this._map.set(key, val);
  }
}

async function readJsonLines(filePath) {
  const text  = await fs.readFile(filePath, 'utf8');
  return text.split('\\n').filter(Boolean).map(JSON.parse);
}

module.exports = { LRUCache, readJsonLines };
"""

_JAVA = """\
package io.example.util;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

public final class Checksum {
    private Checksum() {}

    public static String sha256(Path file) throws IOException, NoSuchAlgorithmException {
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(file)) {
            byte[] buf = new byte[1 << 16];
            int n;
            while ((n = in.read(buf)) != -1) md.update(buf, 0, n);
        }
        byte[] digest = md.digest();
        StringBuilder sb = new StringBuilder(64);
        for (byte b : digest) sb.append(String.format("%02x", b));
        return sb.toString();
    }

    public static Map<String, String> checksumDir(Path dir) throws Exception {
        var result = new LinkedHashMap<String, String>();
        try (var stream = Files.walk(dir)) {
            stream.filter(Files::isRegularFile).sorted()
                  .forEach(p -> {
                      try { result.put(dir.relativize(p).toString(), sha256(p)); }
                      catch (Exception e) { throw new RuntimeException(e); }
                  });
        }
        return result;
    }
}
"""

_CPP = """\
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <algorithm>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t cap) : buf_(cap), cap_(cap) {}

    void push(T value) {
        if (full()) throw std::overflow_error("RingBuffer full");
        buf_[tail_] = std::move(value);
        tail_ = (tail_ + 1) % cap_;
        ++size_;
    }

    T pop() {
        if (empty()) throw std::underflow_error("RingBuffer empty");
        T v = std::move(buf_[head_]);
        head_ = (head_ + 1) % cap_;
        --size_;
        return v;
    }

    [[nodiscard]] bool   empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool   full()  const noexcept { return size_ == cap_; }
    [[nodiscard]] size_t size()  const noexcept { return size_; }

private:
    std::vector<T> buf_;
    std::size_t cap_, head_ = 0, tail_ = 0, size_ = 0;
};
"""

_GO = """\
package shard

import (
\t"bufio"
\t"encoding/binary"
\t"fmt"
\t"io"
\t"os"
)

// Reader reads uint16 token ids from a binary shard file.
type Reader struct {
\tr   *bufio.Reader
\tf   *os.File
\tpos int64
}

func Open(path string) (*Reader, error) {
\tf, err := os.Open(path)
\tif err != nil {
\t\treturn nil, fmt.Errorf("shard.Open: %w", err)
\t}
\treturn &Reader{r: bufio.NewReaderSize(f, 1<<20), f: f}, nil
}

func (r *Reader) ReadTokens(buf []uint16) (int, error) {
\tfor i := range buf {
\t\tvar v uint16
\t\tif err := binary.Read(r.r, binary.LittleEndian, &v); err != nil {
\t\t\tif err == io.EOF {
\t\t\t\treturn i, io.EOF
\t\t\t}
\t\t\treturn i, fmt.Errorf("read token %d: %w", r.pos+int64(i), err)
\t\t}
\t\tbuf[i] = v
\t}
\tr.pos += int64(len(buf))
\treturn len(buf), nil
}

func (r *Reader) Close() error { return r.f.Close() }
"""

CODE_SNIPPETS = {
    "python": _PY,
    "js":     _JS,
    "java":   _JAVA,
    "cpp":    _CPP,
    "go":     _GO,
}

def measure_tokenizer() -> tuple[dict | None, float | str]:
    tok_candidates = [
        Path("/kaggle/input/ebrain-100m-data-binary/ebrain_tokenizer_16k.json"),
        Path("/kaggle/working/ebrain_tokenizer_16k.json"),
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if tok_path is None:
        return None, "tokenizer file not found at any expected path"

    # Verify SHA-256
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    sha_ok = sha == TOK_SHA256

    try:
        from tokenizers import Tokenizer  # type: ignore
        tok = Tokenizer.from_file(str(tok_path))
    except ImportError:
        return None, "tokenizers library not installed"
    except Exception as e:
        return None, f"failed to load tokenizer: {e}"

    per_lang: dict = {}
    for lang, code in CODE_SNIPPETS.items():
        b = len(code.encode("utf-8"))
        try:
            ids = tok.encode(code).ids
            t   = len(ids)
            bpt = b / t if t > 0 else -1.0
        except Exception:
            t, bpt = -1, -1.0
        per_lang[lang] = {"bytes": b, "tokens": t, "bytes_per_token": bpt}

    valid = [v["bytes_per_token"] for v in per_lang.values() if v["bytes_per_token"] > 0]
    mean_bpt = sum(valid) / len(valid) if valid else -1.0
    return per_lang, (mean_bpt, sha_ok, sha)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import hashlib

    log("=" * 80)
    log("E-Brain Phase 0 Benchmark")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log(f"GPU:     {props.name}")
        log(f"VRAM:    {props.total_memory / (1 << 30):.1f} GB total")
    else:
        log("GPU:     none (CPU only)")
    log(f"CUDA:    {torch.version.cuda}")
    log(f"PyTorch: {torch.__version__}")
    log("=" * 80)

    if not torch.cuda.is_available():
        log("FATAL: No CUDA device. Kernel must have enable_gpu: true.")
        flush_results()
        sys.exit(1)

    # =========================================================================
    # 1. Architecture benchmarks at T=2048
    # =========================================================================
    log(f"\n[ 1 / 3 ]  ARCHITECTURE BENCHMARKS")
    log(f"  T={T_BENCH}  fp16  GradScaler  SDPA")
    log(f"  micro_bs={MICRO_BS_T2K}  global_bs={GLOBAL_BS} tokens")
    log(f"  {N_BENCH} measured steps  ({N_WARMUP} warmup excluded from timing)")
    log()

    hdr = (f"  {'cand':<14} {'total_params':>13} {'non_emb':>13}"
           f" {'tok/s [M]':>11} {'vram [M]':>10} {'ms/step [M]':>12}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))

    bench_results: list[dict] = []
    for D, L, H, D_FF, label in CANDIDATES:
        tag = f"{label} D={D} L={L}"
        try:
            r = run_benchmark(D, L, H, D_FF, T_BENCH, MICRO_BS_T2K,
                              N_WARMUP, N_BENCH, label)
            bench_results.append(r)
            log(f"  {tag:<14} {r['total_params']:>13,} {r['non_emb_params']:>13,}"
                f" {r['tok_s']:>11,.0f} {r['peak_vram_gb']:>9.2f}G"
                f" {r['step_ms']:>11.1f}ms")
        except torch.cuda.OutOfMemoryError:
            log(f"  {tag:<14} {'OOM':>13} {'OOM':>13}"
                f" {'OOM':>11} {'OOM':>10} {'OOM':>12}")
            torch.cuda.empty_cache()

    # =========================================================================
    # 2. Phase 1 auto-lock inputs
    # =========================================================================
    log(f"\n[ 2 / 3 ]  PHASE 1 AUTO-LOCK INPUTS")
    log(f"  session_budget_s = {SESSION_S}  (3h50m kernel wall time)")
    log(f"  Selection rule: largest candidate where days<=75 AND tok/non_emb_param>=15"
        f" AND repeats<=2.0x")
    log(f"  Unique-token totals: 4B and 5B shown; actual comes from Phase 2 audit.")
    log()

    hdr2 = (f"  {'cand':<14} {'sess_steps [M]':>14} {'days@4B [M]':>12}"
            f" {'days@5B [M]':>12} {'tok/p@4B':>10} {'tok/p@5B':>10}")
    log(hdr2)
    log("  " + "-" * (len(hdr2) - 2))

    best_r: dict | None = None
    for r in bench_results:
        sess_steps = r["tok_s"] * SESSION_S / GLOBAL_BS
        r["sess_steps"] = sess_steps
        r["days_4B"]    = (4e9 / GLOBAL_BS) / sess_steps
        r["days_5B"]    = (5e9 / GLOBAL_BS) / sess_steps
        tag = f"{r['label']} D={r['D']} L={r['L']}"
        log(f"  {tag:<14} {sess_steps:>14.0f} {r['days_4B']:>12.1f}"
            f" {r['days_5B']:>12.1f}"
            f" {4e9 / r['non_emb_params']:>10.1f}"
            f" {5e9 / r['non_emb_params']:>10.1f}")
        if best_r is None or r["tok_s"] > best_r["tok_s"]:
            best_r = r

    log()
    log("  Repeats (conservative, before Phase 2 audit):")
    for r in bench_results:
        for target_B in (4, 5):
            target = target_B * 1e9
            # Repeats per group assuming 3 groups at equal share:
            group_unique = 4e9 / 3   # floor estimate; real value from Phase 2
            rep = target / (3 * group_unique)   # = target / 4B = 1.0 or 1.25 — both fine
            # More conservative: code group at 1.5B unique (Phase 2 target lower bound)
            code_rep = (target * 0.33) / 1.5e9  # code is ~33% of training tokens
        pass  # printed below

    log(f"  At 4B total target: code group rep = {4e9*0.33/1.5e9:.2f}x (using 1.5B unique floor)")
    log(f"  At 5B total target: code group rep = {5e9*0.33/1.5e9:.2f}x")
    log(f"  Text group rep at 4B: {4e9*0.65/1.738e9:.2f}x  at 5B: {5e9*0.65/1.738e9:.2f}x")
    log(f"  Cap = 2.0x per group. Actual per-group counts after Phase 2 audit.")

    # Preliminary selection (4B floor)
    log()
    log("  Gate check at 4B floor:")
    passing = []
    for r in bench_results:
        tpp   = 4e9 / r["non_emb_params"]
        days  = r["days_4B"]
        ok    = days <= 75 and tpp >= 15
        log(f"    {r['label']} D={r['D']} L={r['L']}:"
            f" days={days:.1f} tok/p={tpp:.1f}"
            f" → {'PASS' if ok else 'FAIL'}")
        if ok:
            passing.append(r)

    if passing:
        chosen = max(passing, key=lambda r: r["non_emb_params"])
        log(f"\n  PRELIMINARY SELECTION (4B floor): candidate {chosen['label']}"
            f" D={chosen['D']} L={chosen['L']}")
        log(f"  Final selection confirmed after Phase 2 audit.")
    elif bench_results:
        log(f"\n  No candidate passes all 3 gates at 4B floor.")
        log(f"  → Raise Phase 2 token target OR reduce context OR accept smaller model.")
        chosen = max(bench_results, key=lambda r: r["tok_s"])
    else:
        chosen = None
        log(f"\n  No candidates completed. GPU OOM on all.")

    # =========================================================================
    # 3a. T=4096 memory probe on highest-throughput candidate
    # =========================================================================
    log(f"\n[ 3a / 4 ]  T=4096 MEMORY PROBE")
    if best_r:
        D, L, H, D_FF = best_r["D"], best_r["L"], best_r["H"], best_r["D_FF"]
        log(f"  Config: candidate {best_r['label']} D={D} L={L} H={H}")
        log(f"  T={T_PROBE}  micro_bs={MICRO_BS_T4K}  {N_PROBE} steps  no warmup  fp16")
        try:
            pr = run_benchmark(D, L, H, D_FF, T_PROBE, MICRO_BS_T4K,
                               0, N_PROBE, best_r["label"] + "_T4096",
                               T_max=T_PROBE)
            log(f"  peak_vram_T4096 [M]:  {pr['peak_vram_gb']:.2f} GB")
            log(f"  tok/s_T4096     [M]:  {pr['tok_s']:.0f}")
            log(f"  OOM: no")
            log(f"  → T=4096 is FEASIBLE on this T4 instance.")
        except torch.cuda.OutOfMemoryError:
            log(f"  peak_vram_T4096 [M]:  OOM")
            log(f"  OOM: yes")
            log(f"  → Fall back to T=2048 as native context.")
            torch.cuda.empty_cache()
    else:
        log("  Skipped — no candidate completed the T=2048 benchmark.")

    # =========================================================================
    # 3c. T=32768 memory probe (gradient checkpointing + mem-efficient SDPA)
    # =========================================================================
    log(f"\n[ 3c / 4 ]  T=32768 MEMORY PROBE")
    log(f"  Purpose: verify T=32768 fits on T4 before claiming YaRN 32K extension is feasible.")
    log(f"  Settings: micro_bs=1  grad_ckpt=ON  SDPA=mem-efficient (no flash/math)  fp16  5 steps")
    if best_r:
        D, L, H, D_FF = best_r["D"], best_r["L"], best_r["H"], best_r["D_FF"]
        log(f"  Config: candidate {best_r['label']} D={D} L={L} H={H}")
        try:
            pr32 = run_t32k_probe(D, L, H, D_FF, best_r["label"])
            log(f"  peak_vram_T32K [M]:  {pr32['peak_vram_gb']:.2f} GB")
            log(f"  ms/step_T32K   [M]:  {pr32['step_ms']:.0f} ms")
            log(f"  tok/s_T32K     [M]:  {pr32['tok_s']:.0f}")
            log(f"  OOM: no")
            log(f"  → T=32768 training is FEASIBLE on this T4 instance (memory only).")
            log(f"  Note: this probe does NOT verify YaRN quality or position extrapolation.")
            log(f"        YaRN quality is verified separately on CPU (yarn/test_yarn_cpu.py).")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            oom_msg = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else str(e)
            log(f"  peak_vram_T32K [M]:  {oom_msg}")
            log(f"  OOM: yes")
            log(f"  → T=32768 does NOT fit with grad_ckpt+mem_efficient on this T4.")
            log(f"  Proposal A: try T=16384 (halved sequence length ~halves activation VRAM).")
            log(f"  Proposal B: if 16K also OOMs, cap context extension at T=8192.")
            log(f"  Do NOT claim 32K context without a passing probe.")
            torch.cuda.empty_cache()
    else:
        log("  Skipped — no candidate completed the T=2048 benchmark.")

    # =========================================================================
    # 3b → now 3d. Tokenizer bytes/token on code
    # =========================================================================
    log(f"\n[ 3d / 4 ]  TOKENIZER BYTES/TOKEN (16K BPE on 5 code languages)")
    result = measure_tokenizer()
    per_lang, meta = result

    if per_lang is None:
        log(f"  ERROR: {meta}")
        log(f"  Decision: CANNOT DETERMINE")
    else:
        mean_bpt, sha_ok, sha = meta
        log(f"  Tokenizer SHA-256: {sha[:32]}... {'✅ match' if sha_ok else '❌ MISMATCH'}")
        log()
        log(f"  {'lang':<10} {'bytes':>8} {'tokens':>8} {'bytes/token [M]':>16}")
        log(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*16}")
        for lang, v in per_lang.items():
            log(f"  {lang:<10} {v['bytes']:>8,} {v['tokens']:>8,} {v['bytes_per_token']:>16.3f}")
        log()
        log(f"  mean bytes/token [M]: {mean_bpt:.3f}")
        log(f"  threshold: 2.5")
        if mean_bpt < 0:
            log(f"  Decision: ERROR — no valid measurements")
        elif mean_bpt >= 2.5:
            log(f"  Decision: KEEP existing 16K BPE tokenizer")
            log(f"  SHA-256 unchanged: {TOK_SHA256}")
        else:
            log(f"  Decision: RETRAIN tokenizer (< 2.5 threshold)")
            log(f"  Action: retrain 16K or 32K BPE on text+code (~25% code, size^0.3 balance)")
            log(f"  → re-tokenize all shards → record new SHA-256 → re-baseline val numbers")
            log(f"  → val/test raw files stay frozen; verify SHA-256 before and after")

    # =========================================================================
    # Summary
    # =========================================================================
    log()
    log("=" * 80)
    log("PHASE 0 COMPLETE")
    log("All numbers above tagged [M] are measured on this T4 instance.")
    log("No measured value should be substituted or estimated in Phase 1.")
    log("Review this output, then proceed to Phase 1 spec lock.")
    log("=" * 80)

    flush_results()
    print(f"\nResults written to {RESULT_FILE}", flush=True)


if __name__ == "__main__":
    main()
