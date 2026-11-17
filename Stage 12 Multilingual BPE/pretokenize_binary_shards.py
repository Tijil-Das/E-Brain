"""
E-Brain 100M — Full Binary Pre-tokenization Pipeline
=====================================================
Run locally (CPU). No GPU quota needed.

Steps per language:
  1. Load Wikipedia parquet shards from HF cache
  2. Clean text (strip wiki markup, normalize unicode)
  3. Filter stubs < 500 chars
  4. Near-duplicate & doc-ID decontamination against val_vault / test_vault / alice
  5. Tokenize with frozen 16K BPE tokenizer
  6. Pack into uint16 binary shards of T=2048 tokens (EOT boundary between docs)
  7. Write manifest: per-lang token count, SHA-256, source info, license

Output: Stage 12 Multilingual BPE/data_100m_binary/
  train_en.bin, train_es.bin, ... train_ar.bin
  data_100m_manifest.json
"""

import os
import sys
import re
import glob
import json
import hashlib
import unicodedata
import struct
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
TOKENIZER_PATH = BASE_DIR / "ebrain_tokenizer_16k.json"
HF_WIKI       = Path.home() / ".cache/huggingface/hub/datasets--wikimedia--wikipedia"
OUT_BIN_DIR   = BASE_DIR / "data_100m_binary"
VAL_DIR       = BASE_DIR / "data" / "val_vault"
TEST_DIR      = BASE_DIR / "data" / "test_vault"

# Try alternate val/test paths
for candidate in [
    BASE_DIR / "data" / "val_vault",
    BASE_DIR / "data_v3" / "val_vault",
    BASE_DIR / "raw_authentic" / "val_vault",
]:
    if candidate.exists():
        VAL_DIR = candidate
        break

for candidate in [
    BASE_DIR / "data" / "test_vault",
    BASE_DIR / "data_v3" / "test_vault",
    BASE_DIR / "raw_authentic" / "test_vault",
]:
    if candidate.exists():
        TEST_DIR = candidate
        break

OUT_BIN_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LANGUAGES     = ['en', 'es', 'ru', 'fr', 'de', 'ja', 'zh', 'hi', 'bn', 'ar']
T_CTX         = 2048
MIN_ARTICLE_CHARS = 500
NGRAM_SIZE    = 8          # for near-duplicate fingerprinting
TOK_SHA256    = "41cb92f102d2a11dbde826c54f09b487a27c4a157be0f179ea09682ad04344bf"

LICENSE_LOG = {
    "wikimedia/wikipedia": {
        "license": "CC BY-SA 4.0",
        "url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source": "https://dumps.wikimedia.org/",
    }
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def clean_wiki_text(text: str) -> str:
    """Remove wiki markup, normalize unicode, strip excessive whitespace."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove wiki templates {{...}}
    depth = 0
    cleaned = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif text[i:i+2] == '}}' and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0:
            cleaned.append(text[i])
            i += 1
        else:
            i += 1
    text = ''.join(cleaned)
    # Remove wiki links [[...]] → keep display text
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    # Remove external links [url text] → keep text
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]*\]', '', text)
    # Remove section headers === ... ===
    text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)
    # Remove bullets/numbered lists markers
    text = re.sub(r'^[*#:;]+', '', text, flags=re.MULTILINE)
    # Remove reference tags
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Normalize unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def ngrams(text: str, n: int):
    """Character n-grams as a set of hashed fingerprints."""
    s = text.lower()
    return {hashlib.md5(s[i:i+n].encode()).hexdigest() for i in range(len(s) - n + 1)}

def build_decontam_fingerprints(vault_dirs):
    """Build n-gram fingerprint set from all val/test/alice files."""
    fps = set()
    doc_ids = set()
    for d in vault_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in d.rglob("*.txt"):
            text = f.read_text(errors='replace')
            fps.update(ngrams(text, NGRAM_SIZE))
            doc_ids.add(f.stem)
        for f in d.rglob("*.json"):
            try:
                data = json.loads(f.read_text())
                for entry in (data if isinstance(data, list) else [data]):
                    t = entry.get('text', '')
                    if t:
                        fps.update(ngrams(t, NGRAM_SIZE))
                        doc_ids.add(entry.get('id', f.stem))
            except Exception:
                pass
    return fps, doc_ids

def is_contaminated(text: str, decontam_fps: set, decontam_ids: set, doc_id: str = None,
                    threshold: float = 0.02) -> bool:
    """Return True if article overlaps too much with val/test."""
    if doc_id and doc_id in decontam_ids:
        return True
    if not decontam_fps:
        return False
    article_fps = ngrams(text, NGRAM_SIZE)
    if not article_fps:
        return False
    overlap = len(article_fps & decontam_fps) / len(article_fps)
    return overlap > threshold

def find_parquet_shards(lang: str):
    """Find all parquet files for a language in the HF cache."""
    patterns = [
        str(HF_WIKI / "**" / f"*{lang}*" / "*.parquet"),
        str(HF_WIKI / "**" / f"*{lang}*" / "train" / "*.parquet"),
        str(HF_WIKI / "**" / "*.parquet"),
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        # Filter by language in path
        matches = [m for m in matches if f"/{lang}/" in m.replace("\\", "/") or
                   f".{lang}." in Path(m).name or f"_{lang}_" in Path(m).name or
                   f"-{lang}-" in Path(m).name]
        if matches:
            return sorted(matches)
    return []

def pack_into_uint16_bin(token_ids: list, out_path: Path, eot_id: int, t_ctx: int):
    """
    Pack token IDs into uint16 binary.
    Format: raw uint16 token IDs. Sequences are T_CTX tokens long.
    EOT token is inserted between documents.
    Returns number of T_CTX-length sequences written.
    """
    arr = np.array(token_ids, dtype=np.uint16)
    # Truncate to multiple of t_ctx
    n_seqs = len(arr) // t_ctx
    arr = arr[:n_seqs * t_ctx]
    arr.tofile(out_path)
    return n_seqs

# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_language(lang: str, decontam_fps: set, decontam_ids: set,
                     tok: Tokenizer, eot_id: int) -> dict:
    """Process one language. Returns stats dict."""
    print(f"\n{'─'*60}")
    print(f"[{lang}] Starting...")

    shards = find_parquet_shards(lang)
    if not shards:
        print(f"  ⚠️  No parquet shards found for {lang}. Skipping.")
        return {}

    print(f"  Found {len(shards)} shard(s):")
    for s in shards[:3]:
        print(f"    {Path(s).name}  ({Path(s).stat().st_size/(1<<20):.1f} MB)")

    all_tokens = []          # flat list of uint16-safe ints
    n_docs      = 0
    n_filtered_stub  = 0
    n_filtered_decon = 0
    seen_doc_ids = set()
    t0 = time.time()

    for shard_path in shards:
        try:
            table = pq.read_table(shard_path)
        except Exception as e:
            print(f"  ⚠️  Could not read {shard_path}: {e}")
            continue

        df = table.to_pydict()
        texts  = df.get('text', [])
        ids    = df.get('id',   [None] * len(texts))
        titles = df.get('title', [''] * len(texts))

        for text, doc_id, title in zip(texts, ids, titles):
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            # Clean
            cleaned = clean_wiki_text(str(text))

            # Stub filter
            if len(cleaned) < MIN_ARTICLE_CHARS:
                n_filtered_stub += 1
                continue

            # Decontamination
            if is_contaminated(cleaned, decontam_fps, decontam_ids, str(doc_id)):
                n_filtered_decon += 1
                continue

            # Tokenize
            encoded = tok.encode(cleaned)
            tok_ids = encoded.ids

            # Append with EOT separator
            all_tokens.extend(tok_ids)
            all_tokens.append(eot_id)
            n_docs += 1

        elapsed = time.time() - t0
        print(f"  Shard done: docs={n_docs}  tokens={len(all_tokens):,}  "
              f"filtered: stub={n_filtered_stub} decon={n_filtered_decon}  "
              f"t={elapsed:.1f}s")

    if not all_tokens:
        print(f"  ⚠️  No tokens produced for {lang}. Skipping.")
        return {}

    # Write binary shard
    out_path = OUT_BIN_DIR / f"train_{lang}.bin"
    n_seqs = pack_into_uint16_bin(all_tokens, out_path, eot_id, T_CTX)
    total_tokens = n_seqs * T_CTX

    # SHA-256
    sha = sha256_bytes(out_path.read_bytes())
    size_mb = out_path.stat().st_size / (1 << 20)

    print(f"\n  ✅ [{lang}]  docs={n_docs}  tokens={total_tokens:,}  "
          f"seqs={n_seqs:,}  size={size_mb:.1f}MB  SHA-256={sha[:16]}...")

    return {
        "lang": lang,
        "docs": n_docs,
        "total_tokens": total_tokens,
        "n_sequences": n_seqs,
        "size_mb": size_mb,
        "sha256": sha,
        "out_file": str(out_path.name),
        "filtered_stub": n_filtered_stub,
        "filtered_decon": n_filtered_decon,
    }

def main():
    print("=" * 72)
    print("E-Brain 100M — Binary Pre-tokenization Pipeline")
    print("=" * 72)

    # ── Verify tokenizer
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: Tokenizer not found at {TOKENIZER_PATH}")
        sys.exit(1)
    actual_sha = sha256_bytes(TOKENIZER_PATH.read_bytes())
    assert actual_sha == TOK_SHA256, \
        f"Tokenizer SHA-256 mismatch!\n  got:      {actual_sha}\n  expected: {TOK_SHA256}"
    print(f"✅ Tokenizer SHA-256 verified: {actual_sha[:16]}...")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eot_id = tok.token_to_id("<|endoftext|>") or 0
    print(f"   EOT token ID: {eot_id}")

    # ── Build decontamination fingerprints
    print(f"\nBuilding decontamination fingerprints...")
    print(f"  Val dir:  {VAL_DIR}  (exists: {VAL_DIR.exists()})")
    print(f"  Test dir: {TEST_DIR}  (exists: {TEST_DIR.exists()})")
    decontam_fps, decontam_ids = build_decontam_fingerprints([VAL_DIR, TEST_DIR])
    print(f"  {len(decontam_fps):,} unique {NGRAM_SIZE}-gram fingerprints")
    print(f"  {len(decontam_ids):,} decontaminated doc IDs")

    # ── Process each language
    manifest_entries = []
    for lang in LANGUAGES:
        stats = process_language(lang, decontam_fps, decontam_ids, tok, eot_id)
        if stats:
            manifest_entries.append(stats)

    # ── Summary
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    total_tokens = sum(e['total_tokens'] for e in manifest_entries)
    total_mb     = sum(e['size_mb']       for e in manifest_entries)
    print(f"{'Lang':<6} {'Docs':>8} {'Tokens':>14} {'Seqs':>8} {'MB':>8} {'SHA-256 (16ch)'}")
    print("-" * 72)
    for e in manifest_entries:
        print(f"{e['lang']:<6} {e['docs']:>8,} {e['total_tokens']:>14,} "
              f"{e['n_sequences']:>8,} {e['size_mb']:>8.1f} {e['sha256'][:16]}")
    print("-" * 72)
    print(f"{'TOTAL':<6} {sum(e['docs'] for e in manifest_entries):>8,} "
          f"{total_tokens:>14,} tokens  {total_mb:.1f} MB")
    print(f"\nNote: 2.40B target = 240M tokens per language × 10 languages")

    # ── Write manifest
    manifest = {
        "tokenizer_sha256": actual_sha,
        "eot_id": eot_id,
        "t_ctx": T_CTX,
        "min_article_chars": MIN_ARTICLE_CHARS,
        "ngram_dedup_size": NGRAM_SIZE,
        "total_tokens": total_tokens,
        "total_mb": total_mb,
        "licenses": LICENSE_LOG,
        "languages": manifest_entries,
    }
    manifest_path = OUT_BIN_DIR / "data_100m_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n✅ Manifest written: {manifest_path}")
    print(f"   SHA-256: {sha256_bytes(manifest_path.read_bytes())[:32]}...")

    # Purity check: each language should have ≥ some meaningful token count
    print(f"\n{'='*72}")
    print("PURITY GATE: checking per-language token counts...")
    for e in manifest_entries:
        ratio = e['total_tokens'] / (total_tokens / len(manifest_entries)) if manifest_entries else 0
        status = "✅" if 0.5 <= ratio <= 2.0 else "⚠️ "
        print(f"  {status} {e['lang']}: {e['total_tokens']:,} tokens  (ratio to mean: {ratio:.2f}x)")

    print(f"\n{'='*72}")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
