"""
Stage 12 - Step 7: High-Performance Multilingual Binary Sharding Engine
====================================================================
Streams text documents from train_vault and val_vault, tokenizes using
the certified FastByteBPE tokenizer, separates documents with <|endoftext|>,
and packs sequences into memory-mapped uint16 binary files (.bin).
"""

import os
import sys
import glob
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_fast_bpe import FastByteBPE

def pack_vault(vault_dir: str, output_bin: str, tokenizer: FastByteBPE, eot_token: int = 16383):
    print(f"\n[Packer] Scanning vault: {vault_dir}...")
    txt_files = sorted(glob.glob(os.path.join(vault_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {vault_dir}")
        
    print(f" -> Found {len(txt_files)} language document files.")
    
    total_tokens = 0
    total_bytes = 0
    start_t = time.time()
    
    # First pass: compute total tokens to allocate exact uint16 buffer
    all_tokens_list = []
    for fpath in txt_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        file_bytes = len(content.encode("utf-8"))
        tokens = tokenizer.encode(content)
        # Append end-of-text separator
        tokens.append(eot_token)
        
        all_tokens_list.extend(tokens)
        total_tokens += len(tokens)
        total_bytes += file_bytes
        
        comp_ratio = file_bytes / max(1, len(tokens) - 1)
        print(f"   * {fname:16s} | Bytes: {file_bytes:8,d} | Tokens: {len(tokens):8,d} | Comp: {comp_ratio:.2f} bytes/token")
        
    elapsed = time.time() - start_t
    print(f" -> Total Raw Bytes:   {total_bytes:,d}")
    print(f" -> Total Packed Tokens:{total_tokens:,d} (inc. <|endoftext|>)")
    print(f" -> Average Compression Ratio: {total_bytes / total_tokens:.2f} bytes/token")
    print(f" -> Tokenization Time: {elapsed:.2f}s ({total_tokens / elapsed:.0f} tokens/s)")
    
    # Save as uint16 binary file
    arr = np.array(all_tokens_list, dtype=np.uint16)
    arr.tofile(output_bin)
    file_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f" -> Saved binary shard: {output_bin} ({file_mb:.2f} MB)")
    
    # Verify memory-mapped read
    mmap_arr = np.memmap(output_bin, dtype=np.uint16, mode='r')
    assert len(mmap_arr) == total_tokens, "Memory map length mismatch!"
    print(f" -> Verified memmap reader: {len(mmap_arr):,d} tokens accessible with 0 RAM overhead.")
    return total_tokens, total_bytes

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "ebrain_tokenizer_16k.json")
    print("=" * 70)
    print("E-BRAIN: MULTILINGUAL BINARY SHARDING PIPELINE")
    print("=" * 70)
    print(f"Loading tokenizer from: {tok_path}")
    tokenizer = FastByteBPE.load(tok_path)
    eot_token = tokenizer.special_tokens.get("<|endoftext|>", 16383)
    print(f"Vocabulary Size: {tokenizer.vocab_size} | EndOfText ID: {eot_token}")
    
    train_vault = os.path.join(base_dir, "data", "train_vault")
    val_vault = os.path.join(base_dir, "data", "val_vault")
    
    out_dir = os.path.join(base_dir, "data", "shards")
    os.makedirs(out_dir, exist_ok=True)
    
    train_bin = os.path.join(out_dir, "train_multilingual.bin")
    val_bin = os.path.join(out_dir, "val_multilingual.bin")
    
    pack_vault(train_vault, train_bin, tokenizer, eot_token)
    pack_vault(val_vault, val_bin, tokenizer, eot_token)
    
    print("\n" + "=" * 70)
    print("BINARY SHARDING COMPLETE! PRETRAINING DATA IS STREAMING-READY.")
    print("=" * 70)

if __name__ == "__main__":
    main()
