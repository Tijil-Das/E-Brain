import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_fast_bpe import FastByteBPE

base_dir = os.path.dirname(os.path.abspath(__file__))
tok_path = os.path.join(base_dir, "ebrain_tokenizer_16k.json")
tok = FastByteBPE.load(tok_path)

print("Special tokens:", tok.special_tokens)
print("Max merge ID:", max(tok.merges.values()) if tok.merges else 0)
print("Vocab dict length:", len(tok.vocab))
print("Max token ID in vocab:", max(tok.vocab.keys()))

val_files = sorted(glob.glob(os.path.join(base_dir, "data", "val_vault", "*.txt")))

for vf in val_files:
    fname = os.path.basename(vf)
    with open(vf, "r", encoding="utf-8") as f:
        text = f.read()
    raw_bytes = len(text.encode("utf-8"))
    chars = len(text)
    tokens = tok.encode(text)
    
    # Analyze token breakdown
    raw_byte_tokens = sum(1 for t in tokens if t < 256)
    merged_tokens = sum(1 for t in tokens if t >= 256 and t not in tok.special_tokens.values())
    special_tokens_count = sum(1 for t in tokens if t in tok.special_tokens.values())
    
    bytes_per_token = raw_bytes / max(1, len(tokens))
    tokens_per_1k_chars = (len(tokens) / max(1, chars)) * 1000
    single_byte_fraction = (raw_byte_tokens / max(1, len(tokens))) * 100
    
    print(f"\n--- {fname} ---")
    print(f"Chars: {chars:,} | Bytes: {raw_bytes:,} | Tokens: {len(tokens):,}")
    print(f"Bytes per token (compression): {bytes_per_token:.2f} bytes/token")
    print(f"Tokens per 1,000 characters:   {tokens_per_1k_chars:.1f}")
    print(f"Single-byte tokens (ID < 256): {raw_byte_tokens:,} ({single_byte_fraction:.2f}%)")
    print(f"Merged tokens (ID >= 256):     {merged_tokens:,} ({100 - single_byte_fraction:.2f}%)")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"Sample raw text: {repr(text[:80])}")
