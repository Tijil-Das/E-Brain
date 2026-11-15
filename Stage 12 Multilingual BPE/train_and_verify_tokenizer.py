"""
Stage 12 - Step 2 Execution:
Trains the 16,384 Byte-Level BPE Tokenizer, verifies lossless roundtrip,
measures tokens-per-word across all 10 languages, and verifies byte fallback.
"""

import os
import glob
import time
import unicodedata
from ebrain_fast_bpe import FastByteBPE

DATA_DIR = "Stage 12 Multilingual BPE/data"
CORPUS_PATH = os.path.join(DATA_DIR, "bpe_training_corpus.txt")
VAL_DIR = os.path.join(DATA_DIR, "val_vault")
TOKENIZER_PATH = "Stage 12 Multilingual BPE/ebrain_tokenizer_16k.json"

LANGUAGES = ["en", "es", "fr", "de", "ru", "hi", "bn", "ar", "zh", "ja"]

def main():
    print("=" * 80)
    print("STAGE 12 - STEP 2: BYTE-LEVEL BPE TRAINING & INVARIANT VERIFICATION")
    print("=" * 80)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        training_text = f.read()

    print(f"Loaded Balanced Multilingual Training Corpus: {len(training_text):,} chars | {len(training_text.encode('utf-8')):,} UTF-8 bytes")

    # Initialize tokenizer and train up to 16,384 tokens (or min_freq=2)
    tok = FastByteBPE(vocab_size=16384)
    tok.train(training_text, target_vocab_size=16384, min_freq=2)

    # Save tokenizer
    tok.save(TOKENIZER_PATH)

    # Reload tokenizer from disk to test serialization
    loaded_tok = FastByteBPE.load(TOKENIZER_PATH)
    print(f"\nReloaded tokenizer from {TOKENIZER_PATH} (Vocab size: {len(loaded_tok.vocab):,})")

    # ==========================================================================
    # TEST 1: STRICT LOSSLESS ROUNDTRIP VERIFICATION
    # ==========================================================================
    print("\n" + "-" * 80)
    print("INVARIANT CHECK 1: EXACT LOSSLESS ROUNDTRIP [decode(encode(text)) == text]")
    print("-" * 80)

    # Test Mixed-Script & Emoji Stress Sample First
    stress_file = os.path.join(DATA_DIR, "mixed_stress_sample.txt")
    with open(stress_file, "r", encoding="utf-8") as f:
        stress_text = f.read()

    stress_encoded = loaded_tok.encode(stress_text)
    stress_decoded = loaded_tok.decode(stress_encoded)
    assert stress_decoded == stress_text, "CRITICAL ERROR: Mixed-script / Emoji roundtrip failed!"
    print(f" -> Mixed-Script, Emojis, Math & Code: {len(stress_encoded):,} tokens -> 100% BITWISE IDENTICAL MATCH (PASSED)")

    # Test Every Validation File per Language
    all_roundtrips_passed = True
    lang_metrics = []

    for lang in LANGUAGES:
        val_file = os.path.join(VAL_DIR, f"val_{lang}.txt")
        with open(val_file, "r", encoding="utf-8") as f:
            val_text = f.read()

        val_bytes = val_text.encode("utf-8")
        tokens = loaded_tok.encode(val_text)
        decoded = loaded_tok.decode(tokens)

        match = (decoded == val_text)
        if not match:
            all_roundtrips_passed = False
            print(f" [FAILED] Roundtrip mismatch in {lang}!")
        
        # Calculate tokens per word
        # In Chinese and Japanese, words are often character-based, so split by space or character count
        words = val_text.split()
        num_words = len(words) if len(words) > 0 else 1
        num_tokens = len(tokens)
        num_bytes = len(val_bytes)

        tokens_per_word = num_tokens / num_words if num_words > 10 else num_tokens / (len(val_text) / 2) # fallback for CJK
        bytes_per_token = num_bytes / num_tokens if num_tokens > 0 else 1.0
        
        # Count fallback bytes (tokens < 256)
        fallback_tokens = sum(1 for t in tokens if t < 256)
        fallback_ratio = (fallback_tokens / num_tokens) * 100 if num_tokens > 0 else 0

        lang_metrics.append({
            "lang": lang,
            "bytes": num_bytes,
            "tokens": num_tokens,
            "words": num_words,
            "tokens_per_word": tokens_per_word,
            "bytes_per_token": bytes_per_token,
            "fallback_pct": fallback_ratio,
            "match": match
        })

        print(f" -> Lang [{lang:2s}]: {num_bytes:,} bytes | {num_tokens:,} tokens | {tokens_per_word:.2f} tok/word | {bytes_per_token:.2f} bytes/tok | Fallback: {fallback_ratio:.1f}% | Roundtrip: {'MATCH' if match else 'FAIL'}")

    assert all_roundtrips_passed, "One or more language roundtrips failed!"

    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 2 REPORT: TOKENS PER WORD & COMPRESSION ACROSS LANGUAGES")
    print("=" * 80)
    print(f"{'Language':<12} | {'Tokens/Word':<12} | {'Bytes/Token':<12} | {'Byte Fallback %':<15} | {'Roundtrip'}")
    print("-" * 70)
    for m in lang_metrics:
        print(f"{m['lang'].upper():<12} | {m['tokens_per_word']:<12.2f} | {m['bytes_per_token']:<12.2f} | {m['fallback_pct']:<15.1f}% | {'EXACT MATCH' if m['match'] else 'MISMATCH'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
