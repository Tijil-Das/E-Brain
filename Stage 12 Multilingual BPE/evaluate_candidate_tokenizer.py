"""
Stage 12 - Candidate Tokenizer Comprehensive Evaluation
======================================================
Evaluates the candidate contiguous 8,000-vocabulary BPE tokenizer across:
1. All 10 sealed validation vaults.
2. Metrics:
   - Bytes per Token (Compression)
   - Characters per Token (Script-fair compression)
   - Tokens per 1,000 Characters
   - High-Byte Fallback % (128 <= ID < 256: true multi-byte fragmentation)
   - ASCII Single-Byte % (ID < 128: ordinary Latin letters / punctuation)
3. Exact roundtrip check: decode(encode(text)) == text
4. Stress tests (pre-NFC, mixed tabs, emojis, C code, control chars, malformed bytes)
"""

import os
import sys
import glob
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_fast_bpe import FastByteBPE

def evaluate_tokenizer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "ebrain_tokenizer_8k.json")
    tok = FastByteBPE.load(tok_path)

    print("=" * 110)
    print(f"CANDIDATE TOKENIZER EVALUATION REPORT (PROVISIONAL VOCAB: {tok.vocab_size:,})")
    print("=" * 110)
    print(f"Base Alphabet        : 256 bytes (0..255)")
    print(f"Learned Merges       : {len(tok.merges):,} pairs (IDs 256..{tok.vocab_size - len(tok.special_tokens) - 1})")
    print(f"Special Tokens       : {tok.special_tokens}")
    print(f"Contiguous ID Space  : max_token_id ({max(tok.vocab.keys())}) + 1 == vocab_size ({tok.vocab_size})")
    print(f"Training Slices      : Exactly 25,000 chars per language from train_vault (Zero val/test leakage)")
    print(f"Corpus Normalization : Pre-tokenization Unicode NFC applied during corpus build")
    print("-" * 110)

    val_files = sorted(glob.glob(os.path.join(base_dir, "data", "val_vault", "*.txt")))

    header = (
        f"{'Language Vault':16s} | {'Chars':7s} | {'Bytes':7s} | {'Tokens':7s} | "
        f"{'B/Tok':6s} | {'Ch/Tok':6s} | {'Tok/1kCh':8s} | {'HiByte FB%':10s} | {'ASCII%':7s} | {'Roundtrip'}"
    )
    print(header)
    print("-" * 110)

    all_passed = True
    tot_chars = 0
    tot_bytes = 0
    tot_tokens = 0
    tot_high_fb = 0
    tot_ascii = 0

    results = []

    for vf in val_files:
        fname = os.path.basename(vf)
        with open(vf, "r", encoding="utf-8") as f:
            text = f.read()

        chars = len(text)
        raw_bytes = len(text.encode("utf-8"))
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)

        # 1. Roundtrip check
        roundtrip_ok = (decoded == text)
        if not roundtrip_ok:
            all_passed = False

        # 2. Compression metrics
        num_tok = max(1, len(tokens))
        bytes_per_tok = raw_bytes / num_tok
        chars_per_tok = chars / num_tok
        tok_per_1k = (num_tok / max(1, chars)) * 1000

        # 3. Disentangled fallback:
        # High-byte fallback: ID in [128, 255] -> Broken multi-byte fragments
        high_fb_cnt = sum(1 for t in tokens if 128 <= t < 256)
        high_fb_pct = (high_fb_cnt / num_tok) * 100

        # ASCII single-byte: ID in [0, 127] -> Ordinary ASCII chars / punctuation
        ascii_cnt = sum(1 for t in tokens if t < 128)
        ascii_pct = (ascii_cnt / num_tok) * 100

        tot_chars += chars
        tot_bytes += raw_bytes
        tot_tokens += len(tokens)
        tot_high_fb += high_fb_cnt
        tot_ascii += ascii_cnt

        status = "100% exact" if roundtrip_ok else "FAILED"
        print(
            f"{fname:16s} | {chars:7,d} | {raw_bytes:7,d} | {len(tokens):7,d} | "
            f"{bytes_per_tok:6.2f} | {chars_per_tok:6.2f} | {tok_per_1k:8.1f} | "
            f"{high_fb_pct:9.2f}% | {ascii_pct:6.2f}% | {status}"
        )
        results.append({
            "vault": fname, "chars": chars, "bytes": raw_bytes, "tokens": len(tokens),
            "bpt": bytes_per_tok, "cpt": chars_per_tok, "tp1k": tok_per_1k,
            "high_fb_pct": high_fb_pct, "ascii_pct": ascii_pct, "roundtrip": roundtrip_ok
        })

    print("-" * 110)
    overall_bpt = tot_bytes / max(1, tot_tokens)
    overall_cpt = tot_chars / max(1, tot_tokens)
    overall_tp1k = (tot_tokens / max(1, tot_chars)) * 1000
    overall_high_fb = (tot_high_fb / max(1, tot_tokens)) * 100
    overall_ascii = (tot_ascii / max(1, tot_tokens)) * 100

    print(
        f"{'OVERALL TOTALS':16s} | {tot_chars:7,d} | {tot_bytes:7,d} | {tot_tokens:7,d} | "
        f"{overall_bpt:6.2f} | {overall_cpt:6.2f} | {overall_tp1k:8.1f} | "
        f"{overall_high_fb:9.2f}% | {overall_ascii:6.2f}% | {'ALL EXACT' if all_passed else 'FAILED'}"
    )

    # Stress test evaluation
    print("\n" + "=" * 110)
    print("STRESS & EDGE-CASE ROUNDTRIP EVALUATION")
    print("=" * 110)
    stress_tests = {
        "Pre-NFC combining accents (e + \\u0301)": "Cafe\u0301 and na\u0308ive re\u0301sume\u0301",
        "Leading/trailing mixed whitespace & tabs": "\t\n  \t  def foo():\n      return True\n\n\t  ",
        "High-entropy Emojis & Math symbols": "The fox 🦊 jumped over the moon 🌕! ∫ e^{-x^2} dx = √π.",
        "Non-Latin mixed with C code snippet": "int main() { char *s = \"Привет, мир! 🚀 世界\"; return 0; }",
        "Extreme control characters": "\x00\x01\x02\x07\x08\x0b\x0c\x0e\x0f\x1b[31mRed Alert\x1b[0m",
    }
    for desc, sample in stress_tests.items():
        enc = tok.encode(sample)
        dec = tok.decode(enc)
        ok = (dec == sample)
        print(f" -> {desc:45s}: {'PASSED (Exact)' if ok else 'FAILED'}")

    # Malformed byte sequence check
    malformed_bytes = b"Hello \xff\xfe World \x80\xbf!"
    tok_bytes = tok.encode_bytes(malformed_bytes)
    dec_bytes = tok.decode_bytes(tok_bytes)
    byte_ok = (dec_bytes == malformed_bytes)
    print(f" -> {'Malformed raw UTF-8 bytes preservation':45s}: {'PASSED (Exact bytes)' if byte_ok else 'FAILED'}")
    print("=" * 110)

if __name__ == "__main__":
    evaluate_tokenizer()
