import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_fast_bpe import FastByteBPE

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tok_path = os.path.join(base_dir, "ebrain_tokenizer_16k.json")
    tok = FastByteBPE.load(tok_path)
    val_files = sorted(glob.glob(os.path.join(base_dir, "data", "val_vault", "*.txt")))

    print("=" * 80)
    print("TOKENIZER CERTIFICATION REPORT (STEP 2 GATE)")
    print("=" * 80)
    print(f"Total Vocabulary Size : {tok.vocab_size}")
    print(f"Base Byte Vocabulary  : 256 byte tokens (ID 0 to 255)")
    print(f"BPE Learned Merges    : {len(tok.merges):,} pairs")
    print(f"Special Tokens        : {tok.special_tokens}")
    print("-" * 80)
    header = f"{'Language File':16s} | {'Words':7s} | {'Tokens':7s} | {'Tok/Word':8s} | {'Byte-Fallback%':14s} | {'Exact Roundtrip'}"
    print(header)
    print("-" * 80)

    total_tokens = 0
    total_words = 0
    total_fallbacks = 0
    all_roundtrips_passed = True

    for vf in val_files:
        fname = os.path.basename(vf)
        with open(vf, "r", encoding="utf-8") as f:
            text = f.read()

        words = len(text.split())
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        exact_match = (decoded == text)
        if not exact_match:
            all_roundtrips_passed = False

        fallback_count = sum(1 for t in tokens if t < 256)
        fallback_pct = (fallback_count / max(1, len(tokens))) * 100
        tok_per_word = len(tokens) / max(1, words)

        total_tokens += len(tokens)
        total_words += words
        total_fallbacks += fallback_count

        status = "EXACT MATCH (100%)" if exact_match else "FAILED"
        print(f"{fname:16s} | {words:7,d} | {len(tokens):7,d} | {tok_per_word:8.2f} | {fallback_pct:13.2f}% | {status}")

    # Stress test: Emoji and mixed code
    stress_file = os.path.join(base_dir, "data", "mixed_stress_sample.txt")
    if os.path.exists(stress_file):
        with open(stress_file, "r", encoding="utf-8") as f:
            stext = f.read()
        stoks = tok.encode(stext)
        sdec = tok.decode(stoks)
        smatch = (sdec == stext)
        sfb = sum(1 for t in stoks if t < 256)
        sfb_pct = (sfb / max(1, len(stoks))) * 100
        swords = len(stext.split())
        print(f"{'stress_mixed.txt':16s} | {swords:7,d} | {len(stoks):7,d} | {len(stoks)/swords:8.2f} | {sfb_pct:13.2f}% | {'EXACT MATCH (100%)' if smatch else 'FAILED'}")

    print("-" * 80)
    avg_fallback = (total_fallbacks / total_tokens) * 100
    avg_tpw = total_tokens / total_words
    print(f"Aggregate Overall : Words: {total_words:,d} | Tokens: {total_tokens:,d} | Avg Tok/Word: {avg_tpw:.2f} | Avg Fallback: {avg_fallback:.2f}%")
    print(f"Lossless Invariant: {'ALL 10 VALIDATION FILES BITWISE IDENTICAL' if all_roundtrips_passed else 'ROUNDTRIP FAILURE'}")
    print("=" * 80)

if __name__ == "__main__":
    generate_report()
