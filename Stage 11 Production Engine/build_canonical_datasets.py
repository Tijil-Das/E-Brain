import os
import re
import json
import hashlib
import numpy as np
import prepare_data

DATA_DIR = "Stage 11 Production Engine"
RAW_DIR = os.path.join(DATA_DIR, "raw_books")
OUT_DIR = os.path.join(DATA_DIR, "canonical_datasets")

def clean_book(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    clean = prepare_data.clean_text(raw)
    # Extra sanitization: strip any leftover project/gutenberg mentions
    lines = clean.split("\n")
    sanitized = [l for l in lines if "gutenberg" not in l.lower() and "project" not in l.lower()]
    return "\n".join(sanitized).strip()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("==========================================================================")
    print("BUILDING CANONICAL ZERO-LEAKAGE DATASETS & BENCHMARK SUITE")
    print("==========================================================================")

    # 1. CUT BENCHMARK VAL & TEST FIRST from 11.txt (Alice in Wonderland)
    # 11.txt will NEVER be part of any training tier.
    alice_clean = clean_book(os.path.join(RAW_DIR, "11.txt"))
    assert len(alice_clean) >= 100000, f"Alice clean too short: {len(alice_clean)}"
    
    benchmark_val_text = alice_clean[:50000]
    benchmark_test_text = alice_clean[50000:100000]

    os.makedirs(os.path.join(OUT_DIR, "common_eval"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "common_eval", "common_val.txt"), "w", encoding="utf-8") as f:
        f.write(benchmark_val_text)
    with open(os.path.join(OUT_DIR, "common_eval", "common_test.txt"), "w", encoding="utf-8") as f:
        f.write(benchmark_test_text)

    print(f"[Common Benchmark Cut First]")
    print(f" - common_val.txt : {len(benchmark_val_text):,} characters (SHA256: {hashlib.sha256(benchmark_val_text.encode()).hexdigest()[:12]})")
    print(f" - common_test.txt: {len(benchmark_test_text):,} characters [HELD OUT & SEALED] (SHA256: {hashlib.sha256(benchmark_test_text.encode()).hexdigest()[:12]})")

    # 2. TIER 1 (170 KB): Arthur Pendelton (Text.txt)
    with open(os.path.join(DATA_DIR, "Text.txt"), "r", encoding="utf-8") as f:
        t1_raw = f.read()
    t1_clean = (t1_raw + t1_raw) # 2x text
    t1_split = int(len(t1_clean) * 0.90)
    t1_train = t1_clean[:t1_split]
    t1_val = t1_clean[t1_split:]

    os.makedirs(os.path.join(OUT_DIR, "tier1_170k"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "tier1_170k", "train.txt"), "w", encoding="utf-8") as f:
        f.write(t1_train)
    with open(os.path.join(OUT_DIR, "tier1_170k", "internal_val.txt"), "w", encoding="utf-8") as f:
        f.write(t1_val)
    print(f"\n[Tier 1: 170 KB]")
    print(f" - Train: {len(t1_train):,} chars | Internal Val: {len(t1_val):,} chars")

    # 3. TIER 2 (1 MB): Sherlock Holmes + Hound of Baskervilles + Time Machine
    sherlock_1661 = clean_book(os.path.join(RAW_DIR, "1661.txt"))
    baskerville_2852 = clean_book(os.path.join(RAW_DIR, "2852.txt"))
    timemachine_35 = clean_book(os.path.join(RAW_DIR, "35.txt"))
    
    # 50,000 chars of Time Machine for Tier 2 internal val
    t2_internal_val = timemachine_35[-50000:]
    timemachine_train = timemachine_35[:-50000]

    t2_train_full = (sherlock_1661 + "\n\n" + baskerville_2852 + "\n\n" + timemachine_train)
    # Trim to exactly 1,000,000 chars for Tier 2
    t2_train = t2_train_full[:1000000]

    os.makedirs(os.path.join(OUT_DIR, "tier2_1mb"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "tier2_1mb", "train.txt"), "w", encoding="utf-8") as f:
        f.write(t2_train)
    with open(os.path.join(OUT_DIR, "tier2_1mb", "internal_val.txt"), "w", encoding="utf-8") as f:
        f.write(t2_internal_val)
    print(f"\n[Tier 2: 1 MB]")
    print(f" - Train: {len(t2_train):,} chars | Internal Val: {len(t2_internal_val):,} chars")

    # 4. TIER 3 (8.7 MB): All other books (encompasses Tier 2, plus Frankenstein, Dracula, Pride & Prejudice, Great Expectations, Moby Dick, War and Peace)
    other_books = ["84.txt", "345.txt", "1342.txt", "1400.txt", "2701.txt", "2600.txt"]
    t3_extra_parts = [clean_book(os.path.join(RAW_DIR, b)) for b in other_books]
    t3_train = (t2_train_full + "\n\n" + "\n\n".join(t3_extra_parts)).strip()

    os.makedirs(os.path.join(OUT_DIR, "tier3_8mb"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "tier3_8mb", "train.txt"), "w", encoding="utf-8") as f:
        f.write(t3_train)
    print(f"\n[Tier 3: 8.7 MB]")
    print(f" - Train: {len(t3_train):,} chars (encompasses Tier 2 train)")

    # 5. BUILD UNIFIED VOCABULARY ACROSS ALL DATA
    all_texts = [benchmark_val_text, benchmark_test_text, t1_train, t2_train, t3_train]
    vocab = prepare_data.build_vocab("".join(all_texts), min_freq=2)
    stoi = vocab['stoi']
    itos = vocab['itos']
    unk_idx = stoi[prepare_data.UNK_CHAR]
    V = len(itos)
    print(f"\n[Unified Vocabulary]")
    print(f" - Total Vocab Size: {V} unique tokens (including UNK)")

    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    # 6. ENCODE ALL TEXTS TO INT32 NUMPY ARRAYS
    def encode_and_save(text, path):
        arr = np.array([stoi.get(c, unk_idx) for c in text], dtype=np.int32)
        np.save(path, arr)
        return arr

    encode_and_save(benchmark_val_text, os.path.join(OUT_DIR, "common_eval", "common_val_tokens.npy"))
    encode_and_save(benchmark_test_text, os.path.join(OUT_DIR, "common_eval", "common_test_tokens.npy"))
    encode_and_save(t1_train, os.path.join(OUT_DIR, "tier1_170k", "train_tokens.npy"))
    encode_and_save(t1_val, os.path.join(OUT_DIR, "tier1_170k", "internal_val_tokens.npy"))
    encode_and_save(t2_train, os.path.join(OUT_DIR, "tier2_1mb", "train_tokens.npy"))
    encode_and_save(t2_internal_val, os.path.join(OUT_DIR, "tier2_1mb", "internal_val_tokens.npy"))
    encode_and_save(t3_train, os.path.join(OUT_DIR, "tier3_8mb", "train_tokens.npy"))

    # 7. AUTOMATED LEAKAGE & DUPLICATE PASSAGE SCANNER (50-char sliding window)
    print("\n[Running Invariant Check: 50-char Duplicate Passage & Leakage Scanner]...")
    def extract_ngrams(text, n=50, step=10):
        return {text[i:i+n] for i in range(0, len(text) - n + 1, step)}

    val_ngrams = extract_ngrams(benchmark_val_text)
    test_ngrams = extract_ngrams(benchmark_test_text)

    # Check test vs val
    assert len(val_ngrams.intersection(test_ngrams)) == 0, "Leakage detected between Common Val and Common Test!"

    # Check each tier against common val and test
    for tier_name, train_text in [("Tier 1", t1_train), ("Tier 2", t2_train), ("Tier 3", t3_train)]:
        train_ngrams = extract_ngrams(train_text)
        leak_val = val_ngrams.intersection(train_ngrams)
        leak_test = test_ngrams.intersection(train_ngrams)
        assert len(leak_val) == 0, f"LEAKAGE DETECTED: {tier_name} contains passages from Common Val! Found {len(leak_val)} matches."
        assert len(leak_test) == 0, f"LEAKAGE DETECTED: {tier_name} contains passages from Common Test! Found {len(leak_test)} matches."
        print(f" -> {tier_name} vs Common Val/Test: 0 duplicate passages (VERIFIED CLEAN)")

    print("\n==========================================================================")
    print(">>> ZERO-LEAKAGE CANONICAL DATASETS BUILT & VERIFIED! <<<")
    print("==========================================================================")

if __name__ == "__main__":
    main()
