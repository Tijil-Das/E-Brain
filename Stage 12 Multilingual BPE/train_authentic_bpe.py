import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebrain_fast_bpe import FastByteBPE

base_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(base_dir, "data", "bpe_training_corpus.txt")

with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded authentic BPE training corpus: {len(text):,} characters ({len(text.encode('utf-8')):,} bytes)")

# Train 8K vocabulary (256 base + 7,741 merges + 3 special tokens = 8,000 exact)
tok = FastByteBPE()
tok.train(text, num_merges=7741, min_freq=2)

out_path = os.path.join(base_dir, "ebrain_tokenizer_8k.json")
tok.save(out_path)

# Verify loaded tokenizer
loaded = FastByteBPE.load(out_path)
print(f"Verified loaded tokenizer: vocab_size={loaded.vocab_size}, max_id={max(loaded.vocab.keys())}")
assert loaded.vocab_size == max(loaded.vocab.keys()) + 1, "Discrepancy in vocab size!"
print("Contiguous ID verification: 100% PASSED.")
