"""
Stage 12 - Step 2: Handcrafted Byte-Level BPE Tokenizer
1. Base vocabulary: 256 byte values (0..255).
2. Merges trained iteratively on balanced 10-language corpus.
3. Special tokens: <|unk|>, <|pad|>, <|endoftext|>.
4. Total vocabulary: 16,384 tokens (16K).
5. Exact lossless roundtrip proof: decode(encode(text)) == text.
"""

import os
import sys
import json
import time
from collections import Counter, defaultdict

class EBrainByteBPE:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|unk|>": 16381,
            "<|pad|>": 16382,
            "<|endoftext|>": 16383
        }
        self.special_token_inv = {v: k for k, v in self.special_tokens.items()}
        # Base vocabulary: 0..255 maps to raw bytes
        self.merges = {} # (p0, p1) -> new_token_id
        self.vocab = {i: bytes([i]) for i in range(256)} # token_id -> bytes
        for s_tok, s_id in self.special_tokens.items():
            self.vocab[s_id] = s_tok.encode('utf-8')

    def get_stats(self, ids):
        counts = defaultdict(int)
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] += 1
        return counts

    def merge_sequence(self, ids, pair, idx):
        new_ids = []
        i = 0
        p0, p1 = pair
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text_bytes, target_vocab_size=16384, min_freq=2, log_interval=1000):
        print(f"Training Byte-Level BPE Tokenizer up to {target_vocab_size:,} tokens...")
        t0 = time.time()
        ids = list(text_bytes)
        num_merges = target_vocab_size - 256 - len(self.special_tokens)
        print(f"Total initial bytes: {len(ids):,} | Target merges to learn: {num_merges:,}")

        current_token_id = 256
        for merge_i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                print(f"No more adjacent pairs to merge at step {merge_i}.")
                break
            top_pair = max(stats, key=stats.get)
            if stats[top_pair] < min_freq:
                print(f"Top pair frequency {stats[top_pair]} < min_freq {min_freq}. Halting at {current_token_id} tokens.")
                break

            ids = self.merge_sequence(ids, top_pair, current_token_id)
            self.merges[top_pair] = current_token_id
            self.vocab[current_token_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            current_token_id += 1

            if (merge_i + 1) % log_interval == 0 or (merge_i + 1) == num_merges:
                elapsed = time.time() - t0
                comp_ratio = len(text_bytes) / len(ids)
                print(f"Merge {merge_i + 1:5d}/{num_merges} | Vocab: {current_token_id:5d} | Tokens: {len(ids):,} | Compression: {comp_ratio:.2f}x | {elapsed:.1f}s", flush=True)

        print(f"Tokenizer trained in {time.time() - t0:.2f}s! Total vocab: {len(self.vocab)} tokens.")

    def encode(self, text):
        if not text:
            return []
        # Convert text to UTF-8 bytes
        b = text.encode("utf-8")
        ids = list(b)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            # Find the pair in current sequence that was merged earliest
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge_sequence(ids, pair, idx)
        return ids

    def decode(self, ids):
        raw_bytes = bytearray()
        for tok in ids:
            if tok in self.vocab:
                raw_bytes.extend(self.vocab[tok])
            else:
                raw_bytes.extend(b"<|unk|>")
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath):
        data = {
            "vocab_size": self.vocab_size,
            "merges": [f"{p[0]},{p[1]},{idx}" for p, idx in self.merges.items()],
            "special_tokens": self.special_tokens
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved tokenizer to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.special_tokens = data["special_tokens"]
        tok.special_token_inv = {v: k for k, v in tok.special_tokens.items()}
        for line in data["merges"]:
            p0, p1, idx = map(int, line.split(","))
            tok.merges[(p0, p1)] = idx
            tok.vocab[idx] = tok.vocab[p0] + tok.vocab[p1]
        for s_tok, s_id in tok.special_tokens.items():
            tok.vocab[s_id] = s_tok.encode("utf-8")
        return tok
