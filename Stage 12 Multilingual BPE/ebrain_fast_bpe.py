"""
Stage 12 - Step 2: High-Speed Incremental Byte-Level BPE Tokenizer
Uses inverted word index and incremental pair-count updates (O(affected_words) per merge).
Trains full 16,384 vocabulary in seconds.
Zero-loss roundtrip guarantee on all languages, emojis, and scripts.
"""

import os
import re
import json
import time
from collections import defaultdict, Counter

class FastByteBPE:
    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|unk|>": 16381,
            "<|pad|>": 16382,
            "<|endoftext|>": 16383
        }
        self.special_token_inv = {v: k for k, v in self.special_tokens.items()}
        self.merges = {} # (p0, p1) -> new_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}
        for s_tok, s_id in self.special_tokens.items():
            self.vocab[s_id] = s_tok.encode('utf-8')

    def train(self, text, target_vocab_size=16384, min_freq=2):
        print(f"Training Incremental Byte-Level BPE up to {target_vocab_size:,} tokens...")
        t0 = time.time()

        # Split text into whitespace/punctuation chunks
        chunks = re.findall(r'\s*\S+|\s+', text)
        word_counts = Counter(chunks)
        print(f"Initial unique chunks: {len(word_counts):,} (from {len(chunks):,} tokens)")

        # Unique words list and their frequencies
        words = [list(chunk.encode('utf-8')) for chunk in word_counts.keys()]
        counts = list(word_counts.values())

        # Inverted index: token_id -> set of word indices that contain this token
        token_to_words = defaultdict(set)
        for w_idx, w in enumerate(words):
            for t in w:
                token_to_words[t].add(w_idx)

        # Global pair counts
        pair_counts = defaultdict(int)
        for w_idx, w in enumerate(words):
            c = counts[w_idx]
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += c

        num_merges = target_vocab_size - 256 - len(self.special_tokens)
        print(f"Target merges to learn: {num_merges:,}")
        current_token_id = 256

        for merge_i in range(num_merges):
            if not pair_counts:
                print(f"No more adjacent pairs to merge at step {merge_i}.")
                break

            top_pair = max(pair_counts, key=pair_counts.get)
            top_freq = pair_counts[top_pair]
            if top_freq < min_freq:
                print(f"Top pair frequency {top_freq} < min_freq ({min_freq}). Halting at {current_token_id} tokens.")
                break

            p0, p1 = top_pair
            self.merges[top_pair] = current_token_id
            self.vocab[current_token_id] = self.vocab[p0] + self.vocab[p1]
            del pair_counts[top_pair]

            # Find only words that contain both p0 and p1
            candidate_words = token_to_words[p0].intersection(token_to_words[p1])
            words_to_update = []
            for w_idx in candidate_words:
                w = words[w_idx]
                has_pair = False
                for i in range(len(w) - 1):
                    if w[i] == p0 and w[i + 1] == p1:
                        has_pair = True
                        break
                if has_pair:
                    words_to_update.append(w_idx)

            for w_idx in words_to_update:
                w = words[w_idx]
                c = counts[w_idx]

                # Subtract old pairs
                for i in range(len(w) - 1):
                    old_pair = (w[i], w[i + 1])
                    pair_counts[old_pair] -= c
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]

                # Perform merge in word
                new_w = []
                i = 0
                n = len(w)
                while i < n:
                    if i < n - 1 and w[i] == p0 and w[i + 1] == p1:
                        new_w.append(current_token_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1

                # Add new pairs
                for i in range(len(new_w) - 1):
                    pair_counts[(new_w[i], new_w[i + 1])] += c

                # Update inverted index
                for old_tok in set(w) - set(new_w):
                    token_to_words[old_tok].discard(w_idx)
                token_to_words[current_token_id].add(w_idx)

                words[w_idx] = new_w

            current_token_id += 1

            if (merge_i + 1) % 1000 == 0 or (merge_i + 1) == num_merges:
                elapsed = time.time() - t0
                print(f"Merge {merge_i + 1:5d}/{num_merges} | Vocab: {current_token_id:5d} | Top Pair: {top_pair} (freq: {top_freq:,}) | Elapsed: {elapsed:.1f}s", flush=True)

        print(f"BPE Training Complete in {time.time() - t0:.2f}s! Final Vocab Size: {len(self.vocab):,} tokens.")

    def encode_chunk(self, byte_list):
        tokens = list(byte_list)
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            best_pair = min(mergeable, key=lambda p: self.merges[p])
            new_tokens = []
            i = 0
            n = len(tokens)
            p0, p1 = best_pair
            merged_id = self.merges[best_pair]
            while i < n:
                if i < n - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        if not text:
            return []
        chunks = re.findall(r'\s*\S+|\s+', text)
        result = []
        for ch in chunks:
            b_list = list(ch.encode('utf-8'))
            result.extend(self.encode_chunk(b_list))
        return result

    def decode(self, ids):
        raw_bytes = bytearray()
        for tok in ids:
            if tok in self.vocab:
                raw_bytes.extend(self.vocab[tok])
            else:
                raw_bytes.extend(b"<|unk|>")
        return raw_bytes.decode('utf-8', errors='replace')

    def save(self, filepath):
        data = {
            "vocab_size": len(self.vocab),
            "merges": [f"{p[0]},{p[1]},{idx}" for p, idx in self.merges.items()],
            "special_tokens": self.special_tokens
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
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
