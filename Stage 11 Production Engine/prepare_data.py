import os
import re
import json
import hashlib
from collections import Counter

UNK_CHAR = '\u25A1'

def clean_text(raw_text):
    """
    Cleans text:
    - Strips Project Gutenberg headers and license footers if present
    - Normalizes curly quotes, unicode dashes, special spaces
    - Collapses excessive blank lines
    - Restricts to clean readable characters
    """
    # 1. Remove Gutenberg boilerplate if present
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG[^\*]*\*\*\*", raw_text, re.IGNORECASE)
    if start_match:
        raw_text = raw_text[start_match.end():]

    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", raw_text, re.IGNORECASE)
    if end_match:
        raw_text = raw_text[:end_match.start()]

    # 2. Standardize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Normalize typography
    replacements = {
        '“': '"', '”': '"', '‘': "'", '’': "'", '`': "'",
        '—': '--', '–': '-', '…': '...',
        '\u00a0': ' ', '\ufeff': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 4. Remove control chars except newline and tab
    cleaned_chars = []
    for c in text:
        if c == '\n' or c == '\t' or (32 <= ord(c) <= 126) or c.isalnum():
            cleaned_chars.append(c)
        else:
            cleaned_chars.append(' ')
    text = "".join(cleaned_chars)

    # 5. Collapse excessive whitespace / newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def build_vocab(text, min_freq=2, max_vocab_size=128):
    counts = Counter(text)
    vocab_chars = [UNK_CHAR]

    for char, freq in counts.most_common():
        if freq >= min_freq and char != UNK_CHAR:
            vocab_chars.append(char)
        if len(vocab_chars) >= max_vocab_size:
            break

    vocab = {
        'itos': vocab_chars,
        'stoi': {ch: i for i, ch in enumerate(vocab_chars)},
        'unk_char': UNK_CHAR
    }
    return vocab


def prepare_dataset(input_file, output_prefix="dataset", train_ratio=0.85, val_ratio=0.10, test_ratio=0.05):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    total_chars = len(cleaned)
    sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    vocab = build_vocab(cleaned, min_freq=2)

    n_train = int(total_chars * train_ratio)
    n_val = int(total_chars * (train_ratio + val_ratio))

    train_text = cleaned[:n_train]
    val_text = cleaned[n_train:n_val]
    test_text = cleaned[n_val:]

    os.makedirs(output_prefix, exist_ok=True)

    with open(os.path.join(output_prefix, "train.txt"), "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(os.path.join(output_prefix, "val.txt"), "w", encoding="utf-8") as f:
        f.write(val_text)
    with open(os.path.join(output_prefix, "test.txt"), "w", encoding="utf-8") as f:
        f.write(test_text)
    with open(os.path.join(output_prefix, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    metadata = {
        'source_file': os.path.basename(input_file),
        'total_chars': total_chars,
        'sha256': sha256,
        'vocab_size': len(vocab['itos']),
        'train_chars': len(train_text),
        'val_chars': len(val_text),
        'test_chars': len(test_text),
        'splits': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    }

    with open(os.path.join(output_prefix, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset prepared successfully in '{output_prefix}/':")
    print(f" - Total characters: {total_chars:,}")
    print(f" - Train: {len(train_text):,} | Val: {len(val_text):,} | Test: {len(test_text):,}")
    print(f" - Vocab size: {len(vocab['itos'])} tokens (UNK token: '{UNK_CHAR}')")
    print(f" - SHA-256 fingerprint: {sha256[:16]}...")
    return metadata

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "Text.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data_tier1"
    prepare_dataset(src, out)
