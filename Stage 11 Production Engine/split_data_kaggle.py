import os
import json
import hashlib
import numpy as np
import prepare_data

def prepare_kaggle_dataset(input_file, dataset_name, out_dir, train_ratio=0.85, val_ratio=0.10, test_ratio=0.05, seed=42):
    """
    Prepares a clean dataset specifically for Kaggle upload:
    1. Cleans text & normalizes typography.
    2. Builds fixed vocabulary with UNK token.
    3. Saves explicit integer token sequences and deterministic split indices.
    4. Generates a comprehensive README.md with SHA-256 fingerprint and licensing details.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()

    cleaned = prepare_data.clean_text(raw_text)
    total_chars = len(cleaned)
    sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    vocab = prepare_data.build_vocab(cleaned, min_freq=2)
    stoi = vocab['stoi']
    unk_idx = stoi[prepare_data.UNK_CHAR]

    # Map all characters to integer token array
    tokens = np.array([stoi.get(c, unk_idx) for c in cleaned], dtype=np.int32)

    # Deterministic sequential split
    n_train = int(total_chars * train_ratio)
    n_val = int(total_chars * (train_ratio + val_ratio))

    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:n_val]
    test_tokens = tokens[n_val:]

    # Save token binaries & split indices
    np.save(os.path.join(out_dir, "train_tokens.npy"), train_tokens)
    np.save(os.path.join(out_dir, "val_tokens.npy"), val_tokens)
    np.save(os.path.join(out_dir, "test_tokens.npy"), test_tokens)

    # Save raw split texts for inspection
    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned[:n_train])
    with open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned[n_train:n_val])
    with open(os.path.join(out_dir, "test.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned[n_val:])

    # Save vocabulary
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    metadata = {
        'dataset_name': dataset_name,
        'source_file': os.path.basename(input_file),
        'total_characters': total_chars,
        'sha256_checksum': sha256,
        'vocab_size': len(vocab['itos']),
        'train_chars': len(train_tokens),
        'val_chars': len(val_tokens),
        'test_chars': len(test_tokens),
        'split_ratio': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'random_seed': seed
    }

    with open(os.path.join(out_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate Kaggle Dataset README.md
    readme_content = f"""# {dataset_name}

## Dataset Overview
This dataset is prepared for training the **E-Brain Causal Transformer** model.

- **Total Characters:** {total_chars:,}
- **Vocabulary Size:** {len(vocab['itos'])} tokens (including `<UNK>` token: `{prepare_data.UNK_CHAR}`)
- **SHA-256 Checksum:** `{sha256}`
- **License:** Public Domain / Project Gutenberg License
- **Splits:**
  - **Train Set (85%):** {len(train_tokens):,} characters
  - **Validation Set (10%):** {len(val_tokens):,} characters
  - **Test Set (5%):** {len(test_tokens):,} characters (held out for final evaluation)

## Directory Contents
- `vocab.json`: Vocabulary dictionary (`itos`, `stoi`, `unk_char`).
- `train_tokens.npy`: Encoded NumPy int32 array for training.
- `val_tokens.npy`: Encoded NumPy int32 array for validation.
- `test_tokens.npy`: Encoded NumPy int32 array for test.
- `train.txt`, `val.txt`, `test.txt`: Raw cleaned text files for inspection.
- `dataset-metadata.json`: Complete provenance metadata and fingerprint.
"""

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Kaggle Dataset '{dataset_name}' packaged in '{out_dir}/':")
    print(f" - Characters: {total_chars:,} (Train: {len(train_tokens):,}, Val: {len(val_tokens):,}, Test: {len(test_tokens):,})")
    print(f" - Vocab: {len(vocab['itos'])} tokens | SHA-256: {sha256[:16]}...")
    return metadata

if __name__ == "__main__":
    # Prepare all 3 tiers for Kaggle
    prepare_kaggle_dataset("Text.txt", "EBrain Tier 1 (170 KB)", "kaggle_datasets/tier1_170k")
    if os.path.exists("cleaned_tier2.txt"):
        prepare_kaggle_dataset("cleaned_tier2.txt", "EBrain Tier 2 (1 MB)", "kaggle_datasets/tier2_1mb")
    if os.path.exists("cleaned_tier3.txt"):
        prepare_kaggle_dataset("cleaned_tier3.txt", "EBrain Tier 3 (5 MB)", "kaggle_datasets/tier3_5mb")
