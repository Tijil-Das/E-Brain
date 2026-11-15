# EBrain Tier 1 (170 KB)

## Dataset Overview
This dataset is prepared for training the **E-Brain Causal Transformer** model.

- **Total Characters:** 85,505
- **Vocabulary Size:** 68 tokens (including `<UNK>` token: `□`)
- **SHA-256 Checksum:** `2711317df6955e0bb8f5b665fd1bd76a90551eb47f56da30c7c2eb12b6da3e74`
- **License:** Public Domain / Project Gutenberg License
- **Splits:**
  - **Train Set (85%):** 72,679 characters
  - **Validation Set (10%):** 8,550 characters
  - **Test Set (5%):** 4,276 characters (held out for final evaluation)

## Directory Contents
- `vocab.json`: Vocabulary dictionary (`itos`, `stoi`, `unk_char`).
- `train_tokens.npy`: Encoded NumPy int32 array for training.
- `val_tokens.npy`: Encoded NumPy int32 array for validation.
- `test_tokens.npy`: Encoded NumPy int32 array for test.
- `train.txt`, `val.txt`, `test.txt`: Raw cleaned text files for inspection.
- `dataset-metadata.json`: Complete provenance metadata and fingerprint.
