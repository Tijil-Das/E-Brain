# EBrain Tier 2 (1 MB)

## Dataset Overview
This dataset is prepared for training the **E-Brain Causal Transformer** model.

- **Total Characters:** 897,329
- **Vocabulary Size:** 85 tokens (including `<UNK>` token: `□`)
- **SHA-256 Checksum:** `5810f57c18267564b670c7b2b8040c31e49a4010d046f407d5bde455f919bdaf`
- **License:** Public Domain / Project Gutenberg License
- **Splits:**
  - **Train Set (85%):** 762,729 characters
  - **Validation Set (10%):** 89,733 characters
  - **Test Set (5%):** 44,867 characters (held out for final evaluation)

## Directory Contents
- `vocab.json`: Vocabulary dictionary (`itos`, `stoi`, `unk_char`).
- `train_tokens.npy`: Encoded NumPy int32 array for training.
- `val_tokens.npy`: Encoded NumPy int32 array for validation.
- `test_tokens.npy`: Encoded NumPy int32 array for test.
- `train.txt`, `val.txt`, `test.txt`: Raw cleaned text files for inspection.
- `dataset-metadata.json`: Complete provenance metadata and fingerprint.
