# EBrain Tier 3 (5 MB)

## Dataset Overview
This dataset is prepared for training the **E-Brain Causal Transformer** model.

- **Total Characters:** 8,682,956
- **Vocabulary Size:** 110 tokens (including `<UNK>` token: `□`)
- **SHA-256 Checksum:** `8669acb43c7c53dcbe80cf5e9e3d46d9fa2b304708f3f92bad0d1ef938b4475f`
- **License:** Public Domain / Project Gutenberg License
- **Splits:**
  - **Train Set (85%):** 7,380,512 characters
  - **Validation Set (10%):** 868,296 characters
  - **Test Set (5%):** 434,148 characters (held out for final evaluation)

## Directory Contents
- `vocab.json`: Vocabulary dictionary (`itos`, `stoi`, `unk_char`).
- `train_tokens.npy`: Encoded NumPy int32 array for training.
- `val_tokens.npy`: Encoded NumPy int32 array for validation.
- `test_tokens.npy`: Encoded NumPy int32 array for test.
- `train.txt`, `val.txt`, `test.txt`: Raw cleaned text files for inspection.
- `dataset-metadata.json`: Complete provenance metadata and fingerprint.
