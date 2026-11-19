import os, json
from pathlib import Path
from typing import List, Optional

class ShardLoader:
    """Deterministic loader over uint16 binary shards."""
    def __init__(self, data_dir: str, seed: int = 42):
        self.rng   = __import__('random').Random(seed)
        self.files = sorted(Path(data_dir).glob('*.bin'))
        if not self.files:
            raise FileNotFoundError(f'No .bin shards in {data_dir}')
        self._idx  = 0

    def next_chunk(self, length: int):
        import numpy as np
        data = np.frombuffer(self.files[self._idx % len(self.files)].read_bytes(),
                             dtype=np.uint16).astype(np.int64)
        self._idx += 1
        return data[:length]

def compute_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
