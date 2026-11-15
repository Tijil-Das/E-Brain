"""
Stage 12 - Multilingual Corpus Quality Audit
===========================================
Audits the raw multilingual corpora across:
1. Native script purity ratio (%)
2. Word counts & unique word diversity
3. Top-10 and Top-100 word token share (Zipfian concentration)
4. Duplicate line rate (%)
5. Formal source attribution & licensing
"""

import os
import glob
import unicodedata
from collections import Counter

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_authentic")
files = sorted(glob.glob(os.path.join(base_dir, "*_raw.txt")))

SCRIPT_RANGES = {
    'en': [(0x0041, 0x005A), (0x0061, 0x007A)], # Latin ASCII
    'es': [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)], # Latin + Accents
    'fr': [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF), (0x0152, 0x0153)], # French
    'de': [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C4, 0x00FC), (0x00DF, 0x00DF)], # German
    'ru': [(0x0400, 0x04FF)], # Cyrillic
    'ja': [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)], # Hiragana, Katakana, Kanji
    'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)], # CJK Unified Ideographs
    'hi': [(0x0900, 0x097F)], # Devanagari
    'bn': [(0x0980, 0x09FF)], # Bengali
    'ar': [(0x0600, 0x06FF)]  # Arabic
}

SOURCES_LICENSES = {
    'en': ('Gutenberg (Conan Doyle: Sherlock)', 'Public Domain'),
    'es': ('Gutenberg (Cervantes: Don Quijote)', 'Public Domain'),
    'fr': ('Gutenberg (Voltaire: Candide)', 'Public Domain'),
    'de': ('Gutenberg (Goethe: Faust)', 'Public Domain'),
    'ru': ('Wikimedia (Russian Wikipedia: Science, History)', 'CC BY-SA 4.0'),
    'ja': ('Wikimedia (Japanese Wikipedia: History, Geo, Science)', 'CC BY-SA 4.0'),
    'zh': ('Gutenberg (Rites of Zhou: Classical Chinese)', 'Public Domain'),
    'hi': ('Wikimedia (Hindi Wikipedia: Science, Geo, Literature)', 'CC BY-SA 4.0'),
    'bn': ('Wikimedia (Bengali Wikipedia: Literature, History)', 'CC BY-SA 4.0'),
    'ar': ('Wikimedia (Arabic Wikipedia: History, Science, Lit)', 'CC BY-SA 4.0')
}

def audit():
    print("=" * 125)
    print("E-BRAIN MULTILINGUAL DATA QUALITY AUDIT & SCRIPT PURITY REPORT")
    print("=" * 125)
    header = (
        f"{'Lang':4s} | {'Chars':8s} | {'Words':7s} | {'Uniq Words':10s} | "
        f"{'Script%':7s} | {'Top-10%':8s} | {'Top-100%':9s} | {'Dup Line%':9s} | {'Source':36s} | {'License'}"
    )
    print(header)
    print("-" * 125)

    for fpath in files:
        lang = os.path.basename(fpath).split('_')[0]
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()

        total_chars = len(text)
        alpha_chars = [c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P')]
        ranges = SCRIPT_RANGES[lang]
        script_chars = 0
        for c in alpha_chars:
            code = ord(c)
            if any(r[0] <= code <= r[1] for r in ranges):
                script_chars += 1
        script_ratio = (script_chars / max(1, len(alpha_chars))) * 100

        words = text.split()
        total_words = len(words)
        counts = Counter(words)
        uniq_words = len(counts)

        top10_words = [cnt for _, cnt in counts.most_common(10)]
        top10_share = (sum(top10_words) / max(1, total_words)) * 100

        top100_words = [cnt for _, cnt in counts.most_common(100)]
        top100_share = (sum(top100_words) / max(1, total_words)) * 100

        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
        dup_lines = len(lines) - len(set(lines))
        dup_rate = (dup_lines / max(1, len(lines))) * 100

        src, lic = SOURCES_LICENSES[lang]
        print(
            f"{lang:4s} | {total_chars:8,d} | {total_words:7,d} | {uniq_words:10,d} | "
            f"{script_ratio:6.2f}% | {top10_share:7.2f}% | {top100_share:8.2f}% | {dup_rate:8.2f}% | {src:36s} | {lic}"
        )
    print("=" * 125)

if __name__ == "__main__":
    audit()
