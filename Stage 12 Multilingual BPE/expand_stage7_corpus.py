import os
import sys
import glob
import json
import hashlib
import re
import html
import unicodedata
import string
import pyarrow.parquet as pq
from collections import Counter
from tokenizers import Tokenizer

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_v3")
TRAIN_DIR = os.path.join(DATA_DIR, "train_vault")
VAL_DIR = os.path.join(DATA_DIR, "val_vault")
TEST_DIR = os.path.join(DATA_DIR, "test_vault")
ALICE_PATH = os.path.join(BASE_DIR, "kaggle_dataset_v3", "alice_common_val.txt")
TOKENIZER_16K_PATH = os.path.join(BASE_DIR, "ebrain_tokenizer_16k.json")
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub/datasets--wikimedia--wikipedia")

LANGUAGES = ['en', 'es', 'ru', 'fr', 'de', 'ja', 'zh', 'hi', 'bn', 'ar']

# Target train MB calibrated to empirical BPE compression ratios
# Sized so that each language yields exactly ~17.63M tokens (Total: ~176.3M tokens = 1 epoch on 8.81M model)
TARGET_TRAIN_BYTES = {
    'en': int(57.5 * 1024 * 1024),  # 3.26 B/tok -> 17.64M tokens
    'es': int(57.3 * 1024 * 1024),  # 3.25 B/tok -> 17.63M tokens
    'ru': int(78.5 * 1024 * 1024),  # 4.45 B/tok -> 17.64M tokens
    'fr': int(55.5 * 1024 * 1024),  # 3.15 B/tok -> 17.62M tokens
    'de': int(55.5 * 1024 * 1024),  # 3.15 B/tok -> 17.62M tokens
    'ja': int(69.1 * 1024 * 1024),  # 3.92 B/tok -> 17.63M tokens
    'zh': int(55.9 * 1024 * 1024),  # 3.17 B/tok -> 17.63M tokens
    'hi': int(65.8 * 1024 * 1024),  # 3.73 B/tok -> 17.64M tokens
    'bn': int(63.6 * 1024 * 1024),  # 3.61 B/tok -> 17.62M tokens
    'ar': int(80.6 * 1024 * 1024),  # 4.57 B/tok -> 17.64M tokens
}

# Frozen floors from Stage 2 Certification
FROZEN_FLOORS = {
    'en': 0.4759,
    'es': 0.4392,
    'ru': 0.2833,
    'fr': 0.2000,
    'de': 0.2000,
    'ja': 4.8740,
    'zh': 0.8423,
    'hi': 3.2872,
    'bn': 7.7017,
    'ar': 0.2000,
}

NEWS_BYTES = {
    'en': 11.329 * 1024 * 1024,
    'es': 11.314 * 1024 * 1024,
    'ru': 11.401 * 1024 * 1024,
}

tok16k = Tokenizer.from_file(TOKENIZER_16K_PATH)

CJK_PUNCT_RANGES = [
    (0x3000, 0x303F),
    (0xFF00, 0xFFEF),
]

def is_cjk_punct(code: int) -> bool:
    return any(start <= code <= end for start, end in CJK_PUNCT_RANGES)

def is_char_valid_for_lang(ch: str, lang: str) -> bool:
    if ch.isspace() or ch.isdigit() or ch in string.punctuation:
        return True
    code = ord(ch)
    if is_cjk_punct(code):
        return True
    if lang in ['en', 'es', 'fr', 'de']:
        return (65 <= code <= 90) or (97 <= code <= 122) or (0x00C0 <= code <= 0x024F) or (0x1E00 <= code <= 0x1EFF)
    elif lang == 'ru':
        return (0x0400 <= code <= 0x052F)
    elif lang == 'hi':
        return (0x0900 <= code <= 0x097F) or (0xA8E0 <= code <= 0xA8FF)
    elif lang == 'bn':
        return (0x0980 <= code <= 0x09FF)
    elif lang == 'ar':
        return (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F) or (0x08A0 <= code <= 0x08FF) or (0xFB50 <= code <= 0xFDFF) or (0xFE70 <= code <= 0xFEFF)
    elif lang == 'zh':
        return (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x20000 <= code <= 0x2A6DF)
    elif lang == 'ja':
        return ((0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or 
                (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or 
                (0x31F0 <= code <= 0x31FF))
    return True

def compute_script_purity(text: str, lang: str) -> float:
    if not text: return 100.0
    sample_text = text if len(text) <= 1_000_000 else text[:1_000_000]
    total = len(sample_text)
    valid = sum(1 for ch in sample_text if is_char_valid_for_lang(ch, lang))
    return (valid / total) * 100.0

def get_tokens_for_overlap(text: str, lang: str):
    if lang in ['ja', 'zh']:
        return [c for c in text if not c.isspace() and c not in string.punctuation and not is_cjk_punct(ord(c))]
    else:
        return text.lower().split()

def get_ngram_hashes(text: str, lang: str, n: int = 8):
    tokens = get_tokens_for_overlap(text, lang)
    if len(tokens) < n:
        return []
    return [hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)]

def compute_duplicate_line_rate(text: str) -> float:
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) >= 20]
    if not lines: return 0.0
    total_lines = len(lines)
    unique_lines = len(set(lines))
    dup_lines = total_lines - unique_lines
    return (dup_lines / total_lines) * 100.0

# Build Alice Exclusion Set
print("Loading Alice Exclusion Hashes...", flush=True)
with open(ALICE_PATH, "r", encoding="utf-8") as f:
    alice_text = f.read()
alice_10_hashes = set(get_ngram_hashes(alice_text, 'en', n=10))

def clean_wiki_article(raw_text: str, lang: str) -> str:
    raw = unicodedata.normalize("NFC", raw_text)
    raw = html.unescape(raw)
    raw = re.sub(r'https?://\S+|www\.\S+', '', raw)
    raw = re.sub(r'\[\d+\]', '', raw)
    raw = re.sub(r'<ref.*?>.*?</ref>', '', raw, flags=re.DOTALL)
    raw = re.sub(r'<ref.*?/>', '', raw)
    raw = re.sub(r'==+\s*(References|External links|See also|Notes|Further reading|Notas|Bibliografía|Enlaces externos|Véase también|Примечания|Ссылки|Литература|См\. также|参考文献|外部链接|参见|ملاحظات|مراجع|وصلات خارجية)\s*==+.*', '', raw, flags=re.DOTALL | re.IGNORECASE)

    bad_patterns = [
        'category:', 'categoría:', 'категория:', 'thumb|', 'file:', 'archivo:', 'файл:',
        'colspan', 'bgcolor', 'cellpadding', 'infobox', 'navbox', 'taxobox'
    ]
    lines = [l.strip() for l in raw.split('\n') if len(l.strip()) > 35 and not any(bp in l.lower() for bp in bad_patterns)]
    
    if lang == 'fr':
        lines = [l for l in lines if not any(b in l.lower() for b in ['insee', 'gentilé', 'population municipale', 'démographie de', 'décoration'])]
    elif lang == 'ar':
        lines = [l for l in lines if not any(b in l.lower() for b in ['تعداد السكان', 'الرمز البريدي', 'إحصائيات', 'علم الأنساب'])]
    elif lang == 'zh':
        lines = [l for l in lines if not any(b in l.lower() for b in ['公元前', '四书五经', '尚书', '周易', '春秋左氏'])]
    elif lang == 'ru':
        lines = [l for l in lines if (sum(1 for ch in l if 65 <= ord(ch) <= 122) / max(1, sum(1 for ch in l if ch.isalpha()))) <= 0.12]
        
    return '\n\n'.join(lines)

def extract_expansion_wiki(lang: str, current_train_text: str, val_text: str, test_text: str, needed_bytes: int):
    files = glob.glob(os.path.join(HF_CACHE, '**', f'*{lang}*/*.parquet'), recursive=True)
    if not files:
        raise FileNotFoundError(f"Cannot find cached Wikipedia parquet for {lang}")
    p_w = files[0]
    
    t_w = pq.read_table(p_w, columns=['id', 'title', 'text'])
    texts = t_w.column('text')
    titles = t_w.column('title')
    ids = t_w.column('id')
    total_docs = len(texts)
    
    # Train slice is strictly 0..85% of parquet rows (identical to Stage 2 partition)
    # Val is 85%..92.5%, Test is 92.5%..100%
    max_train_row = int(total_docs * 0.85)
    
    # Collect Val and Test article IDs and titles to guarantee 0 ID overlap
    val_test_ids = set()
    val_test_titles = set()
    for i in range(max_train_row, total_docs):
        val_test_ids.add(str(ids[i].as_py()))
        val_test_titles.add(str(titles[i].as_py()).lower().strip())
        
    # Also collect existing train document prefixes to prevent duplicate additions
    existing_prefixes = set()
    for block in current_train_text.split('\n\n'):
        b_clean = block.strip()
        if len(b_clean) >= 100:
            existing_prefixes.add(b_clean[:120])
            
    # Contamination hashes for val and test
    val_n_set = set(get_ngram_hashes(val_text, lang, n=16 if lang in ['ja', 'zh'] else 10))
    test_n_set = set(get_ngram_hashes(test_text, lang, n=16 if lang in ['ja', 'zh'] else 10))
    
    docs = []
    accum_b = 0
    
    # Evenly stride through the 0..max_train_row partition
    stride = max(1, max_train_row // 25000)
    for offset in range(stride):
        for i in range(offset, max_train_row, stride):
            raw = texts[i].as_py()
            if len(raw) < 1200 or len(raw) > 50000:
                continue
            doc_id = str(ids[i].as_py())
            title_str = str(titles[i].as_py()).lower().strip()
            
            # 1. Zero doc-ID or title overlap against Val/Test
            if doc_id in val_test_ids or title_str in val_test_titles:
                continue
                
            # Disambiguation / list filters
            if any(bad in title_str for bad in ['list of', 'lists of', 'lista de', 'anexo:', 'список', 'списки', 'desambiguación', 'неоднозначность', 'قائمة', '一覧', '曖昧さ回避', 'सूची', 'তালিকা']):
                continue
                
            cleaned = clean_wiki_article(raw, lang)
            if len(cleaned) < 1000:
                continue
                
            # Prevent duplicate article with existing train
            if cleaned[:120] in existing_prefixes:
                continue
                
            # Contamination check against Val, Test, and Alice
            doc_hashes = get_ngram_hashes(cleaned, lang, n=16 if lang in ['ja', 'zh'] else 10)
            if any(h in val_n_set for h in doc_hashes) or any(h in test_n_set for h in doc_hashes):
                continue
            if lang == 'en' and any(h in alice_10_hashes for h in doc_hashes):
                continue
                
            b = len(cleaned.encode('utf-8'))
            docs.append(cleaned)
            accum_b += b
            if accum_b >= needed_bytes:
                break
        if accum_b >= needed_bytes:
            break
            
    print(f"[{lang.upper()}] Harvested {len(docs)} new clean Wiki docs ({accum_b / (1024*1024):.2f} MB)", flush=True)
    return docs, accum_b

# -------------------------------------------------------------
# Execute Expansion & Audit
# -------------------------------------------------------------
expansion_report = {}
total_final_bytes = 0
total_final_tokens = 0
tokens_by_lang = {}

print("\n" + "=" * 80)
print("EXECUTING STAGE 7 CORPUS EXPANSION & MULTILINGUAL AUDIT")
print("=" * 80 + "\n", flush=True)

for lang in LANGUAGES:
    tp = os.path.join(TRAIN_DIR, f"train_{lang}.txt")
    vp = os.path.join(VAL_DIR, f"val_{lang}.txt")
    te_p = os.path.join(TEST_DIR, f"test_{lang}.txt")
    
    with open(tp, "r", encoding="utf-8", errors="ignore") as f:
        existing_train = f.read()
    with open(vp, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()
    with open(te_p, "r", encoding="utf-8", errors="ignore") as f:
        test_text = f.read()
        
    cur_b = len(existing_train.encode('utf-8'))
    target_b = TARGET_TRAIN_BYTES[lang]
    needed_b = max(0, target_b - cur_b)
    
    print(f"Processing {lang.upper()}: Current {cur_b/(1024*1024):.2f} MB | Target {target_b/(1024*1024):.2f} MB | Needed +{needed_b/(1024*1024):.2f} MB...", flush=True)
    
    new_docs, added_b = extract_expansion_wiki(lang, existing_train, val_text, test_text, needed_b)
    
    expanded_train = existing_train + "\n\n" + "\n\n".join(new_docs)
    expanded_train_b = len(expanded_train.encode('utf-8'))
    
    with open(tp, "w", encoding="utf-8") as f:
        f.write(expanded_train)
        
    # Purity and Duplicates
    purity = compute_script_purity(expanded_train, lang)
    dup_rate = compute_duplicate_line_rate(expanded_train)
    
    # Fast 8-Gram Overlap vs Val and Test
    val_8_hashes = set(get_ngram_hashes(val_text, lang, n=8))
    test_8_hashes = set(get_ngram_hashes(test_text, lang, n=8))
    
    tokens = get_tokens_for_overlap(expanded_train, lang)
    found_val = set()
    found_test = set()
    for i in range(len(tokens) - 7):
        h = hash(tuple(tokens[i : i + 8]))
        if h in val_8_hashes:
            found_val.add(h)
        if h in test_8_hashes:
            found_test.add(h)
            
    val_ov_pct = (len(found_val) / len(val_8_hashes)) * 100.0 if val_8_hashes else 0.0
    test_ov_pct = (len(found_test) / len(test_8_hashes)) * 100.0 if test_8_hashes else 0.0
    
    # Token count with frozen 16K tokenizer
    tok_ids = tok16k.encode(expanded_train).ids
    n_tok = len(tok_ids)
    b_per_tok = expanded_train_b / n_tok
    
    total_final_bytes += expanded_train_b
    total_final_tokens += n_tok
    tokens_by_lang[lang] = n_tok
    
    # Register Share
    news_b = NEWS_BYTES.get(lang, 0.0)
    wiki_b = expanded_train_b - news_b
    wiki_share = (wiki_b / expanded_train_b) * 100.0
    news_share = (news_b / expanded_train_b) * 100.0
    
    # SHA-256
    h_sha = hashlib.sha256(expanded_train.encode('utf-8')).hexdigest()
    
    floor = FROZEN_FLOORS[lang]
    ceiling = floor * 2.0
    ov_pass = (val_ov_pct <= ceiling) and (test_ov_pct <= ceiling)
    
    expansion_report[lang] = {
        "lang": lang,
        "train_mb": expanded_train_b / (1024 * 1024),
        "added_mb": added_b / (1024 * 1024),
        "tokens": n_tok,
        "bytes_per_tok": b_per_tok,
        "purity": purity,
        "dup_rate": dup_rate,
        "val_ov_pct": val_ov_pct,
        "test_ov_pct": test_ov_pct,
        "natural_floor": floor,
        "ceiling_2x": ceiling,
        "overlap_pass": ov_pass,
        "wiki_share_pct": wiki_share,
        "news_share_pct": news_share,
        "doc_id_overlap": 0,
        "sha256": h_sha
    }
    
    print(f"  -> {lang.upper()} Result: {expanded_train_b/(1024*1024):.2f} MB | {n_tok:,d} tokens ({b_per_tok:.2f} B/tok) | Purity: {purity:.2f}% | Dup: {dup_rate:.2f}% | Val Ov: {val_ov_pct:.2f}% (Floor: {floor:.2f}%) | SHA: {h_sha[:16]}...", flush=True)

# -------------------------------------------------------------
# Summary & Sampling Balance
# -------------------------------------------------------------
report_path = os.path.join(DATA_DIR, "stage7_expanded_corpus_audit.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_corpus_bytes": total_final_bytes,
        "total_corpus_tokens": total_final_tokens,
        "average_bytes_per_token": total_final_bytes / total_final_tokens,
        "languages": expansion_report
    }, f, indent=2)

print("\n" + "=" * 80)
print("STAGE 7 CORPUS EXPANSION COMPLETE")
print(f"Total Raw Bytes : {total_final_bytes:,d} ({total_final_bytes/(1024*1024):.2f} MB)")
print(f"Total 16K Tokens: {total_final_tokens:,d} ({total_final_tokens/1e6:.2f}M tokens)")
print(f"Average B/tok   : {total_final_bytes / total_final_tokens:.2f} Bytes/Token")
print(f"Audit Report    : {report_path}")
print("=" * 80 + "\n", flush=True)
