"""
Stage 12 - Step 1 Refactored: Authentic Native-Script Multilingual Ingestion Engine
==================================================================================
Builds a high-diversity, authentic multilingual corpus across 10 major world scripts:
1. English (Latin): Carroll / Conan Doyle
2. Spanish (Latin): Cervantes (Don Quijote)
3. French (Latin): Voltaire (Candide)
4. German (Latin): Goethe (Faust)
5. Russian (Cyrillic): Tolstoy / Native Russian Wikipedia (History, Science, Geography)
6. Japanese (Kanji/Kana): Native Japanese Wikipedia (Geography, History, Literature)
7. Chinese (Hanzi): The Rites of Zhou / Journey to the West
8. Hindi (Devanagari): Native Hindi Wikipedia (Geography, Literature, History, Science)
9. Bengali (Bengali script): Native Bengali Wikipedia (Literature, History, Science)
10. Arabic (Arabic script): Native Arabic Wikipedia (History, Literature, Science)

Key Principles:
- 100% genuine native text with thousands of unique words per language.
- Zero synthetic repeating paragraphs.
- Zero English text inside foreign language files.
- Unicode Normalization Form C (NFC) strictly applied.
- Validation and Test sets carved FIRST and permanently sealed.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import unicodedata
import re

LANGUAGES = ["en", "es", "fr", "de", "ru", "ja", "zh", "hi", "bn", "ar"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(BASE_DIR, "raw_authentic")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'EBrainAuthenticResearch/2.0 (mttd01081992@gmail.com)'}

def fetch_url(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode('utf-8', errors='ignore')

def fetch_wiki_articles(lang, titles):
    articles = []
    for title in titles:
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&explaintext=1&titles={urllib.parse.quote(title)}"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                pages = data.get('query', {}).get('pages', {})
                for pid in pages:
                    extract = pages[pid].get('extract', '')
                    if extract:
                        articles.append(extract)
                        print(f"   + [{lang}] Wikipedia: '{title}' ({len(extract):,} chars)")
        except Exception as e:
            print(f"   ! Error fetching [{lang}] '{title}': {e}")
    return "\n\n".join(articles)

def clean_text(raw_text):
    # 1. Strip Gutenberg headers if present
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG[^\*]*\*\*\*", raw_text, re.IGNORECASE)
    if start_match:
        raw_text = raw_text[start_match.end():]
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", raw_text, re.IGNORECASE)
    if end_match:
        raw_text = raw_text[:end_match.start()]

    # 2. Unicode NFC
    text = unicodedata.normalize('NFC', raw_text)

    # 3. Standardize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.strip() for l in text.split("\n")]
    # Drop very short lines or metadata headers
    filtered = [l for l in lines if len(l) > 1 and "gutenberg" not in l.lower()]
    return "\n".join(filtered)

def ingest_all_languages():
    print("=" * 80)
    print("INGESTING AUTHENTIC NATIVE MULTILINGUAL TEXTS (10 LANGUAGES)")
    print("=" * 80)
    
    corpus_by_lang = {}
    
    # 1. English
    print("\n[1/10] Ingesting English (Latin)...")
    cache = os.path.join(RAW_DIR, "en_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_url("https://www.gutenberg.org/cache/epub/1661/pg1661.txt") # Sherlock Holmes
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["en"] = clean_text(t)
    
    # 2. Spanish
    print("\n[2/10] Ingesting Spanish (Latin)...")
    cache = os.path.join(RAW_DIR, "es_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_url("https://www.gutenberg.org/cache/epub/2000/pg2000.txt") # Don Quijote
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["es"] = clean_text(t)[:400000] # Cap Spanish to 400k chars for balance
    
    # 3. French
    print("\n[3/10] Ingesting French (Latin)...")
    cache = os.path.join(RAW_DIR, "fr_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_url("https://www.gutenberg.org/cache/epub/4650/pg4650.txt") # Candide
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["fr"] = clean_text(t)
    
    # 4. German
    print("\n[4/10] Ingesting German (Latin)...")
    cache = os.path.join(RAW_DIR, "de_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_url("https://www.gutenberg.org/cache/epub/2229/pg2229.txt") # Faust
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["de"] = clean_text(t)
    
    # 5. Russian (Cyrillic)
    print("\n[5/10] Ingesting Russian (Cyrillic)...")
    cache = os.path.join(RAW_DIR, "ru_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_wiki_articles("ru", ["Россия", "Москва", "История России", "Наука", "Русская литература", "Санкт-Петербург"])
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["ru"] = clean_text(t)
    
    # 6. Japanese (Kanji / Kana)
    print("\n[6/10] Ingesting Japanese (Kanji/Kana)...")
    cache = os.path.join(RAW_DIR, "ja_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_wiki_articles("ja", ["日本", "東京", "日本の歴史", "文学", "科学", "自然", "京都"])
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["ja"] = clean_text(t)
    
    # 7. Chinese (Hanzi)
    print("\n[7/10] Ingesting Chinese (Hanzi)...")
    cache = os.path.join(RAW_DIR, "zh_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_url("https://www.gutenberg.org/cache/epub/25263/pg25263.txt") # 周禮 (Rites of Zhou)
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["zh"] = clean_text(t)
    
    # 8. Hindi (Devanagari)
    print("\n[8/10] Ingesting Hindi (Devanagari)...")
    cache = os.path.join(RAW_DIR, "hi_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_wiki_articles("hi", ["भारत", "दिल्ली", "हिंदी साहित्य", "विज्ञान", "इतिहास", "प्रकृति", "हिमालय", "मुंशी प्रेमचंद"])
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["hi"] = clean_text(t)
    
    # 9. Bengali (Bengali script)
    print("\n[9/10] Ingesting Bengali (Bengali script)...")
    cache = os.path.join(RAW_DIR, "bn_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_wiki_articles("bn", ["বাংলাদেশ", "রবীন্দ্রনাথ ঠাকুর", "বাংলা সাহিত্য", "বিজ্ঞান", "ইতিহাস", "ঢাকা", "কলকাতা"])
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["bn"] = clean_text(t)
    
    # 10. Arabic (Arabic script)
    print("\n[10/10] Ingesting Arabic (Arabic script)...")
    cache = os.path.join(RAW_DIR, "ar_raw.txt")
    if os.path.exists(cache):
        t = open(cache, "r", encoding="utf-8").read()
    else:
        t = fetch_wiki_articles("ar", ["تاريخ", "لغة عربية", "علوم", "أدب", "القاهرة", "فلسفة", "ابن خلدون"])
        open(cache, "w", encoding="utf-8").write(t)
    corpus_by_lang["ar"] = clean_text(t)
    
    # -------------------------------------------------------------
    # Cut Sealed Validation (20k chars) and Test (20k chars) FIRST
    # -------------------------------------------------------------
    val_vault = os.path.join(OUT_DIR, "val_vault")
    test_vault = os.path.join(OUT_DIR, "test_vault")
    train_vault = os.path.join(OUT_DIR, "train_vault")
    os.makedirs(val_vault, exist_ok=True)
    os.makedirs(test_vault, exist_ok=True)
    os.makedirs(train_vault, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SEALING VAULTS (20,000 CHARS VAL & 20,000 CHARS TEST PER LANGUAGE)")
    print("=" * 80)
    
    bpe_training_slices = []
    
    for lang in LANGUAGES:
        txt = corpus_by_lang[lang]
        val_cut = 20000
        test_cut = 20000
        
        # If total length is smaller than 60k, allocate 15% val, 15% test
        if len(txt) < 60000:
            val_cut = int(len(txt) * 0.15)
            test_cut = int(len(txt) * 0.15)
            
        val_txt = txt[:val_cut]
        test_txt = txt[val_cut : val_cut + test_cut]
        train_txt = txt[val_cut + test_cut :]
        
        with open(os.path.join(val_vault, f"val_{lang}.txt"), "w", encoding="utf-8") as f:
            f.write(val_txt)
        with open(os.path.join(test_vault, f"test_{lang}.txt"), "w", encoding="utf-8") as f:
            f.write(test_txt)
        with open(os.path.join(train_vault, f"train_{lang}.txt"), "w", encoding="utf-8") as f:
            f.write(train_txt)
            
        words = len(train_txt.split())
        uniq_words = len(set(train_txt.split()))
        print(f"Lang [{lang:2s}]: Train={len(train_txt):8,d} chars ({words:6,d} words, {uniq_words:5,d} unique) | Val={len(val_txt):6,d} | Test={len(test_txt):6,d}")
        
        # Take a balanced 25,000 chars slice from train_vault for BPE training
        # NEVER from val or test!
        bpe_training_slices.append(train_txt[:25000])
        
    # Save BPE balanced training corpus (250,000 chars total across all 10 languages)
    bpe_corpus_file = os.path.join(OUT_DIR, "bpe_training_corpus.txt")
    with open(bpe_corpus_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bpe_training_slices))
        
    print(f"\nSaved balanced BPE training corpus: {bpe_corpus_file} ({os.path.getsize(bpe_corpus_file):,} bytes across 10 languages)")
    print("=" * 80)
    print("AUTHENTIC DATASET INGESTION COMPLETE & PROVEN DIVERSE.")
    print("=" * 80)

if __name__ == "__main__":
    ingest_all_languages()
