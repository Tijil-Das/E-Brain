import os
import urllib.request
import prepare_data

GUTENBERG_BOOKS = {
    # Book name: Project Gutenberg ID
    "sherlock_adventures": "1661",      # ~590 KB
    "hound_baskervilles": "2852",      # ~350 KB
    "frankenstein": "84",              # ~440 KB
    "dracula": "345",                  # ~860 KB
    "time_machine": "35",              # ~180 KB
    "pride_and_prejudice": "1342",      # ~720 KB
    "moby_dick": "2701",               # ~1.2 MB
    "alice_in_wonderland": "11",       # ~170 KB
    "great_expectations": "1400",      # ~1.0 MB
    "war_and_peace": "2600"            # ~3.3 MB
}

def fetch_book(book_id, raw_dir="raw_books"):
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"{book_id}.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    req = urllib.request.Request(url, headers=headers)
    print(f"Downloading Gutenberg Book ID {book_id} from {url}...")
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode('utf-8', errors='ignore')
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

def fetch_and_clean_book(book_id):
    raw = fetch_book(book_id)
    return prepare_data.clean_text(raw)

def build_ladder():
    print("==========================================================================")
    print("PHASE 4 & 5: SCALING LADDER DATASET BUILDER")
    print("==========================================================================")

    # 1. Tier 1: 170 KB (already prepared in data_tier1 from Text.txt)
    print("[Tier 1] 170 KB Baseline: Text.txt -> data_tier1")

    # 2. Tier 2: 1 MB (~1,000,000 characters)
    print("\n[Tier 2] Building ~1 MB Dataset (Sherlock Holmes + Baskervilles)...")
    t2_books = ["1661", "2852"]
    t2_clean = "\n\n".join(fetch_and_clean_book(bid) for bid in t2_books)
    
    with open("cleaned_tier2.txt", "w", encoding="utf-8") as f:
        f.write(t2_clean)
    prepare_data.prepare_dataset("cleaned_tier2.txt", "data_tier2_1mb")

    # 3. Tier 3: 5 MB (~5,000,000 characters)
    print("\n[Tier 3] Building ~5 MB Dataset (Classic Literary Corpus)...")
    t3_books = ["1661", "2852", "84", "345", "35", "1342", "2701", "11", "1400", "2600"]
    t3_clean = "\n\n".join(fetch_and_clean_book(bid) for bid in t3_books)

    with open("cleaned_tier3.txt", "w", encoding="utf-8") as f:
        f.write(t3_clean)
    prepare_data.prepare_dataset("cleaned_tier3.txt", "data_tier3_5mb")

    print("\n==========================================================================")
    print(">>> SCALING LADDER READY! DATA TIERS PREPARED & CLEANED. <<<")
    print("==========================================================================")

if __name__ == "__main__":
    build_ladder()
