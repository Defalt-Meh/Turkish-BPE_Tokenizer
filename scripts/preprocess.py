#!/usr/bin/env python3
"""
preprocess.py — Clean and deduplicate a Turkish text corpus for BPE training.

Pipeline:
  1. Line-level deduplication (exact match via hash set)
  2. Language filtering (drop lines that are mostly non-Turkish)
  3. Length filtering (drop very short / very long lines)
  4. Whitespace normalization
  5. Remove control characters and zero-width chars
  6. Optional: NFC Unicode normalization

Usage:
    python3 scripts/preprocess.py <input.txt> <output.txt> [options]

Options:
    --min-len N       Minimum line length in chars (default: 20)
    --max-len N       Maximum line length in chars (default: 10000)
    --no-dedup        Skip deduplication
    --no-langfilter   Skip language filtering
    --sample N        Only process first N lines (for testing)
    -v, --verbose     Print progress every 100K lines
"""

import argparse
import hashlib
import re
import sys
import unicodedata
from collections import Counter


# ── Turkish character set for language detection ─────────────────────

# Characters specific to Turkish (not in basic ASCII Latin)
TURKISH_SPECIFIC = set("çÇğĞıİöÖşŞüÜ")

# Full Turkish alphabet
TURKISH_LETTERS = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
)

# Common Turkish stopwords for quick language ID
TURKISH_STOPWORDS = {
    "bir", "ve", "bu", "da", "de", "ile", "için", "ama", "var", "olan",
    "den", "dan", "dır", "dir", "gibi", "çok", "daha", "en", "kadar",
    "sonra", "olarak", "hem", "ya", "ise", "ancak", "üzerinde", "arasında",
    "olan", "oldu", "olan", "veya", "ki", "ne", "ben", "sen", "biz",
}


def is_likely_turkish(line: str, threshold: float = 0.3) -> bool:
    """
    Heuristic Turkish language filter.

    Checks:
    1. Presence of Turkish-specific characters (ç, ğ, ı, İ, ö, ş, ü)
    2. Ratio of Turkish alphabet chars to total alphabetic chars
    3. Presence of common Turkish stopwords
    """
    if not line.strip():
        return False

    words = line.lower().split()
    if not words:
        return False

    # Check for Turkish-specific characters
    has_turkish_chars = any(c in TURKISH_SPECIFIC for c in line)

    # Check stopword overlap
    word_set = set(words)
    stopword_hits = len(word_set & TURKISH_STOPWORDS)
    stopword_ratio = stopword_hits / max(len(word_set), 1)

    # Check character ratio
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars:
        turkish_chars = sum(1 for c in alpha_chars if c in TURKISH_LETTERS)
        char_ratio = turkish_chars / len(alpha_chars)
    else:
        char_ratio = 0

    # Combine signals
    score = 0.0
    if has_turkish_chars:
        score += 0.4
    score += char_ratio * 0.3
    score += min(stopword_ratio * 2, 0.3)

    return score >= threshold


def clean_line(line: str) -> str:
    """Clean a single line of text."""

    # Remove zero-width characters
    line = line.replace("\u200b", "")  # zero-width space
    line = line.replace("\u200c", "")  # zero-width non-joiner
    line = line.replace("\u200d", "")  # zero-width joiner
    line = line.replace("\ufeff", "")  # BOM

    # Remove control characters (keep \t and \n for now)
    line = "".join(
        c for c in line
        if not unicodedata.category(c).startswith("C") or c in ("\t", "\n")
    )

    # Normalize whitespace: collapse runs of spaces/tabs to single space
    line = re.sub(r"[ \t]+", " ", line)

    # Strip leading/trailing whitespace
    line = line.strip()

    # NFC normalization (compose characters: e + combining accent → é)
    line = unicodedata.normalize("NFC", line)

    return line


def line_hash(line: str) -> str:
    """Fast hash for deduplication."""
    return hashlib.md5(line.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a Turkish text corpus for BPE training."
    )
    parser.add_argument("input", help="Input text file (UTF-8, one doc per line)")
    parser.add_argument("output", help="Output cleaned text file")
    parser.add_argument("--min-len", type=int, default=20,
                        help="Minimum line length in characters (default: 20)")
    parser.add_argument("--max-len", type=int, default=10000,
                        help="Maximum line length in characters (default: 10000)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip deduplication")
    parser.add_argument("--no-langfilter", action="store_true",
                        help="Skip language filtering")
    parser.add_argument("--sample", type=int, default=0,
                        help="Only process first N lines")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress")
    args = parser.parse_args()

    seen_hashes = set()
    stats = Counter()

    print(f"Preprocessing: {args.input} → {args.output}")
    print(f"  min_len={args.min_len}  max_len={args.max_len}")
    print(f"  dedup={'off' if args.no_dedup else 'on'}")
    print(f"  langfilter={'off' if args.no_langfilter else 'on'}")
    if args.sample > 0:
        print(f"  sample={args.sample} lines")
    print()

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for i, raw_line in enumerate(fin):
            if args.sample > 0 and i >= args.sample:
                break

            stats["total"] += 1

            # Clean
            line = clean_line(raw_line)

            # Length filter
            if len(line) < args.min_len:
                stats["too_short"] += 1
                continue

            if len(line) > args.max_len:
                stats["too_long"] += 1
                continue

            # Language filter
            if not args.no_langfilter and not is_likely_turkish(line):
                stats["not_turkish"] += 1
                continue

            # Deduplication
            if not args.no_dedup:
                h = line_hash(line)
                if h in seen_hashes:
                    stats["duplicate"] += 1
                    continue
                seen_hashes.add(h)

            # Write
            fout.write(line + "\n")
            stats["kept"] += 1

            # Progress
            if args.verbose and (i + 1) % 100000 == 0:
                pct = stats["kept"] / stats["total"] * 100
                mem_mb = len(seen_hashes) * 40 / (1024 * 1024)  # rough estimate
                print(f"  {stats['total']:>10,} processed | "
                      f"{stats['kept']:>10,} kept ({pct:.1f}%) | "
                      f"dedup set ~{mem_mb:.0f} MB")

    # ── Summary ──────────────────────────────────────────────────────

    print()
    print("═══════════════════════════════════════════")
    print("  Preprocessing complete!")
    print("═══════════════════════════════════════════")
    print(f"  Total lines:    {stats['total']:>12,}")
    print(f"  Kept:           {stats['kept']:>12,}  "
          f"({stats['kept']/max(stats['total'],1)*100:.1f}%)")
    print(f"  Too short:      {stats['too_short']:>12,}")
    print(f"  Too long:       {stats['too_long']:>12,}")
    print(f"  Not Turkish:    {stats['not_turkish']:>12,}")
    print(f"  Duplicates:     {stats['duplicate']:>12,}")
    print("═══════════════════════════════════════════")
    print()
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()