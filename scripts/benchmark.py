#!/usr/bin/env python3
"""
benchmark.py — Benchmark the Turkish BPE tokenizer.

Measures:
  1. Fertility (tokens per word / tokens per character)
  2. Encoding throughput (MB/s)
  3. Vocabulary coverage
  4. Comparison with SentencePiece (if available)

Usage:
    python3 scripts/benchmark.py --model models/turkish.tkmodel --text data/wiki_tr_clean.txt

Options:
    --model PATH      Our .tkmodel file
    --text PATH       Test text file (UTF-8)
    --sp-model PATH   SentencePiece .model file for comparison (optional)
    --max-lines N     Limit test to first N lines (default: 10000)
    -v, --verbose     Show per-line details
"""

import argparse
import ctypes
import ctypes.util
import os
import struct
import subprocess
import sys
import time


# ═══════════════════════════════════════════════════════════════════════
#  Load our tokenizer via the CLI tools (subprocess-based)
#
#  In production you'd write Python bindings with ctypes or cffi.
#  For benchmarking, shelling out to ./tools/encode is simple and fair.
# ═══════════════════════════════════════════════════════════════════════

def find_encode_tool():
    """Find the encode binary."""
    candidates = [
        "./tools/encode",
        "tools/encode",
        "../tools/encode",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def encode_with_our_tokenizer(text: str, model_path: str, encode_bin: str) -> list[int]:
    """Encode text using our tokenizer via CLI."""
    result = subprocess.run(
        [encode_bin, "-m", model_path, "-c"],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        return []

    stdout = result.stdout.decode("utf-8").strip()
    if not stdout:
        return []

    return [int(x) for x in stdout.split(",") if x.strip().isdigit()]


def encode_with_sentencepiece(text: str, sp_model_path: str):
    """Encode text using SentencePiece (if available)."""
    try:
        import sentencepiece as spm
    except ImportError:
        return None

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    return sp.Encode(text)


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_fertility(lines: list[str], token_counts: list[int]) -> dict:
    """Compute fertility metrics."""
    total_tokens = sum(token_counts)
    total_chars = sum(len(line) for line in lines)
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)
    total_words = sum(len(line.split()) for line in lines)

    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_words": total_words,
        "tokens_per_char": total_tokens / max(total_chars, 1),
        "tokens_per_word": total_tokens / max(total_words, 1),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "chars_per_token": total_chars / max(total_tokens, 1),
    }


def find_unknown_rate(lines: list[str], token_counts: list[int],
                      byte_counts: list[int]) -> float:
    """
    Estimate the "unknown" rate.
    For byte-level BPE there are no true unknowns, but we can check
    if any single-byte fallbacks dominate (high token count ≈ byte count
    means poor compression / many fallback tokens).
    """
    total_tokens = sum(token_counts)
    total_bytes = sum(byte_counts)

    if total_tokens == 0:
        return 0.0

    # If tokens ≈ bytes, the tokenizer is barely compressing
    compression = 1.0 - (total_tokens / max(total_bytes, 1))
    return max(0.0, 1.0 - compression)  # rough "inefficiency" rate


# ═══════════════════════════════════════════════════════════════════════
#  Turkish-specific analysis
# ═══════════════════════════════════════════════════════════════════════

TURKISH_TEST_WORDS = [
    # Common words
    "merhaba", "güzel", "büyük", "küçük", "çocuk",
    "İstanbul", "Ankara", "Türkiye",
    # Agglutinated forms
    "evlerinizdekilerden",
    "gelebilecekmiydiler",
    "yapabileceklerimizdendi",
    "anlayamayacaklarımızdandır",
    # İ/I/i/ı variations
    "ışık", "ilk", "İzmir", "İğdır",
]


def analyze_turkish_words(model_path: str, encode_bin: str) -> list[dict]:
    """Analyze tokenization of specific Turkish words."""
    results = []
    for word in TURKISH_TEST_WORDS:
        tokens = encode_with_our_tokenizer(word, model_path, encode_bin)
        byte_len = len(word.encode("utf-8"))
        results.append({
            "word": word,
            "tokens": len(tokens),
            "bytes": byte_len,
            "ratio": byte_len / max(len(tokens), 1),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Turkish BPE tokenizer."
    )
    parser.add_argument("--model", required=True,
                        help="Path to our .tkmodel file")
    parser.add_argument("--text", required=True,
                        help="Test text file (UTF-8)")
    parser.add_argument("--sp-model", default=None,
                        help="SentencePiece .model file for comparison")
    parser.add_argument("--max-lines", type=int, default=10000,
                        help="Max lines to process (default: 10000)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Find our encode tool
    encode_bin = find_encode_tool()
    if not encode_bin:
        print("ERROR: cannot find ./tools/encode binary.")
        print("Run 'make tools' first.")
        sys.exit(1)

    # Load test text
    print(f"Loading test text: {args.text}")
    with open(args.text, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= args.max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)

    print(f"  {len(lines)} lines loaded")
    total_bytes = sum(len(l.encode("utf-8")) for l in lines)
    print(f"  {total_bytes:,} bytes total")
    print()

    # ── Our tokenizer ────────────────────────────────────────────────

    print("═══════════════════════════════════════════")
    print("  Our BPE Tokenizer")
    print("═══════════════════════════════════════════")

    # Encode all lines as a single block for throughput measurement
    all_text = "\n".join(lines)

    start = time.perf_counter()
    our_tokens = encode_with_our_tokenizer(all_text, args.model, encode_bin)
    elapsed = time.perf_counter() - start

    our_token_counts = []
    our_byte_counts = []
    for line in lines:
        toks = encode_with_our_tokenizer(line, args.model, encode_bin)
        our_token_counts.append(len(toks))
        our_byte_counts.append(len(line.encode("utf-8")))

    metrics = compute_fertility(lines, our_token_counts)

    throughput = total_bytes / max(elapsed, 0.001) / (1024 * 1024)

    print(f"  Throughput:        {throughput:.1f} MB/s")
    print(f"  Total tokens:      {metrics['total_tokens']:,}")
    print(f"  Tokens/word:       {metrics['tokens_per_word']:.2f}")
    print(f"  Tokens/char:       {metrics['tokens_per_char']:.2f}")
    print(f"  Bytes/token:       {metrics['bytes_per_token']:.2f}")
    print(f"  Chars/token:       {metrics['chars_per_token']:.2f}")
    print()

    # ── Turkish word analysis ────────────────────────────────────────

    print("── Turkish Word Analysis ──")
    print(f"  {'Word':<35} {'Tokens':>6} {'Bytes':>6} {'B/Tok':>6}")
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*6}")

    word_results = analyze_turkish_words(args.model, encode_bin)
    for r in word_results:
        print(f"  {r['word']:<35} {r['tokens']:>6} {r['bytes']:>6} {r['ratio']:>6.1f}")
    print()

    # ── SentencePiece comparison ─────────────────────────────────────

    if args.sp_model:
        try:
            import sentencepiece as spm

            print("═══════════════════════════════════════════")
            print("  SentencePiece Comparison")
            print("═══════════════════════════════════════════")

            sp = spm.SentencePieceProcessor()
            sp.Load(args.sp_model)

            sp_token_counts = []
            start = time.perf_counter()
            for line in lines:
                toks = sp.Encode(line)
                sp_token_counts.append(len(toks))
            sp_elapsed = time.perf_counter() - start

            sp_metrics = compute_fertility(lines, sp_token_counts)
            sp_throughput = total_bytes / max(sp_elapsed, 0.001) / (1024 * 1024)

            print(f"  Throughput:        {sp_throughput:.1f} MB/s")
            print(f"  Total tokens:      {sp_metrics['total_tokens']:,}")
            print(f"  Tokens/word:       {sp_metrics['tokens_per_word']:.2f}")
            print(f"  Bytes/token:       {sp_metrics['bytes_per_token']:.2f}")
            print()

            # Side-by-side
            print("── Head-to-Head ──")
            print(f"  {'Metric':<20} {'Ours':>10} {'SentencePiece':>14} {'Winner':>10}")
            print(f"  {'─'*20} {'─'*10} {'─'*14} {'─'*10}")

            pairs = [
                ("Tokens/word",
                 metrics['tokens_per_word'], sp_metrics['tokens_per_word'], "lower"),
                ("Bytes/token",
                 metrics['bytes_per_token'], sp_metrics['bytes_per_token'], "higher"),
                ("Throughput MB/s",
                 throughput, sp_throughput, "higher"),
            ]

            for name, ours, theirs, better in pairs:
                if better == "lower":
                    winner = "Ours" if ours <= theirs else "SP"
                else:
                    winner = "Ours" if ours >= theirs else "SP"
                print(f"  {name:<20} {ours:>10.2f} {theirs:>14.2f} {winner:>10}")

            print()

            # Per-word comparison
            print("── Turkish Words: Ours vs SentencePiece ──")
            print(f"  {'Word':<30} {'Ours':>6} {'SP':>6} {'Winner':>8}")
            print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*8}")

            for r in word_results:
                sp_toks = sp.Encode(r['word'])
                sp_count = len(sp_toks)
                winner = "Ours" if r['tokens'] <= sp_count else "SP"
                if r['tokens'] == sp_count:
                    winner = "Tie"
                print(f"  {r['word']:<30} {r['tokens']:>6} {sp_count:>6} {winner:>8}")

            print()

        except ImportError:
            print("SentencePiece not installed. Skipping comparison.")
            print("Install with: pip install sentencepiece")
            print()

    elif args.verbose:
        print("(No --sp-model provided, skipping SentencePiece comparison)")
        print()

    # ── Summary ──────────────────────────────────────────────────────

    print("═══════════════════════════════════════════")
    print("  Summary")
    print("═══════════════════════════════════════════")
    print(f"  A good Turkish tokenizer should achieve ~1.2-1.8 tokens/word.")
    print(f"  Our result: {metrics['tokens_per_word']:.2f} tokens/word")
    print()

    if metrics['tokens_per_word'] < 1.5:
        print("  Excellent compression for Turkish!")
    elif metrics['tokens_per_word'] < 2.0:
        print("  Good compression. Consider more training data or larger vocab.")
    elif metrics['tokens_per_word'] < 3.0:
        print("  Moderate. Try increasing vocab size or training corpus.")
    else:
        print("  Poor compression. The vocab may be too small or the training")
        print("  corpus may not be representative of Turkish text.")


if __name__ == "__main__":
    main()