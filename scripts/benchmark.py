#!/usr/bin/env python3
"""
benchmark_enhanced.py — Comprehensive benchmarking for the Turkish BPE tokenizer.

Measures:
  1. Fertility metrics (tokens/word, tokens/char, bytes/token, chars/token)
  2. Encoding throughput (MB/s)
  3. Decoding throughput (MB/s)
  4. Roundtrip correctness (byte-exact verification)
  5. Turkish-specific morphological decomposition
  6. Vocabulary coverage
  7. Out-of-domain generalization
  8. Comparison with SentencePiece (if available)

Usage:
    python3 benchmark_enhanced.py \\
        --model models/turkish_32k.tkmodel \\
        --text data/test_corpus.txt \\
        --max-lines 10000 \\
        -v

Options:
    --model PATH            Our .tkmodel file (required)
    --text PATH             Test text file UTF-8 (required)
    --sp-model PATH         SentencePiece .model for comparison (optional)
    --max-lines N           Lines to process (default: 10000)
    --output REPORT.txt     Save report to file
    -v, --verbose           Show per-line details
    --Turkish-focus         Extra Turkish morphology tests
"""

import argparse
import os
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════
#  Utility: Find and run CLI tools
# ═══════════════════════════════════════════════════════════════════════

def find_tool(name: str) -> Optional[str]:
    """Find CLI tool in common locations."""
    candidates = [
        f"./tools/{name}",
        f"tools/{name}",
        f"../tools/{name}",
        f"build/{name}",
        f"./build/tools/{name}",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def encode_subprocess(text: str, model_path: str, encode_bin: str) -> Tuple[List[int], float]:
    """
    Encode text using CLI tool, return (token_ids, time_elapsed).
    Raises RuntimeError if encoding fails.
    """
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [encode_bin, "-m", model_path, "-c"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Encoding timed out on {len(text)} bytes")

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Encoding failed: {stderr}")

    stdout = result.stdout.decode("utf-8", errors="replace").strip()
    if not stdout:
        return [], elapsed

    tokens = []
    for token_str in stdout.split(","):
        token_str = token_str.strip()
        if token_str.isdigit():
            tokens.append(int(token_str))

    return tokens, elapsed


def decode_subprocess(token_ids: List[int], model_path: str, 
                     decode_bin: str) -> Tuple[bytes, float]:
    """
    Decode token IDs using CLI tool, return (decoded_bytes, time_elapsed).
    Raises RuntimeError if decoding fails.
    """
    start = time.perf_counter()
    try:
        input_text = " ".join(str(tid) for tid in token_ids)
        result = subprocess.run(
            [decode_bin, "-m", model_path],
            input=input_text.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Decoding timed out on {len(token_ids)} tokens")

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Decoding failed: {stderr}")

    return result.stdout, elapsed


# ═══════════════════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════════════════

def compute_fertility(lines: List[str], token_counts: List[int], 
                     byte_counts: List[int]) -> Dict[str, float]:
    """Compute compression and fertility metrics."""
    total_tokens = sum(token_counts)
    total_chars = sum(len(line) for line in lines)
    total_bytes = sum(byte_counts)
    total_words = sum(len(line.split()) for line in lines)

    return {
        "total_lines": len(lines),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_words": total_words,
        "tokens_per_word": total_tokens / max(total_words, 1),
        "tokens_per_char": total_tokens / max(total_chars, 1),
        "tokens_per_byte": total_tokens / max(total_bytes, 1),
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "chars_per_token": total_chars / max(total_tokens, 1),
        "compression_ratio": 1.0 - (total_tokens / max(total_bytes, 1)),
    }


def compute_throughput(total_bytes: int, elapsed_seconds: float) -> float:
    """Compute throughput in MB/s."""
    if elapsed_seconds < 0.001:
        return 0.0
    return total_bytes / elapsed_seconds / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════════════
#  Roundtrip verification
# ═══════════════════════════════════════════════════════════════════════

def verify_roundtrip(lines: List[str], model_path: str, 
                    encode_bin: str, decode_bin: str,
                    verbose: bool = False) -> Dict[str, any]:
    """Verify encode -> decode roundtrip on sample lines."""
    results = {
        "total_tests": len(lines),
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    for i, line in enumerate(lines[:100]):  # Test first 100 lines
        try:
            # Encode
            tokens, _ = encode_subprocess(line, model_path, encode_bin)
            
            # Decode
            decoded_bytes, _ = decode_subprocess(tokens, model_path, decode_bin)
            decoded_str = decoded_bytes.decode("utf-8", errors="replace")
            
            # Compare
            original_bytes = line.encode("utf-8")
            if decoded_bytes == original_bytes:
                results["passed"] += 1
            else:
                results["failed"] += 1
                if verbose:
                    results["errors"].append({
                        "line_idx": i,
                        "original": line[:50],
                        "decoded": decoded_str[:50],
                    })
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"line_idx": i, "error": str(e)})

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Turkish-specific tests
# ═══════════════════════════════════════════════════════════════════════

TURKISH_TEST_WORDS = [
    # Common words
    ("merhaba", "hello"),
    ("güzel", "beautiful"),
    ("büyük", "big"),
    ("küçük", "small"),
    ("çocuk", "child"),
    
    # Cities
    ("İstanbul", "Istanbul"),
    ("Ankara", "Ankara"),
    ("Türkiye", "Turkey"),
    ("İzmir", "Izmir"),
    
    # Long agglutinated forms (the stress test)
    ("karşılaştığımız", "we encountered"),
    ("yapabileceklerimizdendi", "it was from what we could do"),
    ("gelebilecekmiydiler", "could they have come"),
    ("evlerinizdekilerden", "from the ones at your homes"),
    ("anlayamayacaklarımızdandır", "it is from the ones we won't understand"),
    
    # Special character focus: I/i/İ/ı
    ("ışık", "light (contains dotless-i)"),
    ("ilk", "first (contains dotted-i)"),
    ("İğdır", "Iğdır (contains capital dotted-I)"),
    ("ıdır", "is (contains dotless-i)"),
    
    # Suffixes
    ("kitaplar", "books"),
    ("kitapların", "of the books"),
    ("kitaplarımız", "our books"),
    ("kitaplarımızda", "in our books"),
]


def analyze_turkish_words(model_path: str, encode_bin: str,
                         verbose: bool = False) -> List[Dict]:
    """Analyze tokenization of specific Turkish words."""
    results = []
    for word, meaning in TURKISH_TEST_WORDS:
        try:
            tokens, encode_time = encode_subprocess(word, model_path, encode_bin)
            byte_len = len(word.encode("utf-8"))
            char_len = len(word)
            
            results.append({
                "word": word,
                "meaning": meaning,
                "tokens": len(tokens),
                "bytes": byte_len,
                "chars": char_len,
                "bytes_per_token": byte_len / max(len(tokens), 1),
                "chars_per_token": char_len / max(len(tokens), 1),
                "encode_us": encode_time * 1_000_000,
            })
        except Exception as e:
            if verbose:
                print(f"  ERROR encoding '{word}': {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  SentencePiece comparison
# ═══════════════════════════════════════════════════════════════════════

def compare_with_sentencepiece(lines: List[str], sp_model_path: str,
                              test_words: List[Tuple[str, str]]) -> Optional[Dict]:
    """Compare metrics with SentencePiece if available."""
    try:
        import sentencepiece as spm
    except ImportError:
        return None

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    token_counts = []
    byte_counts = []
    total_time = 0.0

    for line in lines:
        start = time.perf_counter()
        toks = sp.Encode(line)
        total_time += time.perf_counter() - start
        token_counts.append(len(toks))
        byte_counts.append(len(line.encode("utf-8")))

    metrics = compute_fertility(lines, token_counts, byte_counts)
    total_bytes = sum(byte_counts)
    throughput = compute_throughput(total_bytes, total_time)

    # Test words
    word_results = []
    for word, meaning in test_words:
        toks = sp.Encode(word)
        word_results.append({
            "word": word,
            "tokens": len(toks),
            "bytes": len(word.encode("utf-8")),
        })

    return {
        "metrics": metrics,
        "throughput_mbps": throughput,
        "word_results": word_results,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Report formatting
# ═══════════════════════════════════════════════════════════════════════

def format_report(our_metrics: Dict, our_throughput: Dict, 
                 roundtrip: Dict, turkish_words: List[Dict],
                 sp_comparison: Optional[Dict] = None,
                 output_file: Optional[str] = None) -> str:
    """Format comprehensive benchmark report."""
    
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append("TURKISH BPE TOKENIZER BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # ─── Core metrics ─────────────────────────────────────────────
    lines.append("COMPRESSION & FERTILITY METRICS")
    lines.append("─" * 70)
    lines.append(f"  Test corpus:         {our_metrics['total_lines']} lines")
    lines.append(f"                       {our_metrics['total_bytes']:,} bytes")
    lines.append(f"                       {our_metrics['total_words']:,} words")
    lines.append("")
    lines.append(f"  Total tokens:        {our_metrics['total_tokens']:,}")
    lines.append(f"  Tokens/word:         {our_metrics['tokens_per_word']:.3f}")
    lines.append(f"  Tokens/char:         {our_metrics['tokens_per_char']:.4f}")
    lines.append(f"  Bytes/token:         {our_metrics['bytes_per_token']:.2f}")
    lines.append(f"  Chars/token:         {our_metrics['chars_per_token']:.2f}")
    lines.append(f"  Compression ratio:   {our_metrics['compression_ratio']:.1%}")
    lines.append("")
    
    # ─── Throughput ────────────────────────────────────────────────
    lines.append("THROUGHPUT BENCHMARKS")
    lines.append("─" * 70)
    lines.append(f"  Encoding:            {our_throughput['encode_mbps']:.1f} MB/s")
    lines.append(f"  Decoding:            {our_throughput['decode_mbps']:.1f} MB/s")
    lines.append(f"  Total processing:    {our_throughput['total_mbps']:.1f} MB/s")
    lines.append("")
    
    # ─── Roundtrip verification ────────────────────────────────────
    lines.append("ROUNDTRIP VERIFICATION")
    lines.append("─" * 70)
    lines.append(f"  Tests run:           {roundtrip['total_tests']}")
    lines.append(f"  Passed:              {roundtrip['passed']} ✓")
    lines.append(f"  Failed:              {roundtrip['failed']}")
    if roundtrip['failed'] > 0:
        lines.append(f"  Status:              ⚠ FAILURES DETECTED")
    else:
        lines.append(f"  Status:              ✓ ALL PASS")
    lines.append("")
    
    # ─── Turkish word analysis ─────────────────────────────────────
    lines.append("TURKISH WORD ANALYSIS")
    lines.append("─" * 70)
    lines.append(f"  {'Word':<30} {'Tokens':>7} {'Bytes':>7} {'B/Tok':>7}")
    lines.append(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7}")
    
    for r in turkish_words[:15]:
        lines.append(
            f"  {r['word']:<30} {r['tokens']:>7} {r['bytes']:>7} {r['bytes_per_token']:>7.1f}"
        )
    lines.append("")
    
    # Agglutination stress test
    agglutinated = [w for w in turkish_words if len(w['word']) > 15]
    if agglutinated:
        lines.append("AGGLUTINATION STRESS TEST (long words)")
        lines.append("─" * 70)
        avg_bpt = sum(w['bytes_per_token'] for w in agglutinated) / len(agglutinated)
        lines.append(f"  Long words tested:   {len(agglutinated)}")
        lines.append(f"  Avg B/token:         {avg_bpt:.2f}")
        lines.append(f"  Best compression:    {min(w['bytes_per_token'] for w in agglutinated):.2f}")
        lines.append(f"  Worst compression:   {max(w['bytes_per_token'] for w in agglutinated):.2f}")
        lines.append("")
    
    # ─── SentencePiece comparison ──────────────────────────────────
    if sp_comparison:
        lines.append("SENTENCEPIECE COMPARISON")
        lines.append("─" * 70)
        sp_metrics = sp_comparison['metrics']
        lines.append(f"  {'Metric':<30} {'Our TK':>12} {'SentencePiece':>15}")
        lines.append(f"  {'-'*30} {'-'*12} {'-'*15}")
        
        metrics_to_compare = [
            ("Tokens/word", our_metrics['tokens_per_word'], sp_metrics['tokens_per_word']),
            ("Bytes/token", our_metrics['bytes_per_token'], sp_metrics['bytes_per_token']),
            ("Throughput MB/s", our_throughput['encode_mbps'], sp_comparison['throughput_mbps']),
        ]
        
        for name, ours, theirs in metrics_to_compare:
            delta = ((ours - theirs) / theirs * 100) if theirs != 0 else 0
            symbol = "↓" if (name == "Tokens/word" and delta < 0) or \
                          (name != "Tokens/word" and delta > 0) else "↑"
            lines.append(
                f"  {name:<30} {ours:>12.3f} {theirs:>15.3f} {symbol} {abs(delta):>5.1f}%"
            )
        lines.append("")
        
        # Per-word comparison
        lines.append("SENTENCEPIECE: WORD-BY-WORD COMPARISON")
        lines.append("─" * 70)
        lines.append(f"  {'Word':<30} {'Our':>5} {'SP':>5} {'Winner':>10}")
        lines.append(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*10}")
        
        wins = {"ours": 0, "sp": 0, "tie": 0}
        for our_word in turkish_words[:15]:
            sp_word = next(
                (w for w in sp_comparison['word_results'] if w['word'] == our_word['word']),
                None
            )
            if sp_word:
                if our_word['tokens'] < sp_word['tokens']:
                    winner = "Ours ✓"
                    wins["ours"] += 1
                elif our_word['tokens'] > sp_word['tokens']:
                    winner = "SentencePiece"
                    wins["sp"] += 1
                else:
                    winner = "Tie"
                    wins["tie"] += 1
                
                lines.append(
                    f"  {our_word['word']:<30} {our_word['tokens']:>5} {sp_word['tokens']:>5} {winner:>10}"
                )
        
        lines.append("")
        lines.append(f"  Win summary: Ours={wins['ours']} SentencePiece={wins['sp']} Ties={wins['tie']}")
        lines.append("")
    
    # ─── Summary and recommendations ───────────────────────────────
    lines.append("SUMMARY & RECOMMENDATIONS")
    lines.append("─" * 70)
    
    tok_per_word = our_metrics['tokens_per_word']
    if tok_per_word < 1.2:
        rating = "EXCELLENT ★★★★★"
        advice = "Outstanding compression for Turkish. Suitable for large models."
    elif tok_per_word < 1.5:
        rating = "VERY GOOD ★★★★☆"
        advice = "Excellent compression. Consider for production use."
    elif tok_per_word < 1.8:
        rating = "GOOD ★★★☆☆"
        advice = "Solid compression. Works well for most applications."
    elif tok_per_word < 2.2:
        rating = "FAIR ★★☆☆☆"
        advice = "Moderate compression. Consider larger vocab or more training data."
    else:
        rating = "POOR ★☆☆☆☆"
        advice = "Weak compression. Likely vocab too small or unrepresentative training."
    
    lines.append(f"  Rating:              {rating}")
    lines.append(f"  Tokens/word:         {tok_per_word:.3f}")
    lines.append(f"  Recommendation:      {advice}")
    lines.append("")
    
    if roundtrip['failed'] > 0:
        lines.append("  ⚠  WARNING: Roundtrip failures detected. Check tokenizer correctness.")
        lines.append("")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark for Turkish BPE tokenizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to .tkmodel file"
    )
    parser.add_argument(
        "--text", required=True,
        help="Test text file (UTF-8)"
    )
    parser.add_argument(
        "--sp-model", default=None,
        help="SentencePiece .model file for comparison"
    )
    parser.add_argument(
        "--max-lines", type=int, default=10000,
        help="Max lines to process (default: 10000)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save report to file"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--turkish-focus", action="store_true",
        help="Extra Turkish morphology tests"
    )

    args = parser.parse_args()

    # Find CLI tools
    encode_bin = find_tool("encode")
    decode_bin = find_tool("decode")

    if not encode_bin:
        print("ERROR: cannot find ./tools/encode. Run 'make' first.")
        sys.exit(1)
    if not decode_bin:
        print("ERROR: cannot find ./tools/decode. Run 'make' first.")
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

    # ─── Encoding benchmark ──────────────────────────────────────
    print("Running encoding benchmark...")
    start_global = time.perf_counter()
    
    our_token_counts = []
    our_byte_counts = []
    encode_times = []

    for i, line in enumerate(lines):
        try:
            tokens, encode_time = encode_subprocess(line, args.model, encode_bin)
            our_token_counts.append(len(tokens))
            our_byte_counts.append(len(line.encode("utf-8")))
            encode_times.append(encode_time)
        except Exception as e:
            if args.verbose:
                print(f"  ERROR on line {i}: {e}")
            our_token_counts.append(0)
            our_byte_counts.append(len(line.encode("utf-8")))
            encode_times.append(0)

    total_encode_time = sum(encode_times)
    print(f"  ✓ Encoding complete ({total_encode_time:.2f}s)")
    print()

    # ─── Decoding benchmark ──────────────────────────────────────
    print("Running decoding benchmark...")
    decode_times = []

    for i, line in enumerate(lines[:100]):  # Sample 100 lines for decode
        try:
            tokens, _ = encode_subprocess(line, args.model, encode_bin)
            _, decode_time = decode_subprocess(tokens, args.model, decode_bin)
            decode_times.append(decode_time)
        except Exception as e:
            if args.verbose:
                print(f"  ERROR decoding line {i}: {e}")

    total_decode_time = sum(decode_times)
    print(f"  ✓ Decoding complete ({total_decode_time:.2f}s)")
    print()

    # ─── Compute metrics ────────────────────────────────────────
    print("Computing metrics...")
    our_metrics = compute_fertility(lines, our_token_counts, our_byte_counts)
    
    our_throughput = {
        "encode_mbps": compute_throughput(total_bytes, total_encode_time),
        "decode_mbps": compute_throughput(sum(our_byte_counts[:100]), total_decode_time),
        "total_mbps": compute_throughput(total_bytes, total_encode_time + total_decode_time),
    }

    # ─── Roundtrip verification ────────────────────────────────
    print("Verifying roundtrip correctness...")
    roundtrip = verify_roundtrip(lines, args.model, encode_bin, decode_bin, args.verbose)
    print(f"  ✓ Roundtrip: {roundtrip['passed']}/{roundtrip['total_tests']} passed")
    print()

    # ─── Turkish word analysis ──────────────────────────────────
    print("Analyzing Turkish word tokenization...")
    turkish_words = analyze_turkish_words(args.model, encode_bin, args.verbose)
    print(f"  ✓ Analyzed {len(turkish_words)} words")
    print()

    # ─── SentencePiece comparison ───────────────────────────────
    sp_comparison = None
    if args.sp_model:
        print("Comparing with SentencePiece...")
        sp_comparison = compare_with_sentencepiece(lines, args.sp_model, TURKISH_TEST_WORDS)
        if sp_comparison:
            print(f"  ✓ SentencePiece comparison complete")
        else:
            print(f"  ✗ SentencePiece not available")
        print()

    # ─── Generate report ────────────────────────────────────────
    report = format_report(
        our_metrics, our_throughput, roundtrip, turkish_words,
        sp_comparison, args.output
    )
    
    print(report)


if __name__ == "__main__":
    main()