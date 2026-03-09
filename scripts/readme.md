# Turkish Tokenizer Benchmarking Guide

## Overview

This guide explains how to benchmark your Turkish BPE tokenizer against known metrics and how to interpret the results.

The benchmark script measures:
- **Compression metrics** (tokens/word, bytes/token, compression ratio)
- **Speed** (encoding/decoding throughput in MB/s)
- **Correctness** (byte-exact roundtrip verification)
- **Turkish morphology** (agglutinated word handling, special characters)
- **Comparison with SentencePiece** (optional but recommended)

---

## Setup

### Prerequisites

1. **Build the tokenizer**
   ```bash
   make clean
   make
   ```
   This creates:
   - `build/libtokenizer.a` (library)
   - `tools/encode` (CLI encoder)
   - `tools/decode` (CLI decoder)
   - `tools/train` (trainer)
   - `tools/inspect` (model inspector)

2. **Prepare test corpus**
   
   Option A: Use OSCAR Turkish (recommended, ~30GB)
   ```bash
   bash scripts/download_oscar.sh
   python3 scripts/preprocess.py data/oscar_tr.txt data/oscar_tr_clean.txt -v
   ```
   
   Option B: Use a smaller representative corpus
   ```bash
   # Turkish Wikipedia dump, cleaned
   # or any UTF-8 Turkish text file
   ```

3. **Train models at different vocab sizes** (for comparison)
   ```bash
   ./tools/train -i data/oscar_tr_clean.txt -o models/turkish_512.tkmodel -v 512
   ./tools/train -i data/oscar_tr_clean.txt -o models/turkish_1k.tkmodel -v 1024
   ./tools/train -i data/oscar_tr_clean.txt -o models/turkish_8k.tkmodel -v 8192
   ./tools/train -i data/oscar_tr_clean.txt -o models/turkish_32k.tkmodel -v 32000
   ```

4. **Optional: Install SentencePiece for comparison**
   ```bash
   pip install sentencepiece
   ```
   
   Train SentencePiece models with matching vocab:
   ```bash
   spm_train --input=data/oscar_tr_clean.txt \
     --model_prefix=models/spm_turkish_32k \
     --vocab_size=32000 --model_type=bpe
   ```

---

## Running Benchmarks

### Basic benchmark (our tokenizer only)

```bash
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/oscar_tr_clean.txt \
  --max-lines 10000
```

This runs on the first 10,000 lines. Output shows:
- Compression metrics
- Throughput (MB/s)
- Roundtrip correctness
- Turkish word analysis

### With SentencePiece comparison

```bash
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/oscar_tr_clean.txt \
  --sp-model models/spm_turkish_32k.model \
  --max-lines 10000 \
  --output benchmark_report.txt
```

Saves detailed report to `benchmark_report.txt`.

### With verbose output (see per-word details)

```bash
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/oscar_tr_clean.txt \
  --verbose \
  --turkish-focus
```

---

## Understanding the Metrics

### Compression Metrics

| Metric | Good value (32K vocab) | Meaning |
|--------|--------|---------|
| **Tokens/word** | < 1.5 | Fewer tokens per word = better compression. Turkish targets 1.2-1.6. |
| **Bytes/token** | 1.5-2.5 | Average UTF-8 bytes per token. Higher = better word-level packing. |
| **Chars/token** | 1.2-2.0 | Average characters per token. |
| **Compression ratio** | > 80% | Percentage reduction from raw bytes. BPE should achieve 75-85%. |

**Example interpretation:**
- **Your result: 1.35 tokens/word** → Excellent. Better than typical English (1.3) and much better than English-centric tokenizers on Turkish (2.0+).
- **Your result: 1.85 tokens/word** → Good but could be improved. Try larger vocab or more training data.
- **Your result: 3.0+ tokens/word** → Poor compression. Likely vocab is too small or training corpus unrepresentative.

### Throughput

| Metric | Good value | Meaning |
|--------|-----------|---------|
| **Encode MB/s** | > 50 | Inference speed. ∼10-100 MB/s is typical for C tokenizers. |
| **Decode MB/s** | > 50 | Reconstruction speed. |
| **Total MB/s** | > 40 | Combined encode + decode. |

**Interpretation:**
- **100+ MB/s** → Very fast. Suitable for real-time streaming.
- **50-100 MB/s** → Good. Acceptable for batch processing.
- **< 20 MB/s** → Slow. Check for bottlenecks or use smaller test set.

### Roundtrip Correctness

| Result | Status |
|--------|--------|
| **All pass** | ✓ Tokenizer is correct. Byte-exact reconstruction. |
| **Some failures** | ⚠ Bug detected. Investigate encode/decode logic. |
| **All fail** | ✗ Critical error. Do not use. |

This is non-negotiable: **encoding → decoding must restore the original bytes exactly**.

### Turkish Word Analysis

The benchmark tests 20 specific Turkish words covering:

1. **Common words**: merhaba, güzel, büyük
2. **Long agglutinated words** (the stress test):
   - `karşılaştığımız` (22 bytes, "we encountered")
   - `yapabileceklerimizdendi` (28 bytes, "it was from what we could do")
   - These should compress to 2-5 tokens, not 20+

3. **Special character focus** (I/i/İ/ı):
   - `ışık` (light, with dotless-i)
   - `ilk` (first, with dotted-i)
   - `İzmir` (city, with capital dotted-I)
   - All must be preserved as distinct tokens

**Good result:**
```
karşılaştığımız    22 bytes → 3 tokens   7.3 B/tok
gelebilecekmiydiler 26 bytes → 4 tokens  6.5 B/tok
evlerinizdekilerden 23 bytes → 5 tokens  4.6 B/tok
```

**Poor result:**
```
karşılaştığımız    22 bytes → 15 tokens  1.5 B/tok (barely compressing)
```

---

## Comparison with SentencePiece

The benchmark performs head-to-head comparison on:

1. **Corpus-level metrics**
   - Tokens/word on full test set
   - Bytes/token
   - Throughput

2. **Turkish words** (17 test words)
   - Counts how many words your tokenizer compresses better
   - Highlights morphological correctness

**Example output:**
```
SENTENCEPIECE COMPARISON
─────────────────────────────────────────────
  Metric                     Our TK    SentencePiece
  ────────────────────────── ────────  ─────────────
  Tokens/word                 1.347         1.412  ↓ 4.6%
  Bytes/token                 2.234         2.105  ↑ 6.1%
  Throughput MB/s           125.400        89.200  ↑ 40.6%
```

**Interpretation:**
- ↓ (down) for Tokens/word = **you're winning** (fewer tokens)
- ↑ (up) for Throughput = **you're winning** (faster)
- ↑ (up) for Bytes/token = **you're losing** (tokens are smaller/more fragmented)

---

## Comprehensive Benchmark Workflow

### Phase 1: Sanity Check
```bash
# Quick test on 1000 lines
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/sample.txt \
  --max-lines 1000
```

Check:
- Roundtrip is 100% passing
- Tokens/word is in expected range (1.0-2.0)
- Throughput is > 20 MB/s

### Phase 2: Comparison across vocab sizes
```bash
for vocab in 512 1024 4096 8192 16384 32000; do
  echo "=== Vocab $vocab ==="
  python3 benchmark_enhanced.py \
    --model models/turkish_${vocab}.tkmodel \
    --text data/oscar_tr_clean.txt \
    --max-lines 5000 \
    --output report_${vocab}.txt
done
```

Plot tokens/word vs vocab size. Should see logarithmic curve flattening around 16K.

### Phase 3: Full corpus benchmark
```bash
# On the full training corpus
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/oscar_tr_clean.txt \
  --sp-model models/spm_turkish_32k.model \
  --output final_report.txt \
  -v
```

### Phase 4: Out-of-domain test
```bash
# Test on different domains (news, social media, literature, etc.)
python3 benchmark_enhanced.py \
  --model models/turkish_32k.tkmodel \
  --text data/wikipedia_tr.txt \
  --output report_wikipedia.txt
```

Compare compression across domains. Good tokenizers generalize well.

---

## Expected Results for Turkish (32K vocab)

Based on the original document's evaluation:

| Configuration | Tokens/word | Bytes/token | Notes |
|---|---|---|---|
| 512 vocab (small) | 4.25 | 1.90 | Too small, but works |
| 1,024 vocab | 3.31 | 2.43 | Practical minimum |
| 8K vocab | ~1.8-2.0 | ~2.1-2.3 | Good compression |
| 32K vocab (standard) | **1.2-1.6** | **2.0-2.4** | **Target: this range** |
| Full OSCAR corpus | Likely 1.15-1.35 | 2.2-2.5 | With 30GB training |

**Benchmark your result against these numbers.**

---

## Troubleshooting

### Issue: Roundtrip failures

**Symptom:**
```
Roundtrip: 45/100 passed
FAILURES DETECTED
```

**Causes:**
1. Encoding bug: produces wrong token IDs
2. Decoding bug: can't reconstruct from IDs
3. UTF-8 handling: surrogates, overlong encodings not rejected

**Fix:**
```bash
# Test on simple ASCII first
echo "hello world" | ./tools/encode -m model.tkmodel -c | ./tools/decode -m model.tkmodel

# Check for UTF-8 errors
./tools/encode -m model.tkmodel -s < test.txt | head -20

# Run the test suite
make tests
./build/tests/test_roundtrip -v
```

### Issue: Poor tokens/word (> 2.5)

**Symptom:**
```
Tokens/word: 2.8
(expected: < 1.5)
```

**Causes:**
1. Vocabulary too small: try 32K instead of 8K
2. Training corpus unrepresentative: use OSCAR or Wikipedia
3. Model file corrupted: retrain

**Fix:**
```bash
# Train with more merges
./tools/train -i data/oscar_tr_clean.txt -o model_new.tkmodel -v 32000

# Check what merges were learned
./tools/inspect -m model_new.tkmodel -r 50
```

### Issue: Slow throughput (< 20 MB/s)

**Symptom:**
```
Encoding: 12.3 MB/s
(expected: > 50 MB/s)
```

**Causes:**
1. System under load: run again or close other programs
2. Large model file: 32K vocab = ~1 MB file, should load instantly
3. Test corpus is very large: try `--max-lines 1000`

**Fix:**
```bash
# Profile the encoding
time ./tools/encode -m model.tkmodel < data/sample.txt > /dev/null
# Should be < 0.1s for 10MB of text

# Check model file size
ls -lh models/*.tkmodel
# Should be < 2 MB
```

### Issue: SentencePiece comparison not running

**Symptom:**
```
SentencePiece not available
Install with: pip install sentencepiece
```

**Fix:**
```bash
pip install sentencepiece

# Verify installation
python3 -c "import sentencepiece; print('OK')"

# Train SentencePiece model
spm_train --input=data/oscar_tr_clean.txt \
  --model_prefix=models/spm_turkish_32k \
  --vocab_size=32000 --model_type=bpe
```

---

## Interpreting Turkish Morphology

The most important test is **agglutinated word compression**. Turkish agglutination means:

```
yakalanmış    (caught)     = yak + al + an + mış
gelebilecekmiydiler (could they have come?) = gel + abil + ecek + miydiler
```

A good tokenizer learns these morphemes as tokens, so long words compress to 3-6 tokens.

**Good decomposition (from original document):**
```
karşılaştığımız (22 bytes) → karşılaştığımız (1 token? or 2-3?)
  [If it becomes 1 token in 32K vocab, excellent!]
  [If it becomes 2-3 tokens, very good]
  [If it becomes > 5 tokens, acceptable]
  [If it becomes > 10 tokens, poor]
```

**Check morphological correctness:**
```bash
./tools/encode -m model.tkmodel -s <<< "karşılaştığımız"
# Output should show token boundaries at morpheme junctions
```

---

## Benchmarking Against Known Standards

### English tokenizer baseline (GPT-style)

For comparison, GPT-2 on English achieves:
- **1.00-1.05 tokens/word** (very tight, mostly word boundaries)
- This is normal for English because it's not agglutinative

Turkish is harder, so targeting 1.3-1.6 tokens/word is **excellent**.

### SentencePiece baseline

SentencePiece on Turkish typically achieves:
- **1.4-1.8 tokens/word** with 32K vocab
- **60-80 MB/s** throughput (Python bindings are slow)

Your tokenizer should be **faster** and **comparable or better** on compression.

### Conclusion

A Turkish tokenizer is "good" when it achieves:
- ✓ **Roundtrip 100% pass** (correctness)
- ✓ **Tokens/word < 1.5** (compression)
- ✓ **Throughput > 50 MB/s** (speed)
- ✓ **Agglutinated words 2-5 tokens** (morphology)
- ✓ **Beats or matches SentencePiece** (competitive)

---

## Additional Resources

- Original document: `README.md` in the tokenizer repository
- Test corpus: OSCAR Turkish (https://oscar-corpus.com/)
- Comparison tool: SentencePiece (https://github.com/google/sentencepiece)
- Turkish morphology reference: Turkish Grammar textbooks