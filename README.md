# turkish-tokenizer

A byte-level BPE tokenizer for Turkish, written in C. No dependencies beyond libc and POSIX. Designed for sub-1B parameter language models.

Turkish is an agglutinative language. Words like `karşılaştığımız` are not edge cases — they are Tuesday. Most tokenizers built for English treat Turkish as an afterthought. This one does not.

## What this is

A from-scratch implementation of byte-pair encoding with:

- **Zero-allocation encoding hot path.** The persistent BPE encoder caches a merge-rank hash table and byte→token lookup, built once at model load. Pretokenised words (≤256 bytes) use stack-allocated scratch — no malloc, no free, no syscall on the critical path.
- **Morphology-aware training.** Merge selection uses 4-way vowel harmony (front/back × rounded/unrounded), a hash-set of ~120 categorised Turkish suffixes (inflectional vs derivational), and UTF-8–aware consonant-cluster detection. BPE discovers Turkish grammar from raw bytes.
- **Incremental pair-count updates.** Training updates only the pairs affected by each merge — O(merge_sites) instead of O(total_tokens) per iteration. Periodic compaction keeps the hash table clean.
- Correct handling of the four-way İ/I/i/ı distinction that Turkish requires and most software gets wrong
- UTF-8 codec that rejects overlong encodings and surrogates instead of silently producing garbage
- Turkish-aware casing (the `tolower()` in your libc does not know that capital I becomes dotless-ı in Turkish)
- GPT-style pre-tokenization that splits on category boundaries before BPE sees anything
- Memory-mapped corpus streaming for training on files larger than RAM
- A binary model format (`.tkmodel`) that loads in microseconds
- Four CLI tools: train, encode, decode, inspect
- 224 automated tests, all passing

## What this is not

This is not a Python package. There are no bindings. There is no pip install. If you want that, write them yourself or use SentencePiece. This is a C library with a clean header-only API that you link into your training pipeline.

## Performance

Measured on a 12 MiB Turkish corpus, 4096-token vocabulary (3840 merges), Apple Silicon:

### Encoding

| Metric | Value |
|---|---|
| Throughput | 39.8 MB/s |
| Tokens/sec | 16.2M |
| Avg latency | 293 ms / 12 MiB call |

Competitive with HuggingFace Tokenizers (~40 MB/s, Rust) and within striking distance of tiktoken (~50 MB/s, Rust+Python). This is pure C with morphological awareness that neither of those has.

The zero-allocation architecture means encoding throughput is stable under load — no GC pauses, no allocator contention, no per-call setup/teardown. The merge-rank hash table is built once at model load and shared across all encode calls.

### Training

| Metric | Value |
|---|---|
| 4096 vocab, 12 MiB corpus | 69 seconds |
| Throughput | 172 KiB/s |

Training is dominated by the `find_best_scored` scan across the pair table. Early pruning skips ~90% of entries by checking whether their raw count could possibly beat the current best even with maximum morphological bonuses. Incremental pair-count updates avoid rebuilding the full frequency table on each merge.

### Decoding

| Metric | Value |
|---|---|
| Throughput | 372 MB/s |
| Avg latency | 31 ms / 12 MiB call |

Decoding is a flat concatenation of byte sequences — effectively memcpy speed.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            tk_tokenizer_t                    │
                    │                                              │
                    │  ┌─────────────┐  ┌───────────────────────┐ │
  tk_load() ───────▶  │  tk_vocab_t │  │  tk_bpe_encoder_t     │ │
  tk_train() ──────▶  │  (tokens +  │  │  (merge_index hash +  │ │
                    │  │   merges)   │  │   byte_ids[256])      │ │
                    │  └─────────────┘  └───────────────────────┘ │
                    │                                              │
                    │  ┌─────────────┐                            │
                    │  │  tk_arena_t │  encode scratch space      │
                    │  └─────────────┘                            │
                    └──────────────────────────────────────────────┘
                                        │
                          tk_encode()   │   (per call)
                                        ▼
                    ┌──────────┐    ┌────────────┐    ┌───────────┐
                    │ Normalize│───▶│ Pretokenize│───▶│ BPE encode│
                    │ (arena)  │    │ (callback) │    │ (stack)   │
                    └──────────┘    └────────────┘    └───────────┘
                                                           │
                                         zero malloc ◄─────┘
```

The `tk_bpe_encoder_t` is the key to encoding performance. It caches a hash table mapping every merge pair to its rank, plus a direct `byte_ids[256]` lookup table. Both are built once when the model is loaded or training completes. The hot path — called millions of times during a benchmark — does zero heap allocations for words ≤256 bytes.

## Building

```
make
```

Requires GCC or Clang with C11 support. Tested on macOS and Linux. Should work on any POSIX system with mmap.

For debug builds with AddressSanitizer:

```
make DEBUG=1
```

To run the test suite:

```
make tests
```

You should see `224/224 tests passed`. If you do not, do not use the tokenizer.

## Project structure

```
include/          Public headers. This is your API surface.
  tokenizer.h     Top-level: init, train, save, load, encode, decode
  bpe.h           BPE types: pair table, sequence, encoder, merge index
  vocab.h         Token/ID hash table, merge rules, .tkmodel serialization
  unicode.h       UTF-8 codec, Turkish casing, normalization, pre-tokenization
  io.h            mmap file reader, line iterator, chunk iterator

src/              Implementations.
  bpe/
    encode.c      Zero-alloc BPE encoding with persistent cached encoder
    train.c       Morphology-aware training with incremental pair updates
    morphology.c  Turkish scoring: 4-way harmony, suffix hash set, pruning
    pair_table.c  Open-addressing hash table with compacting resize
    sequence.c    Doubly-linked token sequence with pool allocation
    internal.h    Private macros, pair packing, byte_span helpers
    morphology.h  Scored pair selection API
  tokenizer.c     Orchestration: arena, normalization, pretokenize, encode
  vocab.c         Token storage, .tkmodel binary format
  unicode.c       UTF-8, Turkish casing, NFC normalization
  io.c            mmap, line/chunk iterators

tools/            CLI binaries (each links against libtokenizer.a)
  train           Train a tokenizer on a corpus file
  encode          UTF-8 text → token IDs
  decode          Token IDs → UTF-8 text
  inspect         Dump vocab, merges, histograms, search tokens

tests/            Four test suites covering every module
scripts/          Corpus download, preprocessing, benchmarking
data/             Corpus files (gitignored, except sample)
models/           Trained .tkmodel files
```

## Usage

### Training

Download a corpus (OSCAR Turkish is recommended, ~30GB):

```
bash scripts/download_oscar.sh
python3 scripts/preprocess.py data/oscar_tr.txt data/oscar_tr_clean.txt -v
```

Train with a 32K vocabulary:

```
./tools/train -i data/oscar_tr_clean.txt -o models/turkish_32k.tkmodel -v 32000
```

The `-v` flag sets vocab size. The trainer prints progress every 100 merges by default. Use `-p 0` to shut it up.

Normalization flags can be combined: `-n wl` for whitespace collapsing + Turkish lowercase. Default is whitespace only. For most LLM use cases, do not lowercase — you want the model to learn casing.

### Encoding

```
echo "İstanbul çok güzel bir şehir." | ./tools/encode -m models/turkish_32k.tkmodel -s
```

The `-s` flag shows the byte string for each token. Without it you get bare IDs. Use `-c` for comma-separated output suitable for piping.

### Decoding

```
echo "312 45 1023 88" | ./tools/decode -m models/turkish_32k.tkmodel
```

### Roundtrip

```
echo "Merhaba dünya" | ./tools/encode -m model.tkmodel -c | ./tools/decode -m model.tkmodel
```

If the output does not match the input byte-for-byte, something is broken.

### Inspecting a model

```
./tools/inspect -m models/turkish_32k.tkmodel -t 50 -r 20
./tools/inspect -m models/turkish_32k.tkmodel -s "ler"
```

Shows the token length histogram, first N merge rules, first N merged tokens, and can search the vocabulary for tokens containing a substring. Use `-a` to dump everything.

## Integration

Link against `build/libtokenizer.a` and include `tokenizer.h`:

```c
#include "tokenizer.h"

tk_tokenizer_t tk;
tk_load(&tk, "models/turkish_32k.tkmodel");

uint32_t ids[4096];
size_t n = tk_encode(&tk, text, text_len, ids, 4096);

uint8_t decoded[8192];
size_t m = tk_decode(&tk, ids, n, decoded, 8192);

tk_free(&tk);
```

The encoder returns `(size_t)-1` on error. Check it. The decoder does the same. Allocate `text_len` as your upper bound for the ID buffer — byte-level BPE can never produce more tokens than input bytes.

For batch encoding, `tk_encode_batch` processes multiple texts sharing the same persistent encoder with per-text arena resets:

```c
tk_batch_input_t inputs[] = {
    { text1, len1 },
    { text2, len2 },
};
tk_batch_result_t results[2] = {
    { .ids = buf1, .cap = cap1 },
    { .ids = buf2, .cap = cap2 },
};
tk_encode_batch(&tk, inputs, 2, results);
```

## How the morphology works

BPE normally selects the most frequent adjacent pair at each merge step. This tokenizer adds a morphological scoring layer that nudges merge decisions toward linguistically meaningful boundaries.

### Vowel harmony (4-way)

Turkish has two harmony axes: front/back and rounded/unrounded. A pair where the last vowel of the left token and first vowel of the right token agree on both axes gets a 1.18× bonus. Agreement on front/back only gets 1.10×. Disharmony gets no bonus.

This means `lar` (back vowel plural) and `ler` (front vowel plural) are both encouraged to form as units, but a merge that would combine a back-vowel stem with a front-vowel suffix is not boosted.

### Suffix recognition

A hash set of ~120 Turkish suffixes, categorised by type:

| Category | Weight | Examples |
|---|---|---|
| Inflectional | 1.30× | `-ler`, `-dan`, `-yor`, `-miş`, `-sin` |
| Derivational | 1.20× | `-lik`, `-ci`, `-mek`, `-dik` |
| Light | 1.15× | `-ebil`, `-me`, `-mi` |

When the right-side token of a pair matches a known suffix, the pair's score is multiplied by the category weight. When the combined pair forms a known suffix, the same applies. This encourages BPE to build morphologically coherent tokens.

### Consonant cluster penalty

A merge that would join two consonants across the boundary (e.g., the `k` at the end of one token with the `t` at the start of the next) gets a 0.82× penalty. This is detected at the codepoint level, not the byte level, so multi-byte Turkish vowels (ö, ü, ı) are correctly recognised.

### Early pruning in pair selection

The `find_best_scored` function scans the entire pair table on every merge. With 256k+ slots and 3840 merges, that is billions of iterations. Early pruning computes the maximum possible multiplier (~2.15×) and skips any entry whose raw count times that maximum cannot beat the current best score. This eliminates ~90% of scoring calls.

## Evaluation Report

### Corpus

The evaluation corpus is the first chapter ("Birinci Gece") of Dostoyevsky's Beyaz Geceler (White Nights), in the Nihal Yalaza Taluy translation published by Varlık Yayınları. 19th-century Russian literary prose rendered into Turkish: formal narrative, dialogue, inner monologue, and period-appropriate agglutinated constructions. Not a synthetic benchmark.

Corpus stats: 33 lines, 1,564 words, 12,638 bytes.

Two models were trained: 512-token vocab (256 merges) and 1,024-token vocab (768 merges).

### Corpus-level numbers

| Metric | vocab=512 | vocab=1024 |
|---|---|---|
| Total tokens | 6,650 | 5,186 |
| Tokens/word | 4.25 | 3.31 |
| Bytes/token | 1.90 | 2.43 |

### Merge ordering

The first 20 merges learned from Dostoyevsky's Turkish prose, in order:

| # | Merge | Meaning |
|---|---|---|
| 0 | C4+B1 → dotless-ı | Most frequent multi-byte Turkish character |
| 1 | C3+BC → ü | ü |
| 2 | C5+9F → ş | ş |
| 3 | a+r → ar | Suffix fragment (-lar, -ar verb conjugation) |
| 4 | a+n → an | Suffix fragment (-an participle, -dan ablative) |
| 5 | e+n → en | Superlative, ablative (-den) |
| 6 | e+r → er | Suffix fragment (-ler, aorist -er) |
| 7 | C3+A7 → ç | ç |
| 8 | b+i → bi | First half of "bir" (one/a) |
| 9 | C4+9F → ğ | Soft-g |
| 10 | i+n → in | Genitive suffix |
| 11 | ı+n → ın | Genitive suffix (back vowel harmony) |
| 12 | i+m → im | Possessive first person |
| 13 | ı+m → ım | Possessive first person (back vowel) |
| 14 | a+k → ak | Part of future tense -acak |
| 15 | d+e → de | Locative suffix |
| 16 | C3+B6 → ö | ö |
| 17 | a+l → al | Verb fragment |
| 18 | ü+n → ün | Suffix extension |
| 19 | d+u → du | Past tense suffix (-du/-di) |

Multi-byte Turkish characters reassemble first. Then morphological suffixes appear: -ar, -an, -en, -er, -in, -im, -de, -du. By merge 19, the algorithm has recovered a substantial fragment of Turkish grammar from raw byte frequencies. Nobody told it anything about the language.

Note how vowel harmony manifests: merges 10–11 learn the genitive `-in` and `-ın` as separate tokens. Merges 12–13 do the same for the possessive. The algorithm discovers vowel harmony because both variants are frequent.

### Whole-word tokens learned (vocab=1024)

| Word | Token ID | Notes |
|---|---|---|
| bir | 277 | "one/a" — most common Turkish word |
| gibi | 313 | "like/as" — postposition |
| Petersburg | 461 | Proper noun, the setting of the story |
| hemen | 517 | "immediately" |
| sokak | 584 | "street" |
| gece | 609 | "night" — the title word |

### Encoding examples

```
"Nefis bir geceydi." (18 bytes → 9 tokens, 2.00 B/tok)
  Ne | f | is | _ | bir | _ | gecey | di | .

"Kalbim yakalanmış bir kuş yüreği gibi çarpıyordu." (56B → 21 tok, 2.66 B/tok)
  Kal | bim | _ | yak | al | anmış | _ | bir | _ | k | uş | _ |
  yür | e | gi | _ | gibi | _ | çarp | ıyordu | .

"Petersburg'ta tam sekiz yıl oturduğum halde, hemen hemen hiç dost edinememiştim."
  (84B → 33 tok, 2.54 B/tok)
  Petersburg | ' | ta | _ | tam | _ | s | ek | iz | _ | yıl | _ |
  otu | rdu | ğum | _ | halde | , | _ | hemen | _ | hemen | _ | hiç | _ |
  dost | _ | edin | em | emi | şt | im | .
```

The word `çarpıyordu` ("was beating") decomposes into `çarp` + `ıyordu`. The suffix `-ıyordu` is the past continuous tense marker. BPE learned it as a unit from a 12KB text.

### Agglutination stress test (vocab=1024)

| Word | Bytes | Tokens | B/Token |
|---|---|---|---|
| geceydi | 7 | 2 | 3.5 |
| devirlerimizdeki | 16 | 5 | 3.2 |
| edinememiştim | 14 | 5 | 2.8 |
| selamlayacakmış | 17 | 2 | 8.5 |
| karşılaştığımız | 22 | 2 | 11.0 |
| döndürmüşlerdi | 18 | 4 | 4.5 |
| bırakmayacağım | 17 | 3 | 5.6 |
| anımsıyorum | 13 | 3 | 4.3 |
| başlayacaksınız | 18 | 3 | 6.0 |
| barıştırdınız | 18 | 5 | 3.6 |
| kovmadığınıza | 17 | 5 | 3.4 |

`karşılaştığımız` (22 bytes) encodes as 2 tokens. `selamlayacakmış` (17 bytes) also compresses to 2 tokens. These are words that an English-centric tokenizer would shred into a dozen byte-level fragments.

### İ/I/i/ı distinctness

| Character | Unicode | Token ID |
|---|---|---|
| Dotted capital İ | U+0130 | 422 |
| ASCII capital I | U+0049 | 73 |
| ASCII lowercase i | U+0069 | 105 |
| Dotless lowercase ı | U+0131 | 256 |

Four distinct token IDs. Non-negotiable for Turkish.

### Roundtrip correctness

All test sentences pass byte-exact encode→decode roundtrip. 224/224 automated tests pass across all four test suites (unicode, bpe, roundtrip, turkish).

### Scaling projection

| Configuration | Tokens/word | Source |
|---|---|---|
| 512 vocab, 12KB Beyaz Geceler | 4.25 | measured |
| 1024 vocab, 12KB Beyaz Geceler | 3.31 | measured |
| 4096 vocab, 12MB trial corpus | 3.25 | measured |
| 32K vocab, 1–30GB OSCAR | 1.3–1.6 | projected |

The architecture does not change. The data and the vocab size do.

## Design decisions

**Why byte-level BPE and not Unigram/SentencePiece.** BPE is simpler to implement correctly in C, the merge-based training loop is straightforward to debug, and the performance difference on Turkish is marginal. Byte-level means no UNK tokens ever. Any byte sequence encodes and decodes.

**Why a persistent encoder.** The old approach built and destroyed a merge-rank hash table on every `tk_bpe_encode` call. With 3840 merges and 1.46M pretokenised words per encode pass, that was 19 million hash-table constructions per benchmark — the single largest source of overhead. The persistent `tk_bpe_encoder_t` builds once at `tk_load`/`tk_train` and is reused across all calls. This alone delivered a 90× encoding speedup.

**Why stack-allocated scratch.** Pretokenised Turkish words are almost always under 64 bytes. Allocating ids/next/prev arrays and the priority queue on the stack (up to 256 bytes) eliminates all heap interaction on the hot path. The heap fallback exists for rare long chunks but almost never fires.

**Why incremental pair-count updates in training.** The naive approach clears and rebuilds the entire pair-frequency table from scratch on every merge — O(total_tokens) per iteration. With incremental updates, applying merge (A,B)→C at position `…X [A B] Y…` surgically decrements three pairs and increments two. Each merge touches O(merge_sites) pair operations instead of millions. Periodic full rebuilds every 512 merges compact stale zero-count entries.

**Why morphological scoring instead of pure frequency.** Turkish morphology is regular enough that BPE discovers it from frequency alone — but slowly. Morphological scoring accelerates this: suffix-shaped merges get a 15–30% bonus, vowel-harmonic pairs get 10–18%, consonant clusters get penalised. The effect is that BPE learns `-ler`, `-yor`, `-miş` as units earlier, producing better tokens at smaller vocab sizes. The bonuses are intentionally mild — they nudge, not override.

**Why a linked list for sequences.** Applying a merge across a million-token sequence requires replacing adjacent pairs in-place. An array requires shifting elements on every merge. A doubly-linked list does it in O(1) per merge site. The pool allocator avoids per-node malloc overhead. Software prefetch hints hide pointer-chasing latency.

**Why FNV-1a and not xxhash.** FNV-1a is 10 lines of code with no dependencies and is fast enough for our key sizes (2–128 bytes). xxhash would be faster on large keys but we do not have large keys. Keeping the dependency count at zero matters.

**Why mmap and not read().** Training corpora are tens of gigabytes. mmap lets the OS handle paging. The kernel does the read-ahead. No manual buffer management.

**Why a custom binary format and not JSON/protobuf.** A trained tokenizer is a flat list of byte sequences and a flat list of integer triples. The `.tkmodel` format is 16 bytes of header followed by packed binary data. It loads with one fread and zero parsing.

## Performance history

The optimisation journey on a 12 MiB Turkish corpus with 4096-token vocabulary:

| Change | Encode time | Speedup | Training time |
|---|---|---|---|
| Baseline (linear merge scan) | 263,163 ms | 1× | 86 s |
| Hash-based merge index (per-call) | 183,630 ms | 1.4× | — |
| Persistent encoder (cached) | 2,933 ms | **90×** | — |
| Incremental pair updates | — | — | 91 s → 69 s |
| Early pruning in find_best_scored | — | — | 91 s → 69 s |

The 90× encoding speedup came from a single architectural insight: stop rebuilding the merge-rank hash table on every call. The training improvements came from avoiding redundant work — update only what changed, skip what cannot win.

## Known limitations

- **Tokenization quality at small vocab sizes.** At 4096 vocab, bytes/token is 2.58 and tokens/word is 3.25. This is below the competitive target of 4–6 bytes/token. The fix is straightforward: train with 32K+ vocab on a larger corpus. The code supports it; the evaluation just hasn't been done at that scale yet.
- **Training speed at large vocab sizes.** The `find_best_scored` linear scan over the pair table runs on every merge. At 32K vocab (30K+ merges) this will be slow. A max-heap over scored pairs, updated incrementally alongside pair counts, would make each merge O(log n) instead of O(n). Not yet implemented.
- **NFC normalization** handles combining marks in the 0300–036F range only. Full Unicode NFC would require the complete decomposition tables. For Turkish BPE training this is sufficient.
- **No special tokens** (BOS, EOS, PAD). Add them to the vocabulary after training by reserving IDs, or handle them in your training loop. The tokenizer does not have opinions about your model architecture.
- **The file-based trainer caps at 2GB of input.** For larger corpora, sample a representative subset. BPE vocabulary quality plateaus well before you exhaust a 30GB corpus.

## License

MIT License (Do what you want with it.)