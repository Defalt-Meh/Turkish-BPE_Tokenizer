# turkish-tokenizer

A byte-level BPE tokenizer for Turkish, written in C. No dependencies beyond libc and POSIX. Designed for sub-1B parameter language models.

Turkish is an agglutinative language. Words like `karsilastigimiz` are not edge cases -- they are Tuesday. Most tokenizers built for English treat Turkish as an afterthought. This one does not.

## What this is

A from-scratch implementation of byte-pair encoding with:

- Correct handling of the four-way I/i distinction (I, i, dotted-I, dotless-i) that Turkish requires and most software gets wrong
- UTF-8 codec that rejects overlong encodings and surrogates instead of silently producing garbage
- Turkish-aware casing (the `tolower()` in your libc does not know that capital I becomes dotless-i in Turkish)
- GPT-style pre-tokenization that splits on category boundaries before BPE sees anything
- Memory-mapped corpus streaming for training on files larger than RAM
- A binary model format (`.tkmodel`) that loads in microseconds
- Four CLI tools: train, encode, decode, inspect
- 224 automated tests, all passing

## What this is not

This is not a Python package. There are no bindings. There is no pip install. If you want that, write them yourself or use SentencePiece. This is a C library with a clean header-only API that you link into your training pipeline.

## Building

```
make
```

That is it. Requires GCC or Clang with C11 support. Tested on Linux. Should work on any POSIX system with mmap.

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
  bpe.h           BPE internals: pair counting, merge logic, sequence ops
  vocab.h         Token/ID hash table, merge rules, .tkmodel serialization
  unicode.h       UTF-8 codec, Turkish casing, normalization, pre-tokenization
  io.h            mmap file reader, line iterator, chunk iterator

src/              Implementations of the above.

tools/            CLI binaries (each links against libtokenizer.a)
  train           Train a tokenizer on a corpus file
  encode          UTF-8 text to token IDs
  decode          Token IDs to UTF-8 text
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

The `-v` flag sets vocab size. The trair prints progress every 100 merges by default. Use `-p 0` to shut it up.

Normalization flags can be combid: `-n wl` for whitespace collapsing + Turkish lowercase. Default is whitespace only. For most LLM use cases, do not lowercase -- you want the model to learn casing.

### Encoding

```
echo "Istanbul cok guzel bir sehir." | ./tools/encode -m models/turkish_32k.tkmodel -s
```

The `-s` flag shows the byte string for each token. Without it you get bare IDs. Use `-c` for comma-separated output suitable for piping.

### Decoding

```
echo "312 45 1023 88" | ./tools/decode -m models/turkish_32k.tkmodel
```

### Roundtrip

```
echo "Merhaba dunya" | ./tools/encode -m model.tkmodel -c | ./tools/decode -m model.tkmodel
```

If the output does not match the input byte-for-byte, something is broken.

### Inspecting a model

```
./tools/inspect -m models/turkish_32k.tkmodel -t 50 -r 20
./tools/inspect -m models/turkish_32k.tkmodel -s "ler"
```

This shows you the token length histogram, first N merge rules, first N merged tokens, and can search the vocabulary for tokens containing a substring. Use `-a` to dump everything.

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

The encoder returns `(size_t)-1` on error. Check it. The decoder does the same. Allocate `text_len` as your upper bound for the ID buffer -- byte-level BPE can never produce more tokens than input bytes.

---

## Evaluation Report

### Corpus

The evaluation corpus is the first chapter ("Birinci Gece") of Dostoyevsky's Beyaz Geceler (White Nights), in the Nihal Yalaza Taluy translation published by Varlik Yayinlari. This is 19th-century Russian literary prose rendered into Turkish. It contains formal narrative, dialogue, inner monologue, and period-appropriate agglutinated constructions. It is not a synthetic benchmark. It is real text that a real tokenizer will encounter.

Corpus stats: 33 lines, 1,564 words, 12,638 bytes.

Two models were trained: a 512-token vocab (256 merges) and a 1,024-token vocab (768 merges). Training times were 0.11s and 0.30s respectively on commodity hardware.

### Corpus-level numbers

| Metric | vocab=512 | vocab=1024 |
|---|---|---|
| Total tokens | 6,650 | 5,186 |
| Tokens/word | 4.25 | 3.31 |
| Bytes/token | 1.90 | 2.43 |

Doubling the vocabulary from 512 to 1024 cut the token count by 22% and improved bytes/token from 1.90 to 2.43. With a 32K vocab trained on 30GB of OSCAR, expect 1.2-1.8 tokens/word. The curve is logarithmic; most of the gain happens in the first few thousand merges.

### Merge ordering

The first 20 merges learned from Dostoyevsky's Turkish prose, in order:

| # | Merge | Meaning |
|---|---|---|
| 0 | C4+B1 -> dotless-i | Dotless-i. Most frequent multi-byte Turkish character. |
| 1 | C3+BC -> u-umlaut | u-umlaut. |
| 2 | C5+9F -> s-cedilla | s-cedilla. |
| 3 | a+r -> ar | Suffix fragment (-lar, -ar verb conjugation). |
| 4 | a+n -> an | Suffix fragment (-an participle, -dan ablative). |
| 5 | e+n -> en | Superlative, ablative (-den). |
| 6 | e+r -> er | Suffix fragment (-ler, aorist -er). |
| 7 | C3+A7 -> c-cedilla | c-cedilla. |
| 8 | b+i -> bi | First half of "bir" (one/a). |
| 9 | C4+9F -> g-breve | Soft-g. |
| 10 | i+n -> in | Genitive suffix. |
| 11 | dotless-i + n -> dotless-i-n | Genitive suffix (back vowel harmony). |
| 12 | i+m -> im | Possessive first person. |
| 13 | dotless-i + m -> dotless-i-m | Possessive first person (back vowel). |
| 14 | a+k -> ak | Part of future tense -acak. |
| 15 | d+e -> de | Locative suffix. |
| 16 | C3+B6 -> o-umlaut | o-umlaut. |
| 17 | a+l -> al | Verb fragment. |
| 18 | u-umlaut + n -> u-umlaut-n | Suffix extension. |
| 19 | d+u -> du | Past tense suffix (-du/-di). |

Multi-byte Turkish characters reassemble first: dotless-i, u-umlaut, s-cedilla, c-cedilla, g-breve, o-umlaut. Then the morphological suffixes appear: -ar, -an, -en, -er, -in, -im, -de, -du. By merge 19, the algorithm has recovered a substantial fragment of Turkish grammar from raw byte frequencies on a chapter of Dostoyevsky. Nobody told it anything about the language.

Note how vowel harmony manifests: merges 10-11 learn the genitive `-in` and `-in` (back vowel) as separate tokens. Merges 12-13 do the same for the possessive `-im` and its back-vowel variant. The algorithm discovers vowel harmony because both variants are frequent.

### Whole-word tokens learned (vocab=1024)

These words from Beyaz Geceler encode as single tokens:

| Word | Token ID | Notes |
|---|---|---|
| bir | 277 | "one/a" -- most common Turkish word |
| gibi | 313 | "like/as" -- postposition |
| Petersburg | 461 | Proper noun. The setting of the story. |
| hemen | 517 | "immediately" |
| sokak | 584 | "street" |
| gece | 609 | "night" -- the title word |

"Petersburg" becoming a single token from a 12KB literary text is worth noting. The word appears often enough in the first chapter that BPE consumed it whole, all 10 bytes. "Nastenka" (the female lead) did not make it as a single token at vocab=1024(Considering the story, she had it coming) -- it decomposes into 5 tokens. With a larger vocabulary trained on more data it would.

### Encoding examples

From the text itself:

```
"Nefis bir geceydi." (18 bytes -> 9 tokens, 2.00 B/tok)
  Ne | f | is | _ | bir | _ | gecey | di | .

"Kalbim yakalanmis bir kus yuregi gibi carpiyordu." (56B -> 21 tok, 2.66 B/tok)
  Kal | bim | _ | yak | al | anmis | _ | bir | _ | k | us | _ |
  yur | e | gi | _ | gibi | _ | carp | iyordu | .

"Petersburg'ta tam sekiz yil oturdugum halde, hemen hemen hic dost edinememistim."
  (84B -> 33 tok, 2.54 B/tok)
  Petersburg | ' | ta | _ | tam | _ | s | ek | iz | _ | yil | _ |
  otu | rdu | gum | _ | halde | , | _ | hemen | _ | hemen | _ | hic | _ |
  dost | _ | edin | em | emi | st | im | .

"Oyle mutluydum ki!" (19B -> 9 tok, 2.11 B/tok)
  Oyle | _ | mut | l | uy | dum | _ | ki | !
```

The phrase `carpiyordu` ("was beating," as in a heart) decomposes into `carp` + `iyordu`. The suffix `-iyordu` is the past continuous tense marker. BPE learned it as a unit from a 12KB text. That is morphologically correct decomposition.

The word `yakalanmis` ("caught/seized") splits as `yak` + `al` + `anmis`. The suffix `-mis` is the reported past tense. Again, correct.

`edinememistim` ("I had not been able to acquire") splits as `edin` + `em` + `emi` + `st` + `im`. The root `edin-` (acquire), inability `-eme-`, reported past `-mis-`, past `-ti-`, first person `-m`. Not perfect segmentation, but reasonable for a 768-merge vocabulary trained on 12KB.

### Agglutination stress test (vocab=1024)

Words drawn directly from the chapter text:

| Word | Bytes | Tokens | B/Token |
|---|---|---|---|
| geceydi | 7 | 2 | 3.5 |
| devirlerimizdeki | 16 | 5 | 3.2 |
| edinememistim | 14 | 5 | 2.8 |
| selamlayacakmis | 17 | 2 | 8.5 |
| karsilastigimiz | 22 | 2 | 11.0 |
| dondurmislerdi | 18 | 4 | 4.5 |
| birakmayacagim | 17 | 3 | 5.6 |
| animsiyorum | 13 | 3 | 4.3 |
| baslayacaksiniz | 18 | 3 | 6.0 |
| baristirdiniz | 18 | 5 | 3.6 |
| kovmadiginiza | 17 | 5 | 3.4 |

`karsilastigimiz` ("the time we encountered each other") -- a 22-byte, 15-character word -- encodes as 2 tokens. `selamlayacakmis` ("apparently was going to greet") also compresses to 2 tokens. These are words that an English-centric tokenizer would shred into a dozen byte-level fragments.

The reason certain long words compress to 2 tokens is simple: they appeared in the training text often enough (or their subwords did) that BPE built them up through successive merges. This is exactly how BPE is supposed to work on agglutinative languages -- common inflected forms become single tokens, rare ones decompose into reusable morphological pieces.

### Dotted-I / Dotless-I distinctness

| Character | Unicode | Token ID |
|---|---|---|
| Dotted capital I | U+0130 | 422 |
| ASCII capital I | U+0049 | 73 |
| ASCII lowercase i | U+0069 | 105 |
| Dotless lowercase i | U+0131 | 256 |

Four distinct token IDs. This is non-negotiable for Turkish. If your tokenizer conflates any of these, downstream NLP performance will degrade silently and you will spend weeks debugging the wrong layer.

### Roundtrip correctness

All test sentences from Beyaz Geceler pass byte-exact encode-decode roundtrip:

```
PASS: "Nefis bir geceydi."
PASS: "Kalbim yakalanmis bir kus yuregi gibi carpiyordu."
PASS: "Oyle mutluydum ki!"
PASS: "Petersburg'ta tam sekiz yil oturdugum halde, hemen hemen hic dost edinememistim."
```

224/224 automated tests pass across all four test suites (unicode, bpe, roundtrip, turkish).

### Scaling projection

| Configuration | Tokens/word | Source |
|---|---|---|
| 512 vocab, 12KB Beyaz Geceler ch.1 | 4.25 | measured |
| 1024 vocab, 12KB Beyaz Geceler ch.1 | 3.31 | measured |
| 32K vocab, 1-30GB OSCAR | 1.3 - 1.6 | projected |

The architecture does not change. The data and the vocab size do.

---

## Design decisions

**Why byte-level BPE and not Unigram/SentencePiece.** BPE is simpler to implement correctly in C, the merge-based training loop is straightforward to debug, and the performance difference on Turkish is marginal. Byte-level means no UNK tokens ever. Any byte sequence encodes and decodes.

**Why a linked list for sequences.** Applying a merge across a million-token sequence requires replacing adjacent pairs in-place. An array would require shifting elements on every merge. A doubly-linked list does it in O(1) per merge site. The pool allocator avoids per-node malloc overhead.

**Why FNV-1a and not xxhash.** FNV-1a is 10 lines of code with no dependencies and is fast enough for our key sizes (2-128 bytes). xxhash would be faster on large keys but we do not have large keys. Keeping the dependency count at zero matters.

**Why mmap and not read().** Training corpora are tens of gigabytes. mmap lets the OS handle paging. MADV_SEQUENTIAL tells the kernel to read ahead. No manual buffer management needed.

**Why a custom binary format and not JSON/protobuf.** A trained tokenizer is a flat list of byte sequences and a flat list of integer triples. Serializing that as JSON is a waste of everyone's time. The `.tkmodel` format is 16 bytes of header followed by packed binary data. It loads with one fread and zero parsing.

## Known limitations

- The naive O(n*m) encoding (n = input length, m = number of merges) is fine for inference on short texts but slow for batch-encoding large corpora. A priority queue approach would be O(n log n). Not implemented yet.
- NFC normalization in `tk_normalize` handles combining marks in the 0300-036F range only. Full Unicode NFC would require the full decomposition tables. For Turkish BPE training this is sufficient.
- No special tokens (BOS, EOS, PAD). Add them to the vocabulary after training by reserving IDs, or handle them in your training loop. The tokenizer does not have opinions about your model architecture.
- The file-based trainer caps at 2GB of input. For larger corpora, sample a representative subset. BPE vocabulary quality plateaus well before you exhaust a 30GB corpus.

## License

Do what you want with it.
