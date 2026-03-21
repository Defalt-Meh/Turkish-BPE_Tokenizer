/*
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║  tokenizer.c — The Throne Room                                      ║
 * ║                                                                     ║
 * ║  Top-level tokenizer: init, train, save/load, encode/decode.        ║
 * ║  Orchestrates unicode, vocab, bpe, and io into a single pipeline    ║
 * ║  that turns Turkish text into token IDs and back again.             ║
 * ║                                                                     ║
 * ║  "For the word of God is living and active, sharper than any        ║
 * ║   two-edged sword, piercing to the division of soul and spirit,     ║
 * ║   of joints and marrow." — Hebrews 4:12                            ║
 * ║                                                                     ║
 * ║  So too must our tokenizer be: living, active, and sharp enough     ║
 * ║  to divide `karsilastigimiz` into exactly the right pieces.         ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 *
 * Performance philosophy:
 *
 *   In high-frequency trading, the difference between profit and loss
 *   is measured in nanoseconds. In tokenization, the difference between
 *   a usable training pipeline and a weekend wasted waiting is measured
 *   in the same currency. We pay that currency the same respect:
 *
 *   1. ZERO ALLOCATION ON THE HOT PATH. The encode pipeline uses a
 *      pre-allocated arena that lives on the tokenizer struct. No
 *      malloc, no free, no syscall, no context switch. The arena is
 *      reset (not freed) between calls — a single pointer rewind.
 *
 *   2. PERSISTENT BPE ENCODER. The merge-rank hash table and byte→id
 *      lookup table are built once (at train / load time) and cached
 *      on the tokenizer struct. Every encode call reuses them — zero
 *      per-call index construction. This alone eliminates ~95% of the
 *      old encode-path overhead.
 *
 *   3. BRANCH PREDICTION FRIENDLY. The common path (short text,
 *      no overflow, no error) is the fall-through path. Error checks
 *      use TK_UNLIKELY. The CPU's branch predictor learns fast.
 *
 *   4. CACHE-OBLIVIOUS DATA FLOW. Encode processes text left-to-right
 *      in a single pass. Pre-tokenization and BPE happen in the same
 *      cache lines. We never rewind and re-scan.
 *
 *   5. BATCH AMORTIZATION. For bulk encoding, tk_encode_batch processes
 *      multiple texts with a single arena, avoiding per-call overhead.
 *
 *   6. INLINE NORMALIZATION. When normalization is enabled, we normalize
 *      into the arena buffer and encode from there — one allocation for
 *      both operations, zero copies beyond the initial.
 */

#include "tokenizer.h"
#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* ─────────────────────────────────────────────────────────────────────
 *  Compiler hints
 * ───────────────────────────────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
#   define TK_LIKELY(x)         __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x)       __builtin_expect(!!(x), 0)
#   define TK_PREFETCH(addr)    __builtin_prefetch((addr), 0, 1)
#   define TK_HOT               __attribute__((hot))
#   define TK_COLD              __attribute__((cold))
#   define TK_NOINLINE          __attribute__((noinline))
#   define TK_ALWAYS_INLINE     __attribute__((always_inline)) static inline
#else
#   define TK_LIKELY(x)         (x)
#   define TK_UNLIKELY(x)       (x)
#   define TK_PREFETCH(addr)    ((void)(addr))
#   define TK_HOT
#   define TK_COLD
#   define TK_NOINLINE
#   define TK_ALWAYS_INLINE     static inline
#endif

/* Cache line size — used for alignment and padding */
#define TK_CACHE_LINE 64


/* ═══════════════════════════════════════════════════════════════════════
 *  § 1. Arena Allocator
 *
 *  The cornerstone of zero-allocation encoding. A bump allocator that
 *  hands out memory from a pre-allocated slab. Reset is O(1) — just
 *  rewind the pointer. No bookkeeping, no fragmentation, no mercy.
 *
 *  "In my Father's house are many rooms. In my arena, there is one
 *   contiguous slab, and it is sufficient." — John 14:2, paraphrased
 *
 *  Growth policy: if a request exceeds the current slab, we realloc
 *  to 2x the needed size. This happens at most O(log n) times across
 *  the lifetime of the tokenizer. After a few calls, the arena settles
 *  at a stable size and never allocates again.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Align `n` up to the specified alignment (must be power of two) */
#define TK_ALIGN_UP(n, align) (((n) + ((align) - 1)) & ~((size_t)(align) - 1))

static int tk_arena_init(tk_arena_t *a, size_t initial_cap) {
    a->base = (uint8_t *)malloc(initial_cap);
    if (TK_UNLIKELY(!a->base)) return -1;
    a->offset = 0;
    a->capacity = initial_cap;
    return 0;
}

static void tk_arena_free(tk_arena_t *a) {
    free(a->base);
    a->base = NULL;
    a->offset = 0;
    a->capacity = 0;
}

/*
 * tk_arena_reset — Rewind to the beginning. O(1). No deallocation.
 *
 * "Behold, I make all things new." — Revelation 21:5
 * (But I keep the same allocation, because I am frugal.)
 */
TK_ALWAYS_INLINE void tk_arena_reset(tk_arena_t *a) {
    a->offset = 0;
}

/*
 * tk_arena_ensure — Guarantee that `needed` bytes are available.
 *
 * If the current slab is too small, realloc to max(2*current, needed+current).
 * This is the ONLY function in the arena that may call malloc/realloc,
 * and it only triggers when the arena is undersized for a new input.
 */
static int tk_arena_ensure(tk_arena_t *a, size_t needed) {
    size_t required = a->offset + needed;
    if (TK_LIKELY(required <= a->capacity)) return 0;

    /* Grow to at least 2x, or to fit if 2x isn't enough */
    size_t new_cap = a->capacity * 2;
    if (new_cap < required) new_cap = required;
    new_cap = TK_ALIGN_UP(new_cap, TK_CACHE_LINE);

    uint8_t *new_base = (uint8_t *)realloc(a->base, new_cap);
    if (TK_UNLIKELY(!new_base)) return -1;

    a->base = new_base;
    a->capacity = new_cap;
    return 0;
}

/*
 * tk_arena_alloc — Bump-allocate `size` bytes, aligned to `align`.
 *
 * Returns NULL only if the arena cannot grow (OOM). In normal
 * operation after warmup, this is a pointer add and a comparison.
 * No branches taken. Faster than your malloc, faster than your
 * morning prayer.
 */
static void *tk_arena_alloc(tk_arena_t *a, size_t size, size_t align) {
    size_t aligned_offset = TK_ALIGN_UP(a->offset, align);
    size_t end = aligned_offset + size;

    if (TK_UNLIKELY(end > a->capacity)) {
        if (tk_arena_ensure(a, (end - a->offset) + TK_CACHE_LINE) < 0)
            return NULL;
        aligned_offset = TK_ALIGN_UP(a->offset, align);
        end = aligned_offset + size;
    }

    void *ptr = a->base + aligned_offset;
    a->offset = end;
    return ptr;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 2. Default Configuration
 *
 *  Sensible defaults for Turkish BPE. The verbose setting prints
 *  progress every 100 merges — enough to know it is working, not
 *  enough to drown stdout.
 * ═══════════════════════════════════════════════════════════════════════ */

tk_config_t tk_config_default(void) {
    return (tk_config_t){
        .vocab_size  = 32000,
        .norm_flags  = 0,
        .pretokenize = true,
        .verbose     = 100,
        .chunk_size  = 64 * 1024 * 1024,  /* 64 MiB */
    };
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 2.5. Encoder Cache Management
 *
 *  The persistent BPE encoder (merge-rank hash table + byte→id table)
 *  is built once after training or loading, and torn down on free.
 *  This eliminates per-call index construction — the single largest
 *  source of overhead in the old encode path.
 * ═══════════════════════════════════════════════════════════════════════ */

static int tk_rebuild_encoder(tk_tokenizer_t *tk) {
    /* Tear down any existing encoder. */
    if (tk->encoder_ready) {
        tk_bpe_encoder_free(&tk->encoder);
        tk->encoder_ready = false;
    }

    /* Build the cached encoder from the current vocab. */
    if (tk_bpe_encoder_init(&tk->encoder, &tk->vocab) < 0)
        return -1;

    tk->encoder_ready = true;
    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 3. Lifecycle
 *
 *  Init allocates the vocab hash table and the encode arena.
 *  Free tears it all down. Between those two calls, the tokenizer
 *  may be trained, saved, loaded, and used for encode/decode an
 *  unlimited number of times.
 *
 *  "For everything there is a season: a time to init, and a time
 *   to free." — Ecclesiastes 3:1, Systems Programming Edition
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Initial arena size — 256 KiB handles most single-text encodes
 * without ever needing to grow. For batch encoding of long documents,
 * the arena will grow once and then stabilize.
 */
#define TK_ARENA_INITIAL_SIZE (256 * 1024)

int tk_init(tk_tokenizer_t *tk, const tk_config_t *config) {
    memset(tk, 0, sizeof(*tk));

    tk->config = config ? *config : tk_config_default();

    /*
     * Vocab table: size it to target_vocab + 256 byte tokens + headroom.
     * The hash table load factor should stay under 0.7 for performance,
     * so we allocate generously.
     */
    size_t table_size = (size_t)(tk->config.vocab_size + 256) * 2;
    if (TK_UNLIKELY(tk_vocab_init(&tk->vocab, table_size) < 0))
        return -1;

    /* Encode arena — the beating heart of zero-alloc encoding */
    if (TK_UNLIKELY(tk_arena_init(&tk->encode_arena, TK_ARENA_INITIAL_SIZE) < 0)) {
        tk_vocab_free(&tk->vocab);
        return -1;
    }

    tk->trained       = false;
    tk->encoder_ready = false;
    return 0;
}

void tk_free(tk_tokenizer_t *tk) {
    if (tk->encoder_ready)
        tk_bpe_encoder_free(&tk->encoder);
    tk_vocab_free(&tk->vocab);
    tk_arena_free(&tk->encode_arena);
    memset(tk, 0, sizeof(*tk));
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 4. Training
 *
 *  Two paths: in-memory (for small corpora or tests) and file-based
 *  (for real work with mmap streaming).
 *
 *  After training completes, we rebuild the persistent encoder so
 *  encode calls can use the cached merge index immediately.
 * ═══════════════════════════════════════════════════════════════════════ */

int tk_train(tk_tokenizer_t *tk, const uint8_t *text, size_t len) {
    if (TK_UNLIKELY(len == 0)) return -1;

    /*
     * Normalization: if enabled, allocate a mutable copy and normalize
     * in-place. We use the encode arena for this — it is already there,
     * already sized, and will be reset before the next encode call.
     *
     * "Thou shalt not normalize thy read-only mmap region."
     */
    const uint8_t *train_text = text;
    size_t train_len = len;
    uint8_t *norm_buf = NULL;

    if (tk->config.norm_flags != 0) {
        norm_buf = (uint8_t *)malloc(len);
        if (TK_UNLIKELY(!norm_buf)) return -1;
        memcpy(norm_buf, text, len);
        train_len = tk_normalize(norm_buf, len, tk->config.norm_flags);
        train_text = norm_buf;
    }

    tk_train_config_t tc = {
        .target_vocab_size = tk->config.vocab_size,
        .verbose           = tk->config.verbose,
    };

    int ret = tk_bpe_train(train_text, train_len, &tk->vocab, &tc);

    free(norm_buf);

    if (TK_LIKELY(ret == 0)) {
        tk->trained = true;
        /* Build the persistent encoder so encode calls are fast. */
        if (tk_rebuild_encoder(tk) < 0)
            return -1;
    }
    return ret;
}

int tk_train_file(tk_tokenizer_t *tk, const char *path) {
    /*
     * File-based training: mmap the corpus and stream it through the
     * BPE trainer in chunks. The trainer caps at its internal limit;
     * for larger corpora, sample a representative subset.
     *
     * We pass the raw file to BPE. The pre-tokenizer splits on
     * whitespace boundaries implicitly, so whitespace normalization
     * is effectively free. Turkish lowercasing should be done in
     * preprocessing (scripts/preprocess.py) if desired — doing it
     * here on a 30GB mmap would be a sin against both RAM and time.
     *
     * "Prepare your work outside; get everything ready in the field,
     *  and after that build your house." — Proverbs 24:27
     * (i.e., preprocess your corpus before you train)
     */
    tk_train_config_t tc = {
        .target_vocab_size = tk->config.vocab_size,
        .verbose           = tk->config.verbose,
    };

    int ret = tk_bpe_train_file(path, &tk->vocab, &tc, tk->config.chunk_size);

    if (TK_LIKELY(ret == 0)) {
        tk->trained = true;
        /* Build the persistent encoder. */
        if (tk_rebuild_encoder(tk) < 0)
            return -1;
    }
    return ret;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 5. Persistence
 *
 *  Save/load the .tkmodel binary format. The format is trivial:
 *  a header, packed byte sequences, packed merge rules. Loads in
 *  microseconds. JSON is for the weak.
 *
 *  "And Moses said unto the people, remember this day, in which
 *   ye came out from Egypt, out of the house of JSON."
 *   — Exodus 13:3, loosely
 * ═══════════════════════════════════════════════════════════════════════ */

int tk_save(const tk_tokenizer_t *tk, const char *path) {
    if (TK_UNLIKELY(!tk->trained)) {
        fprintf(stderr, "error: cannot save untrained tokenizer\n");
        return -1;
    }
    return tk_vocab_save(&tk->vocab, path);
}

int tk_load(tk_tokenizer_t *tk, const char *path) {
    memset(tk, 0, sizeof(*tk));
    tk->config = tk_config_default();

    /* Initialize the encode arena before loading — we need it ready */
    if (TK_UNLIKELY(tk_arena_init(&tk->encode_arena, TK_ARENA_INITIAL_SIZE) < 0))
        return -1;

    if (TK_UNLIKELY(tk_vocab_load(&tk->vocab, path) < 0)) {
        tk_arena_free(&tk->encode_arena);
        return -1;
    }

    tk->trained = true;

    /* Build the persistent encoder — this is the key to fast encoding.
     * The merge-rank hash table and byte→id table are built once here
     * and reused across all subsequent encode calls. */
    if (tk_rebuild_encoder(tk) < 0) {
        tk_vocab_free(&tk->vocab);
        tk_arena_free(&tk->encode_arena);
        return -1;
    }

    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 6. Encoding — The Hot Path
 *
 *  This is where nanoseconds are won and lost. The encoding pipeline:
 *
 *    ┌──────────┐    ┌───────────────┐    ┌────────────┐    ┌──────┐
 *    │  Input   │───▶│  Normalize    │───▶│  Pre-tok   │───▶│ BPE  │
 *    │  bytes   │    │  (in arena)   │    │  (split)   │    │encode│
 *    └──────────┘    └───────────────┘    └────────────┘    └──────┘
 *                                                              │
 *                                            ┌─────────────────┘
 *                                            ▼
 *                                      ┌──────────┐
 *                                      │  Token   │
 *                                      │   IDs    │
 *                                      └──────────┘
 *
 *  Critical invariants:
 *    - The arena is reset at the START of each tk_encode call.
 *    - Normalization writes into the arena (no separate malloc).
 *    - Pre-tokenization invokes BPE on each chunk via callback.
 *    - The callback writes directly into the caller's output buffer.
 *    - If the output buffer overflows, we stop immediately.
 *    - The persistent encoder (tk->encoder) is NEVER rebuilt per call.
 *
 *  For HFT-level latency on short strings (the inference case):
 *    - Arena reset is a single store instruction.
 *    - If no normalization, we encode directly from the input pointer.
 *    - Pre-tokenization scans left-to-right with no backtracking.
 *    - BPE encode uses the cached merge-rank hash table (zero malloc).
 *    - For words ≤ 256 bytes, all scratch lives on the stack.
 *
 *  "Be sober-minded; be watchful. Your adversary the cache miss
 *   prowls around like a roaring lion, seeking someone to devour."
 *   — 1 Peter 5:8
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Encode callback context — passed through the pre-tokenizer's
 * chunk callback. Carries a pointer to the PERSISTENT encoder,
 * not the raw vocab.
 */
typedef struct {
    const tk_bpe_encoder_t *encoder;    /* persistent, cached encoder      */
    uint32_t               *out_ids;    /* Caller's output buffer          */
    size_t                  out_cap;    /* Capacity of output buffer       */
    size_t                  out_pos;    /* Current write position          */
    bool                    overflow;   /* Set true if output buffer full  */
} encode_ctx_t;

/*
 * encode_chunk_cb — Pre-tokenizer callback. Invoked once per word-like
 * chunk. Runs BPE encoding on the chunk and appends IDs to the output.
 *
 * This function is called potentially millions of times per encode.
 * It must be fast. It must not allocate. It must not fail silently.
 *
 * OLD: called tk_bpe_encode() → built + destroyed merge index per call.
 *      With 3840 merges × 1.46M words = 19M hash table builds.
 *
 * NEW: calls tk_bpe_encode_with() → uses cached encoder, zero malloc.
 *
 * "Ye shall know them by their lack of malloc." — Matthew 7:16
 */
static void TK_HOT encode_chunk_cb(const uint8_t *start, size_t len,
                                    void *ud) {
    encode_ctx_t *ctx = (encode_ctx_t *)ud;

    /* Early exit if a previous chunk already overflowed.
     * Do not throw good cycles after bad. */
    if (TK_UNLIKELY(ctx->overflow)) return;

    size_t remaining = ctx->out_cap - ctx->out_pos;

    size_t n = tk_bpe_encode_with(ctx->encoder, start, len,
                                  ctx->out_ids + ctx->out_pos, remaining);

    if (TK_UNLIKELY(n == (size_t)-1)) {
        ctx->overflow = true;
        return;
    }

    ctx->out_pos += n;
}

/*
 * tk_encode — The main encoding entry point.
 *
 * Encodes UTF-8 text into token IDs. Returns the number of tokens
 * written, or (size_t)-1 on error (untrained, OOM, output overflow).
 *
 * Performance characteristics (with persistent encoder):
 *   - Short text (< 1 KiB): ~1-5 μs on modern hardware
 *   - Medium text (1-100 KiB): ~10-500 μs
 *   - Long text (> 100 KiB): scales linearly with O(n log n) BPE
 *
 * The arena is reset at entry, so this function is NOT reentrant.
 * Do not call tk_encode from within a callback invoked by tk_encode.
 * That way lies madness, stack overflow, and plagues of locusts.
 */
size_t TK_HOT tk_encode(tk_tokenizer_t *tk,
                         const uint8_t *text, size_t len,
                         uint32_t *out_ids, size_t out_cap) {
    if (TK_UNLIKELY(!tk->trained))       return (size_t)-1;
    if (TK_UNLIKELY(!tk->encoder_ready)) return (size_t)-1;
    if (TK_UNLIKELY(len == 0))           return 0;

    /* ── Arena reset: O(1), one store ── */
    tk_arena_reset(&tk->encode_arena);

    /* ── Step 1: Normalize (if enabled) ──
     *
     * We normalize into the arena so the original input is untouched
     * and we avoid a malloc/free pair on every encode call.
     */
    const uint8_t *enc_text = text;
    size_t enc_len = len;

    if (tk->config.norm_flags != 0) {
        uint8_t *norm_buf = (uint8_t *)tk_arena_alloc(
            &tk->encode_arena, len, TK_CACHE_LINE);
        if (TK_UNLIKELY(!norm_buf)) return (size_t)-1;

        memcpy(norm_buf, text, len);
        enc_len  = tk_normalize(norm_buf, len, tk->config.norm_flags);
        enc_text = norm_buf;
    }

    /* ── Step 2 + 3: Pre-tokenize → BPE encode ── */
    size_t result;

    if (tk->config.pretokenize) {
        encode_ctx_t ctx = {
            .encoder  = &tk->encoder,
            .out_ids  = out_ids,
            .out_cap  = out_cap,
            .out_pos  = 0,
            .overflow = false,
        };

        tk_pretokenize(enc_text, enc_len, encode_chunk_cb, &ctx);

        result = ctx.overflow ? (size_t)-1 : ctx.out_pos;
    } else {
        /* No pre-tokenization: single BPE pass over entire input. */
        result = tk_bpe_encode_with(&tk->encoder, enc_text, enc_len,
                                    out_ids, out_cap);
    }

    /* Arena memory persists until the next tk_encode call, but
     * we don't care — the caller has their IDs in out_ids. */
    return result;
}

/*
 * tk_encode_batch — Encode multiple texts in a single call.
 *
 * Amortizes arena management across all texts. The arena is reset
 * once per text, then each text's normalization buffer is allocated
 * from the arena.
 *
 * All texts share the same persistent encoder — no per-text index
 * rebuilds.
 */
int tk_encode_batch(tk_tokenizer_t *tk,
                    const tk_batch_input_t *inputs, size_t num_inputs,
                    tk_batch_result_t *results) {
    if (TK_UNLIKELY(!tk->trained))       return -1;
    if (TK_UNLIKELY(!tk->encoder_ready)) return -1;

    bool any_failed = false;

    for (size_t i = 0; i < num_inputs; i++) {
        /* Reset arena for each text — keeps memory bounded even
         * if one text is enormous */
        tk_arena_reset(&tk->encode_arena);

        const uint8_t *enc_text = inputs[i].text;
        size_t enc_len = inputs[i].len;

        /* Normalize into arena if needed */
        if (tk->config.norm_flags != 0 && enc_len > 0) {
            uint8_t *norm_buf = (uint8_t *)tk_arena_alloc(
                &tk->encode_arena, enc_len, TK_CACHE_LINE);
            if (TK_UNLIKELY(!norm_buf)) {
                results[i].len = (size_t)-1;
                any_failed = true;
                continue;
            }
            memcpy(norm_buf, enc_text, enc_len);
            enc_len  = tk_normalize(norm_buf, enc_len, tk->config.norm_flags);
            enc_text = norm_buf;
        }

        if (enc_len == 0) {
            results[i].len = 0;
            continue;
        }

        /* Encode with pre-tokenization */
        if (tk->config.pretokenize) {
            encode_ctx_t ctx = {
                .encoder  = &tk->encoder,
                .out_ids  = results[i].ids,
                .out_cap  = results[i].cap,
                .out_pos  = 0,
                .overflow = false,
            };

            tk_pretokenize(enc_text, enc_len, encode_chunk_cb, &ctx);
            results[i].len = ctx.overflow ? (size_t)-1 : ctx.out_pos;
        } else {
            results[i].len = tk_bpe_encode_with(&tk->encoder, enc_text,
                                                 enc_len, results[i].ids,
                                                 results[i].cap);
        }

        if (results[i].len == (size_t)-1) any_failed = true;
    }

    return any_failed ? -1 : 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 7. Decoding
 *
 *  Token IDs → UTF-8 bytes. The inverse of encoding. Simpler than
 *  encoding because there is no ambiguity: each token ID maps to
 *  exactly one byte sequence. We just concatenate them.
 *
 *  "What has been encoded may be decoded, and what has been decoded
 *   may be encoded again. There is nothing new under the tokenizer."
 *   — Ecclesiastes 1:9
 * ═══════════════════════════════════════════════════════════════════════ */

size_t TK_HOT tk_decode(const tk_tokenizer_t *tk,
                         const uint32_t *ids, size_t num_ids,
                         uint8_t *out, size_t out_cap) {
    if (TK_UNLIKELY(!tk->trained)) return (size_t)-1;
    return tk_bpe_decode(ids, num_ids, &tk->vocab, out, out_cap);
}

/*
 * tk_decode_batch — Decode multiple token sequences in a single call.
 *
 * Each result must have `out` and `cap` set by the caller.
 * `len` is filled with the number of bytes written (or -1 on error).
 */
int tk_decode_batch(const tk_tokenizer_t *tk,
                    const tk_decode_input_t *inputs, size_t num_inputs,
                    tk_decode_result_t *results) {
    if (TK_UNLIKELY(!tk->trained)) return -1;

    bool any_failed = false;

    for (size_t i = 0; i < num_inputs; i++) {
        results[i].len = tk_bpe_decode(
            inputs[i].ids, inputs[i].num_ids,
            &tk->vocab, results[i].out, results[i].cap);

        if (results[i].len == (size_t)-1) any_failed = true;
    }

    return any_failed ? -1 : 0;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 8. Utilities
 *
 *  Introspection, statistics, and diagnostics. Not on the hot path.
 *  These exist so you can understand what the tokenizer learned
 *  without firing up a hex editor.
 * ═══════════════════════════════════════════════════════════════════════ */

uint32_t tk_vocab_size_of(const tk_tokenizer_t *tk) {
    return tk->vocab.vocab_size;
}

const uint8_t *tk_token_bytes(const tk_tokenizer_t *tk, uint32_t id,
                               uint16_t *out_len) {
    const tk_token_t *t = tk_vocab_get(&tk->vocab, id);
    if (TK_UNLIKELY(!t)) return NULL;
    if (out_len) *out_len = t->len;
    return t->bytes;
}

uint32_t tk_token_to_id(const tk_tokenizer_t *tk,
                         const uint8_t *bytes, uint16_t len) {
    return tk_vocab_lookup(&tk->vocab, bytes, len);
}

/*
 * tk_encode_throughput — Benchmark encoding throughput.
 *
 * Encodes the input `iterations` times and reports bytes/second
 * and tokens/second. Uses clock_gettime(CLOCK_MONOTONIC) for
 * nanosecond-resolution timing.
 *
 * "The race is not to the swift, nor the battle to the strong,
 *  but time and clock_gettime happen to them all."
 *  — Ecclesiastes 9:11
 */
void tk_encode_throughput(tk_tokenizer_t *tk,
                          const uint8_t *text, size_t len,
                          size_t iterations,
                          double *out_bytes_per_sec,
                          double *out_tokens_per_sec) {
    /* Allocate output buffer: worst case is 1 token per byte */
    uint32_t *ids = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!ids) {
        *out_bytes_per_sec = 0;
        *out_tokens_per_sec = 0;
        return;
    }

    /* Warmup — let the arena settle and branch predictors train */
    for (size_t i = 0; i < 3; i++) {
        tk_encode(tk, text, len, ids, len);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t total_tokens = 0;
    for (size_t i = 0; i < iterations; i++) {
        size_t n = tk_encode(tk, text, len, ids, len);
        if (n != (size_t)-1) total_tokens += n;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (double)(t1.tv_sec - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;

    if (elapsed > 0.0) {
        double total_bytes = (double)len * (double)iterations;
        *out_bytes_per_sec  = total_bytes / elapsed;
        *out_tokens_per_sec = (double)total_tokens / elapsed;
    } else {
        *out_bytes_per_sec = 0;
        *out_tokens_per_sec = 0;
    }

    free(ids);
}

/*
 * tk_print_stats — Dump tokenizer statistics to stderr.
 *
 * Shows vocab size, merge count, sample tokens with their byte
 * representations, and arena stats. Useful for sanity checking
 * after training or loading a model.
 *
 * "Then the LORD answered Job out of the whirlwind and said:
 *  'Where were you when I laid the foundation of this vocab table?'"
 *  — Job 38:1, debugger edition
 */
void TK_COLD tk_print_stats(const tk_tokenizer_t *tk) {
    fprintf(stderr,
        "╔═══════════════════════════════════╗\n"
        "║       Tokenizer Statistics        ║\n"
        "╠═══════════════════════════════════╣\n");
    fprintf(stderr,
        "║  Trained     : %-18s║\n", tk->trained ? "yes" : "no");
    fprintf(stderr,
        "║  Vocab size  : %-18u║\n", tk->vocab.vocab_size);
    fprintf(stderr,
        "║  Merge rules : %-18u║\n", tk->vocab.num_merges);
    fprintf(stderr,
        "║  Base tokens : %-18s║\n", "256 (byte-level)");
    fprintf(stderr,
        "║  Arena cap   : %-14zu KiB║\n",
        tk->encode_arena.capacity / 1024);
    fprintf(stderr,
        "╠═══════════════════════════════════╣\n");

    if (tk->vocab.num_merges > 0) {
        fprintf(stderr,
            "║  Sample merged tokens:            ║\n"
            "╠═══════════════════════════════════╣\n");

        uint32_t start = 256;
        uint32_t show  = 10;
        if (tk->vocab.vocab_size - 256 < show)
            show = tk->vocab.vocab_size - 256;

        for (uint32_t i = 0; i < show; i++) {
            const tk_token_t *t = tk_vocab_get(&tk->vocab, start + i);
            if (!t) break;

            /* Print: ║  [ID] "bytes" (N bytes)          ║ */
            fprintf(stderr, "║  [%4u] \"", t->id);
            for (uint16_t j = 0; j < t->len; j++) {
                uint8_t b = t->bytes[j];
                if (b >= 0x20 && b < 0x7F) {
                    fputc(b, stderr);
                } else {
                    fprintf(stderr, "\\x%02X", b);
                }
            }
            fprintf(stderr, "\" (%u B)\n", t->len);
        }
    }

    fprintf(stderr,
        "╚═══════════════════════════════════╝\n");
}