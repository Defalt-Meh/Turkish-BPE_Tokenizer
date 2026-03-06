/*
 * tokenizer.c — Top-level tokenizer: init, train, save/load, encode/decode.
 *               Glues together unicode, vocab, bpe, and io modules.
 */

#include "tokenizer.h"
#include "io.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ════════════════════════════════════════════════════════════════════════
 *  Default Configuration
 * ════════════════════════════════════════════════════════════════════════ */

tk_config_t tk_config_default(void) {
    tk_config_t c;
    c.vocab_size   = 32000;
    c.norm_flags   = TK_NORM_WHITESPACE;  /* collapse whitespace by default */
    c.pretokenize  = true;
    c.verbose      = 100;                 /* print every 100 merges */
    c.chunk_size   = 64 * 1024 * 1024;   /* 64 MB */
    return c;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Lifecycle
 * ════════════════════════════════════════════════════════════════════════ */

int tk_init(tk_tokenizer_t *tk, const tk_config_t *config) {
    memset(tk, 0, sizeof(*tk));

    if (config) {
        tk->config = *config;
    } else {
        tk->config = tk_config_default();
    }

    if (tk_vocab_init(&tk->vocab, tk->config.vocab_size + 256) < 0)
        return -1;

    tk->trained = false;
    return 0;
}

void tk_free(tk_tokenizer_t *tk) {
    tk_vocab_free(&tk->vocab);
    memset(tk, 0, sizeof(*tk));
}

/* ════════════════════════════════════════════════════════════════════════
 *  Training
 * ════════════════════════════════════════════════════════════════════════ */

int tk_train(tk_tokenizer_t *tk, const uint8_t *text, size_t len) {
    /* Optionally normalize a copy of the input */
    uint8_t *buf = NULL;
    const uint8_t *train_text = text;
    size_t train_len = len;

    if (tk->config.norm_flags != 0) {
        buf = malloc(len);
        if (!buf) return -1;
        memcpy(buf, text, len);
        train_len = tk_normalize(buf, len, tk->config.norm_flags);
        train_text = buf;
    }

    tk_train_config_t tc;
    tc.target_vocab_size = tk->config.vocab_size;
    tc.verbose = tk->config.verbose;

    int ret = tk_bpe_train(train_text, train_len, &tk->vocab, &tc);

    free(buf);

    if (ret == 0) tk->trained = true;
    return ret;
}

int tk_train_file(tk_tokenizer_t *tk, const char *path) {
    /*
     * For file-based training, we mmap the file and pass it to the
     * in-memory trainer (which caps at 2GB internally).
     *
     * Normalization of the full corpus in-place isn't practical with
     * mmap (read-only mapping). For production, you'd preprocess the
     * corpus file via scripts/preprocess.py first, then train on the
     * clean version. Here we train on raw text — whitespace collapsing
     * is the only normalization that really matters for BPE quality,
     * and the pre-tokenizer handles that implicitly by splitting on
     * whitespace boundaries.
     */
    tk_train_config_t tc;
    tc.target_vocab_size = tk->config.vocab_size;
    tc.verbose = tk->config.verbose;

    int ret = tk_bpe_train_file(path, &tk->vocab, &tc, tk->config.chunk_size);

    if (ret == 0) tk->trained = true;
    return ret;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Persistence
 * ════════════════════════════════════════════════════════════════════════ */

int tk_save(const tk_tokenizer_t *tk, const char *path) {
    if (!tk->trained) {
        fprintf(stderr, "error: cannot save untrained tokenizer\n");
        return -1;
    }
    return tk_vocab_save(&tk->vocab, path);
}

int tk_load(tk_tokenizer_t *tk, const char *path) {
    memset(tk, 0, sizeof(*tk));
    tk->config = tk_config_default();

    if (tk_vocab_load(&tk->vocab, path) < 0) return -1;

    tk->trained = true;
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Encoding
 *
 *  Pipeline:
 *    1. (optional) Normalize the input text.
 *    2. (optional) Pre-tokenize into word-like chunks.
 *    3. For each chunk, apply BPE encoding.
 *    4. Concatenate all token IDs into the output buffer.
 * ════════════════════════════════════════════════════════════════════════ */

/* Callback context for pre-tokenize + encode */
typedef struct {
    const tk_vocab_t *vocab;
    uint32_t         *out_ids;
    size_t            out_cap;
    size_t            out_pos;
    bool              overflow;
} encode_ctx_t;

static void encode_chunk_cb(const uint8_t *start, size_t len, void *ud) {
    encode_ctx_t *ctx = (encode_ctx_t *)ud;
    if (ctx->overflow) return;

    size_t remaining = ctx->out_cap - ctx->out_pos;
    size_t n = tk_bpe_encode(start, len, ctx->vocab,
                             ctx->out_ids + ctx->out_pos, remaining);

    if (n == (size_t)-1) {
        ctx->overflow = true;
        return;
    }

    ctx->out_pos += n;
}

size_t tk_encode(tk_tokenizer_t *tk,
                 const uint8_t *text, size_t len,
                 uint32_t *out_ids, size_t out_cap) {
    if (!tk->trained) return (size_t)-1;
    if (len == 0) return 0;

    /* Step 1: Normalize */
    uint8_t *buf = NULL;
    const uint8_t *enc_text = text;
    size_t enc_len = len;

    if (tk->config.norm_flags != 0) {
        buf = malloc(len);
        if (!buf) return (size_t)-1;
        memcpy(buf, text, len);
        enc_len = tk_normalize(buf, len, tk->config.norm_flags);
        enc_text = buf;
    }

    size_t result;

    if (tk->config.pretokenize) {
        /* Step 2 + 3: Pre-tokenize, then BPE-encode each chunk */
        encode_ctx_t ctx;
        ctx.vocab   = &tk->vocab;
        ctx.out_ids = out_ids;
        ctx.out_cap = out_cap;
        ctx.out_pos = 0;
        ctx.overflow = false;

        tk_pretokenize(enc_text, enc_len, encode_chunk_cb, &ctx);

        result = ctx.overflow ? (size_t)-1 : ctx.out_pos;
    } else {
        /* No pre-tokenization: encode the whole thing at once */
        result = tk_bpe_encode(enc_text, enc_len, &tk->vocab, out_ids, out_cap);
    }

    free(buf);
    return result;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Decoding
 * ════════════════════════════════════════════════════════════════════════ */

size_t tk_decode(const tk_tokenizer_t *tk,
                 const uint32_t *ids, size_t num_ids,
                 uint8_t *out, size_t out_cap) {
    if (!tk->trained) return (size_t)-1;
    return tk_bpe_decode(ids, num_ids, &tk->vocab, out, out_cap);
}

/* ════════════════════════════════════════════════════════════════════════
 *  Utilities
 * ════════════════════════════════════════════════════════════════════════ */

uint32_t tk_vocab_size(const tk_tokenizer_t *tk) {
    return tk->vocab.vocab_size;
}

const uint8_t *tk_token_bytes(const tk_tokenizer_t *tk, uint32_t id, uint16_t *out_len) {
    const tk_token_t *t = tk_vocab_get(&tk->vocab, id);
    if (!t) return NULL;
    if (out_len) *out_len = t->len;
    return t->bytes;
}

uint32_t tk_token_to_id(const tk_tokenizer_t *tk, const uint8_t *bytes, uint16_t len) {
    return tk_vocab_lookup(&tk->vocab, bytes, len);
}

void tk_print_stats(const tk_tokenizer_t *tk) {
    fprintf(stderr, "=== Tokenizer Stats ===\n");
    fprintf(stderr, "  Trained:     %s\n", tk->trained ? "yes" : "no");
    fprintf(stderr, "  Vocab size:  %u\n", tk->vocab.vocab_size);
    fprintf(stderr, "  Merge rules: %u\n", tk->vocab.num_merges);
    fprintf(stderr, "  Base tokens: 256 (byte-level)\n");

    if (tk->vocab.num_merges > 0) {
        /* Show a few example merged tokens */
        fprintf(stderr, "  Sample tokens:\n");
        uint32_t start = 256;
        uint32_t show = 10;
        if (tk->vocab.vocab_size - 256 < show)
            show = tk->vocab.vocab_size - 256;

        for (uint32_t i = 0; i < show; i++) {
            const tk_token_t *t = tk_vocab_get(&tk->vocab, start + i);
            if (!t) break;

            fprintf(stderr, "    [%u] \"", t->id);
            for (uint16_t j = 0; j < t->len; j++) {
                uint8_t b = t->bytes[j];
                if (b >= 0x20 && b < 0x7F) {
                    fputc(b, stderr);
                } else {
                    fprintf(stderr, "\\x%02X", b);
                }
            }
            fprintf(stderr, "\" (%u bytes)\n", t->len);
        }
    }
    fprintf(stderr, "=======================\n");
}