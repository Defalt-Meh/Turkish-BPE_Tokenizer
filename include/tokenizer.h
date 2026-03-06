#ifndef TK_TOKENIZER_H
#define TK_TOKENIZER_H

#include "vocab.h"
#include "bpe.h"
#include "unicode.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * tokenizer.h — Top-level tokenizer API.
 *
 * This is the single entry point for users who just want to:
 *   1. Train a tokenizer on a corpus.
 *   2. Load a trained tokenizer from disk.
 *   3. Encode text → token IDs.
 *   4. Decode token IDs → text.
 *
 * Wraps the lower-level vocab, BPE, unicode, and IO modules into
 * a clean, opaque interface.
 */

/* ── Configuration ───────────────────────────────────────────────────── */

typedef struct {
    uint32_t vocab_size;       /* target vocabulary size (default: 32000) */
    unsigned norm_flags;       /* TK_NORM_* flags applied before encoding */
    bool     pretokenize;      /* run GPT-style pre-tokenization (default: true) */
    int      verbose;          /* print progress every N merges (0=quiet) */
    size_t   chunk_size;       /* corpus chunk size for file training (default: 64MB) */
} tk_config_t;

/* Returns a config with sensible defaults. */
tk_config_t tk_config_default(void);

/* ── Tokenizer handle ────────────────────────────────────────────────── */

typedef struct {
    tk_vocab_t  vocab;
    tk_config_t config;
    bool        trained;
} tk_tokenizer_t;

/* ── Lifecycle ───────────────────────────────────────────────────────── */

/* Initialize a new, untrained tokenizer with the given config.
 * Returns 0 on success. */
int tk_init(tk_tokenizer_t *tk, const tk_config_t *config);

/* Free all resources. */
void tk_free(tk_tokenizer_t *tk);

/* ── Training ────────────────────────────────────────────────────────── */

/* Train on an in-memory UTF-8 buffer. */
int tk_train(tk_tokenizer_t *tk, const uint8_t *text, size_t len);

/* Train on a file (uses mmap, handles large corpora). */
int tk_train_file(tk_tokenizer_t *tk, const char *path);

/* ── Persistence ─────────────────────────────────────────────────────── */

/* Save trained tokenizer to a .tkmodel file. */
int tk_save(const tk_tokenizer_t *tk, const char *path);

/* Load a tokenizer from a .tkmodel file.
 * `tk` should be uninitialized; this sets everything up. */
int tk_load(tk_tokenizer_t *tk, const char *path);

/* ── Encoding / Decoding ─────────────────────────────────────────────── */

/*
 * Encode a UTF-8 string into token IDs.
 *
 * Applies normalization (if configured), pre-tokenization, and BPE.
 *
 *   text / len   — input UTF-8 string
 *   out_ids      — caller-provided output buffer
 *   out_cap      — capacity of out_ids
 *
 * Returns the number of tokens produced, or (size_t)-1 on error.
 * If the output buffer is too small, returns (size_t)-1 and you should
 * retry with a larger buffer (a safe upper bound is `len` tokens).
 */
size_t tk_encode(tk_tokenizer_t *tk,
                 const uint8_t *text, size_t len,
                 uint32_t *out_ids, size_t out_cap);

/*
 * Decode token IDs back into a UTF-8 byte sequence.
 *
 *   ids / num_ids — input token IDs
 *   out           — caller-provided output buffer
 *   out_cap       — capacity of out buffer
 *
 * Returns number of bytes written, or (size_t)-1 on error.
 */
size_t tk_decode(const tk_tokenizer_t *tk,
                 const uint32_t *ids, size_t num_ids,
                 uint8_t *out, size_t out_cap);

/* ── Utilities ───────────────────────────────────────────────────────── */

/* Get the vocabulary size of a trained tokenizer. */
uint32_t tk_vocab_size(const tk_tokenizer_t *tk);

/* Get the byte representation of a token ID. Returns NULL if invalid. */
const uint8_t *tk_token_bytes(const tk_tokenizer_t *tk, uint32_t id, uint16_t *out_len);

/* Look up the ID for a byte sequence. Returns (uint32_t)-1 if not found. */
uint32_t tk_token_to_id(const tk_tokenizer_t *tk, const uint8_t *bytes, uint16_t len);

/* Print tokenizer stats to stderr: vocab size, merge count, etc. */
void tk_print_stats(const tk_tokenizer_t *tk);

#endif /* TK_TOKENIZER_H */