/*
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║  tokenizer.h — Public API for the Turkish BPE Tokenizer             ║
 * ║                                                                     ║
 * ║  Init, train, save, load, encode, decode. This is the only header   ║
 * ║  most consumers need to include.                                    ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 */

#ifndef TK_TOKENIZER_H
#define TK_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Pull in sub-module headers for types used in the tokenizer struct.
 * Consumers who only need the top-level API can ignore these. */
#include "unicode.h"
#include "vocab.h"
#include "bpe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────
 *  Arena Allocator (internal, exposed for struct layout)
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    uint8_t *base;       /* Slab base pointer                          */
    size_t   offset;     /* Current bump position                      */
    size_t   capacity;   /* Total slab size                            */
} tk_arena_t;

/* ─────────────────────────────────────────────────────────────────────
 *  Configuration
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t vocab_size;     /* Target vocabulary size (incl. 256 base)  */
    uint32_t norm_flags;     /* TK_NORM_* flags for normalization        */
    bool     pretokenize;    /* Apply GPT-style pre-tokenization         */
    uint32_t verbose;        /* Print progress every N merges (0=quiet)  */
    size_t   chunk_size;     /* Chunk size for file-based training       */
} tk_config_t;

tk_config_t tk_config_default(void);

/* ─────────────────────────────────────────────────────────────────────
 *  Tokenizer
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    tk_config_t  config;
    tk_vocab_t   vocab;
    tk_arena_t   encode_arena;   /* Zero-alloc encode scratch space    */
    bool         trained;
} tk_tokenizer_t;

/* Lifecycle */
int  tk_init(tk_tokenizer_t *tk, const tk_config_t *config);
void tk_free(tk_tokenizer_t *tk);

/* Training */
int tk_train(tk_tokenizer_t *tk, const uint8_t *text, size_t len);
int tk_train_file(tk_tokenizer_t *tk, const char *path);

/* Persistence */
int tk_save(const tk_tokenizer_t *tk, const char *path);
int tk_load(tk_tokenizer_t *tk, const char *path);

/* Encoding */
size_t tk_encode(tk_tokenizer_t *tk,
                 const uint8_t *text, size_t len,
                 uint32_t *out_ids, size_t out_cap);

/* Decoding */
size_t tk_decode(const tk_tokenizer_t *tk,
                 const uint32_t *ids, size_t num_ids,
                 uint8_t *out, size_t out_cap);

/* ─────────────────────────────────────────────────────────────────────
 *  Batch Encoding / Decoding
 *
 *  For bulk operations. Amortizes overhead across multiple texts.
 * ───────────────────────────────────────────────────────────────────── */

typedef struct {
    const uint8_t *text;     /* Input UTF-8 text                        */
    size_t         len;      /* Input length in bytes                   */
} tk_batch_input_t;

typedef struct {
    uint32_t *ids;           /* Caller-allocated output buffer          */
    size_t    cap;           /* Capacity of ids buffer                  */
    size_t    len;           /* [out] tokens written, or (size_t)-1     */
} tk_batch_result_t;

int tk_encode_batch(tk_tokenizer_t *tk,
                    const tk_batch_input_t *inputs, size_t num_inputs,
                    tk_batch_result_t *results);

typedef struct {
    const uint32_t *ids;     /* Input token IDs                         */
    size_t          num_ids; /* Number of token IDs                     */
} tk_decode_input_t;

typedef struct {
    uint8_t *out;            /* Caller-allocated output buffer          */
    size_t   cap;            /* Capacity of output buffer               */
    size_t   len;            /* [out] bytes written, or (size_t)-1      */
} tk_decode_result_t;

int tk_decode_batch(const tk_tokenizer_t *tk,
                    const tk_decode_input_t *inputs, size_t num_inputs,
                    tk_decode_result_t *results);

/* ─────────────────────────────────────────────────────────────────────
 *  Introspection & Utilities
 * ───────────────────────────────────────────────────────────────────── */

uint32_t        tk_vocab_size_of(const tk_tokenizer_t *tk);
const uint8_t  *tk_token_bytes(const tk_tokenizer_t *tk, uint32_t id,
                                uint16_t *out_len);
uint32_t        tk_token_to_id(const tk_tokenizer_t *tk,
                                const uint8_t *bytes, uint16_t len);

/* Benchmark: encode `iterations` times, report throughput */
void tk_encode_throughput(tk_tokenizer_t *tk,
                          const uint8_t *text, size_t len,
                          size_t iterations,
                          double *out_bytes_per_sec,
                          double *out_tokens_per_sec);

/* Print tokenizer stats to stderr */
void tk_print_stats(const tk_tokenizer_t *tk);

#ifdef __cplusplus
}
#endif

#endif /* TK_TOKENIZER_H */