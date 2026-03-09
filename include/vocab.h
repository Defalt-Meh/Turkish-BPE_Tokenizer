/*
 * vocab.h — Token vocabulary, merge rules, and .tkmodel format.
 */

#ifndef TK_VOCAB_H
#define TK_VOCAB_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Binary format identifiers. */
#define TK_MAGIC   0x544B4D4Cu  /* "TKML" */
#define TK_VERSION 1

/* Maximum byte length of a single token. */
#define TK_MAX_TOKEN_LEN 128

/* ── Token ────────────────────────────────────────────────────────── */

typedef struct {
    uint32_t id;
    uint16_t len;
    uint8_t  bytes[TK_MAX_TOKEN_LEN];
} tk_token_t;

/* ── Merge rule ───────────────────────────────────────────────────── */

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;
} tk_merge_t;

/* ── Vocabulary ───────────────────────────────────────────────────── */

typedef struct {
    /* Dense token array — index == token id. */
    tk_token_t *tokens;
    uint32_t    vocab_size;
    uint32_t    vocab_cap;

    /* Open-addressing hash table (token bytes → token id). */
    uint32_t   *slots;       /* Each slot holds a token id or sentinel. */
    uint32_t    num_slots;   /* Always a power of two.                  */
    uint32_t    slot_mask;   /* num_slots - 1, for branchless wrapping. */

    /* Ordered merge rules. */
    tk_merge_t *merges;
    uint32_t    num_merges;
    uint32_t    merges_cap;
} tk_vocab_t;

/* Lifecycle. */
int  tk_vocab_init(tk_vocab_t *v, uint32_t initial_cap);
void tk_vocab_free(tk_vocab_t *v);

/* Building. */
uint32_t        tk_vocab_add(tk_vocab_t *v, const uint8_t *bytes, uint16_t len);
void            tk_vocab_add_byte_tokens(tk_vocab_t *v);
int             tk_vocab_add_merge(tk_vocab_t *v, uint32_t left,
                                   uint32_t right, uint32_t result);

/* Lookup. */
uint32_t        tk_vocab_lookup(const tk_vocab_t *v,
                                const uint8_t *bytes, uint16_t len);
const tk_token_t *tk_vocab_get(const tk_vocab_t *v, uint32_t id);

/* Serialization. */
int tk_vocab_save(const tk_vocab_t *v, const char *path);
int tk_vocab_load(tk_vocab_t *v, const char *path);

#ifdef __cplusplus
}
#endif

#endif /* TK_VOCAB_H */