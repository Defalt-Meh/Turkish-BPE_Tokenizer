#ifndef TK_VOCAB_H
#define TK_VOCAB_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * vocab.h — Vocabulary and merge-rule storage for byte-level BPE.
 *
 * The vocabulary maps byte sequences (tokens) to integer IDs and back.
 * Merge rules are stored in priority order (earliest = highest priority).
 *
 * Layout of a trained model file (.tkmodel):
 *   [magic 4B] [version 4B] [vocab_size 4B] [num_merges 4B]
 *   [vocab entries: id(4B) len(2B) bytes(len)]...
 *   [merge entries: left_id(4B) right_id(4B)]...   (in priority order)
 */

#define TK_MAGIC       0x544B4250  /* "TKBP" */
#define TK_VERSION     1
#define TK_MAX_TOKEN_LEN  128

/* ── Single token entry ──────────────────────────────────────────────── */

typedef struct {
    uint32_t id;
    uint16_t len;
    uint8_t  bytes[TK_MAX_TOKEN_LEN];
} tk_token_t;

/* ── Merge rule (pair of token IDs that merge into a new token) ─────── */

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;  /* ID of the merged token */
} tk_merge_t;

/* ── Hash table entry for token→id lookup ────────────────────────────── */

typedef struct tk_ht_entry {
    uint8_t  *key;        /* token bytes (heap-allocated) */
    uint16_t  key_len;
    uint32_t  value;      /* token ID */
    struct tk_ht_entry *next;  /* chaining for collisions */
} tk_ht_entry_t;

/* ── Vocabulary ──────────────────────────────────────────────────────── */

typedef struct {
    /* ID → token (dense array, indexed by ID) */
    tk_token_t *tokens;
    uint32_t    vocab_size;
    uint32_t    vocab_cap;

    /* Token bytes → ID (hash table) */
    tk_ht_entry_t **buckets;
    uint32_t        num_buckets;

    /* Merge rules in priority order */
    tk_merge_t *merges;
    uint32_t    num_merges;
    uint32_t    merges_cap;
} tk_vocab_t;

/* ── Lifecycle ───────────────────────────────────────────────────────── */

/* Initialize an empty vocabulary. Returns 0 on success, -1 on error. */
int tk_vocab_init(tk_vocab_t *v, uint32_t initial_cap);

/* Free all memory owned by the vocabulary. */
void tk_vocab_free(tk_vocab_t *v);

/* ── Building the vocabulary ─────────────────────────────────────────── */

/* Add a token (byte sequence) to the vocabulary. Returns its ID.
 * If the token already exists, returns the existing ID.
 * Returns (uint32_t)-1 on allocation failure. */
uint32_t tk_vocab_add(tk_vocab_t *v, const uint8_t *bytes, uint16_t len);

/* Add the 256 single-byte tokens (IDs 0–255). Call this first when
 * building a byte-level BPE vocab from scratch. */
void tk_vocab_add_byte_tokens(tk_vocab_t *v);

/* Record a merge rule: left + right → result (already added via tk_vocab_add).
 * Merges are stored in the order they are added (= priority order). */
int tk_vocab_add_merge(tk_vocab_t *v, uint32_t left, uint32_t right, uint32_t result);

/* ── Lookup ──────────────────────────────────────────────────────────── */

/* Look up a token by its byte content. Returns ID, or (uint32_t)-1 if not found. */
uint32_t tk_vocab_lookup(const tk_vocab_t *v, const uint8_t *bytes, uint16_t len);

/* Get the token entry for a given ID. Returns NULL if ID is out of range. */
const tk_token_t *tk_vocab_get(const tk_vocab_t *v, uint32_t id);

/* ── Serialization ───────────────────────────────────────────────────── */

/* Save vocabulary + merges to a binary .tkmodel file. Returns 0 on success. */
int tk_vocab_save(const tk_vocab_t *v, const char *path);

/* Load vocabulary + merges from a .tkmodel file. `v` should be uninitialized;
 * this function calls tk_vocab_init internally. Returns 0 on success. */
int tk_vocab_load(tk_vocab_t *v, const char *path);

#endif /* TK_VOCAB_H */