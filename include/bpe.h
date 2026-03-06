#ifndef TK_BPE_H
#define TK_BPE_H

#include "vocab.h"
#include <stdint.h>
#include <stddef.h>

/*
 * bpe.h — Byte-Pair Encoding training and encoding.
 *
 * Training:
 *   1. Start with 256 byte-level tokens.
 *   2. Count all adjacent token pairs across the corpus.
 *   3. Merge the most frequent pair into a new token.
 *   4. Repeat until desired vocab size is reached.
 *
 * Encoding:
 *   1. Split input into bytes (each is a token).
 *   2. Apply merge rules in priority order (greedy left-to-right).
 *
 * The pair-counting uses a hash table keyed by (left_id, right_id).
 * For large corpora we process in chunks to bound memory usage.
 */

/* ── Pair frequency entry ────────────────────────────────────────────── */

typedef struct tk_pair_entry {
    uint32_t left;
    uint32_t right;
    int64_t  count;
    struct tk_pair_entry *next;  /* hash chain */
} tk_pair_entry_t;

/* ── Pair frequency hash table ───────────────────────────────────────── */

typedef struct {
    tk_pair_entry_t **buckets;
    uint32_t          num_buckets;
    uint32_t          num_entries;
} tk_pair_table_t;

int  tk_pair_table_init(tk_pair_table_t *pt, uint32_t num_buckets);
void tk_pair_table_free(tk_pair_table_t *pt);
void tk_pair_table_clear(tk_pair_table_t *pt);

/* Increment count for a pair. Creates entry if it doesn't exist. */
int tk_pair_table_add(tk_pair_table_t *pt, uint32_t left, uint32_t right, int64_t count);

/* Find the pair with the highest count. Returns NULL if table is empty. */
const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt);

/* ── Token sequence (linked list for efficient mid-sequence merging) ── */

typedef struct tk_node {
    uint32_t id;
    struct tk_node *prev;
    struct tk_node *next;
} tk_node_t;

typedef struct {
    tk_node_t *head;
    tk_node_t *tail;
    size_t     length;

    /* Pool allocator for nodes to avoid per-node malloc */
    tk_node_t *pool;
    size_t     pool_size;
    size_t     pool_used;
} tk_sequence_t;

int  tk_sequence_init(tk_sequence_t *seq, size_t max_nodes);
void tk_sequence_free(tk_sequence_t *seq);
void tk_sequence_clear(tk_sequence_t *seq);

/* Append a token ID to the end of the sequence. */
int tk_sequence_append(tk_sequence_t *seq, uint32_t id);

/* Apply one merge (left, right → result) across the entire sequence.
 * Modifies the sequence in-place. Returns number of merges applied. */
size_t tk_sequence_apply_merge(tk_sequence_t *seq,
                               uint32_t left, uint32_t right, uint32_t result);

/* Count all adjacent pairs in the sequence into a pair table. */
void tk_sequence_count_pairs(const tk_sequence_t *seq, tk_pair_table_t *pt);

/* Extract the final token IDs into a flat array. Caller provides buffer. 
 * Returns number of IDs written. */
size_t tk_sequence_to_ids(const tk_sequence_t *seq, uint32_t *out, size_t out_cap);

/* ── BPE Trainer ─────────────────────────────────────────────────────── */

typedef struct {
    uint32_t target_vocab_size;  /* stop when vocab reaches this size */
    int      verbose;            /* print progress every N merges (0=quiet) */
} tk_train_config_t;

/*
 * Train BPE on a corpus loaded in memory.
 *
 *   text / text_len   — raw UTF-8 corpus (already pre-tokenized into chunks
 *                        externally, or we pre-tokenize internally)
 *   vocab              — output vocabulary (should be initialized; we add
 *                        byte tokens + merged tokens)
 *   config             — training parameters
 *
 * Returns 0 on success, -1 on error.
 */
int tk_bpe_train(const uint8_t *text, size_t text_len,
                 tk_vocab_t *vocab, const tk_train_config_t *config);

/*
 * Train BPE on a memory-mapped file (for large corpora).
 * Processes the file in chunks to limit RAM usage.
 *
 *   path               — path to the text corpus file
 *   vocab              — output vocabulary
 *   config             — training parameters
 *   chunk_size         — bytes per processing chunk (e.g. 64MB)
 *
 * Returns 0 on success, -1 on error.
 */
int tk_bpe_train_file(const char *path, tk_vocab_t *vocab,
                      const tk_train_config_t *config, size_t chunk_size);

/* ── BPE Encoding (applying learned merges) ──────────────────────────── */

/*
 * Encode a UTF-8 byte sequence into token IDs using the learned merges.
 *
 *   text / text_len — input bytes
 *   vocab           — trained vocabulary with merge rules
 *   out_ids         — output buffer for token IDs
 *   out_cap         — capacity of out_ids
 *
 * Returns number of tokens produced, or (size_t)-1 on error.
 */
size_t tk_bpe_encode(const uint8_t *text, size_t text_len,
                     const tk_vocab_t *vocab,
                     uint32_t *out_ids, size_t out_cap);

/*
 * Decode token IDs back to bytes.
 *
 *   ids / num_ids — input token IDs
 *   vocab         — trained vocabulary
 *   out           — output buffer for bytes
 *   out_cap       — capacity of out
 *
 * Returns number of bytes written, or (size_t)-1 on error.
 */
size_t tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                     const tk_vocab_t *vocab,
                     uint8_t *out, size_t out_cap);

#endif /* TK_BPE_H */