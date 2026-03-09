#ifndef TK_BPE_H
#define TK_BPE_H

#include "vocab.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── pair entry (returned by best-pair queries) ───────────────────── */

typedef struct tk_pair_entry {
    uint32_t left;
    uint32_t right;
    int64_t  count;
    struct tk_pair_entry *next; /* unused in open-addressing; kept for API compat */
} tk_pair_entry_t;

/* ── open-addressing pair frequency table ─────────────────────────── */

typedef struct {
    uint64_t *keys;       /* packed (left<<32|right), EMPTY=0xFF..FF */
    int64_t  *counts;     /* parallel count array */
    uint32_t  num_slots;  /* always power of two */
    uint32_t  mask;       /* num_slots - 1 */
    uint32_t  num_entries;
} tk_pair_table_t;

int  tk_pair_table_init(tk_pair_table_t *pt, uint32_t min_slots);
void tk_pair_table_free(tk_pair_table_t *pt);
void tk_pair_table_clear(tk_pair_table_t *pt);
int  tk_pair_table_add(tk_pair_table_t *pt, uint32_t left,
                       uint32_t right, int64_t count);
const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt);

/* ── token sequence (doubly-linked list, pool-allocated) ──────────── */

typedef struct tk_node {
    uint32_t        id;
    struct tk_node *prev;
    struct tk_node *next;
} tk_node_t;

typedef struct {
    tk_node_t *pool;
    size_t     pool_size;
    size_t     pool_used;
    tk_node_t *head;
    tk_node_t *tail;
    size_t     length;
} tk_sequence_t;

int    tk_sequence_init(tk_sequence_t *seq, size_t max_nodes);
void   tk_sequence_free(tk_sequence_t *seq);
void   tk_sequence_clear(tk_sequence_t *seq);
int    tk_sequence_append(tk_sequence_t *seq, uint32_t id);
size_t tk_sequence_apply_merge(tk_sequence_t *seq, uint32_t left,
                               uint32_t right, uint32_t result);
void   tk_sequence_count_pairs(const tk_sequence_t *seq, tk_pair_table_t *pt);
size_t tk_sequence_to_ids(const tk_sequence_t *seq, uint32_t *out,
                          size_t out_cap);

/* ── training config ──────────────────────────────────────────────── */

typedef struct {
    uint32_t target_vocab_size;
    uint32_t verbose;
} tk_train_config_t;

/* ── training ─────────────────────────────────────────────────────── */

int tk_bpe_train(const uint8_t *text, size_t text_len,
                 tk_vocab_t *vocab, const tk_train_config_t *config);

int tk_bpe_train_file(const char *path, tk_vocab_t *vocab,
                      const tk_train_config_t *config, size_t chunk_size);

/* ── encoding / decoding ──────────────────────────────────────────── */

size_t tk_bpe_encode(const uint8_t *text, size_t text_len,
                     const tk_vocab_t *vocab,
                     uint32_t *out_ids, size_t out_cap);

size_t tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                     const tk_vocab_t *vocab,
                     uint8_t *out, size_t out_cap);

#ifdef __cplusplus
}
#endif

#endif /* TK_BPE_H */