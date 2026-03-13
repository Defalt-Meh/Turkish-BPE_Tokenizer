/*
 * bpe/sequence.c — doubly-linked token sequence with pool allocation.
 */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* ── init / free / clear ──────────────────────────────────────────── */

int tk_sequence_init(tk_sequence_t *seq, size_t max_nodes) {
    memset(seq, 0, sizeof(*seq));
    seq->pool = (tk_node_t *)malloc(max_nodes * sizeof(tk_node_t));
    if (!seq->pool) return -1;
    seq->pool_size = max_nodes;
    return 0;
}

void tk_sequence_free(tk_sequence_t *seq) {
    free(seq->pool);
    memset(seq, 0, sizeof(*seq));
}

void tk_sequence_clear(tk_sequence_t *seq) {
    seq->head      = NULL;
    seq->tail      = NULL;
    seq->length    = 0;
    seq->pool_used = 0;
}

/* ── append ───────────────────────────────────────────────────────── */

static inline tk_node_t *seq_alloc(tk_sequence_t *seq) {
    if (TK_UNLIKELY(seq->pool_used >= seq->pool_size)) return NULL;
    return &seq->pool[seq->pool_used++];
}

int tk_sequence_append(tk_sequence_t *seq, uint32_t id) {
    tk_node_t *n = seq_alloc(seq);
    if (!n) return -1;
    n->id   = id;
    n->next = NULL;
    n->prev = seq->tail;
    if (seq->tail) seq->tail->next = n;
    else           seq->head = n;
    seq->tail = n;
    seq->length++;
    return 0;
}

/* ── apply a single merge across the whole sequence ───────────────── */

size_t TK_HOT tk_sequence_apply_merge(tk_sequence_t *seq,
                                       uint32_t left, uint32_t right,
                                       uint32_t result) {
    size_t count = 0;
    tk_node_t *n = seq->head;

    while (n && n->next) {
        if (n->id == left && n->next->id == right) {
            tk_node_t *dead = n->next;
            n->id   = result;
            n->next = dead->next;
            if (dead->next) dead->next->prev = n;
            else            seq->tail = n;
            seq->length--;
            count++;
            n = n->next;
        } else {
            n = n->next;
        }
    }

    return count;
}

/* ── count adjacent pairs into a pair table ────────────────────────── */

void tk_sequence_count_pairs(const tk_sequence_t *seq, tk_pair_table_t *pt) {
    const tk_node_t *n = seq->head;
    while (n && n->next) {
        tk_pair_table_add(pt, n->id, n->next->id, 1);
        n = n->next;
    }
}

/* ── serialise to a flat id array ─────────────────────────────────── */

size_t tk_sequence_to_ids(const tk_sequence_t *seq, uint32_t *out,
                          size_t out_cap) {
    size_t i = 0;
    const tk_node_t *n = seq->head;
    while (n && i < out_cap) {
        out[i++] = n->id;
        n = n->next;
    }
    return i;
}