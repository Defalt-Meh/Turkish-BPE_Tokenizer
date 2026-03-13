/*
 * bpe/encode.c — priority-queue BPE encoding + decoding.
 */
#include "bpe.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* ── min-heap for merge priority queue ────────────────────────────── */

typedef struct {
    uint32_t pos;
    uint32_t merge_idx;
    int64_t  priority;
} pq_entry_t;

typedef struct {
    pq_entry_t *data;
    size_t       len;
    size_t       cap;
} min_heap_t;

static void heap_swap(pq_entry_t *a, pq_entry_t *b) {
    pq_entry_t tmp = *a; *a = *b; *b = tmp;
}

static void heap_push(min_heap_t *h, pq_entry_t e) {
    if (h->len >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->data = (pq_entry_t *)realloc(h->data,
                                         h->cap * sizeof(pq_entry_t));
    }
    h->data[h->len] = e;
    size_t i = h->len++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (h->data[parent].priority <= h->data[i].priority) break;
        heap_swap(&h->data[parent], &h->data[i]);
        i = parent;
    }
}

static pq_entry_t heap_pop(min_heap_t *h) {
    pq_entry_t top = h->data[0];
    h->data[0] = h->data[--h->len];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->len && h->data[l].priority < h->data[smallest].priority)
            smallest = l;
        if (r < h->len && h->data[r].priority < h->data[smallest].priority)
            smallest = r;
        if (smallest == i) break;
        heap_swap(&h->data[i], &h->data[smallest]);
        i = smallest;
    }
    return top;
}

/* ── merge rank lookup ────────────────────────────────────────────── */

static uint32_t find_merge_rank(const tk_vocab_t *vocab,
                                uint32_t left, uint32_t right) {
    for (uint32_t i = 0; i < vocab->num_merges; i++) {
        if (vocab->merges[i].left == left && vocab->merges[i].right == right)
            return i;
    }
    return (uint32_t)-1;
}

/* ── helper: validate byte id against vocab ───────────────────────── */

static inline uint32_t safe_byte_id(const tk_vocab_t *vocab, uint8_t byte) {
    uint32_t id = (uint32_t)byte;
    if (!tk_vocab_get(vocab, id)) id = (uint32_t)' ';
    return id;
}

/* ── BPE encoding ─────────────────────────────────────────────────── */

size_t TK_HOT tk_bpe_encode(const uint8_t *text, size_t text_len,
                              const tk_vocab_t *vocab,
                              uint32_t *out_ids, size_t out_cap) {
    if (text_len == 0) return 0;

    /* Fast path: no merges, just emit byte tokens. */
    if (vocab->num_merges == 0) {
        if (text_len > out_cap) return (size_t)-1;
        for (size_t i = 0; i < text_len; i++)
            out_ids[i] = safe_byte_id(vocab, text[i]);
        return text_len;
    }

    /* Small inputs: linear scan through merge list. */
    if (text_len <= 128) {
        tk_sequence_t seq;
        if (tk_sequence_init(&seq, text_len) < 0) return (size_t)-1;
        for (size_t i = 0; i < text_len; i++)
            tk_sequence_append(&seq, safe_byte_id(vocab, text[i]));
        for (uint32_t m = 0; m < vocab->num_merges; m++) {
            const tk_merge_t *mg = &vocab->merges[m];
            tk_sequence_apply_merge(&seq, mg->left, mg->right, mg->result);
            if (seq.length <= 1) break;
        }
        size_t n = tk_sequence_to_ids(&seq, out_ids, out_cap);
        tk_sequence_free(&seq);
        return n;
    }

    /* General path: priority-queue driven merge. */
    uint32_t *ids      = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    uint32_t *next_arr = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    uint32_t *prev_arr = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    if (!ids || !next_arr || !prev_arr) {
        free(ids); free(next_arr); free(prev_arr);
        return (size_t)-1;
    }

    for (size_t i = 0; i < text_len; i++) {
        ids[i]      = safe_byte_id(vocab, text[i]);
        next_arr[i] = (uint32_t)(i + 1);
        prev_arr[i] = (uint32_t)(i - 1);
    }
    next_arr[text_len - 1] = (uint32_t)-1;
    prev_arr[0]            = (uint32_t)-1;

    min_heap_t heap = { NULL, 0, 0 };

    for (size_t i = 0; i + 1 < text_len; i++) {
        uint32_t rank = find_merge_rank(vocab, ids[i], ids[i + 1]);
        if (rank != (uint32_t)-1) {
            pq_entry_t e = { (uint32_t)i, rank, (int64_t)rank };
            heap_push(&heap, e);
        }
    }

    size_t remaining = text_len;

    while (heap.len > 0 && remaining > 1) {
        pq_entry_t top = heap_pop(&heap);
        uint32_t pos = top.pos;
        uint32_t mi  = top.merge_idx;

        if (ids[pos] == (uint32_t)-1) continue;
        uint32_t nxt = next_arr[pos];
        if (nxt == (uint32_t)-1 || ids[nxt] == (uint32_t)-1) continue;

        if (ids[pos] != vocab->merges[mi].left ||
            ids[nxt] != vocab->merges[mi].right) continue;

        ids[pos] = vocab->merges[mi].result;
        ids[nxt] = (uint32_t)-1;
        next_arr[pos] = next_arr[nxt];
        if (next_arr[nxt] != (uint32_t)-1)
            prev_arr[next_arr[nxt]] = pos;
        remaining--;

        if (prev_arr[pos] != (uint32_t)-1) {
            uint32_t p = prev_arr[pos];
            uint32_t rank = find_merge_rank(vocab, ids[p], ids[pos]);
            if (rank != (uint32_t)-1) {
                pq_entry_t e = { p, rank, (int64_t)rank };
                heap_push(&heap, e);
            }
        }

        if (next_arr[pos] != (uint32_t)-1) {
            uint32_t n2 = next_arr[pos];
            uint32_t rank = find_merge_rank(vocab, ids[pos], ids[n2]);
            if (rank != (uint32_t)-1) {
                pq_entry_t e = { pos, rank, (int64_t)rank };
                heap_push(&heap, e);
            }
        }
    }

    size_t out_len = 0;
    uint32_t cur = 0;
    while (cur != (uint32_t)-1 && out_len < out_cap) {
        if (ids[cur] != (uint32_t)-1)
            out_ids[out_len++] = ids[cur];
        cur = next_arr[cur];
    }

    free(ids);
    free(next_arr);
    free(prev_arr);
    free(heap.data);
    return out_len;
}

/* ── BPE decoding ─────────────────────────────────────────────────── */

size_t TK_HOT tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                              const tk_vocab_t *vocab,
                              uint8_t *out, size_t out_cap) {
    size_t pos = 0;
    for (size_t i = 0; i < num_ids; i++) {
        const tk_token_t *tok = tk_vocab_get(vocab, ids[i]);
        if (TK_UNLIKELY(!tok))                    return (size_t)-1;
        if (TK_UNLIKELY(pos + tok->len > out_cap)) return (size_t)-1;
        memcpy(out + pos, tok->bytes, tok->len);
        pos += tok->len;
    }
    return pos;
}