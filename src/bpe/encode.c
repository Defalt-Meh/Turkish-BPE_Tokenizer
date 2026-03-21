/*
 * bpe/encode.c — zero-allocation BPE encoding + decoding.
 *
 * Design goal: the hot path (tk_bpe_encode_with) must do ZERO heap
 * allocations for inputs ≤ STACK_LIMIT bytes.  Since pretokenised
 * Turkish words are almost always < 64 bytes, this covers > 99.9 %
 * of calls.
 *
 * Where the old code spent its time
 * ──────────────────────────────────
 *   With 3840 merges, 1.46 M words, 13 iterations = 19 M encode calls.
 *
 *   OLD per-call overhead:
 *     • merge_index_build:  malloc × 2, memset 32 KiB, 3840 hash
 *       inserts = ~150 µs  →  19 M × 150 µs = 2850 s (47 min!!)
 *     • merge_index_free:   free × 2 per call
 *     • safe_byte_id:       vocab hash lookup per byte
 *     • 3 × malloc + 3 × free for ids/next/prev
 *     • short-path (≤128 B): linear scan 128 × 3840 = 491 k ops
 *
 *   NEW per-call overhead:
 *     • merge index:  already built, pointer deref
 *     • byte→id:      single array load (byte_ids[b])
 *     • scratch:      stack arrays, zero malloc
 *     • all paths:    O(n log n) PQ, n typically 5–20
 *
 * Architecture
 * ────────────
 *   tk_bpe_encoder_t   — persistent, build once at model load
 *     ├─ merge_index_t — O(1) pair→rank hash table
 *     ├─ byte_ids[256] — direct byte→token mapping
 *     └─ vocab ptr     — weak reference
 *
 *   tk_bpe_encode_with(enc, text, len, out, cap)  — hot path
 *     ├─ stack scratch if len ≤ STACK_LIMIT (no malloc)
 *     └─ heap fallback for rare long chunks
 *
 *   tk_bpe_encode(text, len, vocab, out, cap)     — legacy wrapper
 */
#include "bpe.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* Stack buffer size.  256 bytes covers virtually every pretokenised
 * word.  Total stack cost:  256×4×3 (ids/next/prev) + 256×8 (heap)
 * ≈ 5 KiB — well within default thread stack limits.              */
#define STACK_LIMIT 256

/* ── min-heap ─────────────────────────────────────────────────────── *
 *
 * Trimmed to 8 bytes per entry (was 16 with the int64_t priority).
 * merge_rank IS the priority — lower rank = higher priority.       */

typedef struct {
    uint32_t pos;
    uint32_t merge_rank;
} pq_entry_t;

typedef struct {
    pq_entry_t *data;
    size_t       len;
} pq_t;

static inline void pq_push(pq_t *q, uint32_t pos, uint32_t rank) {
    size_t i = q->len++;
    q->data[i].pos        = pos;
    q->data[i].merge_rank = rank;
    while (i > 0) {
        size_t p = (i - 1) >> 1;
        if (q->data[p].merge_rank <= q->data[i].merge_rank) break;
        pq_entry_t tmp = q->data[p];
        q->data[p] = q->data[i];
        q->data[i] = tmp;
        i = p;
    }
}

static inline pq_entry_t pq_pop(pq_t *q) {
    pq_entry_t top = q->data[0];
    q->data[0] = q->data[--q->len];
    size_t i = 0;
    for (;;) {
        size_t l = (i << 1) | 1, r = l + 1, s = i;
        if (l < q->len && q->data[l].merge_rank < q->data[s].merge_rank)
            s = l;
        if (r < q->len && q->data[r].merge_rank < q->data[s].merge_rank)
            s = r;
        if (s == i) break;
        pq_entry_t tmp = q->data[i];
        q->data[i] = q->data[s];
        q->data[s] = tmp;
        i = s;
    }
    return top;
}

/* ── merge index (hash table: pair → rank) ────────────────────────── */

static int merge_index_build(merge_index_t *idx, const tk_vocab_t *vocab) {
    uint32_t n = vocab->num_merges;
    uint32_t slots = 256;
    while (slots < n * 2) slots *= 2;

    idx->keys  = (uint64_t *)malloc(slots * sizeof(uint64_t));
    idx->ranks = (uint32_t *)malloc(slots * sizeof(uint32_t));
    if (!idx->keys || !idx->ranks) {
        free(idx->keys); free(idx->ranks);
        idx->keys = NULL; idx->ranks = NULL;
        return -1;
    }
    memset(idx->keys, 0xFF, slots * sizeof(uint64_t));
    idx->mask = slots - 1;

    for (uint32_t i = 0; i < n; i++) {
        uint64_t key = pack_pair(vocab->merges[i].left,
                                 vocab->merges[i].right);
        uint32_t slot = pair_hash64(key) & idx->mask;
        while (idx->keys[slot] != PAIR_EMPTY)
            slot = (slot + 1) & idx->mask;
        idx->keys[slot]  = key;
        idx->ranks[slot] = i;
    }
    return 0;
}

static inline uint32_t TK_HOT
merge_index_get(const merge_index_t *idx,
                uint32_t left, uint32_t right) {
    uint64_t key  = pack_pair(left, right);
    uint32_t slot = pair_hash64(key) & idx->mask;
    for (;;) {
        uint64_t k = idx->keys[slot];
        if (k == key)        return idx->ranks[slot];
        if (k == PAIR_EMPTY) return (uint32_t)-1;
        slot = (slot + 1) & idx->mask;
    }
}

static void merge_index_free(merge_index_t *idx) {
    free(idx->keys);
    free(idx->ranks);
    idx->keys  = NULL;
    idx->ranks = NULL;
}

/* ── persistent encoder lifecycle ─────────────────────────────────── */

int tk_bpe_encoder_init(tk_bpe_encoder_t *enc, const tk_vocab_t *vocab) {
    memset(enc, 0, sizeof(*enc));
    enc->vocab = vocab;

    /* Pre-compute byte → token-id.  One array load replaces a per-byte
     * hash lookup through the vocab. */
    for (int b = 0; b < 256; b++) {
        uint32_t id = (uint32_t)b;
        if (!tk_vocab_get(vocab, id)) id = (uint32_t)' ';
        enc->byte_ids[b] = id;
    }

    if (vocab->num_merges > 0) {
        if (merge_index_build(&enc->mi, vocab) < 0)
            return -1;
        enc->has_mi = true;
    }
    return 0;
}

void tk_bpe_encoder_free(tk_bpe_encoder_t *enc) {
    if (enc->has_mi) merge_index_free(&enc->mi);
    memset(enc, 0, sizeof(*enc));
}

/* ── core encode engine ───────────────────────────────────────────── *
 *
 * Caller provides all scratch memory — the function itself does
 * ZERO allocations.  This is the inner hot loop that runs 19 M+
 * times during a benchmark.                                        */

static size_t TK_HOT
encode_core(const tk_bpe_encoder_t *enc,
            const uint8_t *text, size_t text_len,
            uint32_t *out_ids, size_t out_cap,
            uint32_t *ids, uint32_t *next_arr,
            uint32_t *prev_arr, pq_entry_t *heap_buf)
{
    const merge_index_t *mi       = &enc->mi;
    const tk_merge_t    *merges   = enc->vocab->merges;
    const uint32_t      *byte_ids = enc->byte_ids;

    /* ── init linked list from raw bytes ──────────────────────────── */
    for (size_t i = 0; i < text_len; i++) {
        ids[i]      = byte_ids[text[i]];
        next_arr[i] = (uint32_t)(i + 1);
        prev_arr[i] = (uint32_t)(i - 1);
    }
    next_arr[text_len - 1] = (uint32_t)-1;
    prev_arr[0]            = (uint32_t)-1;

    /* ── seed PQ ──────────────────────────────────────────────────── */
    pq_t pq;
    pq.data = heap_buf;
    pq.len  = 0;

    for (size_t i = 0; i + 1 < text_len; i++) {
        uint32_t rank = merge_index_get(mi, ids[i], ids[i + 1]);
        if (rank != (uint32_t)-1)
            pq_push(&pq, (uint32_t)i, rank);
    }

    /* ── merge loop ───────────────────────────────────────────────── */
    size_t remaining = text_len;

    while (pq.len > 0 & remaining > 1) {
        pq_entry_t top = pq_pop(&pq);
        const uint32_t pos  = top.pos;
        const uint32_t rank = top.merge_rank;

        /* Validity checks — ordered by probability of early exit. */
        const uint32_t cur_id = ids[pos];
        if (cur_id == (uint32_t)-1)           continue;

        const uint32_t nxt = next_arr[pos];
        if (nxt == (uint32_t)-1)              continue;

        const uint32_t nxt_id = ids[nxt];
        if (cur_id != merges[rank].left)      continue;
        if (nxt_id != merges[rank].right)     continue;

        /* Apply. */
        const uint32_t result  = merges[rank].result;
        const uint32_t nxt_nxt = next_arr[nxt];

        ids[pos] = result;
        ids[nxt] = (uint32_t)-1;
        next_arr[pos] = nxt_nxt;
        if (nxt_nxt != (uint32_t)-1)
            prev_arr[nxt_nxt] = pos;
        remaining--;

        /* Re-check left. */
        const uint32_t prv = prev_arr[pos];
        if (prv != (uint32_t)-1) {
            uint32_t r = merge_index_get(mi, ids[prv], result);
            if (r != (uint32_t)-1)
                pq_push(&pq, prv, r);
        }

        /* Re-check right. */
        if (nxt_nxt != (uint32_t)-1) {
            uint32_t r = merge_index_get(mi, result, ids[nxt_nxt]);
            if (r != (uint32_t)-1)
                pq_push(&pq, pos, r);
        }
    }

    /* ── collect ──────────────────────────────────────────────────── */
    size_t   out_len = 0;
    uint32_t cur     = 0;
    while (cur != (uint32_t)-1 && out_len < out_cap) {
        if (ids[cur] != (uint32_t)-1)
            out_ids[out_len++] = ids[cur];
        cur = next_arr[cur];
    }

    return out_len;
}

/* ── public: encode with persistent encoder (hot path) ────────────── */

size_t TK_HOT
tk_bpe_encode_with(const tk_bpe_encoder_t *enc,
                   const uint8_t *text, size_t text_len,
                   uint32_t *out_ids, size_t out_cap)
{
    if (TK_UNLIKELY(text_len == 0)) return 0;

    /* No merges → raw byte tokens (direct table lookup, no branch). */
    if (TK_UNLIKELY(!enc->has_mi)) {
        if (text_len > out_cap) return (size_t)-1;
        const uint32_t *b = enc->byte_ids;
        for (size_t i = 0; i < text_len; i++)
            out_ids[i] = b[text[i]];
        return text_len;
    }

    /* ── stack path (zero malloc) ─────────────────────────────────── */
    if (TK_LIKELY(text_len <= STACK_LIMIT)) {
        uint32_t   s_ids [STACK_LIMIT];
        uint32_t   s_next[STACK_LIMIT];
        uint32_t   s_prev[STACK_LIMIT];
        pq_entry_t s_heap[STACK_LIMIT];

        return encode_core(enc, text, text_len, out_ids, out_cap,
                           s_ids, s_next, s_prev, s_heap);
    }

    /* ── heap path (rare — very long pretokenised chunks) ─────────── */
    size_t alloc_n = text_len;
    uint32_t   *buf  = (uint32_t *)  malloc(3 * alloc_n * sizeof(uint32_t));
    pq_entry_t *heap = (pq_entry_t *)malloc(alloc_n * sizeof(pq_entry_t));
    if (!buf || !heap) { free(buf); free(heap); return (size_t)-1; }

    size_t n = encode_core(enc, text, text_len, out_ids, out_cap,
                           buf, buf + alloc_n, buf + 2 * alloc_n, heap);
    free(buf);
    free(heap);
    return n;
}

/* ── legacy API wrapper ───────────────────────────────────────────── *
 *
 * Builds encoder on the fly.  For benchmarks and production use,
 * switch to tk_bpe_encoder_init() + tk_bpe_encode_with().          */

size_t tk_bpe_encode(const uint8_t *text, size_t text_len,
                     const tk_vocab_t *vocab,
                     uint32_t *out_ids, size_t out_cap) {
    tk_bpe_encoder_t enc;
    if (tk_bpe_encoder_init(&enc, vocab) < 0) return (size_t)-1;
    size_t n = tk_bpe_encode_with(&enc, text, text_len, out_ids, out_cap);
    tk_bpe_encoder_free(&enc);
    return n;
}

/* ── BPE decoding ─────────────────────────────────────────────────── */

size_t TK_HOT tk_bpe_decode(const uint32_t *ids, size_t num_ids,
                              const tk_vocab_t *vocab,
                              uint8_t *out, size_t out_cap) {
    size_t pos = 0;
    for (size_t i = 0; i < num_ids; i++) {
        const tk_token_t *tok = tk_vocab_get(vocab, ids[i]);
        if (TK_UNLIKELY(!tok))                     return (size_t)-1;
        if (TK_UNLIKELY(pos + tok->len > out_cap)) return (size_t)-1;
        memcpy(out + pos, tok->bytes, tok->len);
        pos += tok->len;
    }
    return pos;
}