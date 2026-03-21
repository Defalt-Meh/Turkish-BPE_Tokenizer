/*
 * bpe/pair_table.c — open-addressing hash table for (left,right) pair counts.
 *
 * Improvements over the original:
 *
 *   1. Single contiguous allocation for keys[] and counts[].  One
 *      malloc / free / realloc instead of two; the two arrays land in
 *      adjacent memory so the prefetcher can pull both on the same
 *      cache-line stream.
 *
 *   2. Tombstone-aware entry counting.  Inserting into a PAIR_TOMB
 *      slot no longer inflates num_entries, so the load-factor check
 *      stays accurate across long incremental-update runs.
 *
 *   3. Compacting resize — entries with count ≤ 0 are silently dropped
 *      during resize, acting as a free garbage-collection pass.  This
 *      matters for the incremental training loop in train.c which
 *      decrements counts to zero without removing the key.
 *
 *   4. tk_pair_table_best() no longer uses a static local.  The caller
 *      passes a tk_pair_entry_t* and gets back a bool.  This is
 *      thread-safe and lets the caller hold multiple results.
 *      (A thin wrapper preserves the old return-pointer API so nothing
 *      else needs to change right now.)
 *
 *   5. New tk_pair_table_get() for direct count queries.
 */
#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* ── helpers ──────────────────────────────────────────────────────── */

static uint32_t next_pow2_32(uint32_t v) {
    v--; v |= v >> 1; v |= v >> 2;
    v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/* Allocate a single buffer holding keys[n] followed by counts[n].
 * Returns the raw pointer (caller frees).  Sets *out_keys, *out_counts
 * to the two sub-arrays and fills keys with PAIR_EMPTY. */
static void *alloc_table(uint32_t n, uint64_t **out_keys,
                         int64_t **out_counts) {
    /* Ensure counts[] starts on an 8-byte boundary.  Both uint64_t and
     * int64_t are 8 bytes, and n is always a power of two ≥ 1024, so
     * keys[n] already ends on an aligned address.  No padding needed. */
    size_t total = (size_t)n * sizeof(uint64_t) +
                   (size_t)n * sizeof(int64_t);
    void *buf = malloc(total);
    if (!buf) return NULL;

    *out_keys   = (uint64_t *)buf;
    *out_counts = (int64_t *)((uint64_t *)buf + n);

    memset(*out_keys, 0xFF, n * sizeof(uint64_t));   /* PAIR_EMPTY */
    return buf;
}

/* ── init / free / clear ──────────────────────────────────────────── */

int tk_pair_table_init(tk_pair_table_t *pt, uint32_t min_slots) {
    uint32_t n = next_pow2_32(min_slots < 1024 ? 1024 : min_slots);

    uint64_t *keys;
    int64_t  *counts;
    void *buf = alloc_table(n, &keys, &counts);
    if (!buf) {
        pt->keys = NULL; pt->counts = NULL;
        return -1;
    }

    pt->keys        = keys;
    pt->counts      = counts;
    pt->num_slots   = n;
    pt->mask        = n - 1;
    pt->num_entries = 0;
    return 0;
}

void tk_pair_table_free(tk_pair_table_t *pt) {
    /* keys points to the base of the single allocation. */
    free(pt->keys);
    pt->keys        = NULL;
    pt->counts      = NULL;
    pt->num_entries = 0;
}

void tk_pair_table_clear(tk_pair_table_t *pt) {
    memset(pt->keys, 0xFF, pt->num_slots * sizeof(uint64_t));
    pt->num_entries = 0;
}

/* ── resize (with compaction) ─────────────────────────────────────── */

static int pair_table_resize(tk_pair_table_t *pt) {
    uint32_t  old_n    = pt->num_slots;
    uint64_t *old_keys = pt->keys;
    int64_t  *old_cnt  = pt->counts;

    uint32_t new_n    = old_n * 2;
    uint32_t new_mask = new_n - 1;

    uint64_t *new_keys;
    int64_t  *new_cnt;
    void *buf = alloc_table(new_n, &new_keys, &new_cnt);
    if (!buf) return -1;

    uint32_t live = 0;

    for (uint32_t i = 0; i < old_n; i++) {
        uint64_t k = old_keys[i];
        if (k == PAIR_EMPTY || k == PAIR_TOMB) continue;

        /* Compaction: drop entries whose count has fallen to zero or
         * below (stale artefacts from incremental decrements). */
        if (old_cnt[i] <= 0) continue;

        uint32_t idx = pair_hash64(k) & new_mask;
        while (new_keys[idx] != PAIR_EMPTY)
            idx = (idx + 1) & new_mask;
        new_keys[idx] = k;
        new_cnt[idx]  = old_cnt[i];
        live++;
    }

    /* old_keys is the base of the single allocation. */
    free(old_keys);
    pt->keys        = new_keys;
    pt->counts      = new_cnt;
    pt->num_slots   = new_n;
    pt->mask        = new_mask;
    pt->num_entries = live;
    return 0;
}

/* ── add ──────────────────────────────────────────────────────────── */

int TK_HOT tk_pair_table_add(tk_pair_table_t *pt, uint32_t left,
                              uint32_t right, int64_t count) {
    if (TK_UNLIKELY(pt->num_entries * 2 >= pt->num_slots)) {
        if (pair_table_resize(pt) < 0) return -1;
    }

    uint64_t key = pack_pair(left, right);
    uint32_t idx = pair_hash64(key) & pt->mask;
    uint32_t first_tomb = (uint32_t)-1;

    for (;;) {
        uint64_t k = pt->keys[idx];

        if (k == key) {
            pt->counts[idx] += count;
            return 0;
        }

        if (k == PAIR_TOMB) {
            /* Remember the first tombstone for possible reuse, but
             * keep probing — the key might exist further along. */
            if (first_tomb == (uint32_t)-1)
                first_tomb = idx;
            idx = (idx + 1) & pt->mask;
            continue;
        }

        if (k == PAIR_EMPTY) {
            /* Key not found.  Insert at the first tombstone if we
             * passed one; otherwise insert here.  Reusing a tombstone
             * does NOT increment num_entries — the slot was already
             * counted when it was first occupied. */
            if (first_tomb != (uint32_t)-1) {
                pt->keys[first_tomb]   = key;
                pt->counts[first_tomb] = count;
                /* No num_entries++ : tombstone was already counted. */
            } else {
                pt->keys[idx]   = key;
                pt->counts[idx] = count;
                pt->num_entries++;
            }
            return 0;
        }

        idx = (idx + 1) & pt->mask;
    }
}

/* ── get ──────────────────────────────────────────────────────────── */

int64_t tk_pair_table_get(const tk_pair_table_t *pt,
                          uint32_t left, uint32_t right) {
    uint64_t key = pack_pair(left, right);
    uint32_t idx = pair_hash64(key) & pt->mask;

    for (;;) {
        uint64_t k = pt->keys[idx];
        if (k == key)        return pt->counts[idx];
        if (k == PAIR_EMPTY) return 0;
        idx = (idx + 1) & pt->mask;
    }
}

/* ── best entry (raw count, no morphological scoring) ─────────────── */

bool tk_pair_table_best_r(const tk_pair_table_t *pt,
                          tk_pair_entry_t *out) {
    int64_t  best_count = 0;
    uint64_t best_key   = PAIR_EMPTY;

    for (uint32_t i = 0; i < pt->num_slots; i++) {
        uint64_t k = pt->keys[i];
        if (k == PAIR_EMPTY || k == PAIR_TOMB) continue;
        int64_t c = pt->counts[i];
        if (c > best_count || (c == best_count && k < best_key)) {
            best_count = c;
            best_key   = k;
        }
    }

    if (best_key == PAIR_EMPTY) return false;

    out->left  = pair_left(best_key);
    out->right = pair_right(best_key);
    out->count = best_count;
    out->next  = NULL;
    return true;
}

/* Legacy API wrapper — returns pointer to caller-invisible static.
 * Kept so existing call sites don't need to change yet. */
const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt) {
    static tk_pair_entry_t result;
    if (!tk_pair_table_best_r(pt, &result)) return NULL;
    return &result;
}