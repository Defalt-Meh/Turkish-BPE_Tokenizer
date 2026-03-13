/*
 * bpe/pair_table.c — open-addressing hash table for (left,right) pair counts.
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

/* ── init / free / clear ──────────────────────────────────────────── */

int tk_pair_table_init(tk_pair_table_t *pt, uint32_t min_slots) {
    uint32_t n = next_pow2_32(min_slots < 1024 ? 1024 : min_slots);
    pt->keys   = (uint64_t *)malloc(n * sizeof(uint64_t));
    pt->counts = (int64_t  *)malloc(n * sizeof(int64_t));
    if (!pt->keys || !pt->counts) {
        free(pt->keys); free(pt->counts);
        pt->keys = NULL; pt->counts = NULL;
        return -1;
    }
    memset(pt->keys, 0xFF, n * sizeof(uint64_t));
    pt->num_slots   = n;
    pt->mask        = n - 1;
    pt->num_entries = 0;
    return 0;
}

void tk_pair_table_free(tk_pair_table_t *pt) {
    free(pt->keys);
    free(pt->counts);
    pt->keys        = NULL;
    pt->counts      = NULL;
    pt->num_entries = 0;
}

void tk_pair_table_clear(tk_pair_table_t *pt) {
    memset(pt->keys, 0xFF, pt->num_slots * sizeof(uint64_t));
    pt->num_entries = 0;
}

/* ── resize ───────────────────────────────────────────────────────── */

static int pair_table_resize(tk_pair_table_t *pt) {
    uint32_t  old_n    = pt->num_slots;
    uint64_t *old_keys = pt->keys;
    int64_t  *old_cnt  = pt->counts;

    uint32_t new_n    = old_n * 2;
    uint32_t new_mask = new_n - 1;

    uint64_t *new_keys = (uint64_t *)malloc(new_n * sizeof(uint64_t));
    int64_t  *new_cnt  = (int64_t  *)malloc(new_n * sizeof(int64_t));
    if (!new_keys || !new_cnt) {
        free(new_keys); free(new_cnt);
        return -1;
    }
    memset(new_keys, 0xFF, new_n * sizeof(uint64_t));

    for (uint32_t i = 0; i < old_n; i++) {
        if (old_keys[i] == PAIR_EMPTY || old_keys[i] == PAIR_TOMB) continue;
        uint32_t idx = pair_hash64(old_keys[i]) & new_mask;
        while (new_keys[idx] != PAIR_EMPTY)
            idx = (idx + 1) & new_mask;
        new_keys[idx] = old_keys[i];
        new_cnt[idx]  = old_cnt[i];
    }

    free(old_keys);
    free(old_cnt);
    pt->keys      = new_keys;
    pt->counts    = new_cnt;
    pt->num_slots = new_n;
    pt->mask      = new_mask;
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

    for (;;) {
        uint64_t k = pt->keys[idx];
        if (k == key) {
            pt->counts[idx] += count;
            return 0;
        }
        if (k == PAIR_EMPTY || k == PAIR_TOMB) {
            pt->keys[idx]   = key;
            pt->counts[idx] = count;
            pt->num_entries++;
            return 0;
        }
        idx = (idx + 1) & pt->mask;
    }
}

/* ── best entry (raw count, no morphological scoring) ─────────────── */

const tk_pair_entry_t *tk_pair_table_best(const tk_pair_table_t *pt) {
    static tk_pair_entry_t result;
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

    if (best_key == PAIR_EMPTY) return NULL;

    result.left  = pair_left(best_key);
    result.right = pair_right(best_key);
    result.count = best_count;
    return &result;
}