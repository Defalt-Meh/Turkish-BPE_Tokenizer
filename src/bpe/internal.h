/*
 * bpe/internal.h — private macros and inline helpers for the BPE subsystem.
 *
 * NOT part of the public API. Include only from src/bpe/*.c files.
 * All public types and function declarations live in ../bpe.h.
 */
#ifndef BPE_INTERNAL_H
#define BPE_INTERNAL_H

#include "bpe.h"        /* public types: tk_pair_table_t, tk_sequence_t, … */

/* ── compiler hints ───────────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
#   define TK_LIKELY(x)   __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#   define TK_HOT          __attribute__((hot))
#else
#   define TK_LIKELY(x)   (x)
#   define TK_UNLIKELY(x) (x)
#   define TK_HOT
#endif

/* ── sentinel values for the pair hash table ──────────────────────── */

#define PAIR_EMPTY  ((uint64_t)-1)
#define PAIR_TOMB   ((uint64_t)-2)

/* ── pair key packing ─────────────────────────────────────────────── */

static inline uint64_t pack_pair(uint32_t left, uint32_t right) {
    return ((uint64_t)left << 32) | (uint64_t)right;
}

static inline uint32_t pair_left(uint64_t key) {
    return (uint32_t)(key >> 32);
}

static inline uint32_t pair_right(uint64_t key) {
    return (uint32_t)(key & 0xFFFFFFFF);
}

static inline uint32_t pair_hash64(uint64_t key) {
    key ^= key >> 33;
    key *= 0xFF51AFD7ED558CCDull;
    key ^= key >> 33;
    key *= 0xC4CEB9FE1A85EC53ull;
    key ^= key >> 33;
    return (uint32_t)key;
}

/* ── byte_span helper (used by morphology + train) ────────────────── */

typedef struct {
    const uint8_t *bytes;
    uint16_t       len;
} byte_span_t;

#endif /* BPE_INTERNAL_H */