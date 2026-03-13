/*
 * bpe/morphology.h — Turkish morphological scoring (internal API).
 * Used by train.c for pair selection.  NOT part of the public API.
 */
#ifndef BPE_MORPHOLOGY_H
#define BPE_MORPHOLOGY_H

#include "internal.h"       /* byte_span_t */
#include "bpe.h"         /* tk_pair_table_t, tk_vocab_t */

#include <stdbool.h>

typedef struct {
    uint32_t left;
    uint32_t right;
    int64_t  raw_count;
    double   score;
} scored_pair_t;

/* Individual queries (also used for verbose logging in train.c). */
bool   pair_respects_vowel_harmony(byte_span_t left, byte_span_t right);
bool   is_morphological_suffix    (byte_span_t span);

/* Morphology-aware scoring of a single pair. */
double morphological_score(byte_span_t left, byte_span_t right, int64_t raw);

/* Scan the whole pair table and return the best-scored pair. */
scored_pair_t find_best_scored(const tk_pair_table_t *pt,
                               const tk_vocab_t *vocab);

#endif /* BPE_MORPHOLOGY_H */