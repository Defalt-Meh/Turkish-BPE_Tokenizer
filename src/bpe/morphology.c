/*
 * bpe/morphology.c — Turkish morphological scoring for BPE pair selection.
 */
#include "morphology.h"
#include "internal.h"
#include "unicode.h"
#include "bpe.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* ── byte_span helpers ────────────────────────────────────────────── */

static bool span_eq(byte_span_t a, const char *s) {
    size_t slen = strlen(s);
    if (a.len != slen) return false;
    return memcmp(a.bytes, s, slen) == 0;
}

/* (kept for future use) */
static bool span_ends_with(byte_span_t span, const char *suffix) {
    size_t slen = strlen(suffix);
    if (span.len < slen) return false;
    return memcmp(span.bytes + span.len - slen, suffix, slen) == 0;
    (void)span_ends_with; /* suppress unused warning */
}

/* ── vowel classification ─────────────────────────────────────────── */

static bool is_turkish_vowel_byte(uint8_t b) {
    switch (b) {
    case 'a': case 'e': case 'o': case 'u':
    case 'A': case 'E': case 'O': case 'U':
    case 'i': case 'I':
        return true;
    default:
        return false;
    }
}

static bool is_turkish_vowel_cp(uint32_t cp) {
    switch (cp) {
    case 'a': case 'e': case 'i': case 'o': case 'u':
    case 'A': case 'E': case 'I': case 'O': case 'U':
    case 0x00F6: case 0x00FC: case 0x0131: case 0x00D6:
    case 0x00DC: case 0x0130:
        return true;
    default:
        return false;
    }
}

static bool is_back_vowel(uint32_t cp) {
    switch (cp) {
    case 'a': case 'A':
    case 0x0131:            /* ı */
    case 'o': case 'O':
    case 'u': case 'U':
        return true;
    default:
        return false;
    }
}

static bool is_front_vowel(uint32_t cp) {
    switch (cp) {
    case 'e': case 'E':
    case 'i': case 0x0130:    /* İ */
    case 0x00F6: case 0x00D6: /* ö Ö */
    case 0x00FC: case 0x00DC: /* ü Ü */
        return true;
    default:
        return false;
    }
}

static uint32_t last_vowel_in_span(byte_span_t span) {
    const uint8_t *p   = span.bytes;
    const uint8_t *end = p + span.len;
    uint32_t last_v = 0;
    while (p < end) {
        uint32_t cp = utf8_decode(&p, end);
        if (is_turkish_vowel_cp(cp)) last_v = cp;
    }
    return last_v;
}

/* ── vowel harmony ────────────────────────────────────────────────── */

bool pair_respects_vowel_harmony(byte_span_t left_span,
                                 byte_span_t right_span) {
    uint32_t lv = last_vowel_in_span(left_span);
    if (lv == 0) return true;

    const uint8_t *p   = right_span.bytes;
    const uint8_t *end = p + right_span.len;
    uint32_t first_rv  = 0;
    while (p < end) {
        uint32_t cp = utf8_decode(&p, end);
        if (is_turkish_vowel_cp(cp)) { first_rv = cp; break; }
    }
    if (first_rv == 0) return true;

    if (is_back_vowel(lv)  && is_back_vowel(first_rv))  return true;
    if (is_front_vowel(lv) && is_front_vowel(first_rv)) return true;

    return false;
}

/* ── Turkish suffix table ─────────────────────────────────────────── */

static const char *turkish_suffix_table[] = {
    /* case suffixes */
    "de", "da", "den", "dan", "te", "ta", "ten", "tan",
    "ye", "ya", "e", "a",
    "in", "un", "\xC4\xB1n", "\xC3\xBCn",
    "nin", "nun", "n\xC4\xB1n", "n\xC3\xBCn",
    /* plural */
    "ler", "lar",
    /* possessive */
    "im", "\xC4\xB1m", "um", "\xC3\xBCm",
    "in", "\xC4\xB1n", "un", "\xC3\xBCn",
    "si", "s\xC4\xB1",
    "miz", "m\xC4\xB1z", "muz", "m\xC3\xBCz",
    "niz", "n\xC4\xB1z", "nuz", "n\xC3\xBCz",
    "leri", "lar\xC4\xB1",
    /* tense */
    "di", "d\xC4\xB1", "du", "d\xC3\xBC",
    "ti", "t\xC4\xB1", "tu", "t\xC3\xBC",
    "mis", "m\xC4\xB1s", "mus", "m\xC3\xBCs",
    "mi\xC5\x9F", "m\xC4\xB1\xC5\x9F",
    "yor", "iyor", "\xC4\xB1yor", "uyor", "\xC3\xBCyor",
    "ecek", "acak",
    "ir", "\xC4\xB1r", "ur", "\xC3\xBCr",
    "er", "ar",
    /* verbal noun / participle */
    "mek", "mak",
    "en", "an",
    "dik", "d\xC4\xB1k", "duk", "d\xC3\xBCk",
    "tik", "t\xC4\xB1k", "tuk", "t\xC3\xBCk",
    /* ability */
    "ebil", "abil",
    /* negative */
    "me", "ma",
    /* copula */
    "dir", "d\xC4\xB1r", "dur", "d\xC3\xBCr",
    "tir", "t\xC4\xB1r", "tur", "t\xC3\xBCr",
    /* question */
    "mi", "m\xC4\xB1", "mu", "m\xC3\xBC",
    NULL
};

bool is_morphological_suffix(byte_span_t span) {
    for (int i = 0; turkish_suffix_table[i]; i++) {
        if (span_eq(span, turkish_suffix_table[i])) return true;
    }
    return false;
}

/* ── merge / cluster helpers ──────────────────────────────────────── */

static bool merge_forms_suffix(byte_span_t left, byte_span_t right) {
    if (left.len + right.len > TK_MAX_TOKEN_LEN) return false;
    uint8_t combined[TK_MAX_TOKEN_LEN];
    memcpy(combined, left.bytes, left.len);
    memcpy(combined + left.len, right.bytes, right.len);
    byte_span_t comb = { combined, (uint16_t)(left.len + right.len) };
    return is_morphological_suffix(comb);
}

static bool is_consonant_cluster_break(byte_span_t left, byte_span_t right) {
    if (left.len == 0 || right.len == 0) return false;
    uint8_t lb = left.bytes[left.len - 1];
    uint8_t rb = right.bytes[0];
    if (!is_turkish_vowel_byte(lb) && !is_turkish_vowel_byte(rb))
        return true;
    return false;
}

/* ── morphological scoring ────────────────────────────────────────── */

double morphological_score(byte_span_t left_span,
                           byte_span_t right_span,
                           int64_t raw_count) {
    double score = (double)raw_count;

    if (pair_respects_vowel_harmony(left_span, right_span))
        score *= 1.15;

    if (is_morphological_suffix(right_span))
        score *= 1.25;

    if (merge_forms_suffix(left_span, right_span))
        score *= 1.30;

    if (is_consonant_cluster_break(left_span, right_span))
        score *= 0.85;

    bool left_has_vowel  = (last_vowel_in_span(left_span) != 0);
    bool right_has_vowel = (last_vowel_in_span(right_span) != 0);
    if (left_has_vowel && right_has_vowel)
        score *= 1.05;

    if (left_span.len >= 2 && right_span.len >= 2)
        score *= 1.02;

    return score;
}

/* ── scored best-pair selection ───────────────────────────────────── */

scored_pair_t find_best_scored(const tk_pair_table_t *pt,
                               const tk_vocab_t *vocab) {
    scored_pair_t best = { 0, 0, 0, -1.0 };

    for (uint32_t i = 0; i < pt->num_slots; i++) {
        uint64_t k = pt->keys[i];
        if (k == PAIR_EMPTY || k == PAIR_TOMB) continue;

        int64_t count = pt->counts[i];
        if (count < 2) continue;

        uint32_t l = pair_left(k);
        uint32_t r = pair_right(k);

        const tk_token_t *lt = tk_vocab_get(vocab, l);
        const tk_token_t *rt = tk_vocab_get(vocab, r);
        if (!lt || !rt) continue;

        byte_span_t ls = { lt->bytes, lt->len };
        byte_span_t rs = { rt->bytes, rt->len };

        double sc = morphological_score(ls, rs, count);

        if (sc > best.score ||
            (sc == best.score && k < pack_pair(best.left, best.right))) {
            best.left      = l;
            best.right     = r;
            best.raw_count = count;
            best.score     = sc;
        }
    }

    return best;
}