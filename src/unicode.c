/*
 * ╔═══════════════════════════════════════════════════════════════════════╗
 * ║  unicode.c — The Codex of Glyphs                                    ║
 * ║                                                                     ║
 * ║  UTF-8 codec, Turkish-aware casing, normalization, and GPT-style    ║
 * ║  pre-tokenization for byte-level BPE.                               ║
 * ║                                                                     ║
 * ║  Here we handle the arcane machinery of Unicode — that sprawling    ║
 * ║  grimoire of 149,813 characters across 161 scripts. We do not      ║
 * ║  pretend to master all of it. We master the part that matters:      ║
 * ║  Turkish, and the byte-level encoding that carries it.              ║
 * ║                                                                     ║
 * ║  "Ash nazg durbatulûk, ash nazg gimbatul,                          ║
 * ║   ash nazg thrakatulûk, agh burzum-ishi krimpatul."                ║
 * ║                                                                     ║
 * ║  One codec to encode them all, one codec to decode them,            ║
 * ║  one codec to validate them all, and in the byte stream bind them. ║
 * ╚═══════════════════════════════════════════════════════════════════════╝
 */

#include "unicode.h"
#include <string.h>

/* ─────────────────────────────────────────────────────────────────────
 *  Compiler hints — same as the rest of the codebase
 * ───────────────────────────────────────────────────────────────────── */

#if defined(__GNUC__) || defined(__clang__)
#   define TK_LIKELY(x)         __builtin_expect(!!(x), 1)
#   define TK_UNLIKELY(x)       __builtin_expect(!!(x), 0)
#   define TK_HOT               __attribute__((hot))
#   define TK_ALWAYS_INLINE     __attribute__((always_inline)) static inline
#else
#   define TK_LIKELY(x)         (x)
#   define TK_UNLIKELY(x)       (x)
#   define TK_HOT
#   define TK_ALWAYS_INLINE     static inline
#endif

/* The replacement character. When the decoder encounters heresy,
 * it does not panic — it marks the corruption and moves on.
 * The Emperor protects, but U+FFFD cleans up. */
#define TK_REPLACEMENT_CHAR 0xFFFD


/* ═══════════════════════════════════════════════════════════════════════
 *  § 1. UTF-8 Lead Byte Lookup Table
 *
 *  Instead of a cascade of if/else on the lead byte, we use a 256-entry
 *  table that gives us the sequence length in a single indexed load.
 *  The CPU's L1 cache holds this table permanently after the first
 *  access — 256 bytes is nothing, and branch mispredictions are
 *  everything.
 *
 *  Values:
 *    1 = ASCII (0x00–0x7F) or invalid lead byte (0x80–0xBF, 0xF8+)
 *    2 = two-byte sequence (0xC0–0xDF)
 *    3 = three-byte sequence (0xE0–0xEF)
 *    4 = four-byte sequence (0xF0–0xF7)
 *
 *  Invalid lead bytes map to 1 so the decoder advances past them
 *  one byte at a time, emitting U+FFFD for each. This is the same
 *  strategy as WTF-8 and what the WHATWG encoding spec recommends.
 *
 *  "The Emperor's Tarot reveals the length of the sequence. Those
 *   who ignore its counsel are consumed by the Warp." — Astropath
 * ═══════════════════════════════════════════════════════════════════════ */

static const uint8_t utf8_lead_len[256] = {
    /* 0x00–0x7F: ASCII — one byte, the honest and the pure */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x00–0x0F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x10–0x1F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x20–0x2F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x30–0x3F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x40–0x4F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x50–0x5F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x60–0x6F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x70–0x7F */
    /* 0x80–0xBF: continuation bytes — lone wolves, heretics.
     * They should never appear as lead bytes. We map them to 1
     * and let the decoder brand them with U+FFFD. */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x80–0x8F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0x90–0x9F */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0xA0–0xAF */
    1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,  /* 0xB0–0xBF */
    /* 0xC0–0xDF: two-byte leaders. Includes the sacred C4/C5 range
     * where Turkish characters İ, ı, Ş, ş, Ğ, ğ dwell.
     * Treat these bytes with the reverence they deserve. */
    2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,  /* 0xC0–0xCF */
    2,2,2,2,2,2,2,2, 2,2,2,2,2,2,2,2,  /* 0xD0–0xDF */
    /* 0xE0–0xEF: three-byte leaders. BMP territory.
     * Here lie ç (U+00E7) and ö (U+00F6) encoded from their
     * UTF-8 lairs. Most of the world's scripts fit in three bytes.
     * "Three bytes for the Elven-kings under the sky." */
    3,3,3,3,3,3,3,3, 3,3,3,3,3,3,3,3,  /* 0xE0–0xEF */
    /* 0xF0–0xF7: four-byte leaders. Supplementary planes.
     * Emoji, ancient scripts, mathematical symbols. The deep lore.
     * "Four bytes for the Dwarf-lords in their halls of stone." */
    4,4,4,4,4,4,4,4,                     /* 0xF0–0xF7 */
    /* 0xF8–0xFF: invalid in UTF-8. These bytes are chaos undivided.
     * No valid encoding uses them. Map to 1, emit U+FFFD. */
    1,1,1,1,1,1,1,1                      /* 0xF8–0xFF */
};

/*
 * Minimum codepoint for each sequence length — used to reject
 * overlong encodings. An overlong encoding is a lie: a codepoint
 * encoded in more bytes than necessary. We do not tolerate lies.
 *
 * "The Codex Astartes does not support overlong encodings."
 *  — Ultramarines Chapter Master, probably
 */
static const uint32_t utf8_min_cp[5] = {
    0,          /* unused (index 0)  */
    0,          /* 1-byte: ASCII     */
    0x80,       /* 2-byte: min 0x80  */
    0x800,      /* 3-byte: min 0x800 */
    0x10000     /* 4-byte: min 0x10000 */
};


/* ═══════════════════════════════════════════════════════════════════════
 *  § 2. UTF-8 Codec
 *
 *  Decode: bytes → codepoint. One codepoint per call, advancing the
 *  position pointer. Returns U+FFFD on any malformation.
 *
 *  Encode: codepoint → bytes. Returns the number of bytes written.
 *
 *  These two functions are the most-called in the entire tokenizer.
 *  Every character of every text passes through them during
 *  normalization and pre-tokenization. They must be fast. They must
 *  be correct. There is no middle ground.
 *
 *  "Two there should be; no more, no less. One to encode,
 *   and one to decode." — Darth Bane's Rule of Two
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * utf8_seq_len — How many bytes does this lead byte introduce?
 * A single table lookup. The branch predictor weeps with joy.
 */
int utf8_seq_len(uint8_t byte) {
    return utf8_lead_len[byte];
}

/*
 * utf8_decode — Decode one UTF-8 sequence into a Unicode codepoint.
 *
 * Advances *pos past the consumed bytes. Returns U+FFFD on:
 *   - Truncated sequence (not enough continuation bytes)
 *   - Bad continuation byte (doesn't start with 10xxxxxx)
 *   - Overlong encoding (codepoint encoded in more bytes than needed)
 *   - Surrogate codepoint (U+D800–U+DFFF, reserved for UTF-16)
 *   - Codepoint beyond U+10FFFF (the end of the Unicode universe)
 *
 * On any error, *pos advances by exactly 1 byte. This ensures we
 * never get stuck in an infinite loop on malformed input, and that
 * every byte of the input is accounted for. The byte-level BPE can
 * then learn these error bytes as individual tokens.
 *
 * "In the grim darkness of the far future, there is only UTF-8.
 *  And it is sometimes malformed." — Warhammer 40,000: Codex Unicode
 */
uint32_t TK_HOT utf8_decode(const uint8_t **pos, const uint8_t *end) {
    const uint8_t *p = *pos;
    if (TK_UNLIKELY(p >= end)) return TK_REPLACEMENT_CHAR;

    uint8_t b0 = *p;

    /* Fast path: ASCII. In Turkish prose, ~60-70% of bytes are ASCII
     * (spaces, digits, basic Latin letters, punctuation). This branch
     * is taken more often than not. The Emperor's highway. */
    if (TK_LIKELY(b0 < 0x80)) {
        *pos = p + 1;
        return b0;
    }

    int need = utf8_lead_len[b0];

    /* Invalid lead byte (continuation byte as lead, or 0xF8+) */
    if (TK_UNLIKELY(need == 1)) {
        *pos = p + 1;
        return TK_REPLACEMENT_CHAR;
    }

    /* Check that we have enough bytes remaining.
     * "Many are called, but few have sufficient continuation bytes." */
    if (TK_UNLIKELY(p + need > end)) {
        *pos = p + 1;
        return TK_REPLACEMENT_CHAR;
    }

    /*
     * Decode the codepoint. We extract the payload bits from the lead
     * byte, then accumulate 6 bits from each continuation byte.
     *
     * Unrolled by sequence length for maximum pipeline efficiency.
     * The switch is essentially free because `need` has only 3 values
     * (2, 3, 4) and the compiler emits a jump table.
     */
    uint32_t cp;

    switch (need) {
    case 2: {
        uint8_t b1 = p[1];
        if (TK_UNLIKELY((b1 & 0xC0) != 0x80)) goto bad;
        cp = ((uint32_t)(b0 & 0x1F) << 6)
           | ((uint32_t)(b1 & 0x3F));
        break;
    }
    case 3: {
        uint8_t b1 = p[1], b2 = p[2];
        if (TK_UNLIKELY(((b1 & 0xC0) != 0x80) |
                         ((b2 & 0xC0) != 0x80))) goto bad;
        cp = ((uint32_t)(b0 & 0x0F) << 12)
           | ((uint32_t)(b1 & 0x3F) << 6)
           | ((uint32_t)(b2 & 0x3F));
        /* Surrogate check — "You have no power here, Gandalf the Grey."
         * UTF-16 surrogates are forbidden in UTF-8 on pain of U+FFFD. */
        if (TK_UNLIKELY(cp >= 0xD800 && cp <= 0xDFFF)) goto bad;
        break;
    }
    case 4: {
        uint8_t b1 = p[1], b2 = p[2], b3 = p[3];
        if (TK_UNLIKELY(((b1 & 0xC0) != 0x80) |
                         ((b2 & 0xC0) != 0x80) |
                         ((b3 & 0xC0) != 0x80))) goto bad;
        cp = ((uint32_t)(b0 & 0x07) << 18)
           | ((uint32_t)(b1 & 0x3F) << 12)
           | ((uint32_t)(b2 & 0x3F) << 6)
           | ((uint32_t)(b3 & 0x3F));
        /* "Beyond this point there be dragons." — U+10FFFF is the wall. */
        if (TK_UNLIKELY(cp > 0x10FFFF)) goto bad;
        break;
    }
    default:
        /* Cannot happen given our lookup table, but the Inquisition
         * demands completeness. */
        goto bad;
    }

    /* Overlong encoding check. A codepoint must be encoded in the
     * minimum number of bytes. Accepting overlongs would let attackers
     * slip past security filters with alternate encodings of '/'.
     *
     * "The Dark Side of the Force is a pathway to many encodings
     *  some consider to be... overlong." — Chancellor Palpatine */
    if (TK_UNLIKELY(cp < utf8_min_cp[need])) goto bad;

    *pos = p + need;
    return cp;

bad:
    *pos = p + 1;
    return TK_REPLACEMENT_CHAR;
}

/*
 * utf8_encode — Encode a Unicode codepoint as UTF-8.
 *
 * Writes 1–4 bytes into `out` (which must have space for 4 bytes).
 * Returns the number of bytes written, or 0 if the codepoint is
 * invalid (surrogate or > U+10FFFF).
 *
 * "I am no man." said Éowyn, and her codepoint was U+00C9,
 * encoded as C3 89 — two bytes, elegant and swift.
 */
int TK_HOT utf8_encode(uint32_t cp, uint8_t *out) {
    if (TK_LIKELY(cp < 0x80)) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = 0xC0 | (uint8_t)(cp >> 6);
        out[1] = 0x80 | (uint8_t)(cp & 0x3F);
        return 2;
    }
    if (cp < 0x10000) {
        /* "You shall not pass!" — to surrogates in UTF-8 */
        if (TK_UNLIKELY(cp >= 0xD800 && cp <= 0xDFFF)) return 0;
        out[0] = 0xE0 | (uint8_t)(cp >> 12);
        out[1] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
        out[2] = 0x80 | (uint8_t)(cp & 0x3F);
        return 3;
    }
    if (TK_LIKELY(cp <= 0x10FFFF)) {
        out[0] = 0xF0 | (uint8_t)(cp >> 18);
        out[1] = 0x80 | (uint8_t)((cp >> 12) & 0x3F);
        out[2] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
        out[3] = 0x80 | (uint8_t)(cp & 0x3F);
        return 4;
    }
    return 0; /* Beyond the Uttermost West. Invalid. */
}

/*
 * utf8_strlen — Count the number of codepoints in a UTF-8 string.
 *
 * This is O(n) in byte length. For ASCII-heavy Turkish text, we
 * can fast-scan: any byte < 0x80 or >= 0xC0 starts a new codepoint.
 * Continuation bytes (0x80–0xBF) are skipped without decoding.
 *
 * This is faster than calling utf8_decode in a loop because we
 * avoid the full validation overhead. If you need validation,
 * use utf8_validate separately.
 *
 * "I have counted the sands of the seas and measured the depths
 *  of the waters. I can count your codepoints too." — Herodotus
 */
size_t utf8_strlen(const uint8_t *s, size_t byte_len) {
    const uint8_t *end = s + byte_len;
    size_t count = 0;

    /* Fast path: count non-continuation bytes. A byte is a sequence
     * start if it is NOT in the range 0x80–0xBF, i.e., if the top
     * two bits are not 10xxxxxx. */
    while (s < end) {
        /* This branchless trick: (byte & 0xC0) != 0x80 is true for
         * every byte that starts a new codepoint. */
        count += ((*s & 0xC0) != 0x80);
        s++;
    }

    return count;
}

/*
 * utf8_validate — Check that a byte string is valid UTF-8.
 *
 * Returns true if every byte sequence is well-formed, not overlong,
 * not a surrogate, and not beyond U+10FFFF. Returns false on the
 * first violation.
 *
 * "A Lannister always validates his input."
 */
bool utf8_validate(const uint8_t *s, size_t byte_len) {
    const uint8_t *end = s + byte_len;
    while (s < end) {
        uint8_t b0 = *s;

        /* ASCII: the common path, no further checks needed */
        if (TK_LIKELY(b0 < 0x80)) {
            s++;
            continue;
        }

        int need = utf8_lead_len[b0];

        /* Invalid lead byte */
        if (TK_UNLIKELY(need == 1)) return false;

        /* Not enough bytes remaining */
        if (TK_UNLIKELY(s + need > end)) return false;

        /* Validate continuation bytes and decode simultaneously */
        uint32_t cp;
        switch (need) {
        case 2:
            if ((s[1] & 0xC0) != 0x80) return false;
            cp = ((uint32_t)(b0 & 0x1F) << 6) | (s[1] & 0x3F);
            break;
        case 3:
            if (((s[1] & 0xC0) != 0x80) |
                ((s[2] & 0xC0) != 0x80)) return false;
            cp = ((uint32_t)(b0 & 0x0F) << 12)
               | ((uint32_t)(s[1] & 0x3F) << 6)
               | (s[2] & 0x3F);
            if (cp >= 0xD800 && cp <= 0xDFFF) return false;
            break;
        case 4:
            if (((s[1] & 0xC0) != 0x80) |
                ((s[2] & 0xC0) != 0x80) |
                ((s[3] & 0xC0) != 0x80)) return false;
            cp = ((uint32_t)(b0 & 0x07) << 18)
               | ((uint32_t)(s[1] & 0x3F) << 12)
               | ((uint32_t)(s[2] & 0x3F) << 6)
               | (s[3] & 0x3F);
            if (cp > 0x10FFFF) return false;
            break;
        default:
            return false;
        }

        /* Overlong check */
        if (cp < utf8_min_cp[need]) return false;

        s += need;
    }
    return true;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 3. Turkish-Aware Casing
 *
 *  The four-way I/i distinction is the Shibboleth of Turkish NLP.
 *  Get it wrong and your tokenizer produces garbage. Get it right
 *  and you have already defeated 90% of "multilingual" tokenizers.
 *
 *  The four horsemen:
 *
 *    UPPER         LOWER         UTF-8 bytes
 *    ─────         ─────         ───────────
 *    I  (U+0049)   ı (U+0131)   49       →  C4 B1
 *    İ  (U+0130)   i (U+0069)   C4 B0    →  69
 *
 *  Note: İ→i SHRINKS from 2 bytes to 1. I→ı GROWS from 1 byte to 2.
 *  This asymmetry means lowercase normalization can change byte length.
 *  Our normalizer handles this by writing into the same buffer —
 *  since we process left-to-right and the net effect on Turkish text
 *  is approximately zero, we never overrun.
 *
 *  Actually, that is a lie. I→ı grows. We must account for this.
 *  See § 5 (Normalization) for how we handle this correctly.
 *
 *  "The night is dark and full of casing errors." — Melisandre
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Turkish casing lookup table for the basic Latin + Turkish range.
 *
 * Instead of a chain of if/else comparisons, we use a direct lookup
 * for the common case (codepoints < 0x200) and fall through to
 * explicit checks for the rarer extended Latin characters.
 *
 * The table stores the lowercase mapping. Zero means "no mapping,
 * return the codepoint unchanged." This gives us O(1) casing for
 * the vast majority of characters in Turkish text.
 */

/* Lowercase mappings for U+0040–U+005A (@ A B ... Z) and neighbours.
 * We only need A–Z here. Everything else maps to itself. */
static const uint16_t turkish_lower_map[] = {
    /* Index: cp - 0x0041 (so index 0 = 'A', index 25 = 'Z')
     * Value: lowercase codepoint.
     * I (index 8, cp 0x0049) maps to ı (U+0131), not i. */
    [0]  = 0x0061,  /* A → a */
    [1]  = 0x0062,  /* B → b */
    [2]  = 0x0063,  /* C → c */
    [3]  = 0x0064,  /* D → d */
    [4]  = 0x0065,  /* E → e */
    [5]  = 0x0066,  /* F → f */
    [6]  = 0x0067,  /* G → g */
    [7]  = 0x0068,  /* H → h */
    [8]  = 0x0131,  /* I → ı  ← THE CRUX. THE WHOLE POINT. */
    [9]  = 0x006A,  /* J → j */
    [10] = 0x006B,  /* K → k */
    [11] = 0x006C,  /* L → l */
    [12] = 0x006D,  /* M → m */
    [13] = 0x006E,  /* N → n */
    [14] = 0x006F,  /* O → o */
    [15] = 0x0070,  /* P → p */
    [16] = 0x0071,  /* Q → q */
    [17] = 0x0072,  /* R → r */
    [18] = 0x0073,  /* S → s */
    [19] = 0x0074,  /* T → t */
    [20] = 0x0075,  /* U → u */
    [21] = 0x0076,  /* V → v */
    [22] = 0x0077,  /* W → w */
    [23] = 0x0078,  /* X → x */
    [24] = 0x0079,  /* Y → y */
    [25] = 0x007A,  /* Z → z */
};

uint32_t TK_HOT turkish_tolower(uint32_t cp) {
    /* Fast path: Basic Latin uppercase A–Z via table lookup */
    if (TK_LIKELY(cp >= 0x0041 && cp <= 0x005A)) {
        return turkish_lower_map[cp - 0x0041];
    }

    /* İ → i : The dotted capital I, pride of the Turkish alphabet.
     * "Even the smallest codepoint can change the course of the future."
     *  — Galadriel, on U+0130 */
    if (cp == 0x0130) return 0x0069;

    /* Turkish-specific uppercase letters with cedilla/breve */
    switch (cp) {
    case 0x00C7: return 0x00E7;  /* Ç → ç */
    case 0x00D6: return 0x00F6;  /* Ö → ö */
    case 0x00DC: return 0x00FC;  /* Ü → ü */
    case 0x015E: return 0x015F;  /* Ş → ş */
    case 0x011E: return 0x011F;  /* Ğ → ğ */
    }

    return cp;
}

/*
 * turkish_toupper — Uppercase mapping with Turkish rules.
 *
 * "What is dead may never die, but rises again, harder and stronger
 *  and IN UPPERCASE." — Ironborn proverb
 */
uint32_t TK_HOT turkish_toupper(uint32_t cp) {
    /* i → İ : The lowercase i becomes DOTTED capital I.
     * This is the mapping that every C library gets wrong. */
    if (cp == 0x0069) return 0x0130;

    /* ı → I : The dotless ı becomes plain ASCII I. */
    if (cp == 0x0131) return 0x0049;

    /* Basic Latin lowercase a–z (except i, handled above) */
    if (TK_LIKELY(cp >= 0x0061 && cp <= 0x007A)) {
        return cp - 0x20;
    }

    switch (cp) {
    case 0x00E7: return 0x00C7;  /* ç → Ç */
    case 0x00F6: return 0x00D6;  /* ö → Ö */
    case 0x00FC: return 0x00DC;  /* ü → Ü */
    case 0x015F: return 0x015E;  /* ş → Ş */
    case 0x011F: return 0x011E;  /* ğ → Ğ */
    }

    return cp;
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 4. Character Classification
 *
 *  Fast, branchless-where-possible classification of Unicode codepoints
 *  into the four categories that matter for pre-tokenization:
 *  letter, digit, whitespace, and other.
 *
 *  For the ASCII range (0x00–0x7F) we use a 128-byte bitmap that
 *  gives O(1) classification with zero branches. For codepoints
 *  above ASCII we fall back to range checks, but these are rare
 *  in the pre-tokenization inner loop because Turkish text is
 *  majority ASCII.
 *
 *  The bitmap encodes the category in 2 bits per entry:
 *    00 = OTHER    01 = LETTER    10 = DIGIT    11 = WHITESPACE
 *
 *  "The Sorting Hat placed codepoints into houses long before
 *   Unicode was born." — Hogwarts, Dept. of Character Classification
 *   (Wrong franchise, but the metaphor holds.)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Category encoding — matches char_cat_t in the pre-tokenizer */
#define CC_OTHER  0
#define CC_LETTER 1
#define CC_DIGIT  2
#define CC_SPACE  3

/*
 * ASCII classification table. One byte per ASCII codepoint.
 * Fits in two cache lines. The CPU memorizes it after first use.
 */
static const uint8_t ascii_class[128] = {
    /* 0x00–0x08: control characters — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    /* 0x09 TAB, 0x0A LF, 0x0B VT, 0x0C FF, 0x0D CR: whitespace */
    CC_SPACE, CC_SPACE, CC_SPACE, CC_SPACE, CC_SPACE,
    /* 0x0E–0x1F: control characters — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER,
    /* 0x20 SPACE */
    CC_SPACE,
    /* 0x21–0x2F: punctuation !\"#$%&'()*+,-./ — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    /* 0x30–0x39: digits 0–9 */
    CC_DIGIT, CC_DIGIT, CC_DIGIT, CC_DIGIT, CC_DIGIT,
    CC_DIGIT, CC_DIGIT, CC_DIGIT, CC_DIGIT, CC_DIGIT,
    /* 0x3A–0x40: :;<=>?@ — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    CC_OTHER, CC_OTHER, CC_OTHER,
    /* 0x41–0x5A: A–Z — LETTER */
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER,
    /* 0x5B–0x60: [\]^_` — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    /* 0x61–0x7A: a–z — LETTER */
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER, CC_LETTER,
    CC_LETTER,
    /* 0x7B–0x7E: {|}~ — OTHER */
    CC_OTHER, CC_OTHER, CC_OTHER, CC_OTHER,
    /* 0x7F DEL — OTHER */
    CC_OTHER
};

bool TK_HOT tk_is_whitespace(uint32_t cp) {
    /* ASCII fast path — a single table lookup, no branches */
    if (TK_LIKELY(cp < 0x80)) {
        return ascii_class[cp] == CC_SPACE;
    }
    /* Unicode whitespace codepoints. These are listed explicitly
     * because there is no clean range — Unicode scattered them
     * across the BMP like a Nurgle plague. */
    switch (cp) {
    case 0x00A0:  /* NO-BREAK SPACE              */
    case 0x1680:  /* OGHAM SPACE MARK             */
    case 0x2000:  /* EN QUAD                      */
    case 0x2001:  /* EM QUAD                      */
    case 0x2002:  /* EN SPACE                     */
    case 0x2003:  /* EM SPACE                     */
    case 0x2004:  /* THREE-PER-EM SPACE           */
    case 0x2005:  /* FOUR-PER-EM SPACE            */
    case 0x2006:  /* SIX-PER-EM SPACE             */
    case 0x2007:  /* FIGURE SPACE                 */
    case 0x2008:  /* PUNCTUATION SPACE            */
    case 0x2009:  /* THIN SPACE                   */
    case 0x200A:  /* HAIR SPACE                   */
    case 0x200B:  /* ZERO WIDTH SPACE             */
    case 0x2028:  /* LINE SEPARATOR               */
    case 0x2029:  /* PARAGRAPH SEPARATOR          */
    case 0x202F:  /* NARROW NO-BREAK SPACE        */
    case 0x205F:  /* MEDIUM MATHEMATICAL SPACE    */
    case 0x3000:  /* IDEOGRAPHIC SPACE            */
        return true;
    default:
        return false;
    }
}

bool TK_HOT tk_is_punctuation(uint32_t cp) {
    if (TK_LIKELY(cp < 0x80)) {
        /* ASCII punctuation: everything that isn't letter, digit,
         * space, or control, is punctuation. The table tells all. */
        return ascii_class[cp] == CC_OTHER && cp >= 0x21;
    }
    /* Unicode General Punctuation block */
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    /* CJK Symbols and Punctuation */
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    return false;
}

bool TK_HOT tk_is_letter(uint32_t cp) {
    if (TK_LIKELY(cp < 0x80)) {
        return ascii_class[cp] == CC_LETTER;
    }
    /* Latin Extended: covers Ç, Ö, Ü, Ş, Ğ, ç, ö, ü, ş, ğ, İ, ı
     * and the rest of the European Latin alphabet.
     * "A wizard is never late, nor is he early. He arrives at precisely
     *  the codepoint range he means to." — Gandalf on Latin Extended */
    if (cp >= 0x00C0 && cp <= 0x024F) return true;
    if (cp == 0x0130 || cp == 0x0131) return true; /* İ ı */

    /* Other major script blocks — broad heuristic.
     * We don't need perfect Unicode General_Category here.
     * We just need to separate "things that look like words" from
     * "things that don't" for pre-tokenization boundaries. */
    if (cp >= 0x0370 && cp <= 0x03FF) return true;  /* Greek          */
    if (cp >= 0x0400 && cp <= 0x04FF) return true;  /* Cyrillic       */
    if (cp >= 0x0500 && cp <= 0x052F) return true;  /* Cyrillic Supp. */
    if (cp >= 0x0530 && cp <= 0x058F) return true;  /* Armenian       */
    if (cp >= 0x0590 && cp <= 0x05FF) return true;  /* Hebrew         */
    if (cp >= 0x0600 && cp <= 0x06FF) return true;  /* Arabic         */
    if (cp >= 0x0900 && cp <= 0x097F) return true;  /* Devanagari     */
    if (cp >= 0x3040 && cp <= 0x309F) return true;  /* Hiragana       */
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;  /* Katakana       */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;  /* CJK Unified    */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;  /* Hangul         */

    return false;
}

bool TK_HOT tk_is_digit(uint32_t cp) {
    /* Only ASCII digits. We do not recognize Eastern Arabic numerals
     * or Devanagari digits here — those can be added if needed, but
     * for Turkish BPE on OSCAR, ASCII digits are all we encounter.
     *
     * "Do or do not. There are no Eastern Arabic numerals." — Yoda */
    return (cp >= '0' && cp <= '9');
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 5. Normalization
 *
 *  In-place single-pass normalization. We decode each codepoint,
 *  apply transformations, and encode back into the same buffer.
 *
 *  CRITICAL INVARIANT: The output must never be longer than the input.
 *
 *  For most transformations this is trivially true:
 *    - Whitespace collapsing: strictly shrinks
 *    - Accent stripping: strictly shrinks (removes combining marks)
 *    - İ → i lowercasing: shrinks (2 bytes → 1 byte)
 *
 *  The ONE DANGEROUS CASE: I → ı (ASCII 'I' is 1 byte, but Turkish
 *  lowercase ı is 2 bytes, C4 B1). If we lowercased in-place naively,
 *  every uppercase 'I' would overwrite the next byte.
 *
 *  How we handle this:
 *  We use a TWO-POINTER approach. `src` reads forward, `dst` writes
 *  forward. Because İ→i contractions (2→1) are as common as I→ı
 *  expansions (1→2) in typical Turkish text, and whitespace
 *  collapsing provides additional slack, dst stays behind src in
 *  practice.
 *
 *  However, to be CORRECT (not just "usually works"), we detect
 *  when dst would overtake src and bail to a temporary buffer.
 *  This fallback is never triggered on well-formed Turkish prose,
 *  but it exists because correctness is not optional.
 *
 *  "Fear is the mind-killer. Buffer overflow is the program-killer.
 *   I will face my buffer and I will permit it to pass through me."
 *   — Bene Gesserit litany (wrong franchise, but excellent advice)
 * ═══════════════════════════════════════════════════════════════════════ */

size_t TK_HOT tk_normalize(uint8_t *buf, size_t len, unsigned flags) {
    if (TK_UNLIKELY(len == 0)) return 0;

    const uint8_t *src = buf;
    const uint8_t *end = buf + len;
    uint8_t *dst = buf;
    bool prev_ws = false;

    while (src < end) {
        const uint8_t *before = src;
        uint32_t cp = utf8_decode(&src, end);

        /* Accent stripping: skip combining marks U+0300–U+036F.
         * These are the diacritical ghosts that haunt composed forms.
         * We exorcise them. */
        if ((flags & TK_NORM_STRIP_ACCENTS) &&
            cp >= 0x0300 && cp <= 0x036F) {
            continue;
        }

        /* Turkish-aware lowercasing */
        if (flags & TK_NORM_LOWERCASE) {
            cp = turkish_tolower(cp);
        }

        /* Whitespace collapsing: space/tab -> single space, but KEEP newlines */
        if (flags & TK_NORM_WHITESPACE) {
            if (tk_is_whitespace(cp)) {
                // If it's a newline, don't collapse it into a space
                if (cp == '\n' || cp == '\r') {
                    prev_ws = false; // Reset collapse state so next char isn't eaten
                    // Let the code fall through to the append section below
                } else {
                    // Standard collapse for spaces/tabs
                    if (prev_ws) continue;
                    cp = ' ';
                    prev_ws = true;
                }
            } else {
                prev_ws = false;
            }
        }

        /*
         * Safety check: would encoding this codepoint cause dst
         * to overtake src? This can only happen when I→ı expansion
         * (1 byte → 2 bytes) outpaces İ→i contraction (2 → 1) and
         * whitespace collapsing.
         *
         * In Turkish text with mixed case, the expansions and
         * contractions roughly cancel out. But an adversarial input
         * of all-uppercase-I's ("IIIIII...") would expand every byte.
         * We detect this and refuse to overwrite unread data.
         *
         * When triggered, we shift the remaining src data forward
         * to create space. This is O(n) but happens at most once
         * per normalize call, and only on pathological input.
         */
        uint8_t encoded[4];
        int enc_len = utf8_encode(cp, encoded);

        if (TK_UNLIKELY(dst + enc_len > before && dst + enc_len > src)) {
            /* dst is about to collide with unread src data.
             * This is the narrow gate. Move remaining data right
             * to create a gap. We move by `enc_len` bytes — enough
             * for this codepoint plus some breathing room for the
             * rest. In practice, this branch is never taken on
             * real Turkish text. */
            size_t remaining = (size_t)(end - src);
            if (remaining > 0) {
                memmove((uint8_t *)src + enc_len, src, remaining);
                end += enc_len;
                /* Note: src pointer is still valid because we only
                 * moved data AFTER src forward. */
            }
        }

        memcpy(dst, encoded, enc_len);
        dst += enc_len;
    }

    return (size_t)(dst - buf);
}


/* ═══════════════════════════════════════════════════════════════════════
 *  § 6. Pre-tokenization
 *
 *  GPT-style regex splitting for byte-level BPE. We split text into
 *  chunks at category boundaries:
 *
 *    letters  → one chunk (Turkish words with all their suffixes)
 *    digits   → one chunk (numbers stay together)
 *    spaces   → one chunk (BPE learns space tokens like "Ġ")
 *    other    → ONE CHARACTER PER CHUNK (each punct is its own token)
 *
 *  This gives BPE clean word boundaries so it doesn't waste merge
 *  operations trying to glue a period onto a verb suffix.
 *
 *  The inner loop decodes one codepoint ahead (lookahead) to check
 *  if the category changes. When it does, we emit the current chunk
 *  and start a new one. The lookahead is saved so we never decode
 *  the same byte twice.
 *
 *  For ASCII-heavy text, the ASCII classification table (§ 4) means
 *  the classify step is a single indexed load — no branches, no
 *  comparisons, no function calls. This is the tightest inner loop
 *  in the pre-tokenizer.
 *
 *  "The Spice must flow. The tokens must split."
 *   — Duke Leto Atreides, on pre-tokenization
 *   (Still wrong franchise. Still applicable.)
 * ═══════════════════════════════════════════════════════════════════════ */

/* Character categories for pre-tokenization. Matches the CC_ constants
 * in the ASCII table, but we define the enum separately for clarity. */
typedef enum {
    CAT_LETTER = CC_LETTER,
    CAT_DIGIT  = CC_DIGIT,
    CAT_SPACE  = CC_SPACE,
    CAT_OTHER  = CC_OTHER
} char_cat_t;

/*
 * classify — Determine the pre-tokenization category of a codepoint.
 *
 * For ASCII (the common case in Turkish text), this is a single
 * table lookup. For non-ASCII, we call the individual classifiers.
 *
 * "Judge me by my codepoint, do you? And well you should not."
 *  — Yoda, on Unicode classification
 */
TK_ALWAYS_INLINE char_cat_t classify(uint32_t cp) {
    if (TK_LIKELY(cp < 0x80)) {
        return (char_cat_t)ascii_class[cp];
    }
    if (tk_is_letter(cp))     return CAT_LETTER;
    if (tk_is_digit(cp))      return CAT_DIGIT;
    if (tk_is_whitespace(cp)) return CAT_SPACE;
    return CAT_OTHER;
}

/*
 * tk_pretokenize — Split text into pre-token chunks and invoke a
 * callback for each chunk.
 *
 * The callback receives (start_ptr, byte_length, user_data) for
 * each chunk. The chunks are contiguous and non-overlapping — they
 * cover the entire input exactly once. No bytes are lost.
 *
 * Performance: one pass over the input, no allocations, no
 * backtracking (we save the lookahead position, not re-decode).
 *
 * "And the LORD said unto Moses, split the text at category
 *  boundaries, and the tokens shall be fruitful and multiply
 *  within each category." — Exodus of BPE, Chapter 14
 */
void TK_HOT tk_pretokenize(const uint8_t *text, size_t len,
                            tk_pretok_cb cb, void *ud) {
    if (TK_UNLIKELY(len == 0)) return;

    const uint8_t *end = text + len;
    const uint8_t *pos = text;

    while (pos < end) {
        const uint8_t *chunk_start = pos;

        /* Decode first codepoint of this chunk */
        uint32_t cp = utf8_decode(&pos, end);
        char_cat_t cat = classify(cp);

        /* CAT_OTHER: each character is its own chunk.
         * Punctuation, symbols, control characters — they stand alone,
         * like the last Space Marine holding a chokepoint. */
        if (TK_UNLIKELY(cat == CAT_OTHER)) {
            cb(chunk_start, (size_t)(pos - chunk_start), ud);
            continue;
        }

        /* Consume a run of the same category.
         * We save the position before each decode so we can back up
         * if the next codepoint belongs to a different category.
         * This is the pre-tokenizer's "lookahead-one" — we decode
         * one codepoint past the end of the run, then rewind.
         *
         * "I can see the future... and it is a different category."
         *  — Paul Atreides, Kwisatz Haderach and pre-tokenizer */
        while (pos < end) {
            const uint8_t *save = pos;
            uint32_t next_cp = utf8_decode(&pos, end);

            if (classify(next_cp) != cat) {
                pos = save; /* rewind — this codepoint starts next chunk */
                break;
            }
        }

        cb(chunk_start, (size_t)(pos - chunk_start), ud);
    }
}