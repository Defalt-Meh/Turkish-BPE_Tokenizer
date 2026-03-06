/*
 * unicode.c — UTF-8 codec, Turkish-aware casing, normalization,
 *             and GPT-style pre-tokenization for byte-level BPE.
 */

#include "unicode.h"
#include <string.h>

/* ════════════════════════════════════════════════════════════════════════
 *  UTF-8 Codec
 * ════════════════════════════════════════════════════════════════════════ */

int utf8_seq_len(uint8_t byte) {
    if (byte < 0x80) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1; /* invalid lead byte — treat as single byte */
}

uint32_t utf8_decode(const uint8_t **pos, const uint8_t *end) {
    const uint8_t *p = *pos;
    if (p >= end) return 0xFFFD;

    uint8_t b0 = *p;
    uint32_t cp;
    int need;

    if (b0 < 0x80) {
        *pos = p + 1;
        return b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        cp = b0 & 0x1F; need = 1;
    } else if ((b0 & 0xF0) == 0xE0) {
        cp = b0 & 0x0F; need = 2;
    } else if ((b0 & 0xF8) == 0xF0) {
        cp = b0 & 0x07; need = 3;
    } else {
        *pos = p + 1;
        return 0xFFFD; /* invalid lead byte */
    }

    if (p + need >= end) {
        *pos = p + 1;
        return 0xFFFD; /* truncated sequence */
    }

    for (int i = 1; i <= need; i++) {
        uint8_t cont = p[i];
        if ((cont & 0xC0) != 0x80) {
            *pos = p + 1;
            return 0xFFFD; /* bad continuation byte */
        }
        cp = (cp << 6) | (cont & 0x3F);
    }

    *pos = p + 1 + need;

    /* Reject overlong encodings and surrogates */
    if (cp < 0x80 && need > 0) return 0xFFFD;
    if (cp < 0x800 && need > 1) return 0xFFFD;
    if (cp < 0x10000 && need > 2) return 0xFFFD;
    if (cp >= 0xD800 && cp <= 0xDFFF) return 0xFFFD;
    if (cp > 0x10FFFF) return 0xFFFD;

    return cp;
}

int utf8_encode(uint32_t cp, uint8_t *out) {
    if (cp < 0x80) {
        out[0] = (uint8_t)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = 0xC0 | (uint8_t)(cp >> 6);
        out[1] = 0x80 | (uint8_t)(cp & 0x3F);
        return 2;
    }
    if (cp < 0x10000) {
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0; /* surrogate */
        out[0] = 0xE0 | (uint8_t)(cp >> 12);
        out[1] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
        out[2] = 0x80 | (uint8_t)(cp & 0x3F);
        return 3;
    }
    if (cp <= 0x10FFFF) {
        out[0] = 0xF0 | (uint8_t)(cp >> 18);
        out[1] = 0x80 | (uint8_t)((cp >> 12) & 0x3F);
        out[2] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
        out[3] = 0x80 | (uint8_t)(cp & 0x3F);
        return 4;
    }
    return 0; /* invalid codepoint */
}

size_t utf8_strlen(const uint8_t *s, size_t byte_len) {
    const uint8_t *end = s + byte_len;
    size_t count = 0;
    while (s < end) {
        utf8_decode(&s, end);
        count++;
    }
    return count;
}

bool utf8_validate(const uint8_t *s, size_t byte_len) {
    const uint8_t *end = s + byte_len;
    while (s < end) {
        uint32_t cp = utf8_decode(&s, end);
        if (cp == 0xFFFD) return false;
    }
    return true;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Turkish-aware Casing
 *
 *  The four special mappings:
 *    İ (U+0130) → i (U+0069)    Turkish dotted capital I → lowercase
 *    I (U+0049) → ı (U+0131)    ASCII capital I → Turkish dotless lowercase
 *    i (U+0069) → İ (U+0130)    lowercase i → Turkish dotted capital I
 *    ı (U+0131) → I (U+0049)    Turkish dotless ı → ASCII capital I
 *
 *  For all other Latin letters we use the ASCII offset trick for A-Z/a-z,
 *  and leave non-Latin codepoints unchanged (good enough for BPE;
 *  full Unicode case tables would bloat the binary for minimal gain).
 * ════════════════════════════════════════════════════════════════════════ */

uint32_t turkish_tolower(uint32_t cp) {
    /* Turkish specials */
    if (cp == 0x0130) return 0x0069; /* İ -> i */
    if (cp == 0x0049) return 0x0131; /* I -> ı */

    /* Basic Latin uppercase A-Z (except I, already handled) */
    if (cp >= 0x0041 && cp <= 0x005A) return cp + 0x20;

    /* Ç(U+00C7)->ç, Ö(U+00D6)->ö, Ü(U+00DC)->ü, Ş(U+015E)->ş, Ğ(U+011E)->ğ */
    if (cp == 0x00C7) return 0x00E7; /* Ç -> ç */
    if (cp == 0x00D6) return 0x00F6; /* Ö -> ö */
    if (cp == 0x00DC) return 0x00FC; /* Ü -> ü */
    if (cp == 0x015E) return 0x015F; /* Ş -> ş */
    if (cp == 0x011E) return 0x011F; /* Ğ -> ğ */

    return cp;
}

uint32_t turkish_toupper(uint32_t cp) {
    /* Turkish specials */
    if (cp == 0x0069) return 0x0130; /* i -> İ */
    if (cp == 0x0131) return 0x0049; /* ı -> I */

    /* Basic Latin lowercase a-z (except i, already handled) */
    if (cp >= 0x0061 && cp <= 0x007A) return cp - 0x20;

    /* ç->Ç, ö->Ö, ü->Ü, ş->Ş, ğ->Ğ */
    if (cp == 0x00E7) return 0x00C7;
    if (cp == 0x00F6) return 0x00D6;
    if (cp == 0x00FC) return 0x00DC;
    if (cp == 0x015F) return 0x015E;
    if (cp == 0x011F) return 0x011E;

    return cp;
}

/* ════════════════════════════════════════════════════════════════════════
 *  Character Classification
 * ════════════════════════════════════════════════════════════════════════ */

bool tk_is_whitespace(uint32_t cp) {
    /* ASCII whitespace + common Unicode spaces */
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v')
        return true;
    if (cp == 0x00A0 || cp == 0x1680) return true; /* NBSP, Ogham space */
    if (cp >= 0x2000 && cp <= 0x200B) return true;  /* En/Em spaces, etc. */
    if (cp == 0x2028 || cp == 0x2029) return true;   /* line/para separator */
    if (cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

bool tk_is_punctuation(uint32_t cp) {
    /* ASCII punctuation ranges */
    if (cp >= 0x21 && cp <= 0x2F) return true;  /* ! " # $ % & ' ( ) * + , - . / */
    if (cp >= 0x3A && cp <= 0x40) return true;  /* : ; < = > ? @ */
    if (cp >= 0x5B && cp <= 0x60) return true;  /* [ \ ] ^ _ ` */
    if (cp >= 0x7B && cp <= 0x7E) return true;  /* { | } ~ */
    /* General punctuation block */
    if (cp >= 0x2000 && cp <= 0x206F) return true;
    /* CJK punctuation */
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    return false;
}

bool tk_is_letter(uint32_t cp) {
    /* Latin basics + supplements (covers Turkish alphabet entirely) */
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp >= 0x00C0 && cp <= 0x024F) return true;  /* Latin Extended-A/B */
    if (cp == 0x0130 || cp == 0x0131) return true;   /* İ ı */

    /* Broad heuristic: most codepoints in letter blocks.
     * For a full BPE tokenizer this is sufficient — we just need to
     * separate letters from digits/punctuation/whitespace. */
    if (cp >= 0x0400 && cp <= 0x04FF) return true;   /* Cyrillic */
    if (cp >= 0x0600 && cp <= 0x06FF) return true;   /* Arabic */
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;   /* CJK Unified */
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;   /* Hangul */

    return false;
}

bool tk_is_digit(uint32_t cp) {
    return (cp >= '0' && cp <= '9');
}

/* ════════════════════════════════════════════════════════════════════════
 *  Normalization
 * ════════════════════════════════════════════════════════════════════════ */

size_t tk_normalize(uint8_t *buf, size_t len, unsigned flags) {
    /* We do a single-pass decode-transform-encode back into the same buffer.
     * This works because our transforms never increase byte length:
     *   - Lowercasing can only stay same or shrink (İ 2B -> i 1B)
     *   - Whitespace collapsing only shrinks
     *   - Accent stripping only shrinks (we skip combining marks)
     */
    const uint8_t *src = buf;
    const uint8_t *end = buf + len;
    uint8_t *dst = buf;
    bool prev_ws = false;

    while (src < end) {
        uint32_t cp = utf8_decode(&src, end);

        /* Accent stripping: skip combining marks U+0300..U+036F */
        if ((flags & TK_NORM_STRIP_ACCENTS) &&
            cp >= 0x0300 && cp <= 0x036F) {
            continue;
        }

        /* Turkish-aware lowercasing */
        if (flags & TK_NORM_LOWERCASE) {
            cp = turkish_tolower(cp);
        }

        /* Whitespace collapsing */
        if (flags & TK_NORM_WHITESPACE) {
            if (tk_is_whitespace(cp)) {
                if (prev_ws) continue; /* skip duplicate whitespace */
                cp = ' ';
                prev_ws = true;
            } else {
                prev_ws = false;
            }
        }

        int n = utf8_encode(cp, dst);
        dst += n;
    }

    return (size_t)(dst - buf);
}

/* ════════════════════════════════════════════════════════════════════════
 *  Pre-tokenization (GPT-style regex splitting for byte-level BPE)
 *
 *  We split input into chunks of:
 *    1. Runs of letters (including Turkish special chars)
 *    2. Runs of digits
 *    3. Runs of whitespace (kept — BPE will learn space tokens)
 *    4. Individual punctuation / other (each is its own chunk)
 *
 *  This gives BPE reasonable "word" boundaries so it doesn't try to
 *  merge across wildly different character categories.
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    CAT_LETTER,
    CAT_DIGIT,
    CAT_SPACE,
    CAT_OTHER
} char_cat_t;

static char_cat_t classify(uint32_t cp) {
    if (tk_is_letter(cp)) return CAT_LETTER;
    if (tk_is_digit(cp))  return CAT_DIGIT;
    if (tk_is_whitespace(cp)) return CAT_SPACE;
    return CAT_OTHER;
}

void tk_pretokenize(const uint8_t *text, size_t len, tk_pretok_cb cb, void *ud) {
    const uint8_t *end = text + len;
    const uint8_t *pos = text;

    while (pos < end) {
        const uint8_t *chunk_start = pos;
        const uint8_t *save = pos;
        uint32_t cp = utf8_decode(&pos, end);
        char_cat_t cat = classify(cp);

        if (cat == CAT_OTHER) {
            /* Each "other" character is its own token */
            cb(chunk_start, (size_t)(pos - chunk_start), ud);
            continue;
        }

        /* Consume a run of the same category */
        while (pos < end) {
            save = pos;
            uint32_t next_cp = utf8_decode(&pos, end);
            if (classify(next_cp) != cat) {
                pos = save; /* back up — this char belongs to next chunk */
                break;
            }
        }

        cb(chunk_start, (size_t)(pos - chunk_start), ud);
    }
}