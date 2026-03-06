/*
 * test_unicode.c — Tests for UTF-8 codec, Turkish casing, normalization,
 *                  character classification, and pre-tokenization.
 */

#include "unicode.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL [%d]: %s\n", tests_run, msg); \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ── UTF-8 Codec ────────────────────────────────────────────────────── */

static void test_utf8_encode_decode(void) {
    fprintf(stderr, "  utf8 encode/decode...\n");

    /* ASCII */
    uint8_t buf[4];
    int n = utf8_encode('A', buf);
    ASSERT(n == 1 && buf[0] == 0x41, "encode ASCII 'A'");

    const uint8_t *p = buf;
    uint32_t cp = utf8_decode(&p, buf + n);
    ASSERT(cp == 'A', "decode ASCII 'A'");

    /* Turkish İ (U+0130) → 2 bytes: C4 B0 */
    n = utf8_encode(0x0130, buf);
    ASSERT(n == 2 && buf[0] == 0xC4 && buf[1] == 0xB0, "encode İ U+0130");

    p = buf;
    cp = utf8_decode(&p, buf + n);
    ASSERT(cp == 0x0130, "decode İ U+0130");

    /* Turkish ı (U+0131) → 2 bytes: C4 B1 */
    n = utf8_encode(0x0131, buf);
    ASSERT(n == 2 && buf[0] == 0xC4 && buf[1] == 0xB1, "encode ı U+0131");

    p = buf;
    cp = utf8_decode(&p, buf + n);
    ASSERT(cp == 0x0131, "decode ı U+0131");

    /* 3-byte: Euro sign € (U+20AC) */
    n = utf8_encode(0x20AC, buf);
    ASSERT(n == 3, "encode € is 3 bytes");
    p = buf;
    cp = utf8_decode(&p, buf + n);
    ASSERT(cp == 0x20AC, "decode € U+20AC");

    /* 4-byte: Emoji 😀 (U+1F600) */
    n = utf8_encode(0x1F600, buf);
    ASSERT(n == 4, "encode 😀 is 4 bytes");
    p = buf;
    cp = utf8_decode(&p, buf + n);
    ASSERT(cp == 0x1F600, "decode 😀 U+1F600");

    /* Invalid: surrogate */
    n = utf8_encode(0xD800, buf);
    ASSERT(n == 0, "reject surrogate U+D800");

    /* Beyond Unicode */
    n = utf8_encode(0x110000, buf);
    ASSERT(n == 0, "reject U+110000");
}

static void test_utf8_seq_len(void) {
    fprintf(stderr, "  utf8 seq_len...\n");

    ASSERT(utf8_seq_len(0x00) == 1, "NUL is 1 byte");
    ASSERT(utf8_seq_len(0x41) == 1, "ASCII 'A' is 1 byte");
    ASSERT(utf8_seq_len(0x7F) == 1, "0x7F is 1 byte");
    ASSERT(utf8_seq_len(0xC4) == 2, "0xC4 lead is 2 bytes");
    ASSERT(utf8_seq_len(0xE2) == 3, "0xE2 lead is 3 bytes");
    ASSERT(utf8_seq_len(0xF0) == 4, "0xF0 lead is 4 bytes");
    ASSERT(utf8_seq_len(0x80) == 1, "continuation 0x80 treated as 1");
    ASSERT(utf8_seq_len(0xFF) == 1, "invalid 0xFF treated as 1");
}

static void test_utf8_strlen_validate(void) {
    fprintf(stderr, "  utf8 strlen/validate...\n");

    /* "İstanbul" in UTF-8 */
    const uint8_t istanbul[] = {
        0xC4, 0xB0,                         /* İ */
        0x73, 0x74, 0x61, 0x6E, 0x62, 0x75, 0x6C  /* stanbul */
    };
    ASSERT(utf8_strlen(istanbul, sizeof(istanbul)) == 8, "İstanbul is 8 codepoints");
    ASSERT(utf8_validate(istanbul, sizeof(istanbul)), "İstanbul is valid UTF-8");

    /* Invalid sequence */
    const uint8_t bad[] = { 0xC4 }; /* truncated 2-byte seq */
    ASSERT(!utf8_validate(bad, 1), "truncated sequence is invalid");

    /* Empty */
    ASSERT(utf8_strlen((const uint8_t *)"", 0) == 0, "empty string is 0 codepoints");
}

/* ── Turkish Casing ──────────────────────────────────────────────────── */

static void test_turkish_casing(void) {
    fprintf(stderr, "  turkish casing...\n");

    /* The four special mappings */
    ASSERT(turkish_tolower(0x0130) == 0x0069, "İ -> i");
    ASSERT(turkish_tolower(0x0049) == 0x0131, "I -> ı");
    ASSERT(turkish_toupper(0x0069) == 0x0130, "i -> İ");
    ASSERT(turkish_toupper(0x0131) == 0x0049, "ı -> I");

    /* Regular Latin should still work */
    ASSERT(turkish_tolower('A') == 'a', "A -> a");
    ASSERT(turkish_tolower('Z') == 'z', "Z -> z");
    ASSERT(turkish_toupper('a') == 'A', "a -> A");
    ASSERT(turkish_toupper('z') == 'Z', "z -> Z");

    /* Turkish-specific letters */
    ASSERT(turkish_tolower(0x00C7) == 0x00E7, "Ç -> ç");
    ASSERT(turkish_tolower(0x015E) == 0x015F, "Ş -> ş");
    ASSERT(turkish_tolower(0x011E) == 0x011F, "Ğ -> ğ");
    ASSERT(turkish_tolower(0x00D6) == 0x00F6, "Ö -> ö");
    ASSERT(turkish_tolower(0x00DC) == 0x00FC, "Ü -> ü");

    ASSERT(turkish_toupper(0x00E7) == 0x00C7, "ç -> Ç");
    ASSERT(turkish_toupper(0x015F) == 0x015E, "ş -> Ş");
    ASSERT(turkish_toupper(0x011F) == 0x011E, "ğ -> Ğ");
    ASSERT(turkish_toupper(0x00F6) == 0x00D6, "ö -> Ö");
    ASSERT(turkish_toupper(0x00FC) == 0x00DC, "ü -> Ü");

    /* Non-letter should pass through */
    ASSERT(turkish_tolower('5') == '5', "digit unchanged");
    ASSERT(turkish_toupper('!') == '!', "punct unchanged");
}

/* ── Character Classification ────────────────────────────────────────── */

static void test_classification(void) {
    fprintf(stderr, "  character classification...\n");

    ASSERT(tk_is_whitespace(' '), "space is whitespace");
    ASSERT(tk_is_whitespace('\n'), "newline is whitespace");
    ASSERT(tk_is_whitespace('\t'), "tab is whitespace");
    ASSERT(tk_is_whitespace(0x00A0), "NBSP is whitespace");
    ASSERT(!tk_is_whitespace('a'), "'a' not whitespace");

    ASSERT(tk_is_punctuation('.'), "'.' is punctuation");
    ASSERT(tk_is_punctuation(','), "',' is punctuation");
    ASSERT(tk_is_punctuation('!'), "'!' is punctuation");
    ASSERT(!tk_is_punctuation('a'), "'a' not punctuation");

    ASSERT(tk_is_letter('a'), "'a' is letter");
    ASSERT(tk_is_letter('Z'), "'Z' is letter");
    ASSERT(tk_is_letter(0x0130), "İ is letter");
    ASSERT(tk_is_letter(0x0131), "ı is letter");
    ASSERT(!tk_is_letter('5'), "'5' not letter");
    ASSERT(!tk_is_letter(' '), "space not letter");

    ASSERT(tk_is_digit('0'), "'0' is digit");
    ASSERT(tk_is_digit('9'), "'9' is digit");
    ASSERT(!tk_is_digit('a'), "'a' not digit");
}

/* ── Normalization ───────────────────────────────────────────────────── */

static void test_normalization(void) {
    fprintf(stderr, "  normalization...\n");

    /* Whitespace collapsing */
    uint8_t ws[] = "hello   world\t\nnew";
    size_t len = tk_normalize(ws, strlen((char *)ws), TK_NORM_WHITESPACE);
    ws[len] = '\0';
    ASSERT(strcmp((char *)ws, "hello world new") == 0,
           "whitespace collapsed to single spaces");

    /* Turkish lowercase: "İSTANBUL" → "istanbul"
     * İ(C4 B0) S T A N B U L → i s t a n b u l
     * But I → ı(C4 B1), so we need to be careful.
     * Actually: İ→i, S→s, T→t, A→a, N→n, B→b, U→u, L→l
     */
    uint8_t city[] = {0xC4, 0xB0, 'S', 'T', 'A', 'N', 'B', 'U', 'L', 0};
    size_t city_len = 9; /* not counting NUL */
    size_t new_len = tk_normalize(city, city_len, TK_NORM_LOWERCASE);
    city[new_len] = '\0';
    /* İ(2 bytes) becomes i(1 byte), so total shrinks by 1 */
    ASSERT(new_len == 8, "İSTANBUL lowercase is 8 bytes");
    ASSERT(strcmp((char *)city, "istanbul") == 0, "İSTANBUL -> istanbul");
}

/* ── Pre-tokenization ────────────────────────────────────────────────── */

typedef struct {
    char chunks[32][64];
    int count;
} pretok_result_t;

static void pretok_cb(const uint8_t *start, size_t len, void *ud) {
    pretok_result_t *r = (pretok_result_t *)ud;
    if (r->count < 32 && len < 64) {
        memcpy(r->chunks[r->count], start, len);
        r->chunks[r->count][len] = '\0';
        r->count++;
    }
}

static void test_pretokenize(void) {
    fprintf(stderr, "  pre-tokenization...\n");

    const char *text = "Hello, world! 123";
    pretok_result_t r;
    memset(&r, 0, sizeof(r));

    tk_pretokenize((const uint8_t *)text, strlen(text), pretok_cb, &r);

    /* Expected chunks: "Hello" "," " " "world" "!" " " "123" */
    ASSERT(r.count == 7, "pretokenize 'Hello, world! 123' -> 7 chunks");

    if (r.count >= 7) {
        ASSERT(strcmp(r.chunks[0], "Hello") == 0, "chunk 0 = 'Hello'");
        ASSERT(strcmp(r.chunks[1], ",") == 0, "chunk 1 = ','");
        ASSERT(strcmp(r.chunks[2], " ") == 0, "chunk 2 = ' '");
        ASSERT(strcmp(r.chunks[3], "world") == 0, "chunk 3 = 'world'");
        ASSERT(strcmp(r.chunks[4], "!") == 0, "chunk 4 = '!'");
        ASSERT(strcmp(r.chunks[5], " ") == 0, "chunk 5 = ' '");
        ASSERT(strcmp(r.chunks[6], "123") == 0, "chunk 6 = '123'");
    }

    /* Turkish text with mixed categories */
    const char *tr = "Merhaba!";
    memset(&r, 0, sizeof(r));
    tk_pretokenize((const uint8_t *)tr, strlen(tr), pretok_cb, &r);

    ASSERT(r.count == 2, "pretokenize 'Merhaba!' -> 2 chunks");
    if (r.count >= 2) {
        ASSERT(strcmp(r.chunks[0], "Merhaba") == 0, "chunk 0 = 'Merhaba'");
        ASSERT(strcmp(r.chunks[1], "!") == 0, "chunk 1 = '!'");
    }
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "test_unicode:\n");

    test_utf8_encode_decode();
    test_utf8_seq_len();
    test_utf8_strlen_validate();
    test_turkish_casing();
    test_classification();
    test_normalization();
    test_pretokenize();

    fprintf(stderr, "\n  %d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}