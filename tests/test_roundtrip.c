/*
 * test_roundtrip.c — Roundtrip tests: encode then decode must produce
 *                    the original bytes for any valid input.
 */

#include "tokenizer.h"
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

/* Train a tokenizer on a reasonable Turkish-ish corpus */
static tk_tokenizer_t *make_tokenizer(void) {
    static tk_tokenizer_t tk;

    const char *corpus =
        "Merhaba dünya! İstanbul'da güzel bir gün. "
        "Türkiye'nin en büyük şehri İstanbul'dur. "
        "Çocuklar okuldan eve geldiler. "
        "Öğretmenler sınıfta ders anlatıyorlar. "
        "Bugün hava çok güzel, dışarı çıkalım. "
        "Kediler ve köpekler en sevilen evcil hayvanlardır. "
        "Merhaba dünya! İstanbul'da güzel bir gün. "
        "Türkiye'nin en büyük şehri İstanbul'dur. "
        "Çocuklar okuldan eve geldiler. "
        "Öğretmenler sınıfta ders anlatıyorlar. "
        "Bugün hava çok güzel, dışarı çıkalım. "
        "Kediler ve köpekler en sevilen evcil hayvanlardır. "
        "Merhaba dünya! İstanbul'da güzel bir gün. "
        "Türkiye'nin en büyük şehri İstanbul'dur. "
        "Çocuklar okuldan eve geldiler. "
        "Öğretmenler sınıfta ders anlatıyorlar. "
        "Bugün hava çok güzel, dışarı çıkalım. "
        "Kediler ve köpekler en sevilen evcil hayvanlardır. ";

    tk_config_t config = tk_config_default();
    config.vocab_size = 300;
    config.verbose = 0;
    config.norm_flags = 0; /* no normalization — we want exact roundtrip */

    if (tk_init(&tk, &config) != 0) return NULL;
    if (tk_train(&tk, (const uint8_t *)corpus, strlen(corpus)) != 0) return NULL;

    return &tk;
}

/* Generic roundtrip test helper */
static void check_roundtrip(tk_tokenizer_t *tk, const uint8_t *input,
                            size_t len, const char *label) {
    uint32_t ids[4096];
    uint8_t decoded[4096];

    size_t num_tokens = tk_encode(tk, input, len, ids, 4096);
    if (num_tokens == (size_t)-1) {
        tests_run++;
        fprintf(stderr, "  FAIL [%d]: %s — encode failed\n", tests_run, label);
        return;
    }

    size_t dec_len = tk_decode(tk, ids, num_tokens, decoded, 4096);
    if (dec_len == (size_t)-1) {
        tests_run++;
        fprintf(stderr, "  FAIL [%d]: %s — decode failed\n", tests_run, label);
        return;
    }

    ASSERT(dec_len == len, label);
    if (dec_len == len) {
        char msg[128];
        snprintf(msg, sizeof(msg), "%s (byte content)", label);
        ASSERT(memcmp(input, decoded, len) == 0, msg);
    }
}

/* ── Test Cases ──────────────────────────────────────────────────────── */

static void test_ascii_roundtrip(tk_tokenizer_t *tk) {
    fprintf(stderr, "  ascii roundtrip...\n");

    const char *cases[] = {
        "hello",
        "Hello, World!",
        "abc123",
        "   spaces   ",
        "line1\nline2\nline3",
        "",
        "a",
        "!@#$%^&*()",
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        char label[64];
        snprintf(label, sizeof(label), "ascii roundtrip [%d]", i);
        check_roundtrip(tk, (const uint8_t *)cases[i], strlen(cases[i]), label);
    }
}

static void test_turkish_roundtrip(tk_tokenizer_t *tk) {
    fprintf(stderr, "  turkish text roundtrip...\n");

    const char *cases[] = {
        "İstanbul",
        "güzel",
        "çocuklar",
        "öğretmen",
        "şehir",
        "Türkiye",
        "ığüşöç",
        "İstanbul'da güzel bir gün.",
        "Çocuklar okuldan eve geldiler.",
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        char label[64];
        snprintf(label, sizeof(label), "turkish roundtrip [%d]", i);
        check_roundtrip(tk, (const uint8_t *)cases[i], strlen(cases[i]), label);
    }
}

static void test_raw_bytes_roundtrip(tk_tokenizer_t *tk) {
    fprintf(stderr, "  raw bytes roundtrip...\n");

    /* All 256 byte values — byte-level BPE must handle them all */
    uint8_t all_bytes[256];
    for (int i = 0; i < 256; i++) all_bytes[i] = (uint8_t)i;

    check_roundtrip(tk, all_bytes, 256, "all 256 byte values");

    /* Repeated patterns */
    uint8_t pattern[128];
    for (int i = 0; i < 128; i++) pattern[i] = (uint8_t)(i % 4);
    check_roundtrip(tk, pattern, 128, "repeating 4-byte pattern");

    /* Single bytes */
    for (int i = 0; i < 256; i += 37) { /* sample every 37th byte */
        uint8_t b = (uint8_t)i;
        char label[64];
        snprintf(label, sizeof(label), "single byte 0x%02X", i);
        check_roundtrip(tk, &b, 1, label);
    }
}

static void test_unicode_roundtrip(tk_tokenizer_t *tk) {
    fprintf(stderr, "  unicode roundtrip...\n");

    /* Multi-byte UTF-8 sequences */
    const char *cases[] = {
        "\xC4\xB0stanbul",                    /* İstanbul */
        "\xC3\xBC\xC3\xB6\xC3\xA7",         /* üöç */
        "\xE2\x82\xAC\xE2\x82\xAC",         /* €€ */
        "\xF0\x9F\x98\x80\xF0\x9F\x87\xB9\xF0\x9F\x87\xB7", /* 😀🇹🇷 */
        "mixed: abc \xC4\xB0 123 \xE2\x82\xAC end",
    };

    for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        char label[64];
        snprintf(label, sizeof(label), "unicode roundtrip [%d]", i);
        check_roundtrip(tk, (const uint8_t *)cases[i], strlen(cases[i]), label);
    }
}

static void test_long_text_roundtrip(tk_tokenizer_t *tk) {
    fprintf(stderr, "  long text roundtrip...\n");

    /* Build a ~2KB string by repeating Turkish phrases */
    const char *phrase = "Merhaba dünya, bugün hava çok güzel! ";
    size_t plen = strlen(phrase);
    size_t reps = 2048 / plen;

    size_t total = reps * plen;
    uint8_t *buf = malloc(total);
    if (!buf) { tests_run++; fprintf(stderr, "  FAIL: malloc\n"); return; }

    for (size_t i = 0; i < reps; i++) {
        memcpy(buf + i * plen, phrase, plen);
    }

    check_roundtrip(tk, buf, total, "2KB repeated Turkish text");
    free(buf);
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(void) {
    fprintf(stderr, "test_roundtrip:\n");

    tk_tokenizer_t *tk = make_tokenizer();
    if (!tk) {
        fprintf(stderr, "  FAIL: could not create tokenizer\n");
        return 1;
    }

    test_ascii_roundtrip(tk);
    test_turkish_roundtrip(tk);
    test_raw_bytes_roundtrip(tk);
    test_unicode_roundtrip(tk);
    test_long_text_roundtrip(tk);

    fprintf(stderr, "\n  %d/%d tests passed.\n", tests_passed, tests_run);

    tk_free(tk);
    return (tests_passed == tests_run) ? 0 : 1;
}